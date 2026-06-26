import os
import sys
import json
import gzip
import time
import glob
import hashlib
import logging
import threading
from dataclasses import asdict
from datetime import datetime, timezone

import bittensor as bt
import requests

from compelle.engine import (
    LLM, Elo, run_tournament, resolve_strategy, _GIST_REVISIONED_RE,
    parse_quota_reset,
)
from compelle.eligibility import fetch_records
from compelle.intent_classifier import classify_records
from compelle import provenance


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("compelle.validator")

# `or "data"` rather than the dict default so an empty env var (operator sets
# `COMPELLE_DATA_DIR=` with no value) doesn't silently resolve to "" and write
# state files to the filesystem root.
DATA_DIR = os.environ.get("COMPELLE_DATA_DIR") or "data"
# Local-only health snapshot for operator monitoring. Written atomically once
# per loop. Override the path via COMPELLE_HEALTH_PATH; default sits next to
# the epoch archive in DATA_DIR.
HEALTH_PATH = os.environ.get("COMPELLE_HEALTH_PATH") or f"{DATA_DIR}/health.json"
# Canonical R2 ingest endpoint. Used as the fallback when neither
# COMPELLE_PUSH_URL nor config.json's push_url is set to a non-empty value.
DEFAULT_PUSH_URL = "https://compelle-ingest.compelle.workers.dev/ingest"
# 2hr default: with Chutes timeout=120s and up to 11 LLM calls per game, a
# slow-Chutes tournament can plausibly run >30 min even at 10 miners. Tournament
# work doesn't pulse the heartbeat, so a tight watchdog false-fires mid-game.
WATCHDOG_TIMEOUT_SECONDS = int(os.environ.get("COMPELLE_WATCHDOG_SECONDS") or "7200")


def load_config() -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "config.json")) as f:
        cfg = json.load(f)
    cfg["wallet_name"] = os.environ.get("BT_WALLET_NAME", cfg.get("wallet_name", ""))
    cfg["hotkey"] = os.environ.get("BT_HOTKEY", cfg.get("hotkey", ""))
    cfg["netuid"] = int(os.environ.get("BT_NETUID", cfg.get("netuid", 82)))
    cfg["network"] = os.environ.get("BT_NETWORK", cfg.get("network", "finney"))
    cfg["chutes_api_url"] = os.environ.get(
        "CHUTES_BASE_URL", cfg.get("chutes_api_url", "https://llm.chutes.ai/v1")
    )
    # Push URL precedence: non-empty env > non-empty config > hardcoded default.
    # An empty env var no longer disables uploading — that was a footgun where
    # operators using a clean .env file would silently stop contributing
    # transcripts to the public archive. To explicitly opt out, set
    # COMPELLE_PUSH_URL=disabled (or set push_url to "disabled" in config.json).
    env_push = os.environ.get("COMPELLE_PUSH_URL")
    cfg_push = cfg.get("push_url")
    cfg["push_url"] = (env_push or cfg_push or DEFAULT_PUSH_URL).strip()
    return cfg


def get_active_config(cfg: dict, block: int) -> dict:
    nb = cfg.get("new_config_start_block")
    if cfg.get("new_config") and nb is not None and block >= int(nb):
        return cfg["new_config"]
    return cfg["old_config"]


ROTATABLE_KEYS = {"topics"}
MAX_TOPICS = 100
MAX_TOPIC_BYTES = 4000


VALID_FRAMINGS = {"direct", "probability", "market_trajectory"}


def _validate_topics(topics) -> bool:
    if not isinstance(topics, list) or not (1 <= len(topics) <= MAX_TOPICS):
        log.error(f"gist topics invalid: must be a list of 1..{MAX_TOPICS} objects")
        return False
    for t in topics:
        if not isinstance(t, dict) or not isinstance(t.get("motion"), str):
            log.error("gist topic missing required 'motion' string field")
            return False
        if len(json.dumps(t).encode("utf-8")) > MAX_TOPIC_BYTES:
            log.error(f"gist topic exceeds {MAX_TOPIC_BYTES} bytes")
            return False
        if t.get("framing") and t["framing"] not in VALID_FRAMINGS:
            log.error(f"gist topic framing must be one of {VALID_FRAMINGS} or omitted")
            return False
    return True


def fetch_config(gist_id: str, expected_owner: str, current_block: int) -> tuple[dict | None, str]:
    try:
        headers = {}
        token = os.environ.get("GITHUB_TOKEN", "").strip()
        if token:
            headers["Authorization"] = f"token {token}"
        r = requests.get(f"https://api.github.com/gists/{gist_id}", headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        revision = (data.get("history") or [{}])[0].get("version", "")
        owner = (data.get("owner") or {}).get("login", "")
        if expected_owner and owner != expected_owner:
            log.error(f"gist owner mismatch: got {owner!r}, expected {expected_owner!r}")
            return None, revision
        files = data.get("files") or {}
        if not files:
            return None, revision
        # Reject ambiguous multi-file config gists for the same reason
        # resolve_strategy does: dict-iteration order is not stable
        # cross-validator consensus.
        if len(files) > 1:
            log.error(f"config gist has {len(files)} files (must be single-file)")
            return None, revision
        content = next(iter(files.values())).get("content", "") or ""
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            log.error("gist content not a JSON object")
            return None, revision
        if int(parsed.get("min_block", 0)) > current_block:
            return None, revision
        if "topics" in parsed and not _validate_topics(parsed["topics"]):
            return None, revision
        overrides = {k: v for k, v in parsed.items() if k in ROTATABLE_KEYS}
        return overrides, revision
    except Exception as e:
        log.error(f"config gist fetch failed: {e}")
    return None, ""


def _within_rate_limit(sub, netuid: int, my_uid: int) -> int:
    """Return blocks remaining until set_weights is allowed again, or 0 if free.

    The chain enforces SubtensorModule::WeightsRateLimit between successive
    set_weights / commit_weights calls per UID. Calling within the window
    silently fails with success=False / message=None — useless for diagnosis
    and burns retries. Query LastUpdate + WeightsRateLimit and skip preemptively.
    """
    try:
        last_update = sub.substrate.query("SubtensorModule", "LastUpdate", [netuid]).value
        rate_limit = sub.weights_rate_limit(netuid)
        current = sub.get_current_block()
    except Exception as e:
        log.warning(f"rate-limit pre-check failed: {e}; will attempt anyway")
        return 0
    if not last_update or my_uid >= len(last_update) or rate_limit is None:
        return 0
    elapsed = current - int(last_update[my_uid])
    if elapsed >= int(rate_limit):
        return 0
    return int(rate_limit) - elapsed


def set_weights(sub, wallet, netuid, uids, vals) -> bool:
    try:
        my_uid = sub.substrate.query(
            "SubtensorModule", "Uids",
            [netuid, wallet.hotkey.ss58_address]
        ).value
    except Exception:
        my_uid = None
    if my_uid is not None:
        wait = _within_rate_limit(sub, netuid, my_uid)
        if wait > 0:
            log.info(f"skip set_weights: chain rate-limit, "
                     f"{wait} blocks remaining (~{wait * 12}s)")
            return False
    for attempt in range(3):
        # Re-query weights_version_key per attempt — chain rotates it under
        # commit-reveal and a stale value gets the extrinsic rejected.
        try:
            wvk = sub.substrate.query("SubtensorModule", "WeightsVersionKey", [netuid]).value
        except Exception:
            wvk = None
        extra = {"version_key": wvk} if wvk is not None else {}
        try:
            # bittensor 10.x set_weights handles commit-reveal automatically. Don't
            # block on reveal execution — the reveal happens in a future epoch and
            # we only need the commit phase to land THIS epoch. Without this flag
            # the call hangs waiting for reveal, eventually times out as "not-ok"
            # even though the commit succeeded.
            resp = sub.set_weights(wallet=wallet, netuid=netuid, uids=uids, weights=vals,
                                   wait_for_inclusion=True,
                                   wait_for_revealed_execution=False,
                                   **extra)
            # bittensor 10.x returns a tuple (success, message) or an
            # ExtrinsicResponse with .success when raise_error=False. Treat
            # falsy and "success=False" both as failure.
            ok = True
            msg = ""
            if isinstance(resp, tuple) and len(resp) >= 1:
                ok, msg = bool(resp[0]), (str(resp[1]) if len(resp) > 1 else "")
            elif hasattr(resp, "success"):
                ok, msg = bool(resp.success), str(getattr(resp, "message", "") or "")
            if ok:
                log.info(f"set_weights ok: {len(uids)} uids (version_key={wvk}) {msg}")
                return True
            log.warning(f"set_weights attempt {attempt + 1} returned not-ok: {msg or resp}")
        except Exception as e:
            log.error(f"set_weights attempt {attempt + 1} threw: {e}")
        if attempt < 2:
            time.sleep(5)
    log.error("set_weights FAILED after 3 attempts; epoch weights NOT on chain")
    return False


def write_epoch(epoch, epoch_block, topics_revision, records, weights, results, elo,
                set_weights_status, real_strategies=None):
    os.makedirs(DATA_DIR, exist_ok=True)
    from compelle import FULL_VERSION
    # All games in a tournament share one topic now, so surface it at the top.
    tournament_topic = None
    if results:
        first = results[0][2]
        tournament_topic = {
            "id": getattr(first, "topic_id", ""),
            "motion": first.topic,
        }
    # `played` is the source of truth for "actually entered the tournament":
    # tier=real reflects only is_real (timing + non-epsilon + non-vdata) and
    # silently lets through hotkeys whose gist commitment doesn't resolve. A
    # downstream consumer reading the archive shouldn't have to derive that
    # by intersecting `tier` with `weight>0` and Elo state.
    played_set = set((real_strategies or {}).keys())
    payload = {
        "epoch": epoch,
        "epoch_block": epoch_block,
        "config_revision": topics_revision,
        "validator_version": FULL_VERSION,
        "tournament_topic": tournament_topic,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "set_weights_status": set_weights_status,
        "miners": {
            hk: {
                "uid": r.uid,
                "tier": (
                    "real" if r.is_real_and_clean
                    else "intent_locked" if (r.is_real and r.intent_verdict == "BAD")
                    else "epsilon" if r.is_placeholder
                    else "ineligible"
                ),
                "intent_verdict": r.intent_verdict,
                "played": hk in played_set,
                "elo": elo.ratings.get(hk),
                "weight": weights.get(hk, 0.0),
            }
            for hk, r in records.items()
        },
        "results": [{"pro": pro, "con": con, **asdict(r)} for pro, con, r in results],
    }
    path = f"{DATA_DIR}/epoch_{epoch_block:010d}.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f)


def prune_epochs(keep: int):
    if keep <= 0:
        return
    for old in sorted(glob.glob(f"{DATA_DIR}/epoch_*.json.gz"))[:-keep]:
        try:
            os.remove(old)
        except OSError:
            pass


PUSHED_PATH = f"{DATA_DIR}/pushed.json"


def _block_from_filename(name: str) -> int | None:
    s = name.removeprefix("epoch_").removesuffix(".json.gz")
    return int(s) if s.isdigit() else None


def push_startup_ping(push_url: str, wallet, block: int) -> None:
    """At validator startup, push a tiny `kind=startup` payload to /ingest
    so the backend can record `(hotkey, version, ts)` without waiting for
    the next epoch's tournament-completion push. Non-blocking, swallows
    all errors — version visibility is observability, not a hard requirement.

    Filename uses unix_ts (10 digits) as the identifier rather than chain
    block, to avoid R2-key collisions with regular per-tournament epoch
    files (which use 7-8-digit chain blocks padded to 10). Same cf-worker
    `epoch_\\d{10}.json.gz` regex accepts both; the indexer branches on
    payload.kind to route into the right table.
    """
    if not push_url or push_url.lower() in ("disabled", "off", "none"):
        return
    from compelle import FULL_VERSION
    now_ts = int(time.time())
    payload = {
        "kind": "startup",
        "validator_version": FULL_VERSION,
        "ts": now_ts,
        "epoch_block": int(block),    # chain block at startup, informational
    }
    body = gzip.compress(json.dumps(payload).encode("utf-8"))
    name = f"epoch_{now_ts:010d}.json.gz"
    try:
        sig = wallet.hotkey.sign(hashlib.sha256(body).digest()).hex()
        r = requests.post(push_url, data=body, headers={
            "User-Agent": "compelle-validator/0.1",
            "Content-Type": "application/gzip",
            "X-Compelle-Hotkey": wallet.hotkey.ss58_address,
            "X-Compelle-Signature": sig,
            "X-Compelle-Filename": name,
        }, timeout=10)
    except Exception as e:
        log.warning(f"startup ping exception (non-fatal): {e}")
        return
    if r.status_code >= 400:
        log.warning(f"startup ping {name} -> {r.status_code}: {r.text[:200]}")
        return
    log.info(f"startup ping ok: version={FULL_VERSION} block={block}")


def push_pending(push_url: str, wallet) -> None:
    if not push_url or push_url.lower() in ("disabled", "off", "none"):
        return
    on_disk = {}
    for path in sorted(glob.glob(f"{DATA_DIR}/epoch_*.json.gz")):
        block = _block_from_filename(os.path.basename(path))
        if block is not None:
            on_disk[block] = path
    try:
        with open(PUSHED_PATH) as f:
            pushed = {int(b) for b in json.load(f)} & set(on_disk)
    except (OSError, ValueError):
        pushed = set()
    pending = sorted(set(on_disk) - pushed)
    if not pending:
        return
    log.info(f"push: {len(pending)} pending")
    hotkey_addr = wallet.hotkey.ss58_address
    for block in pending:
        path = on_disk[block]
        name = os.path.basename(path)
        try:
            with open(path, "rb") as f:
                body = f.read()
        except OSError as e:
            log.warning(f"push read failed {path}: {e}")
            continue
        sig = wallet.hotkey.sign(hashlib.sha256(body).digest()).hex()
        try:
            r = requests.post(push_url, data=body, headers={
                "User-Agent": "compelle-validator/0.1",
                "Content-Type": "application/gzip",
                "X-Compelle-Hotkey": hotkey_addr,
                "X-Compelle-Signature": sig,
                "X-Compelle-Filename": name,
            }, timeout=60)
        except Exception as e:
            log.warning(f"push {name} exception: {e}; will retry next epoch")
            break
        if r.status_code >= 400:
            log.warning(f"push {name} -> {r.status_code}: {r.text[:200]}; will retry next epoch")
            break
        pushed.add(block)
        log.info(f"push {name} ok")
    try:
        with open(PUSHED_PATH, "w") as f:
            json.dump(sorted(pushed), f)
    except OSError as e:
        log.warning(f"pushed.json save failed: {e}")


def _count_pending_pushes() -> int:
    on_disk = set()
    for path in glob.glob(f"{DATA_DIR}/epoch_*.json.gz"):
        block = _block_from_filename(os.path.basename(path))
        if block is not None:
            on_disk.add(block)
    try:
        with open(PUSHED_PATH) as f:
            pushed = {int(b) for b in json.load(f)} & on_disk
    except (OSError, ValueError):
        pushed = set()
    return len(on_disk - pushed)


def write_health(epoch: int, last_progress_ts: float, current_block: int,
                 last_sw_block, last_sw_status, last_sw_ts,
                 chutes_quota_reset_at=None) -> None:
    """Atomically write a snapshot of validator state for operator monitoring.

    Strictly local file. No HTTP, no network. Operators can `cat
    data/health.json` to debug or wire up their own monitoring however they
    like. Override the path via COMPELLE_HEALTH_PATH if needed.
    """
    try:
        from compelle import FULL_VERSION
        os.makedirs(os.path.dirname(HEALTH_PATH) or ".", exist_ok=True)
        now = int(time.time())
        snapshot = {
            "ts": now,
            "version": FULL_VERSION,
            "epoch": epoch,
            "current_block": current_block,
            "last_progress_age_s": max(0, now - int(last_progress_ts)),
            "last_set_weights_block": last_sw_block,
            "last_set_weights_status": last_sw_status,
            "last_set_weights_at": int(last_sw_ts) if last_sw_ts else None,
            "pending_pushes": _count_pending_pushes(),
            "chutes_quota_reset_at": chutes_quota_reset_at,
        }
        tmp = HEALTH_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(snapshot, f, indent=2)
        os.replace(tmp, HEALTH_PATH)
    except Exception as e:
        log.warning(f"health.json write failed (non-fatal): {e}")


ELO_STATE_PATH = f"{DATA_DIR}/elo_state.json"
LAST_TEMPO_PATH = f"{DATA_DIR}/last_completed_tempo.txt"


def save_elo(elo) -> None:
    """Persist Elo ratings so a restart resumes from last state."""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        tmp = ELO_STATE_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"k": elo.k, "initial": elo.initial, "ratings": elo.ratings}, f)
        os.replace(tmp, ELO_STATE_PATH)
    except Exception as e:
        log.warning(f"elo save failed: {e}")


def prune_stale_elo(elo, records) -> int:
    """Drop Elo entries whose hotkey is unreachable from this epoch's chain
    state. Two cuts, both safe:

      1. Hotkey not in current metagraph records (deregistered). The slot may
         later be re-registered to a different operator on the same UID — we
         must not carry their predecessor's rating forward.
      2. Hotkey is registered but its current commitment is structurally
         malformed (gist:<id>/<rev> regex rejection). This is a deterministic
         rejection, not a transient fetch failure, so it's safe to act on.
         If the miner repairs their commitment, they start fresh at `initial`.

    Hotkeys whose gist passes the regex but fails to fetch (e.g., transient
    GitHub 403) are NOT pruned — their stored rating is held in case the next
    fetch succeeds. The post-tournament filter already excludes them from
    weights for this epoch via real_strategies.
    """
    to_remove = []
    for hk in list(elo.ratings.keys()):
        if hk not in records:
            to_remove.append(hk)
            continue
        text = (records[hk].commitment_text or "").strip()
        if text.startswith("gist:") and not _GIST_REVISIONED_RE.match(text):
            to_remove.append(hk)
    for hk in to_remove:
        del elo.ratings[hk]
    return len(to_remove)


def load_elo(default_k: float, default_initial: float):
    """Load persisted Elo ratings; if missing/corrupt, return fresh."""
    try:
        with open(ELO_STATE_PATH) as f:
            d = json.load(f)
        e = Elo(k=d.get("k", default_k), initial=d.get("initial", default_initial))
        e.ratings = dict(d.get("ratings", {}))
        log.info(f"loaded persisted Elo: {len(e.ratings)} hotkeys")
        return e
    except FileNotFoundError:
        return Elo(k=default_k, initial=default_initial)
    except Exception as e:
        log.warning(f"elo load failed ({e}); starting fresh")
        return Elo(k=default_k, initial=default_initial)


KOTH_DEFENSE_MARGIN = 200.0  # a contender must lead the king's Elo by this much,
KOTH_DETHRONE_EPOCHS = 20    # and hold that lead this many consecutive epochs (~1 day), to be crowned
KOTH_STATE_PATH = f"{DATA_DIR}/koth_state.json"


def _load_koth() -> dict:
    """King-of-the-hill streak state: {tempo, challenger, streak}. `tempo` is the
    tempo index the streak was last advanced for, so a mid-tempo restart reuses
    the stored decision instead of double-counting the same epoch."""
    try:
        with open(KOTH_STATE_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"tempo": -1, "challenger": None, "streak": 0}
    except Exception as e:
        log.warning(f"koth state load failed ({e}); starting fresh")
        return {"tempo": -1, "challenger": None, "streak": 0}


def _save_koth(state: dict) -> None:
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        tmp = KOTH_STATE_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, KOTH_STATE_PATH)
    except Exception as e:
        log.warning(f"koth state save failed: {e}")


def _incumbent_king(sub, netuid):
    """Hotkey with the highest consensus weight last epoch = the reigning king.
    Read from chain so every validator anchors to the SAME incumbent (consensus
    is shared state; local Elo is not). Returns None only on a chain-read
    failure; koth_weights then holds the prior on-chain weights rather than
    crowning a divergent local leader."""
    try:
        mg = sub.metagraph(netuid, lite=True)
        top_uid = max(range(len(mg.consensus)), key=lambda i: mg.consensus[i])
        return list(mg.hotkeys)[top_uid]
    except Exception as e:
        log.warning(f"KOTH incumbent read failed: {e}")
        return None


def koth_weights(sub, netuid, elo, eligible, current_tempo,
                 margin=KOTH_DEFENSE_MARGIN, hold=KOTH_DETHRONE_EPOCHS):
    """Winner-take-all king of the hill.

    King = highest consensus weight last epoch. A challenger is crowned only if it
    holds Elo >= king + 200 for one day (20 consecutive epochs). The crowned king
    takes literally 100%.

    The streak is keyed on hotkey, never on UID: UIDs recycle on deregistration,
    so a fresh miner could otherwise inherit a dead slot's streak. It resets to
    zero on any single epoch the top contender falls below the line, or if the
    contender's hotkey changes. A miner can't swap its strategy mid-climb to
    launder a streak: eligibility requires the commitment to land within
    ELIGIBILITY_WINDOW_BLOCKS of registration (see eligibility.py), so a later
    swap pushes the hotkey out of `eligible` entirely, which resets its streak by
    the identity rule, rather than carrying it forward under a new strategy.

    King is read from chain so every validator anchors to the SAME incumbent
    (consensus is shared; local Elo is not). On a consensus-read failure the king
    is unknown, so we return {} and let the prior on-chain weights stand; we never
    fall back to the local Elo leader, which would have each validator crown its
    own argmax and diverge.

    The streak advances at most once per tempo: `current_tempo` is checked against
    the persisted tempo so a mid-tempo restart (which re-sets weights for an
    already-processed tempo) does not re-increment it. The downstream
    real_strategies filter still applies, so a defending king that isn't competing
    this epoch drops to empty weights and the prior weights stand.
    """
    incumbent = _incumbent_king(sub, netuid)
    if incumbent is None:
        return {}
    king_elo = elo.ratings.get(incumbent, elo.initial)
    # Top contender = highest-Elo eligible miner other than the reigning king.
    rated = [hk for hk in eligible if hk in elo.ratings and hk != incumbent]
    contender = max(rated, key=lambda hk: (elo.ratings[hk], hk)) if rated else None
    above = contender is not None and elo.ratings[contender] >= king_elo + margin

    state = _load_koth()
    if state.get("tempo") != current_tempo:  # new tempo: advance the streak once
        if above and state.get("challenger") == contender:
            state["streak"] = int(state.get("streak", 0)) + 1
        elif above:  # new challenger hotkey -> fresh streak at 1
            state.update(challenger=contender, streak=1)
        else:        # nobody above the line this epoch -> reset
            state.update(challenger=None, streak=0)
        state["tempo"] = current_tempo
        _save_koth(state)

    if (above and state.get("challenger") == contender
            and state.get("streak", 0) >= hold):
        log.info(f"KOTH: {contender} dethrones {incumbent} "
                 f"(led by >= {margin:.0f} Elo for {state['streak']} epochs)")
        return {contender: 1.0}
    return {incumbent: 1.0}


def get_last_tempo() -> int:
    """Return the highest tempo_index already processed, or -1 if never."""
    try:
        with open(LAST_TEMPO_PATH) as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return -1


def set_last_tempo(tempo_index: int) -> None:
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        tmp = LAST_TEMPO_PATH + ".tmp"
        with open(tmp, "w") as f:
            f.write(str(tempo_index))
        os.replace(tmp, LAST_TEMPO_PATH)
    except Exception as e:
        log.warning(f"last_tempo save failed: {e}")


def _preflight(cfg: dict) -> tuple[bt.Wallet, str]:
    """Validate startup requirements. Exits with code 2 on misconfiguration so
    the operator sees a clear error in journalctl instead of a cryptic stack
    trace from systemd's restart loop."""
    api_key = os.environ.get("CHUTES_API_KEY", "").strip()
    if not api_key:
        log.error("preflight: CHUTES_API_KEY missing or empty")
        sys.exit(2)
    if not cfg.get("wallet_name") or not cfg.get("hotkey"):
        log.error("preflight: BT_WALLET_NAME and BT_HOTKEY must be set")
        sys.exit(2)
    try:
        wallet = bt.Wallet(name=cfg["wallet_name"], hotkey=cfg["hotkey"])
        # Force-load the hotkey from disk so a missing/corrupt key fails now,
        # not 90 minutes later inside push_pending().
        addr = wallet.hotkey.ss58_address
    except Exception as e:
        log.error(f"preflight: wallet load failed ({cfg['wallet_name']}/{cfg['hotkey']}): {e}")
        sys.exit(2)
    log.info(f"preflight ok: wallet={cfg['wallet_name']}/{cfg['hotkey']} ss58={addr}")
    return wallet, api_key


def _start_watchdog(last_progress: list) -> None:
    """Kill the process if the main loop hasn't made progress in
    WATCHDOG_TIMEOUT_SECONDS. Substrate calls have no socket-level timeout,
    so a wedged Finney peer can block forever; systemd Restart=always then
    gives us a fresh Subtensor connection."""
    def _tick():
        while True:
            time.sleep(60)
            stalled = time.time() - last_progress[0]
            if stalled > WATCHDOG_TIMEOUT_SECONDS:
                log.error(f"watchdog: no progress in {stalled:.0f}s, exiting for restart")
                os._exit(3)
    threading.Thread(target=_tick, daemon=True).start()


def _heartbeat_sleep(last_progress: list, total_secs: float) -> None:
    """Sleep total_secs while pulsing the watchdog every 60s."""
    end = time.time() + total_secs
    while True:
        remaining = end - time.time()
        if remaining <= 0:
            return
        last_progress[0] = time.time()
        time.sleep(min(60.0, remaining))


def _next_cycle_sleep(elapsed: float, epoch_seconds: float,
                       sw_failed: bool) -> float:
    """Return how long to sleep before the next epoch loop iteration.

    Goal: total cycle length (tournament + sleep) equals one subnet tempo
    (`epoch_seconds`). Without this, a 40-min tournament + 72-min sleep gives
    a 112-min cycle, which is longer than the SN82 tempo (72 min) — so the
    chain ends up using stale weights for whole tempos. By filling the cycle
    instead of padding it, every tempo gets a fresh set_weights call.

    Edge cases:
      - `sw_failed`: short retry window so we recover within the same tempo.
      - elapsed >= epoch_seconds (tournament longer than a tempo): clamp to 0
        and start the next cycle immediately. Skipping tempos in this regime
        is unavoidable without making the tournament itself faster.
    """
    if sw_failed:
        return 60.0
    return max(0.0, epoch_seconds - elapsed)


def main():
    cfg = load_config()
    wallet, api_key = _preflight(cfg)
    netuid = cfg["netuid"]
    llm = LLM(cfg["chutes_api_url"], api_key,
              thinking_models=(cfg.get("old_config") or cfg)["game"].get("thinking_models", []))
    elo = load_elo(
        default_k=cfg["old_config"]["elo"]["k_factor"],
        default_initial=cfg["old_config"]["elo"]["initial_rating"],
    )

    last_progress = [time.time()]
    _start_watchdog(last_progress)

    from compelle import FULL_VERSION
    log.info(f"compelle validator {FULL_VERSION} starting on netuid {netuid} ({cfg['network']})")

    # Startup ping to /ingest so the backend sees the new version within seconds
    # rather than waiting for the next epoch's payload push. Fresh Subtensor
    # (closed right away) so this doesn't tangle with the per-epoch one. Non-fatal.
    try:
        _vsub = bt.Subtensor(network=cfg["network"])
        try:
            try:
                _block = int(_vsub.get_current_block())
            except Exception:
                _block = 0
            push_startup_ping(cfg["push_url"], wallet, _block)
        finally:
            try: _vsub.close()
            except Exception: pass
    except Exception as e:
        log.warning(f"startup ping skipped (chain unreachable at startup): {e}")

    gist_id = cfg.get("config_gist_id", "")
    gist_owner = cfg.get("config_gist_owner", "")
    keep_epochs = int(cfg.get("keep_epochs", 1000))
    cached_overrides: dict = {}
    cached_revision = "bundled"
    epoch = 0
    # Most recent set_weights attempt — surfaced via data/health.json so
    # operators can monitor "is mine stuck?" without scraping logs.
    last_sw_block = None
    last_sw_status = None
    last_sw_ts = None
    # Most recent Chutes 402 reset timestamp (ISO string) — populated when the
    # epoch preflight detects quota exhaustion; surfaces in health.json so
    # operators can see "next inference window opens at..." without log scraping.
    last_chutes_quota_reset_at = None

    while True:
        cycle_start = time.time()
        last_progress[0] = cycle_start
        epoch += 1
        log.info(f"=== epoch {epoch} ===")

        # Fresh Subtensor per epoch dodges stuck-socket hangs, but the websocket
        # must be released or it accumulates against the public RPC endpoint and
        # eventually trips per-IP 429s. Close at every exit path below.
        sub = None
        try:
            sub = bt.Subtensor(network=cfg["network"])
            epoch_start_block = sub.get_current_block()
            block_hash = sub.substrate.get_block_hash(epoch_start_block)
            records = fetch_records(sub, netuid, block=epoch_start_block, block_hash=block_hash)
        except Exception as e:
            log.error(f"chain unreachable: {e}; retry 60s")
            if sub is not None:
                try: sub.close()
                except Exception: pass
            time.sleep(60)
            epoch -= 1
            continue

        try:
            cfg = load_config()
        except Exception as e:
            log.warning(f"config reload failed, using last good: {e}")

        active = get_active_config(cfg, epoch_start_block)
        cfg.update(active)

        if gist_id:
            overrides, rev = fetch_config(gist_id, gist_owner, epoch_start_block)
            if overrides:
                cached_overrides, cached_revision = overrides, rev
        cfg.update(cached_overrides)

        elo.k = cfg["elo"]["k_factor"]
        elo.initial = cfg["elo"]["initial_rating"]

        which = "new" if active is cfg.get("new_config") else "old"
        log.info(f"config: {which} ({len(cfg.get('topics', []))} topics, "
                 f"gist_revision={cached_revision[:8]})")

        # Commit-time intent classifier — pluggable, feature-flagged via cfg.
        intent_cfg = cfg.get("intent_classifier") or {}
        if intent_cfg.get("enabled", True):
            try:
                intent_results = classify_records(
                    records,
                    llm=llm,
                    resolve_fn=resolve_strategy,
                    panel=intent_cfg.get("models"),
                    max_tokens=int(intent_cfg.get("max_tokens", 8192)),
                    temperature=float(intent_cfg.get("temperature", 0.1)),
                    parallel_workers=int(intent_cfg.get("parallel_workers", 6)),
                    uid_parallel_workers=int(intent_cfg.get("uid_parallel_workers", 6)),
                    thinking_tags=cfg.get("thinking_tags"),
                )
                for hk, res in intent_results.items():
                    records[hk].intent_verdict = res.verdict
                n_bad = sum(1 for r in records.values() if r.intent_verdict == "BAD")
                n_good = sum(1 for r in records.values() if r.intent_verdict == "GOOD")
                n_pending = sum(1 for r in records.values() if r.is_real and r.intent_verdict == "PENDING")
                log.info(f"intent classifier: {n_good} good, {n_bad} bad, {n_pending} pending")
                for hk, res in intent_results.items():
                    if res.is_bad:
                        log.warning(f"intent BAD: uid={records[hk].uid} hk={hk[:12]} reasons={[v.reason for v in res.votes]}")
            except Exception as e:
                log.warning(f"intent classifier failed at epoch level: {e}; treating all as PENDING")

        real_strategies = {
            hk: resolve_strategy(r.commitment_text)
            for hk, r in records.items() if r.is_real_and_clean
        }
        real_strategies = {hk: t for hk, t in real_strategies.items() if t.strip()}

        if provenance.enabled():
            for hk, r in records.items():
                try:
                    resolved = real_strategies.get(hk)
                    if resolved is None:
                        resolved = resolve_strategy(r.commitment_text)
                    provenance.log_miner(epoch_start_block, r, resolved)
                except Exception:
                    pass

        n_real = len(real_strategies)
        n_eps = sum(1 for r in records.values() if r.is_placeholder)
        n_zero = sum(1 for r in records.values() if not r.is_eligible)
        n_intent_locked = sum(1 for r in records.values() if r.is_real and r.intent_verdict == "BAD")
        log.info(f"miners: {n_real} real, {n_eps} epsilon, {n_zero} ineligible, {n_intent_locked} intent-locked")

        # Idempotency: skip the tournament if we already completed THIS tempo.
        # Prevents double-applying Elo if the validator restarts mid-tempo.
        # Topic-index uses the same tempo computation as engine.run_tournament.
        tempo_blocks = cfg.get("tempo_blocks", 360)
        current_tempo = epoch_start_block // tempo_blocks
        last_tempo = get_last_tempo()

        results = []
        real_weights: dict[str, float] = {}
        if current_tempo <= last_tempo:
            log.info(f"tempo {current_tempo} already processed (last={last_tempo}); "
                     f"skipping tournament, reusing existing Elo for weights")
            real_weights = koth_weights(sub, netuid, elo, real_strategies, current_tempo)
        elif n_real >= 2:
            # Preflight the whole failover chain, not just the primary: a 429 on
            # the primary alone must not skip the epoch when a fallback is
            # reachable. Only a total outage (every model down) skips the round;
            # per-game _chat then routes within the tournament.
            chain = [cfg["game"]["model"], *cfg["game"].get("model_fallbacks", [])]
            ok, working, err = llm.ping_chain(chain)
            if not ok:
                log.warning(f"LLM preflight failed (all {len(chain)} models down): {err[:200]}")
                # Surface Chutes 402 quota reset timestamps in health.json so the
                # operator sees when inference resumes. We don't sleep here; the
                # next epoch loop will naturally re-preflight.
                reset_iso = parse_quota_reset(err) if "402" in err else None
                if reset_iso:
                    last_chutes_quota_reset_at = reset_iso
                    log.error(f"Chutes quota exhausted; reset_at={reset_iso}")
            else:
                if working != chain[0]:
                    log.warning(f"LLM preflight: primary {chain[0]} unavailable; "
                                f"running this epoch on fallback {working}")
                results, elo = run_tournament(
                    llm, cfg, real_strategies, epoch_start_block, elo=elo,
                    # Pulse the watchdog after each completed game so a long
                    # tournament (many miners × many turns) doesn't false-kill
                    # under a fixed-wall-clock watchdog timeout.
                    on_progress=lambda: last_progress.__setitem__(0, time.time()),
                )
                if results:
                    real_weights = koth_weights(sub, netuid, elo, real_strategies, current_tempo)
                    # Mark tempo BEFORE saving Elo: a crash between the two writes
                    # is preferable in this order — on restart we'd skip the
                    # tournament (stale Elo) instead of double-counting matches.
                    set_last_tempo(current_tempo)
                    n_pruned = prune_stale_elo(elo, records)
                    if n_pruned:
                        log.info(f"pruned {n_pruned} stale Elo entries (deregistered or malformed gist)")
                    save_elo(elo)

        # Drop hotkeys whose CURRENT commitment doesn't resolve, even if they
        # carry stale Elo from earlier epochs. real_strategies (built above) is
        # already filtered to non-empty resolutions, so it's the canonical set
        # of "currently usable" miners. Without this guard, a miner that earned
        # Elo under a previous (looser) regex keeps getting weight after their
        # commitment is rejected.
        real_weights = {hk: w for hk, w in real_weights.items()
                        if hk in real_strategies}
        weights = dict(real_weights)  # KOTH: king takes literally 100%; no epsilon carve-out
        if weights:
            uids = [records[hk].uid for hk in weights]
            vals = list(weights.values())
            sw_status = "succeeded" if set_weights(sub, wallet, netuid, uids, vals) else "failed"
        else:
            # No king to weight this epoch: chain consensus was unreadable, or the
            # reigning king isn't a currently-competing real miner. Leave the prior
            # epoch's weights standing on chain rather than zeroing them out.
            log.info("no weights to set this epoch")
            sw_status = "skipped"
        last_progress[0] = time.time()  # set_weights / chain ops finished, watchdog should see it
        last_sw_block = epoch_start_block
        last_sw_status = sw_status
        last_sw_ts = time.time()

        write_epoch(epoch, epoch_start_block, cached_revision, records, weights, results, elo,
                    sw_status, real_strategies=real_strategies)
        prune_epochs(keep_epochs)
        push_pending(cfg["push_url"], wallet)
        write_health(epoch, last_progress[0], epoch_start_block,
                     last_sw_block, last_sw_status, last_sw_ts,
                     chutes_quota_reset_at=last_chutes_quota_reset_at)

        # On set_weights failure, retry within the same tempo by waking up
        # quickly. The next iteration will see current_tempo <= last_tempo,
        # skip the tournament, recompute weights from cached Elo, and retry
        # the chain submission. Once the tempo rolls over those weights are
        # gone, so a tight retry window is the only way to recover.
        elapsed = time.time() - cycle_start
        sleep_secs = _next_cycle_sleep(elapsed, cfg["tournament"]["epoch_seconds"],
                                        sw_status == "failed")
        log.info(f"epoch {epoch} done; next in {sleep_secs:.0f}s "
                 f"(cycle elapsed {elapsed:.0f}s)")
        try: sub.close()
        except Exception as e: log.warning(f"subtensor close failed: {e}")
        _heartbeat_sleep(last_progress, sleep_secs)


if __name__ == "__main__":
    main()
