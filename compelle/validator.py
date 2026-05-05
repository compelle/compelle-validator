import os
import json
import gzip
import time
import glob
import hashlib
import logging
from dataclasses import asdict
from datetime import datetime, timezone

import bittensor as bt
import requests

from compelle.engine import LLM, Elo, run_tournament, resolve_strategy
from compelle.eligibility import fetch_records, assign_weights


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("compelle.validator")

DATA_DIR = "data"


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
    cfg["push_url"] = os.environ.get("COMPELLE_PUSH_URL", cfg.get("push_url", ""))
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
        r = requests.get(f"https://api.github.com/gists/{gist_id}", timeout=10)
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
        content = next(iter(files.values())).get("content", "")
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


def set_weights(sub, wallet, netuid, uids, vals) -> bool:
    try:
        wvk = sub.substrate.query("SubtensorModule", "WeightsVersionKey", [netuid]).value
    except Exception:
        wvk = None
    extra = {"version_key": wvk} if wvk is not None else {}
    for attempt in range(3):
        try:
            sub.set_weights(wallet=wallet, netuid=netuid, uids=uids, weights=vals,
                            wait_for_inclusion=True, **extra)
            log.info(f"set_weights ok: {len(uids)} uids (version_key={wvk})")
            return True
        except Exception as e:
            log.error(f"set_weights attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(5)
    log.error("set_weights FAILED after 3 attempts; epoch weights NOT on chain")
    return False


def write_epoch(epoch, epoch_block, topics_revision, records, weights, results, elo,
                set_weights_status):
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
                "tier": "real" if r.is_real else ("epsilon" if r.is_placeholder else "ineligible"),
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


def push_pending(push_url: str, wallet) -> None:
    if not push_url:
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


def main():
    cfg = load_config()
    api_key = os.environ["CHUTES_API_KEY"]
    wallet = bt.Wallet(name=cfg["wallet_name"], hotkey=cfg["hotkey"])
    netuid = cfg["netuid"]
    llm = LLM(cfg["chutes_api_url"], api_key)
    elo = Elo(
        k=cfg["old_config"]["elo"]["k_factor"],
        initial=cfg["old_config"]["elo"]["initial_rating"],
    )

    from compelle import FULL_VERSION
    log.info(f"compelle validator {FULL_VERSION} starting on netuid {netuid} ({cfg['network']})")
    gist_id = cfg.get("config_gist_id", "")
    gist_owner = cfg.get("config_gist_owner", "")
    keep_epochs = int(cfg.get("keep_epochs", 1000))
    cached_overrides: dict = {}
    cached_revision = "bundled"
    epoch = 0

    while True:
        epoch += 1
        log.info(f"=== epoch {epoch} ===")

        try:
            sub = bt.Subtensor(network=cfg["network"])
            epoch_start_block = sub.get_current_block()
            block_hash = sub.substrate.get_block_hash(epoch_start_block)
            records = fetch_records(sub, netuid, block=epoch_start_block, block_hash=block_hash)
        except Exception as e:
            log.error(f"chain unreachable: {e}; retry 60s")
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

        real_strategies = {
            hk: resolve_strategy(r.commitment_text)
            for hk, r in records.items() if r.is_real
        }
        real_strategies = {hk: t for hk, t in real_strategies.items() if t.strip()}

        n_real = len(real_strategies)
        n_eps = sum(1 for r in records.values() if r.is_placeholder)
        n_zero = sum(1 for r in records.values() if not r.is_eligible)
        log.info(f"miners: {n_real} real, {n_eps} epsilon, {n_zero} ineligible")

        results = []
        real_weights: dict[str, float] = {}
        if n_real >= 2:
            ok, err = llm.ping(cfg["game"]["model"])
            if not ok:
                log.warning(f"LLM preflight failed: {err[:200]}")
            else:
                results, elo = run_tournament(llm, cfg, real_strategies,
                                              epoch_start_block, elo=elo)
                if results:
                    real_weights = elo.weights(cfg["elo"]["temperature"])

        real_weights = {hk: w for hk, w in real_weights.items()
                        if hk in records and records[hk].is_real}
        weights = assign_weights(records, real_weights, epoch_start_block)
        if weights:
            uids = [records[hk].uid for hk in weights]
            vals = list(weights.values())
            sw_status = "succeeded" if set_weights(sub, wallet, netuid, uids, vals) else "failed"
        else:
            log.info("no weights to set this epoch")
            sw_status = "skipped"

        write_epoch(epoch, epoch_start_block, cached_revision, records, weights, results, elo,
                    sw_status)
        prune_epochs(keep_epochs)
        push_pending(cfg["push_url"], wallet)

        sleep_secs = cfg["tournament"]["epoch_seconds"]
        log.info(f"epoch {epoch} done; next in {sleep_secs}s")
        time.sleep(sleep_secs)


if __name__ == "__main__":
    main()
