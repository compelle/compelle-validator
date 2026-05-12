import re
import os
import json
import math
import time
import random
import logging
import itertools
import hashlib
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import requests
from openai import OpenAI

log = logging.getLogger(__name__)

MAX_STRATEGY_BYTES = 65536


def topic_id(topic_obj: dict) -> str:
    """Stable, content-addressed ID for a topic. Hash of normalized motion text.
    Same motion text across days/gist-revisions yields the same id, so analytics
    can group games by topic without string-matching the motion."""
    motion = (topic_obj.get("motion") or "").strip()
    return hashlib.sha256(motion.encode("utf-8")).hexdigest()[:12]

_NOT_FOUND_MARKERS = ("404", "no_such_model", "model_not_found", "model not found",
                      "model does not exist", "unknown model")
# Substrings that indicate a transient upstream condition worth retrying. Shared
# by LLM.chat (same-model retry with backoff) and _chat (fallback to next model
# after exhausting retries). 429 = rate limit, 5xx = upstream outage, "timeout"
# catches bare httpx.ReadTimeout strings that don't carry an HTTP status.
_TRANSIENT_MARKERS = ("429", "503", "502", "504", "rate limit", "overloaded",
                      "service unavailable", "timeout")
_VERDICT_WORDS = {
    "PRO": "Pro", "PROPOSITION": "Pro", "PROP": "Pro", "YES": "Pro", "AFFIRMATIVE": "Pro",
    "CON": "Con", "OPPOSITION": "Con", "OPP": "Con", "NO": "Con", "NEGATIVE": "Con",
    "AGAINST": "Con",
}


class ModelNotAvailableError(Exception):
    pass


def parse_quota_reset(reason_str: str) -> str | None:
    """Extract the ISO `quota_reset_timestamp` from a Chutes 402 error.

    Chutes 402 responses include a `quota_reset_timestamp` field; on hit, return
    the ISO string so the caller can decide how long to wait. Returns None if
    the input isn't a 402 quota error or the timestamp can't be located.
    """
    if not reason_str or "quota_reset_timestamp" not in reason_str:
        return None
    idx = reason_str.find("quota_reset_timestamp")
    tail = reason_str[idx:idx + 120]
    q1 = tail.find("'", tail.find(":") + 1)
    q2 = tail.find("'", q1 + 1) if q1 >= 0 else -1
    if q1 >= 0 and q2 > q1:
        return tail[q1 + 1:q2]
    return None


class LLM:
    def __init__(self, base_url: str, api_key: str, timeout: float = 120.0):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def ping(self, model: str) -> tuple[bool, str]:
        try:
            self.client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": "hi"}],
                max_tokens=1, temperature=0.0,
            )
            return True, ""
        except Exception as e:
            return False, str(e)

    def chat(self, system, messages, model, max_tokens=2048, temperature=0.6) -> str:
        full = [{"role": "system", "content": system}] + messages
        for attempt in range(8):
            try:
                r = self.client.chat.completions.create(
                    model=model, messages=full,
                    max_tokens=max_tokens, temperature=temperature,
                )
                return r.choices[0].message.content or ""
            except Exception as e:
                s = str(e).lower()
                if any(m in s for m in _NOT_FOUND_MARKERS):
                    raise ModelNotAvailableError(str(e)) from e
                if any(m in s for m in _TRANSIENT_MARKERS) and attempt < 7:
                    time.sleep(min(2 ** attempt + random.random(), 60))
                    continue
                raise


def strip_thinking(text: str, tags: list[str]) -> str:
    for tag in tags:
        text = re.sub(rf"<{re.escape(tag)}>[\s\S]*?</{re.escape(tag)}>", "",
                      text, flags=re.IGNORECASE)
    return text.strip()


# Strategies must reference an immutable gist revision (40-hex SHA-1). Mutable
# refs like `gist:<id>` are rejected so a miner can't swap their committed
# strategy after the fact.
_GIST_REVISIONED_RE = re.compile(r"^gist:([0-9a-f]{20,40})/([0-9a-f]{40})$")
# Cache (gist_id, revision) → resolved text. Persisted to disk so a validator
# restart doesn't re-spend the GitHub anonymous-API quota (60 req/hr/IP) on
# fetches we've already done. Revisions are immutable by SHA-1, so a hit is
# safe forever; we never invalidate.
_GIST_CACHE_PATH = os.environ.get("COMPELLE_GIST_CACHE") or "data/gist_cache.json"
_gist_cache: dict[tuple[str, str], str] = {}
_gist_cache_loaded = False


def _load_gist_cache() -> None:
    global _gist_cache_loaded
    if _gist_cache_loaded:
        return
    _gist_cache_loaded = True
    try:
        with open(_GIST_CACHE_PATH) as f:
            data = json.load(f)
        for k, v in data.items():
            if "/" in k and isinstance(v, str):
                gid, rev = k.split("/", 1)
                _gist_cache[(gid, rev)] = v
        log.info(f"loaded gist cache: {len(_gist_cache)} entries from {_GIST_CACHE_PATH}")
    except FileNotFoundError:
        pass
    except Exception as e:
        log.warning(f"gist cache load failed ({e}); starting empty")


def _save_gist_cache() -> None:
    try:
        d = os.path.dirname(_GIST_CACHE_PATH)
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = _GIST_CACHE_PATH + ".tmp"
        out = {f"{gid}/{rev}": v for (gid, rev), v in _gist_cache.items()}
        with open(tmp, "w") as f:
            json.dump(out, f)
        os.replace(tmp, _GIST_CACHE_PATH)
    except Exception as e:
        log.warning(f"gist cache save failed: {e}")


def resolve_strategy(commitment: str) -> str:
    if not commitment.startswith("gist:"):
        return commitment if len(commitment.encode("utf-8")) <= MAX_STRATEGY_BYTES else ""

    m = _GIST_REVISIONED_RE.match(commitment)
    if not m:
        log.warning(f"reject gist commitment: malformed or missing revision: {commitment[:80]!r}")
        return ""

    gist_id, revision = m.group(1), m.group(2)
    cache_key = (gist_id, revision)
    _load_gist_cache()
    if cache_key in _gist_cache:
        return _gist_cache[cache_key]

    try:
        # Anonymous GitHub API is 60 req/hr/IP — fine when the cache is warm,
        # painful at first boot. Operators who set GITHUB_TOKEN get 5000 req/hr.
        # No token = unchanged behavior.
        headers = {}
        token = os.environ.get("GITHUB_TOKEN", "").strip()
        if token:
            headers["Authorization"] = f"token {token}"
        r = requests.get(f"https://api.github.com/gists/{gist_id}/{revision}",
                         headers=headers, timeout=10)
        r.raise_for_status()  # GitHub 422s on revisions not in history
        files = r.json().get("files", {})
        if not files:
            return ""
        # Reject ambiguous multi-file gists. Picking by dict-iteration order
        # depends on GitHub's JSON-encoder behavior and is not a stable
        # consensus primitive — different validators could read different
        # files. Strategies must live in a single-file gist.
        if len(files) > 1:
            log.warning(f"reject gist {commitment}: has {len(files)} files (must be single-file)")
            return ""
        content = next(iter(files.values())).get("content", "") or ""
        if len(content.encode("utf-8")) > MAX_STRATEGY_BYTES:
            return ""
        _gist_cache[cache_key] = content
        _save_gist_cache()
        return content
    except Exception as e:
        log.error(f"gist {commitment} fetch failed: {e}")
        return ""


@dataclass
class GameResult:
    topic: str
    winner: str
    reason: str
    turns: int
    transcript: list
    duration: float
    judge_explanation: str = ""
    completed_at: str = ""
    topic_data: dict | None = None
    topic_id: str = ""
    # Per-judge votes when a panel decides the verdict (empty for concession-decided
    # games or single-judge fallback). Each entry: {"model","verdict","reason"}.
    judge_panel_votes: list = field(default_factory=list)


def _chat(llm, system, history, primary, fallbacks, max_tok, temp) -> str:
    """Try `primary` first, then each fallback. Falls back on:
      - ModelNotAvailableError (404 / unknown model)
      - persistent throttling (429) or upstream outage (503/502/504) — these
        are exhausted by `LLM.chat`'s internal retry first, so by the time we
        catch one here the primary has already burned its retries.
    """
    models = [primary] + [m for m in fallbacks if m != primary]
    last: Exception | None = None
    for m in models:
        try:
            return llm.chat(system, history, m, max_tok, temp)
        except ModelNotAvailableError as e:
            log.warning("model %s unavailable, falling back", m)
            last = e
        except Exception as e:
            s = str(e).lower()
            transient = any(c in s for c in _TRANSIENT_MARKERS)
            if transient and m != models[-1]:
                log.warning("model %s transient error (%s), falling back", m, s[:80])
                last = e
                continue
            raise
    raise last  # type: ignore[misc]


def play_game(llm, config, topic_obj, strategy_pro, strategy_con) -> GameResult:
    cfg = config["game"]
    motion = topic_obj["motion"]
    ctx_text = topic_obj.get("context", "")
    context = f"\nBackground context: {ctx_text}\n" if ctx_text else ""
    today = datetime.now().strftime("%B %d, %Y")
    template = config["game_prompt"]
    fallbacks = cfg.get("model_fallbacks", [])
    max_tok, temp = cfg["max_tokens_per_turn"], cfg["temperature"]
    tags = config.get("thinking_tags") or ["think"]
    sym = config["concession"]["symbol"]
    min_len = config["concession"]["min_length"]

    prompts = {
        "Pro": template.format(topic=motion, side="Pro (in favor of the motion)",
                               strategy=strategy_pro, context=context, date=today),
        "Con": template.format(topic=motion, side="Con (opposed to the motion)",
                               strategy=strategy_con, context=context, date=today),
    }
    histories = {"Pro": [], "Con": []}
    transcript: list = []
    start = time.time()

    tid = topic_id(topic_obj)

    def _mk(winner, reason):
        return GameResult(motion, winner, reason, len(transcript), transcript,
                          time.time() - start, topic_data=topic_obj, topic_id=tid)

    for _ in range(cfg["max_turns"]):
        for side in ("Pro", "Con"):
            opp = "Con" if side == "Pro" else "Pro"
            try:
                raw = _chat(llm, prompts[side], histories[side], cfg["model"],
                            fallbacks, max_tok, temp)
            except Exception as e:
                return _mk("draw", f"LLM error: {e}")
            visible = strip_thinking(raw, tags)
            transcript.append({"speaker": side, "text": raw})
            histories[side].append({"role": "assistant", "content": visible})
            histories[opp].append({"role": "user", "content": visible})
            v = visible.strip()
            if v.startswith(sym) and len(v) >= min_len:
                return _mk(opp, f"{side} conceded")

    if cfg.get("allow_draws", False):
        return _mk("draw", "Max turns reached")

    return judge_game(llm, config, topic_obj, motion, transcript, time.time() - start)


def _judge_one(llm, config, judge_model: str, prompt: str, tags: list) -> tuple[str, str] | None:
    """Run a single judge model. Returns (verdict, reason) or None on failure.

    Each panel member is queried with itself as primary and NO fallback — we
    want OTHER panel members to vote, not for a single judge's failure to
    cascade across the panel and pollute the diversity. Two attempts at
    increasing temperature in case the model fails to produce a clean PRO/CON.
    """
    # Thinking-heavy models (Kimi, Qwen3-Thinking, DeepSeek) routinely burn
    # 2000+ reasoning tokens before producing visible output. A 2048 ceiling
    # truncates them mid-`<think>` block with no PRO/CON line. Match the
    # intent_classifier ceiling so the panel has consistent headroom.
    judge_max_tokens = int(config.get("game", {}).get("judge_max_tokens", 8192))
    for attempt in range(2):
        try:
            verdict = llm.chat(
                "You are an impartial debate judge. You MUST pick a winner. "
                "Respond with exactly PRO or CON on the first line.",
                [{"role": "user", "content": prompt}],
                judge_model, max_tokens=judge_max_tokens, temperature=0.3 + attempt * 0.2,
            )
        except Exception as e:
            log.warning("judge %s attempt %d failed: %s", judge_model, attempt + 1, str(e)[:80])
            if attempt == 1:
                return None
            continue
        clean = strip_thinking(verdict, tags) or verdict.strip()
        lines = [line.strip() for line in clean.split("\n") if line.strip()]
        if not lines:
            continue
        match = re.match(r"^[*\"]*(\w+)", lines[0].upper())
        winner = _VERDICT_WORDS.get(match.group(1)) if match else None
        if winner:
            return (winner, " ".join(lines[1:]))
    return None


def _single_judge(llm, config, topic_obj, motion, transcript, duration, tid,
                  panel_record=None, reason_prefix=""):
    """Single-judge decision via the legacy primary+fallback chain.

    Used in two paths:
      1. judge_panel is absent or empty in config (legacy operators)
      2. Panel verdict ambiguous — couldn't produce a clear majority
    """
    # Normalize panel_record to a list so GameResult.judge_panel_votes
    # serializes as [] (the dataclass default) rather than null when the
    # legacy no-panel path runs.
    panel_record = panel_record if panel_record is not None else []
    cfg = config["game"]
    transcript_text = "\n\n".join(f"[{e['speaker']}]: {e['text']}" for e in transcript)
    prompt = config["judge_prompt"].format(topic=motion, transcript=transcript_text)
    tags = config.get("thinking_tags") or ["think"]
    primary = cfg.get("judge_model", cfg["model"])
    fallbacks = cfg.get("judge_model_fallbacks", cfg.get("model_fallbacks", []))
    # Same ceiling as the panel path — debater/judge models share lineage and
    # thinking budgets, so the single-judge fallback needs equal headroom.
    judge_max_tokens = int(cfg.get("judge_max_tokens", 8192))
    for attempt in range(3):
        try:
            verdict = _chat(
                llm,
                "You are an impartial debate judge. You MUST pick a winner. "
                "Respond with exactly PRO or CON on the first line.",
                [{"role": "user", "content": prompt}],
                primary, fallbacks, max_tok=judge_max_tokens, temp=0.3 + attempt * 0.2,
            )
        except Exception as e:
            if attempt == 2:
                return GameResult(motion, "draw", f"{reason_prefix}Judge error: {e}",
                                  len(transcript), transcript, duration,
                                  topic_data=topic_obj, topic_id=tid,
                                  judge_panel_votes=panel_record)
            continue
        clean = strip_thinking(verdict, tags) or verdict.strip()
        lines = [line.strip() for line in clean.split("\n") if line.strip()]
        if not lines:
            continue
        match = re.match(r"^[*\"]*(\w+)", lines[0].upper())
        winner = _VERDICT_WORDS.get(match.group(1)) if match else None
        if winner:
            return GameResult(motion, winner, f"{reason_prefix}Judge decision",
                              len(transcript), transcript, duration,
                              " ".join(lines[1:]), topic_data=topic_obj, topic_id=tid,
                              judge_panel_votes=panel_record)
    return GameResult(motion, "draw", f"{reason_prefix}Judge indecisive",
                      len(transcript), transcript, duration,
                      topic_data=topic_obj, topic_id=tid,
                      judge_panel_votes=panel_record)


def judge_game(llm, config, topic_obj, motion, transcript, duration) -> GameResult:
    """Judge a debate via configurable judge panel.

    Reads `judge_panel` from game config (list of model IDs). Falls back to the
    legacy single-judge config (judge_model + judge_model_fallbacks) when
    judge_panel is absent or empty. Panel members vote independently in
    parallel and the majority verdict wins. With odd panel sizes ties are
    impossible barring all-judge failure.

    Safety net: if fewer than 3 panel members produce a valid vote (i.e. one
    or more models are down or returning malformed verdicts), fall through to
    the single-judge path. Threshold of 3 is the smallest sample for a
    statistically meaningful majority.
    """
    cfg = config["game"]
    transcript_text = "\n\n".join(f"[{e['speaker']}]: {e['text']}" for e in transcript)
    prompt = config["judge_prompt"].format(topic=motion, transcript=transcript_text)
    tags = config.get("thinking_tags") or ["think"]
    tid = topic_id(topic_obj)

    panel = cfg.get("judge_panel") or []
    if not panel:
        return _single_judge(llm, config, topic_obj, motion, transcript, duration, tid)

    # Panel path: run each judge in parallel, take majority vote. Panel members
    # do NOT use fallbacks — the panel itself is the resilience mechanism.
    votes: list = []  # list of (verdict, reason, model)
    with ThreadPoolExecutor(max_workers=min(len(panel), 8)) as ex:
        futs = {ex.submit(_judge_one, llm, config, m, prompt, tags): m for m in panel}
        for fut in as_completed(futs):
            r = fut.result()
            if r is not None:
                votes.append((r[0], r[1], futs[fut]))

    panel_record = [{"model": m, "verdict": w, "reason": (rsn or "")[:280]}
                    for (w, rsn, m) in votes]

    # Safety net: only fall back to single-judge when the partial panel can't
    # produce a clear winner — i.e., max side-count < 2. With a 3-judge panel
    # plus one judge failing, a 2-0 partial result IS a clear majority and we
    # should accept it; only 0/1-vote or split-1-1 cases need rejudging.
    pro_count = sum(1 for (w, _, _) in votes if w == "Pro")
    con_count = sum(1 for (w, _, _) in votes if w == "Con")
    if max(pro_count, con_count) < 2:
        log.warning(f"panel only {pro_count}-{con_count} ({len(votes)} valid votes); "
                    f"falling back to single-judge")
        return _single_judge(llm, config, topic_obj, motion, transcript, duration, tid,
                             panel_record=panel_record,
                             reason_prefix="Panel-insufficient → ")

    if pro_count == con_count:
        # Reachable with even-sized panel and split votes (e.g., 4-judge 2-2).
        # With the configured 5-judge panel this only fires if exactly one
        # judge fails AND the remaining 4 split evenly, which is rare.
        return GameResult(motion, "draw", f"Panel split {pro_count}-{con_count}",
                          len(transcript), transcript, duration,
                          topic_data=topic_obj, topic_id=tid,
                          judge_panel_votes=panel_record)

    winner = "Pro" if pro_count > con_count else "Con"
    explanation = next(rsn for (w, rsn, _) in votes if w == winner)
    return GameResult(motion, winner, f"Panel verdict {pro_count}-{con_count}",
                      len(transcript), transcript, duration, explanation,
                      topic_data=topic_obj, topic_id=tid,
                      judge_panel_votes=panel_record)


class Elo:
    def __init__(self, k: float = 32.0, initial: float = 1000.0):
        self.k = k
        self.initial = initial
        self.ratings: dict[str, float] = {}

    def get(self, p: str) -> float:
        return self.ratings.setdefault(p, self.initial)

    def expected(self, ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

    def update(self, winner: str, loser: str):
        ra, rb = self.get(winner), self.get(loser)
        self.ratings[winner] = ra + self.k * (1.0 - self.expected(ra, rb))
        self.ratings[loser] = rb + self.k * (0.0 - self.expected(rb, ra))

    def update_draw(self, a: str, b: str, penalty: float = 0):
        ra, rb = self.get(a), self.get(b)
        self.ratings[a] = ra + self.k * (0.5 - self.expected(ra, rb)) - penalty
        self.ratings[b] = rb + self.k * (0.5 - self.expected(rb, ra)) - penalty

    def weights(self, temperature: float = 100.0) -> dict[str, float]:
        if not self.ratings:
            return {}
        max_r = max(self.ratings.values())
        e = {k: math.exp((v - max_r) / temperature) for k, v in self.ratings.items()}
        total = sum(e.values())
        return {k: v / total for k, v in e.items()}


def _stable_coin_flip(a: str, b: str, tempo_index: int, round_num: int) -> int:
    """Deterministic 0/1 from (sorted pair, tempo, round). Stable across
    processes (no PYTHONHASHSEED dependency) and across validators (no
    shared rng-stream state). round_num in the key means a pair that meets
    again in a later round gets the opposite assignment."""
    key = f"{min(a,b)}|{max(a,b)}|t{tempo_index}|r{round_num}".encode()
    return hashlib.sha256(key).digest()[0] & 1


def _swiss_pair_round(hotkeys, scores, color_diff, played, elo, round_num,
                       tempo_index):
    """Generate pairings for one Swiss round. Returns list of (pro, con).

    Round 1 uses standard Swiss seeding: top-half-by-Elo plays bottom-half.
    Round 2+: pair within score groups (highest score first), preferring
    opponents not yet played; assign Pro to the player with the lower
    color_diff so per-player Pro/Con balance trends to zero across the
    tournament. Ties (equal CD, equal Elo) resolved by a content-addressed
    coin flip so all validators converge on the same pairings without
    depending on shared rng-stream state (which would desync across
    validators when one tiebreak fires for some validators but not others).
    """
    from collections import defaultdict

    if round_num == 1:
        sorted_hk = sorted(hotkeys, key=lambda h: (-elo.get(h), h))
        n = len(sorted_hk)
        half = n // 2
        top = sorted_hk[:half]
        bot = sorted_hk[half:half * 2]
        # Top half plays Pro; later rounds will flip them via color_diff.
        # Odd miner (sorted_hk[2*half:]) sits out this round (a "bye").
        if n % 2 == 1:
            log.info(f"swiss R1 bye: UID-like-hotkey {sorted_hk[-1][:8]}… "
                     f"(odd miner count N={n})")
        return [(top[i], bot[i]) for i in range(half)]

    # Score group keys are floats (0.0 / 0.5 / 1.0 / 1.5 / …). IEEE 754 sums
    # of half-integers are exact, so equality across validators is safe.
    groups = defaultdict(list)
    for hk in hotkeys:
        groups[scores[hk]].append(hk)

    pairs = []
    floaters = []  # players who couldn't be paired in their score group
    for score_val in sorted(groups.keys(), reverse=True):
        bucket = list(groups[score_val]) + floaters
        bucket.sort(key=lambda h: (-elo.get(h), h))
        floaters = []
        while len(bucket) >= 2:
            top_player = bucket.pop(0)
            opp_idx = None
            for i, candidate in enumerate(bucket):
                if candidate not in played[top_player]:
                    opp_idx = i
                    break
            if opp_idx is None:
                # All same-score peers have been played; carry to next group.
                floaters.append(top_player)
                continue
            opp = bucket.pop(opp_idx)
            cd_top, cd_opp = color_diff[top_player], color_diff[opp]
            if cd_top < cd_opp:
                pro, con = top_player, opp
            elif cd_top > cd_opp:
                pro, con = opp, top_player
            else:
                # Equal color preference: stable coin flip from sorted pair.
                if _stable_coin_flip(top_player, opp, tempo_index, round_num):
                    pro, con = top_player, opp
                else:
                    pro, con = opp, top_player
            pairs.append((pro, con))
        floaters.extend(bucket)

    # Last-resort: pair leftover floaters even if rematch. Rare path,
    # logged so vtrust regressions are easy to root-cause.
    if floaters:
        log.info(f"swiss R{round_num} floater fallback: {len(floaters)} "
                 f"players, rematches allowed")
    floaters.sort(key=lambda h: (-elo.get(h), h))
    while len(floaters) >= 2:
        a, b = floaters.pop(0), floaters.pop(0)
        if color_diff[a] <= color_diff[b]:
            pairs.append((a, b))
        else:
            pairs.append((b, a))
    return pairs


def run_tournament(llm, config, strategies, epoch_start_block: int, elo=None,
                   on_progress=None):
    hotkeys = list(strategies.keys())
    if len(hotkeys) < 2:
        return [], elo or Elo()

    elo_cfg = config["elo"]
    if elo is None:
        elo = Elo(k=elo_cfg["k_factor"], initial=elo_cfg["initial_rating"])
    for hk in hotkeys:
        elo.get(hk)

    # Seed off the tempo bucket, NOT the sampled epoch_start_block. Validators
    # that observe the chain at slightly different blocks within the same tempo
    # would otherwise produce different orderings (and thus different Elo
    # paths). Topic selection uses the same bucket so all validators converge.
    tempo = config.get("tempo_blocks", 360)
    tempo_index = epoch_start_block // tempo

    # ONE topic per tournament — fairer Elo signal because every miner is
    # tested under identical conditions. Topic index is chain-derived. Wraps
    # when exceeding the topic list. After 00/06/12/18 UTC topic-refresh the
    # gist content changes; validators briefly diverge but realign within an
    # epoch.
    topics = config.get("topics") or []
    if not topics:
        log.warning("no topics in config; skipping tournament")
        return [], elo
    topic_index = tempo_index % len(topics)
    chosen_topic = topics[topic_index]
    log.info(f"tournament topic: index={topic_index}/{len(topics)} "
             f"id={topic_id(chosen_topic)} motion={chosen_topic.get('motion','')[:80]!r}")

    # Swiss tournament: N rounds, score-based pairing, per-player color balance.
    # Drops compute from O(N²) round-robin (90 games for 10 miners) to a small
    # multiple of N (20 games for 10 miners at 4 rounds) while preserving
    # per-player Pro/Con balance.
    num_rounds = config["tournament"].get("swiss_rounds", 4)
    workers = config["tournament"].get("max_concurrent_games", 5)
    log.info(f"swiss tournament: {len(hotkeys)} miners × {num_rounds} rounds")

    scores = {hk: 0.0 for hk in hotkeys}
    color_diff = {hk: 0 for hk in hotkeys}  # (# games as Pro) − (# games as Con)
    played = {hk: set() for hk in hotkeys}
    all_results = []

    for round_num in range(1, num_rounds + 1):
        pairs = _swiss_pair_round(hotkeys, scores, color_diff, played,
                                  elo, round_num, tempo_index)
        for (pro, con) in pairs:
            log.info(f"swiss R{round_num} pair: pro={pro[:8]}… "
                     f"con={con[:8]}… pro_cd={color_diff[pro]:+d} "
                     f"con_cd={color_diff[con]:+d} pro_elo={elo.get(pro):.1f} "
                     f"con_elo={elo.get(con):.1f}")
        log.info(f"swiss R{round_num}/{num_rounds}: {len(pairs)} games")
        games = list(enumerate((pair, chosen_topic) for pair in pairs))

        def _play(item):
            idx, ((pro, con), topic) = item
            result = play_game(llm, config, topic, strategies[pro], strategies[con])
            result.completed_at = datetime.now(timezone.utc).isoformat()
            return idx, pro, con, result

        completed: list[tuple[int, str, str, GameResult]] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for fut in as_completed({pool.submit(_play, g): g for g in games}):
                try:
                    completed.append(fut.result())
                except Exception as e:
                    log.error(f"game error: {e}")
                # Pulse the watchdog after each game (success or failure).
                # Without this, large tournaments can exceed the watchdog
                # timeout even while making steady progress, causing systemd
                # to kill+restart mid-tournament and lose completed games.
                if on_progress is not None:
                    try:
                        on_progress()
                    except Exception as e:
                        log.warning(f"on_progress callback failed: {e}")

        completed.sort(key=lambda x: x[0])
        for _, pro, con, result in completed:
            reason = (result.reason or "").lower()
            # `in` instead of `startswith` so the panel-fallback path's
            # prefixed form ("Panel-insufficient → Judge error: ...") is
            # also recognized as an infrastructure failure.
            errored = "llm error" in reason or "judge error" in reason
            if errored:
                # Don't burn the matchup slot on infra failure: the same pair
                # may succeed in a later round when Chutes recovers. Without
                # this, a sustained Chutes outage causes every round to mark
                # all pairs as played, forcing later rounds into the floater
                # rematch fallback and producing no useful Elo updates.
                all_results.append((pro, con, result))
                continue
            played[pro].add(con)
            played[con].add(pro)
            color_diff[pro] += 1
            color_diff[con] -= 1
            if result.winner == "Pro":
                elo.update(pro, con)
                scores[pro] += 1.0
            elif result.winner == "Con":
                elo.update(con, pro)
                scores[con] += 1.0
            else:
                elo.update_draw(pro, con, elo_cfg.get("draw_penalty", 0))
                scores[pro] += 0.5
                scores[con] += 0.5
            all_results.append((pro, con, result))

    return all_results, elo
