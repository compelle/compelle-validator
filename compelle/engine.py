import re
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
_VERDICT_WORDS = {
    "PRO": "Pro", "PROPOSITION": "Pro", "PROP": "Pro", "YES": "Pro", "AFFIRMATIVE": "Pro",
    "CON": "Con", "OPPOSITION": "Con", "OPP": "Con", "NO": "Con", "NEGATIVE": "Con",
    "AGAINST": "Con",
}


class ModelNotAvailableError(Exception):
    pass


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
                if ("429" in s or "503" in s) and attempt < 7:
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
_gist_cache: dict[tuple[str, str], str] = {}


def resolve_strategy(commitment: str) -> str:
    if not commitment.startswith("gist:"):
        return commitment if len(commitment.encode("utf-8")) <= MAX_STRATEGY_BYTES else ""

    m = _GIST_REVISIONED_RE.match(commitment)
    if not m:
        log.warning(f"reject gist commitment: malformed or missing revision: {commitment[:80]!r}")
        return ""

    gist_id, revision = m.group(1), m.group(2)
    cache_key = (gist_id, revision)
    if cache_key in _gist_cache:
        return _gist_cache[cache_key]

    try:
        r = requests.get(f"https://api.github.com/gists/{gist_id}/{revision}", timeout=10)
        r.raise_for_status()  # GitHub 422s on revisions not in history
        files = r.json().get("files", {})
        content = next(iter(files.values())).get("content", "") if files else ""
        if len(content.encode("utf-8")) > MAX_STRATEGY_BYTES:
            return ""
        _gist_cache[cache_key] = content
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
    models = [primary] + [m for m in fallbacks if m != primary]
    last: Exception | None = None
    for m in models:
        try:
            return llm.chat(system, history, m, max_tok, temp)
        except ModelNotAvailableError as e:
            log.warning("model %s unavailable, falling back", m)
            last = e
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
    for attempt in range(2):
        try:
            verdict = llm.chat(
                "You are an impartial debate judge. You MUST pick a winner. "
                "Respond with exactly PRO or CON on the first line.",
                [{"role": "user", "content": prompt}],
                judge_model, max_tokens=2048, temperature=0.3 + attempt * 0.2,
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


def judge_game(llm, config, topic_obj, motion, transcript, duration) -> GameResult:
    """Judge a debate via configurable judge panel.

    Reads `judge_panel` from game config (list of model IDs). Falls back to the
    legacy single-judge config (judge_model + judge_model_fallbacks) when
    judge_panel is absent or empty. Panel members vote independently in
    parallel and the majority verdict wins. With odd panel sizes ties are
    impossible barring all-judge failure.
    """
    cfg = config["game"]
    transcript_text = "\n\n".join(f"[{e['speaker']}]: {e['text']}" for e in transcript)
    prompt = config["judge_prompt"].format(topic=motion, transcript=transcript_text)
    tags = config.get("thinking_tags") or ["think"]
    tid = topic_id(topic_obj)

    panel = cfg.get("judge_panel") or []
    if not panel:
        # Legacy path: single-judge with fallbacks via _chat.
        primary = cfg.get("judge_model", cfg["model"])
        fallbacks = cfg.get("judge_model_fallbacks", cfg.get("model_fallbacks", []))
        for attempt in range(3):
            try:
                verdict = _chat(
                    llm,
                    "You are an impartial debate judge. You MUST pick a winner. "
                    "Respond with exactly PRO or CON on the first line.",
                    [{"role": "user", "content": prompt}],
                    primary, fallbacks, max_tok=2048, temp=0.3 + attempt * 0.2,
                )
            except Exception as e:
                if attempt == 2:
                    return GameResult(motion, "draw", f"Judge error: {e}",
                                      len(transcript), transcript, duration,
                                      topic_data=topic_obj, topic_id=tid)
                continue
            clean = strip_thinking(verdict, tags) or verdict.strip()
            lines = [line.strip() for line in clean.split("\n") if line.strip()]
            if not lines:
                continue
            match = re.match(r"^[*\"]*(\w+)", lines[0].upper())
            winner = _VERDICT_WORDS.get(match.group(1)) if match else None
            if winner:
                return GameResult(motion, winner, "Judge decision",
                                  len(transcript), transcript, duration,
                                  " ".join(lines[1:]), topic_data=topic_obj, topic_id=tid)
        return GameResult(motion, "draw", "Judge indecisive",
                          len(transcript), transcript, duration,
                          topic_data=topic_obj, topic_id=tid)

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

    if not votes:
        return GameResult(motion, "draw", "Panel all failed",
                          len(transcript), transcript, duration,
                          topic_data=topic_obj, topic_id=tid,
                          judge_panel_votes=panel_record)

    pro_count = sum(1 for (w, _, _) in votes if w == "Pro")
    con_count = sum(1 for (w, _, _) in votes if w == "Con")
    if pro_count == con_count:
        # Only reachable with even-sized panel and split votes.
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


def run_tournament(llm, config, strategies, epoch_start_block: int, elo=None):
    hotkeys = list(strategies.keys())
    if len(hotkeys) < 2:
        return [], elo or Elo()

    elo_cfg = config["elo"]
    if elo is None:
        elo = Elo(k=elo_cfg["k_factor"], initial=elo_cfg["initial_rating"])
    for hk in hotkeys:
        elo.get(hk)

    rng = random.Random(epoch_start_block)
    matchups = [m for a, b in itertools.combinations(hotkeys, 2) for m in ((a, b), (b, a))]
    rng.shuffle(matchups)

    # ONE topic per tournament — fairer Elo signal because every miner is tested
    # under identical conditions. Topic index is chain-derived so all validators
    # converge: tempo_index = epoch_start_block // tempo. Wraps around when we
    # exceed the topic-list length. After the 00/06/12/18 UTC topic-refresh the
    # gist content changes; validators briefly diverge but realign within an epoch.
    topics = config.get("topics") or []
    if not topics:
        log.warning("no topics in config; skipping tournament")
        return [], elo
    tempo = config.get("tempo_blocks", 360)
    topic_index = (epoch_start_block // tempo) % len(topics)
    chosen_topic = topics[topic_index]
    log.info(f"tournament topic: index={topic_index}/{len(topics)} "
             f"id={topic_id(chosen_topic)} motion={chosen_topic.get('motion','')[:80]!r}")
    games = list(enumerate((pair, chosen_topic) for pair in matchups))
    workers = config["tournament"].get("max_concurrent_games", 5)

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

    completed.sort(key=lambda x: x[0])
    results = []
    for _, pro, con, result in completed:
        reason = (result.reason or "").lower()
        if reason.startswith(("llm error", "judge error")):
            pass
        elif result.winner == "Pro":
            elo.update(pro, con)
        elif result.winner == "Con":
            elo.update(con, pro)
        else:
            elo.update_draw(pro, con, elo_cfg.get("draw_penalty", 0))
        results.append((pro, con, result))

    return results, elo
