import re
import os
import json
import math
import time
import random
import logging
import threading
import itertools
import hashlib
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import httpx
import requests
from openai import OpenAI

from compelle import provenance

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
# catches bare httpx.ReadTimeout strings that don't carry an HTTP status,
# "connection" catches httpx.ConnectError / RemoteProtocolError / TCP resets
# that Chutes' load balancer surfaces as bare "Connection error." strings —
# without this, every dropped HTTP/2 stream became "LLM error" and the game
# ended in a draw without retry, voiding ~70% of every tournament.
_TRANSIENT_MARKERS = ("429", "503", "502", "504", "rate limit", "overloaded",
                      "service unavailable", "timeout", "timed out",
                      "connection error", "remote protocol")
# Subset of _TRANSIENT_MARKERS that indicate the request took too long.
# These get a tighter retry cap (see LLM.chat) because each attempt blocks for
# the full httpx ReadTimeout window (~120s), and without a cap a single stuck
# call can burn ~18 min before falling through.
_TIMEOUT_MARKERS = ("timeout", "timed out")
_MAX_TIMEOUT_ATTEMPTS = 3
# Subset of _TRANSIENT_MARKERS that mean the model is saturated right now
# (HTTP 429 / "infrastructure is at maximum capacity"). Unlike a one-off
# network blip, retrying the same model only burns Retry-After / backoff
# sleeps while a healthy fallback sits idle, and our 1 req/s limiter
# (_acquire_rate_token) already rules out self-inflicted bursts, so a 429 is
# real upstream capacity rather than our own rate. Fail over to the next model
# on the first hit. The primary is still tried first on every call, so a model
# with capacity answers on attempt 0 and the common path is unchanged.
_CAPACITY_MARKERS = ("429", "too many requests", "rate limit",
                     "at capacity", "maximum capacity")
_MAX_CAPACITY_ATTEMPTS = 1
# A 200 response carrying blank content is not a usable turn (a provider hiccup,
# or a reasoning model that spent its budget on hidden reasoning_content and
# emitted nothing visible). It raises no exception, so without this path it was
# silently stored as an empty turn: the "broken opening" / blank mid-debate turn
# seen on the site. LLM.chat re-rolls the same model up to this many times
# (empties are intermittent), then _chat falls over to the next model.
_MAX_EMPTY_ATTEMPTS = 2
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


_RATE_LIMIT_INTERVAL = float(os.environ.get("COMPELLE_LLM_MIN_INTERVAL", "1.0"))
_rate_lock = threading.Lock()
_next_send_ts = 0.0


def _acquire_rate_token() -> None:
    """Block until the global LLM dispatch token-bucket allows another call.

    Process-wide token bucket: every chat.completions.create() must wait
    until at least _RATE_LIMIT_INTERVAL seconds have elapsed since the
    previous dispatch. With the default 1.0s, the validator emits at most
    one outbound LLM request per second across all threads, panel members,
    tournament games, and judges. This eliminates 429 burst-throttling
    from Chutes because we never fire calls faster than the steady-state
    quota allows.
    """
    global _next_send_ts
    with _rate_lock:
        now = time.time()
        if _next_send_ts > now:
            time.sleep(_next_send_ts - now)
            _next_send_ts = _next_send_ts + _RATE_LIMIT_INTERVAL
        else:
            _next_send_ts = now + _RATE_LIMIT_INTERVAL


def _extract_retry_after(exc: Exception) -> float | None:
    """Best-effort parse of Retry-After from a Chutes rate-limit exception.

    The openai SDK exposes the httpx Response on RateLimitError via
    `exc.response`. RFC 7231 allows Retry-After as either seconds or
    HTTP-date; Chutes returns seconds. Returns None if the header is
    absent or unparseable; callers fall back to exponential backoff.
    """
    try:
        resp = getattr(exc, "response", None)
        if resp is None:
            return None
        headers = getattr(resp, "headers", None)
        if headers is None:
            return None
        ra = headers.get("Retry-After") or headers.get("retry-after")
        if ra is None:
            return None
        v = float(str(ra).strip())
        return max(0.0, min(v, 120.0))
    except (ValueError, TypeError, AttributeError):
        return None


class LLM:
    def __init__(self, base_url: str, api_key: str, timeout: float = 60.0,
                 thinking_models=None):
        # HTTP/1.1 with a keep-alive pool. We originally picked HTTP/2 to
        # multiplex many requests over fewer sockets (friendlier to Chutes'
        # per-IP throttle), but empirical head-to-head benchmarks showed
        # HTTP/2 had a ~30% connection-error rate (Chutes' load balancer
        # drops whole multiplex sockets, killing every in-flight stream)
        # versus 0% under HTTP/1.1, plus 2-9x worse p90 latency. Per-phase
        # timeouts let us bail quickly on stuck pool/connect/write while
        # still giving Chutes 45s to start streaming. max_retries=0 disables
        # the OpenAI SDK's sub-second retry storm so LLM.chat's exponential
        # backoff is the only retry policy in play. A process-wide token
        # bucket (see _acquire_rate_token) caps outbound rate; see the
        # COMPELLE_LLM_MIN_INTERVAL env var to tune.
        http = httpx.Client(
            http2=False,
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=200,
                                keepalive_expiry=30.0),
            # read=90s: empirical GLM-5.1 latency distribution on realistic 7-turn
            # debate prompts under 20-concurrent load was 51-93s (median 65s,
            # p90 91s, max 93s). Our prior 45s timeout was bailing on 100% of
            # GLM calls, forcing the fallback chain to do the work every time
            # ("timeouts" weren't real — GLM was completing fine, we just cut
            # it off too soon). 90s catches >=90% of GLM calls at their median
            # so the primary model actually gets used. Worst case on a
            # genuinely-stuck call: ~45s more wait than before; that's small
            # compared to the ~10-20s fallback adds anyway.
            timeout=httpx.Timeout(connect=10.0, read=90.0, write=10.0, pool=5.0),
        )
        self.client = OpenAI(base_url=base_url, api_key=api_key,
                             http_client=http, max_retries=0)
        # Models that take a chat_template_kwargs thinking flag (DeepSeek-V3.2 is
        # hybrid: reasoning is off by default). For these we send extra_body to
        # turn the trace on. It returns in reasoning_content, never in the answer
        # content, so nothing leaks to viewers and there is nothing to strip.
        self.thinking_models = set(thinking_models or ())

    def ping(self, model: str) -> tuple[bool, str]:
        try:
            _acquire_rate_token()
            self.client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": "hi"}],
                max_tokens=1, temperature=0.0,
            )
            return True, ""
        except Exception as e:
            return False, str(e)

    def ping_chain(self, models: list[str]) -> tuple[bool, str, str]:
        """Ping each model in order; succeed on the first that answers.

        Returns (ok, working_model, last_error). Lets the validator start a
        tournament whenever ANY model in the failover chain is reachable,
        instead of skipping the whole epoch when only the primary is at
        capacity. Per-game _chat handles routing once the tournament runs.
        Models marked permanently dead (a prior 404) are skipped here too — we
        do not ping a model the catalog has already told us does not exist.
        """
        last_err = ""
        for m in models:
            if _breaker_for(m).dead:
                continue
            ok, err = self.ping(m)
            if ok:
                return True, m, ""
            last_err = err
        return False, "", last_err or "all models unavailable"

    def chat(self, system, messages, model, max_tokens=2048, temperature=0.6) -> str:
        full = [{"role": "system", "content": system}] + messages
        timeout_attempts = 0
        capacity_attempts = 0
        empty_attempts = 0
        for attempt in range(8):
            try:
                _acquire_rate_token()
                extra = ({"extra_body": {"chat_template_kwargs": {"thinking": True}}}
                         if model in self.thinking_models else {})
                r = self.client.chat.completions.create(
                    model=model, messages=full,
                    max_tokens=max_tokens, temperature=temperature, **extra,
                )
                content = r.choices[0].message.content or ""
                if content.strip():
                    return content
                # Blank 200: re-roll the same model (empties are intermittent),
                # then give up so _chat falls over to the next model.
                empty_attempts += 1
                if empty_attempts >= _MAX_EMPTY_ATTEMPTS:
                    return ""
                time.sleep(min(2 ** empty_attempts + random.random(), 8))
                continue
            except Exception as e:
                s = str(e).lower()
                if any(m in s for m in _NOT_FOUND_MARKERS):
                    raise ModelNotAvailableError(str(e)) from e
                # Timeouts get a tighter cap: each attempt blocks for the full
                # client ReadTimeout (~120s), so 8 attempts could stall ~18 min
                # per call. Three is enough to ride out a transient blip while
                # bounding worst-case wall-clock for stuck panel members.
                if any(m in s for m in _TIMEOUT_MARKERS):
                    timeout_attempts += 1
                    if timeout_attempts >= _MAX_TIMEOUT_ATTEMPTS:
                        raise
                if any(m in s for m in _CAPACITY_MARKERS):
                    capacity_attempts += 1
                    if capacity_attempts >= _MAX_CAPACITY_ATTEMPTS:
                        raise
                if any(m in s for m in _TRANSIENT_MARKERS) and attempt < 7:
                    # Prefer the server's Retry-After when it tells us
                    # exactly how long to wait; fall back to generic
                    # exponential when the header is absent.
                    ra = _extract_retry_after(e)
                    if ra is not None:
                        time.sleep(ra)
                    else:
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


# --- Per-model circuit breaker -----------------------------------------------
# A model is preferred whenever it is healthy, but under a Chutes capacity crunch
# a model can straddle our 90s read timeout: most calls still complete, just
# slowly, so a naive caller pays up to the full timeout before falling over.
# Across a tournament that turns a ~50-minute run into hours. The 429 fast-
# failover (see _CAPACITY_MARKERS) handles outright rejection but not the slow-
# but-completing case. This breaker watches a model's consecutive failures
# (timeouts and capacity) and, once it is clearly degraded, OPENS: callers skip
# it for a cooldown window. The cooldown grows exponentially as it keeps failing.
# When it expires the breaker goes HALF-OPEN and the next call verifies recovery
# with a cheap one-token ping (see LLM.ping); success CLOSES it, failure re-opens
# with a longer cooldown. State is process-wide, one breaker per model id (see
# _breaker_for). The JUDGE path gates on this directly; the debate path consults
# the reliability EWMA below instead (which the same record() calls also feed).
_BREAKER_FAIL_THRESHOLD = 3      # consecutive primary failures before opening
_BREAKER_BASE_COOLDOWN = 30.0    # seconds the primary is skipped on the first open
_BREAKER_MAX_COOLDOWN = 300.0    # cap on the exponential cooldown growth

# --- Adaptive model health (debate path ranking) -----------------------------
# The breaker above is a short-memory gate: three strikes and the primary is
# skipped for a cooldown, then bounced back the instant a cheap ping answers.
# That is right for a one-off blip but wrong for a chronically flapping primary
# (GLM-5.2 under a sustained Chutes brownout: blank storms + 90s timeouts that
# intermittently succeed, so the breaker keeps closing and re-leading a model
# that is still mostly broken while a healthy, faster fallback sits behind it).
# The debate path therefore also keeps a longer-memory reliability EWMA per
# model and ranks the failover chain by it: a model that has been failing sinks
# below one that has been answering, so _chat automatically leads with the
# reliable model (e.g. Qwen) without an operator touching config. The EWMA
# decays back toward neutral when a model stops being called, so a recovered
# model climbs back to the lead on its own. A 404 marks a model permanently
# dead: it is dropped from the chain and never tried or pinged again.
_HEALTH_NEUTRAL = 0.85           # reliability prior for a model we have not tried
_HEALTH_ALPHA = 0.4              # EWMA weight on the newest outcome (one fail shows)
_HEALTH_DECAY_HALFLIFE = 180.0   # seconds for the EWMA to drift halfway back to neutral
_CONFIG_PRIOR_STEP = 0.05        # per-position bias keeping config order when health ties


class _PrimaryBreaker:
    """Thread-safe per-model health unit.

    Carries two views of one model's health, fed by the same `record` calls:
      - the short-memory circuit breaker (state / fails / cooldown) the JUDGE
        path gates on via `gate()` — three strikes skip the model for a growing
        cooldown, a cheap ping recovers it. Unchanged behavior.
      - a long-memory reliability EWMA (`health`, `score`) the DEBATE path ranks
        the failover chain by, plus a permanent `dead` flag for 404s. This is
        what lets _chat lead with whichever model has been reliable lately
        instead of bouncing back to a flapping primary.
    """

    def __init__(self, threshold: int, base_cooldown: float, max_cooldown: float):
        self._threshold = threshold
        self._base = base_cooldown
        self._max = max_cooldown
        self._lock = threading.Lock()
        self._state = "closed"     # closed | open | half_open
        self._fails = 0            # consecutive primary failures while closed
        self._opens = 0            # consecutive opens, drives backoff growth
        self._open_until = 0.0     # monotonic deadline of the current cooldown
        self._probing = False      # a half-open recovery probe is in flight
        self._dead = False         # permanently unavailable (404); dropped from the chain
        self._ok = _HEALTH_NEUTRAL # reliability EWMA in [0, 1]
        self._ok_at = 0.0          # monotonic time of the last EWMA update (0 = never)

    def gate(self) -> str:
        """Return "use" (try the primary normally), "probe" (it may have
        recovered, verify with a cheap ping first), or "skip" (primary is down,
        go straight to the fallback)."""
        with self._lock:
            if self._state == "open" and time.monotonic() >= self._open_until:
                self._state = "half_open"
                self._probing = False
            if self._state == "closed":
                return "use"
            if self._state == "half_open" and not self._probing:
                self._probing = True
                log.info("primary circuit breaker HALF-OPEN: probing recovery")
                return "probe"
            return "skip"

    def _decayed_locked(self, now: float) -> float:
        """EWMA decayed toward neutral by the time since the last update. Caller
        holds the lock. A model nobody has called drifts back to neutral so a
        recovered model becomes eligible to lead again without live traffic."""
        if self._ok_at == 0.0:
            return _HEALTH_NEUTRAL
        elapsed = max(0.0, now - self._ok_at)
        factor = 0.5 ** (elapsed / _HEALTH_DECAY_HALFLIFE)
        return _HEALTH_NEUTRAL + (self._ok - _HEALTH_NEUTRAL) * factor

    def record(self, ok: bool, permanent: bool = False) -> None:
        """Report a model outcome (a real call while closed, or the ping while
        half-open). Updates the reliability EWMA, then drives the circuit
        breaker: closes on success; opens or re-opens with a growing cooldown on
        failure. Failures that arrive after the breaker already opened (stale
        in-flight calls) are ignored so they do not inflate the backoff.
        `permanent=True` (a 404) marks the model dead and short-circuits the
        rest: a dead model is never tried, ranked, or pinged again."""
        with self._lock:
            if permanent:
                self._dead = True
                return
            now = time.monotonic()
            sample = 1.0 if ok else 0.0
            blended = (1 - _HEALTH_ALPHA) * self._decayed_locked(now) + _HEALTH_ALPHA * sample
            # Cap at neutral: health is a demerit signal. A model is neutral when
            # it behaves and drops below when it fails; success only restores it
            # toward neutral, never above. This keeps a reliable fallback from
            # pumping its score past the configured primary, so once the primary
            # recovers it reclaims the lead on the config-order prior alone.
            self._ok = min(_HEALTH_NEUTRAL, blended)
            self._ok_at = now
            if ok:
                if self._state != "closed":
                    log.info("primary circuit breaker CLOSED: primary recovered")
                self._state, self._fails, self._opens, self._probing = "closed", 0, 0, False
                return
            if self._state == "half_open":
                self._probing = False
                self._opens += 1
            elif self._state == "closed":
                self._fails += 1
                if self._fails < self._threshold:
                    return
                self._opens += 1
            else:  # already open: a stale in-flight failure, ignore
                return
            self._state = "open"
            cooldown = min(self._base * (2 ** (self._opens - 1)), self._max)
            self._open_until = now + cooldown
            log.warning("primary circuit breaker OPEN: skipping primary for %.0fs", cooldown)

    @property
    def dead(self) -> bool:
        return self._dead

    def health(self, now: float | None = None) -> float:
        """Current decayed reliability EWMA in [0, 1]. Higher = more reliable."""
        with self._lock:
            return self._decayed_locked(time.monotonic() if now is None else now)

    def score(self, config_prior: float, now: float | None = None) -> float:
        """Ranking score for the debate chain: decayed reliability plus a small
        config-order prior (keeps config order when health ties). A dead model
        scores -inf so it sorts out of the chain entirely. Reads `dead` without
        the lock (atomic bool) and takes the lock once inside `health`, so it
        never nests locks."""
        if self._dead:
            return float("-inf")
        return self.health(now) + config_prior


# One breaker per model id, created on first use. A model's health is shared
# across every role it plays (debate primary in _chat and judge panel slot in
# _judge_one), so a model circuit-broken as a judge is also skipped on the
# debate path, and vice versa.
_breakers: dict = {}
_breakers_lock = threading.Lock()


def _breaker_for(model: str) -> _PrimaryBreaker:
    with _breakers_lock:
        b = _breakers.get(model)
        if b is None:
            b = _PrimaryBreaker(_BREAKER_FAIL_THRESHOLD,
                                _BREAKER_BASE_COOLDOWN, _BREAKER_MAX_COOLDOWN)
            _breakers[model] = b
        return b


# Last-known lead model per configured primary, so the adaptive selector logs a
# lead change once per transition instead of on every degraded call (10 workers
# x many turns would bury the log during exactly the brownout an operator is
# trying to read). Best-effort: a racing duplicate line is harmless.
_adaptive_lead: dict = {}


def _rank(models: list[str]) -> list[str]:
    """Order the failover chain by live model health, dropping dead models.

    `models` arrives in config order (primary first). Each non-dead model scores
    its decayed reliability EWMA plus a small per-position prior, so equal health
    preserves config order but a model that has been failing sinks below a
    healthier one — that is how the debate path leads with the reliable model
    (e.g. Qwen) while the configured primary flaps, and hands the lead back once
    the primary recovers. Dead (404) models are removed entirely; the debate path
    never asks for them again. An empty return means every model is dead."""
    now = time.monotonic()
    scored = []
    n = len(models)
    for idx, m in enumerate(models):
        b = _breaker_for(m)
        if b.dead:
            continue
        scored.append((b.score(_CONFIG_PRIOR_STEP * (n - idx), now), idx, m))
    scored.sort(key=lambda t: (-t[0], t[1]))  # health desc, config order breaks ties
    return [m for _, _, m in scored]


def _chat(llm, system, history, primary, fallbacks, max_tok, temp) -> str:
    """Try `primary` first, then each fallback. Falls back on:
      - ModelNotAvailableError (404 / unknown model)
      - persistent throttling (429) or upstream outage (503/502/504) — these
        are exhausted by `LLM.chat`'s internal retry first, so by the time we
        catch one here the primary has already burned its retries.
    The failover chain is ordered by live health (see _rank): a model that has
    been failing sinks below a healthier one, so the lead automatically moves to
    the reliable model (e.g. Qwen) while the configured primary flaps, and hands
    back to the primary once it recovers. Every attempt feeds that model's
    health, so the order self-corrects call by call.
    """
    models = _rank([primary] + [m for m in fallbacks if m != primary])
    if not models:
        raise ModelNotAvailableError(
            f"every model in the failover chain is permanently unavailable "
            f"(primary={primary}, fallbacks={fallbacks})")
    has_fallback = len(models) > 1
    lead = models[0]
    if _adaptive_lead.get(primary) != lead:
        _adaptive_lead[primary] = lead
        if primary not in models:
            log.warning("adaptive selector: primary %s unavailable, leading with %s", primary, lead)
        elif lead != primary:
            log.warning("adaptive selector: primary %s degraded, leading with %s", primary, lead)
        else:
            log.info("adaptive selector: primary %s leads the chain", primary)
    last: Exception | None = None
    for i, m in enumerate(models):
        h = _breaker_for(m)
        # Probationary lead: a model that climbed back to the front purely by
        # health decay (still below neutral, i.e. it failed recently) gets a
        # cheap one-token reality check before we risk a full debate-turn call
        # and its read-timeout on it. A bad ping re-demotes it for ~free; only a
        # healthy ping lets the real call proceed.
        if i == 0 and has_fallback and h.health() < _HEALTH_NEUTRAL:
            try:
                ok, _ = llm.ping(m)
            except Exception:
                ok = False  # a raising ping must not strand us on the probe
            h.record(ok)
            if not ok:
                log.warning("model %s recovery probe failed, falling back", m)
                last = RuntimeError(f"probe failed for {m}")
                continue
        try:
            out = llm.chat(system, history, m, max_tok, temp)
        except ModelNotAvailableError as e:
            # 404 / unknown model: gone from the catalog. Mark it permanently
            # dead so _rank drops it from every future call — never asked again.
            h.record(False, permanent=True)
            log.warning("model %s unavailable (404), dropping from the chain permanently", m)
            last = e
            continue
        except Exception as e:
            h.record(False)
            s = str(e).lower()
            transient = any(c in s for c in _TRANSIENT_MARKERS)
            if transient and m != models[-1]:
                log.warning("model %s transient error (%s), falling back", m, s[:80])
                last = e
                continue
            raise
        else:
            if out.strip():
                h.record(True)
                return out
            # Blank content survived LLM.chat's same-model retries. Treat it
            # like a transient failure: penalize health and fall over to the next
            # model instead of storing an empty turn.
            h.record(False)
            last = RuntimeError(f"empty completion from {m}")
            if m != models[-1]:
                log.warning("model %s returned blank content, falling back", m)
                continue
            raise last
    raise last or RuntimeError("no model produced output")  # type: ignore[misc]


# Pro speaks first, so its opening _chat call has no prior turns. Sending a
# model only a system message with no user turn makes many models (DeepSeek
# included) return empty content about half the time, or parrot the missing
# instruction scaffolding ("Just start speaking to your opponent", "just
# begin..."). A single user kickoff fixes it. It seeds only the model-visible
# history, never the stored transcript, so nothing leaks to viewers.
_FIRST_TURN_KICKOFF = "The debate now begins. Present your opening argument directly."


def _play_round_lockstep(llm, config, pairs, topic_obj, strategies, workers,
                         on_progress=None, on_game_turn=None):
    """Play one Swiss round with cross-game turn-batching.

    Instead of N independent per-game turn loops (one slow turn stalls the
    whole game serially), we fire all Pro turn-1 calls across active games
    as one wave, then all Con turn-1, then Pro turn-2, etc. A slow call only
    blocks its own wave, not other games' progression. The per-wave thread
    pool uses `workers` for concurrency. Concessions short-circuit one pair
    without affecting others; unfinished pairs are judged in parallel after
    the final wave.

    on_game_turn(idx, pair, side, text, transcript, conceded) is an optional
    per-turn hook (e.g., for live-state UIs). on_progress() is pulsed after
    each wave so the systemd watchdog stays fed.
    """
    cfg = config["game"]
    motion = topic_obj["motion"]
    ctx_text = topic_obj.get("context", "")
    context = f"\nBackground context: {ctx_text}\n" if ctx_text else ""
    today = datetime.now().strftime("%B %d, %Y")
    template = config["game_prompt"]
    fallbacks = cfg.get("model_fallbacks", [])
    max_tok, temp = cfg["max_tokens_per_turn"], cfg["temperature"]
    max_turns = cfg["max_turns"]
    tags = config.get("thinking_tags") or ["think"]
    sym = config["concession"]["symbol"]
    min_len = config["concession"]["min_length"]
    allow_draws = cfg.get("allow_draws", False)
    tid = topic_id(topic_obj)

    n = len(pairs)
    prompts: dict[tuple[int, str], str] = {}
    histories: dict[tuple[int, str], list] = {}
    transcripts: list[list[dict]] = [[] for _ in range(n)]
    start_times: list[float] = [time.time()] * n
    outcomes: list[tuple[str, str] | None] = [None] * n  # (winner, reason) once decided

    for idx, (pro, con) in enumerate(pairs):
        prompts[(idx, "Pro")] = template.format(
            topic=motion, side="Pro (in favor of the motion)",
            strategy=strategies[pro], context=context, date=today)
        prompts[(idx, "Con")] = template.format(
            topic=motion, side="Con (opposed to the motion)",
            strategy=strategies[con], context=context, date=today)
        # Pro opens; seed a kickoff user turn (see _FIRST_TURN_KICKOFF). Con's
        # first turn already has Pro's opening as its user message, so only Pro
        # needs the seed. The kickoff never enters transcripts[idx].
        histories[(idx, "Pro")] = [{"role": "user", "content": _FIRST_TURN_KICKOFF}]
        histories[(idx, "Con")] = []

    for wave in range(max_turns * 2):
        side = "Pro" if wave % 2 == 0 else "Con"
        opp = "Con" if side == "Pro" else "Pro"
        active = [i for i in range(n) if outcomes[i] is None]
        if not active:
            break
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futs = {pool.submit(_chat, llm, prompts[(i, side)],
                                histories[(i, side)], cfg["model"],
                                fallbacks, max_tok, temp): i
                    for i in active}
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    raw = fut.result()
                except Exception as e:
                    # Infra failure across the whole fallback chain: void, not draw.
                    # run_tournament drops voids from Elo and downstream W/L excludes
                    # them, so a provider outage never scores as a real tie.
                    outcomes[i] = ("void", f"LLM error: {e}")
                    if on_game_turn is not None:
                        try:
                            on_game_turn(idx=i, pair=pairs[i], side=side,
                                         text="", transcript=transcripts[i],
                                         conceded=True)
                        except Exception as ce:
                            log.warning(f"on_game_turn callback failed: {ce}")
                    continue
                visible = strip_thinking(raw, tags)
                transcripts[i].append({"speaker": side, "text": visible})
                histories[(i, side)].append({"role": "assistant", "content": visible})
                histories[(i, opp)].append({"role": "user", "content": visible})
                v = visible.strip()
                if v.startswith(sym) and len(v) >= min_len:
                    outcomes[i] = (opp, f"{side} conceded")
                if on_game_turn is not None:
                    try:
                        on_game_turn(idx=i, pair=pairs[i], side=side,
                                     text=visible, transcript=transcripts[i],
                                     conceded=(outcomes[i] is not None))
                    except Exception as ce:
                        log.warning(f"on_game_turn callback failed: {ce}")
        if on_progress is not None:
            try:
                on_progress()
            except Exception as e:
                log.warning(f"on_progress callback failed: {e}")

    to_judge = [i for i in range(n) if outcomes[i] is None]
    judge_results: dict[int, GameResult] = {}
    if to_judge and allow_draws:
        for i in to_judge:
            outcomes[i] = ("draw", "Max turns reached")
    elif to_judge:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futs = {pool.submit(judge_game, llm, config, topic_obj, motion,
                                transcripts[i],
                                time.time() - start_times[i]): i
                    for i in to_judge}
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    judge_results[i] = fut.result()
                except Exception as e:
                    outcomes[i] = ("void", f"Judge error: {e}")  # judge infra failure: void, not draw
        if on_progress is not None:
            try:
                on_progress()
            except Exception as e:
                log.warning(f"on_progress callback failed: {e}")

    out: list[tuple[int, str, str, GameResult]] = []
    for i, (pro, con) in enumerate(pairs):
        if i in judge_results:
            gr = judge_results[i]
            if not gr.topic_id:
                gr.topic_id = tid
            if not gr.topic_data:
                gr.topic_data = topic_obj
        else:
            winner, reason = outcomes[i] or ("draw", "unknown")
            gr = GameResult(
                topic=motion, winner=winner, reason=reason,
                turns=len(transcripts[i]), transcript=transcripts[i],
                duration=time.time() - start_times[i],
                topic_data=topic_obj, topic_id=tid,
            )
        gr.completed_at = datetime.now(timezone.utc).isoformat()
        out.append((i, pro, con, gr))
    return out


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
    # Pro opens; seed a kickoff user turn so the model has something to answer
    # (empty history yields an empty or echoed first turn). Stays out of the
    # transcript. See _FIRST_TURN_KICKOFF.
    histories = {"Pro": [{"role": "user", "content": _FIRST_TURN_KICKOFF}], "Con": []}
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
                return _mk("void", f"LLM error: {e}")  # infra failure: void, not draw
            visible = strip_thinking(raw, tags)
            transcript.append({"speaker": side, "text": visible})
            histories[side].append({"role": "assistant", "content": visible})
            histories[opp].append({"role": "user", "content": visible})
            v = visible.strip()
            if v.startswith(sym) and len(v) >= min_len:
                return _mk(opp, f"{side} conceded")

    if cfg.get("allow_draws", False):
        return _mk("draw", "Max turns reached")

    return judge_game(llm, config, topic_obj, motion, transcript, time.time() - start)


def _judge_deliberate(llm, config, judge_model: str, prompt: str,
                      my_verdict: str, my_reason: str,
                      other_verdict: str, other_reason: str,
                      tags: list) -> tuple[str, str] | None:
    """Second-round judge call after a 1-1 panel split. Shows the judge the
    same transcript, their prior verdict + reasoning, and the other panel
    member's verdict + reasoning, then asks them to reconsider.

    Empirically resolves ~50% of splits to a clean verdict (sample n=13).
    Failed-to-converge cases fall through to draw-with-penalty unchanged,
    so the worst case is the same as today's no-deliberation behavior.
    Cost: 2 extra LLM calls per split game (~60-180s wall time).
    """
    cfg = config.get("game", {})
    judge_max_tokens = int(cfg.get("judge_max_tokens", 8192))
    # Sanitize: empty reasons and embedded double-quotes produce visually
    # broken prompts and may bias the model. Replace quotes with single
    # quotes; substitute a neutral phrase when the original reason is empty.
    my_reason_safe = (my_reason or "(no stated reason)").replace('"', "'")
    other_reason_safe = (other_reason or "(no stated reason)").replace('"', "'")
    deliberate_prompt = (
        prompt
        + f"\n\nYour first reading produced: {my_verdict}. Your stated reason: \"{my_reason_safe}\""
        + f"\n\nA second independent judge produced: {other_verdict}. Their reason: \"{other_reason_safe}\""
        + "\n\nRe-examine the transcript with both readings on the table. "
        "If your original verdict still strikes you as correct, keep it; "
        "if the other reading exposes a flaw in yours, switch. Do not flip "
        "merely because another judge disagreed, only flip if their argument "
        "is genuinely stronger than your own.\n"
        "Output exactly PRO or CON on line 1, then one sentence naming the "
        "specific transcript moment that decides it."
    )
    try:
        verdict = llm.chat(
            "You are an impartial debate judge reconsidering after a panel split. "
            "You MUST pick a winner. Respond with exactly PRO or CON on the first line.",
            [{"role": "user", "content": deliberate_prompt}],
            judge_model, max_tokens=judge_max_tokens, temperature=0.3,
        )
    except Exception as e:
        log.warning("deliberation %s failed: %s", judge_model, str(e)[:80])
        return None
    clean = strip_thinking(verdict, tags) or verdict.strip()
    lines = [line.strip() for line in clean.split("\n") if line.strip()]
    if not lines:
        return None
    match = re.match(r"^[*\"]*(\w+)", lines[0].upper())
    winner = _VERDICT_WORDS.get(match.group(1)) if match else None
    if winner:
        return (winner, " ".join(lines[1:]))
    return None


def _judge_one(llm, config, judge_model: str, prompt: str, tags: list) -> tuple[str, str] | None:
    """Run a single judge model. Returns (verdict, reason) or None on failure.

    Each panel member is queried with itself as primary. Two things can divert
    a slot to `game.judge_fallback_model` (a fast, lightly loaded model with its
    own Chutes queue, typically the debater primary):
      - the per-model circuit breaker (see _breaker_for): if this judge model
        is already known degraded, attempt 0 skips straight to the fallback, or
        cheap-pings to verify recovery first. A judge model marked permanently
        dead (a prior 404) is likewise routed to the fallback.
      - a failed attempt 0: attempt 1 retries on the fallback.
    Without `judge_fallback_model` configured there is no breaker, and attempt 1
    repeats the original model (legacy behavior).

    Unlike the debate path, the panel is NOT reordered by health: each slot is a
    distinct vote and reshuffling would collapse panel diversity. The only
    adaptive move here is dropping a dead model to the fallback.

    Diversity rationale: even when the fallback fires, the panel still gets one
    vote per slot. Falling through to a single different model beats losing the
    vote entirely, which pushes the panel toward voiding the game.
    """
    cfg = config.get("game", {})
    # Thinking-heavy models (Kimi, Qwen3-Thinking, DeepSeek) routinely burn
    # 2000+ reasoning tokens before producing visible output. A 2048 ceiling
    # truncates them mid-`<think>` block with no PRO/CON line. Match the
    # intent_classifier ceiling so the panel has consistent headroom.
    judge_max_tokens = int(cfg.get("judge_max_tokens", 8192))
    fallback_model = cfg.get("judge_fallback_model") or None
    # Belt: a circuit breaker on the judge model, the same one the debate path
    # uses. When this judge model is degraded (e.g. GLM under a Chutes crunch)
    # we skip it straight to the fallback instead of burning the read-timeout
    # budget on a call that will fail. Only meaningful when a fallback exists.
    breaker = _breaker_for(judge_model) if fallback_model else None
    gate = breaker.gate() if breaker else "use"
    judge_dead = bool(breaker and breaker.dead)
    for attempt in range(2):
        # Attempt 0 normally uses the judge model; a dead model or an open
        # breaker redirects it to the fallback (skip), or a half-open breaker
        # verifies recovery with a cheap ping (probe). Attempt 1 is the fallback
        # model, sidestepping per-model throttling.
        if attempt == 0 and (judge_dead or gate == "skip"):
            model_to_use = fallback_model
        elif attempt == 0 and gate == "probe":
            try:
                ok, _ = llm.ping(judge_model)
            except Exception:
                ok = False  # a raising ping must still clear the half-open prober
            breaker.record(ok)
            model_to_use = judge_model if ok else fallback_model
        elif attempt == 0:
            model_to_use = judge_model
        else:
            model_to_use = fallback_model or judge_model
        try:
            verdict = llm.chat(
                "You are an impartial debate judge. You MUST pick a winner. "
                "Respond with exactly PRO or CON on the first line.",
                [{"role": "user", "content": prompt}],
                model_to_use, max_tokens=judge_max_tokens, temperature=0.3 + attempt * 0.2,
            )
        except ModelNotAvailableError as e:
            # 404: mark whichever model we actually called permanently dead so it
            # is dropped everywhere (this judge slot next time, and the debate
            # path), then fall through to the fallback / give up.
            _breaker_for(model_to_use).record(False, permanent=True)
            log.warning("judge model %s unavailable (404), dropping permanently: %s",
                        model_to_use, str(e)[:80])
            if attempt == 1:
                return None
            continue
        except Exception as e:
            if breaker and attempt == 0 and model_to_use == judge_model and gate in ("use", "probe"):
                breaker.record(False)
            log.warning("judge %s attempt %d (model=%s) failed: %s",
                        judge_model, attempt + 1, model_to_use, str(e)[:80])
            if attempt == 1:
                return None
            continue
        if breaker and attempt == 0 and model_to_use == judge_model and gate in ("use", "probe"):
            breaker.record(True)  # the model answered; healthy regardless of parse
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
    legacy single-judge config (judge_model + judge_model_fallbacks) only when
    judge_panel is absent or empty. Panel members vote independently in
    parallel; the majority verdict wins.

    Resolution, in order:
      - fewer than 2 valid votes (models down or returning malformed verdicts):
        the game is VOIDED rather than decided by a single judge. Each panel
        slot already retries once via game.judge_fallback_model (see
        _judge_one), so reaching this means at least two slots failed both
        attempts. Voids are recorded for audit but skip Elo / W-L in
        _play_round.
      - a tie (e.g. 1-1 on this 2-model panel): run one deliberation round in
        which each judge sees another's verdict, then take the converged
        majority, the lone second-round vote as a tiebreaker, or a draw with
        penalty if it stays split.
      - otherwise: the first-round majority wins.
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

    panel_record = [{"model": m, "verdict": w, "reason": (rsn or "")[:280],
                     "round": "first"}
                    for (w, rsn, m) in votes]

    pro_count = sum(1 for (w, _, _) in votes if w == "Pro")
    con_count = sum(1 for (w, _, _) in votes if w == "Con")
    total_votes = pro_count + con_count

    # Insufficient: fewer than 2 valid votes after per-judge retries. Void the
    # game rather than fall back to a single-judge tiebreaker — a flaky panel
    # shouldn't silently degrade to a less-deliberated verdict. Void games are
    # still recorded for audit but skip Elo / W-L bookkeeping in _play_round.
    if total_votes < 2:
        log.warning(f"panel incomplete: {total_votes}/{len(panel)} valid votes; voiding")
        return GameResult(motion, "void",
                          f"Panel incomplete {pro_count}-{con_count} → voided",
                          len(transcript), transcript, duration,
                          topic_data=topic_obj, topic_id=tid,
                          judge_panel_votes=panel_record)

    # Split: every panel member voted but they disagreed (1-1 in a 2-judge
    # panel, 2-2 in a 4-judge panel, etc.). Run a deliberation round first:
    # each judge sees the others' verdicts + reasoning and reconsiders.
    # Empirically converges ~50% of splits to a clean verdict; non-converging
    # cases fall through to draw-with-penalty (same as the no-deliberation
    # baseline).
    if pro_count == con_count:
        log.warning(f"panel split {pro_count}-{con_count}; running deliberation")
        new_votes: list = []
        # Pairing: each judge sees the next judge's verdict (mod panel size).
        # TODO: revisit if judge_panel grows past 2 — for larger panels, each
        # judge should see ALL other votes, not just one neighbor.
        with ThreadPoolExecutor(max_workers=len(votes)) as ex:
            futs = {}
            for i, (w, rsn, m) in enumerate(votes):
                other = votes[(i + 1) % len(votes)]
                futs[ex.submit(_judge_deliberate, llm, config, m, prompt,
                              w, rsn, other[0], other[1], tags)] = m
            for fut in as_completed(futs):
                r = fut.result()
                if r is not None:
                    new_votes.append((r[0], r[1], futs[fut]))

        # Append the deliberation round to panel_record so the archive shows both
        panel_record = panel_record + [
            {"model": m, "verdict": w, "reason": (rsn or "")[:280],
             "round": "deliberation"}
            for (w, rsn, m) in new_votes
        ]

        new_pro = sum(1 for (w, _, _) in new_votes if w == "Pro")
        new_con = sum(1 for (w, _, _) in new_votes if w == "Con")
        total_new = new_pro + new_con

        # Converged: 2+ votes, clear majority
        if total_new >= 2 and new_pro != new_con:
            winner = "Pro" if new_pro > new_con else "Con"
            explanation = next(rsn for (w, rsn, _) in new_votes if w == winner)
            log.info(f"deliberation converged: {new_pro}-{new_con} → {winner}")
            return GameResult(
                motion, winner,
                f"Panel split {pro_count}-{con_count} → deliberation {new_pro}-{new_con} → {winner}",
                len(transcript), transcript, duration, explanation,
                topic_data=topic_obj, topic_id=tid,
                judge_panel_votes=panel_record)

        # Partial: only one judge returned a valid second-round vote. Use it
        # as a tiebreaker rather than discarding informed signal. The single
        # responder saw both first-round verdicts and chose deliberately.
        if total_new == 1:
            winner = "Pro" if new_pro == 1 else "Con"
            explanation = new_votes[0][1]
            log.info(f"deliberation partial: 1 valid vote → {winner} (tiebreaker)")
            return GameResult(
                motion, winner,
                f"Panel split {pro_count}-{con_count} → deliberation 1-vote → {winner}",
                len(transcript), transcript, duration, explanation,
                topic_data=topic_obj, topic_id=tid,
                judge_panel_votes=panel_record)

        log.warning(f"deliberation also split {new_pro}-{new_con}; draw with penalty")
        return GameResult(
            motion, "draw",
            f"Panel split {pro_count}-{con_count} → deliberation {new_pro}-{new_con} → draw",
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


def run_title_fight(llm, config, king_hk, challenger_hk, strategies, topics,
                    workers, on_progress=None):
    """A quick final check that the new king actually beats the old king before
    the crown transfers.

    Every topic is played both ways (each fighter takes Pro on it once and Con
    once), so the two sides get an equal number of Pro and Con seats and the
    seat advantage cancels. Draws and infra voids are not counted either way.
    Elo is not read or updated here. The caller applies the win bar: the win
    count must be significant under a fair coin, so an evenly-matched challenger
    (50/50 vs the king) almost never clears it on variance; the new king must
    clearly beat the old one.

    Why a match and not Elo alone: the crown is winner-take-all, so it should
    turn on beating the sitting king head-to-head, the most direct merit test
    there is, rather than on an Elo number accrued against the wider field. It
    stays winnable: a clearly stronger challenger clears the bar easily; only a
    coin-flip-equal one is held back. This head-to-head is also the verification
    step sealed (private-strategy) commitments build on, where a challenge is
    settled by match outcome without either strategy being made public.

    Returns (ch_wins, king_wins, decided).
    """
    workers = max(1, workers)
    ch_wins = king_wins = 0
    for topic_obj in topics:
        # (pro, con) both ways: challenger Pro then challenger Con.
        pairs = [(challenger_hk, king_hk), (king_hk, challenger_hk)]
        out = _play_round_lockstep(llm, config, pairs, topic_obj, strategies,
                                   workers, on_progress=on_progress)
        for _i, pro, con, gr in out:
            if gr.winner == "Pro":
                w = pro
            elif gr.winner == "Con":
                w = con
            else:
                continue  # draw or void: proves nothing, excluded
            if w == challenger_hk:
                ch_wins += 1
            elif w == king_hk:
                king_wins += 1
    decided = ch_wins + king_wins
    log.info(f"title check: challenger {challenger_hk[:8]}… {ch_wins}-{king_wins} "
             f"king {king_hk[:8]}… ({decided}/{2 * len(topics)} decided)")
    return ch_wins, king_wins, decided


def run_tournament(llm, config, strategies, epoch_start_block: int, elo=None,
                   on_progress=None, on_game_turn=None):
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
        for idx, (pro, con) in enumerate(pairs):
            log.info(f"swiss R{round_num} pair: pro={pro[:8]}… "
                     f"con={con[:8]}… pro_cd={color_diff[pro]:+d} "
                     f"con_cd={color_diff[con]:+d} pro_elo={elo.get(pro):.1f} "
                     f"con_elo={elo.get(con):.1f}")
            provenance.log_game(epoch_start_block, round_num, idx, chosen_topic,
                                pro, con, strategies[pro], strategies[con],
                                elo.get(pro), elo.get(con))
        log.info(f"swiss R{round_num}/{num_rounds}: {len(pairs)} games")

        # Lockstep round: all games advance turn-by-turn in cross-game waves
        # so one slow LLM call blocks only its own wave, not the slowest
        # game's whole 10-turn arc. on_progress is pulsed per wave to keep
        # the systemd watchdog fed during long rounds.
        completed = _play_round_lockstep(
            llm, config, pairs, chosen_topic, strategies, workers,
            on_progress=on_progress, on_game_turn=on_game_turn,
        )

        completed.sort(key=lambda x: x[0])
        for _, pro, con, result in completed:
            reason = (result.reason or "").lower()
            # `in` instead of `startswith` so the panel-fallback path's
            # prefixed form ("Panel-insufficient → Judge error: ...") is
            # also recognized as an infrastructure failure.
            errored = "llm error" in reason or "judge error" in reason
            voided = result.winner == "void"
            if errored or voided:
                # Don't burn the matchup slot on infra failure or panel void:
                # the same pair may succeed in a later round when Chutes
                # recovers or the panel produces enough valid votes. Without
                # this, a sustained outage causes every round to mark all
                # pairs as played, forcing later rounds into the floater
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
