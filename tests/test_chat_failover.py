"""Retry/failover behavior of LLM.chat and _chat under Chutes capacity errors.

The business rule these tests encode: when a model returns a capacity 429 it is
saturated *now*, so we must fail over to a healthy fallback on the first hit
rather than burning the 8-attempt backoff loop on it. The primary stays first
on every call, so a model that has capacity is still used (we "prefer GLM if it
is up"). Timeouts keep their separate, deliberate 3-attempt cap.
"""
import logging

import pytest

from compelle import engine
from compelle.engine import LLM, _chat, _judge_one


def _resp(content):
    msg = type("M", (), {"content": content})()
    choice = type("C", (), {"message": msg})()
    return type("R", (), {"choices": [choice]})()


class FakeClient:
    """Stands in for the OpenAI client. `by_model` maps a model id to either
    ('raise', exc) or ('ok', text); every call is recorded in `calls` so a test
    can assert exactly how many attempts each model got."""

    def __init__(self, by_model):
        self.by_model = by_model
        self.calls = []
        self.token_budgets = []  # max_tokens per call; a 1 marks a recovery ping
        self.ping_ok = set()     # models whose 1-token ping succeeds even if heavy calls fail
        self.chat = self
        self.completions = self

    def create(self, model, messages, max_tokens, temperature):
        self.calls.append(model)
        self.token_budgets.append(max_tokens)
        if max_tokens == 1 and model in self.ping_ok:
            return _resp("pong")  # cheap recovery ping succeeds even when heavy calls fail
        kind, payload = self.by_model[model]
        if kind == "raise":
            raise payload
        if kind == "seq":
            # A sequence of responses for one model: pop until the last remains,
            # so ('seq', ['', 'answer']) returns '' once then 'answer' after.
            payload = payload.pop(0) if len(payload) > 1 else payload[0]
        return _resp(payload)


@pytest.fixture
def fast(monkeypatch):
    # Neutralize the global 1 req/s limiter and all backoff sleeps so the
    # retry/failover logic runs at full speed and deterministically.
    monkeypatch.setattr(engine, "_acquire_rate_token", lambda: None)
    monkeypatch.setattr(engine.time, "sleep", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _isolate_breaker():
    # Per-model breakers and the adaptive-lead log state are process-wide; clear
    # both before each test so neither they nor the pre-existing tests leak state
    # across the suite.
    engine._breakers.clear()
    engine._adaptive_lead.clear()


@pytest.fixture
def breaker(monkeypatch):
    # Controllable monotonic clock so cooldown / half-open transitions are
    # deterministic. Breakers are created lazily by _breaker_for (threshold 3,
    # 30s base, 300s cap, the module defaults), so we just clear the registry
    # and drive the clock. The returned handle's .advance(seconds) moves time.
    clock = {"t": 1000.0}
    monkeypatch.setattr(engine.time, "monotonic", lambda: clock["t"])
    engine._breakers.clear()

    class _Clock:
        def advance(self, seconds):
            clock["t"] += seconds

    return _Clock()


@pytest.fixture
def engine_logs():
    # Capture compelle.engine messages straight off its logger. We can't use
    # pytest's caplog here: compelle.validator runs logging.basicConfig() at
    # import, so once the full suite imports it, caplog's root-handler capture
    # breaks (assertions then pass in isolation but fail in-suite). Reading
    # directly off the engine logger is order-independent.
    logger = logging.getLogger("compelle.engine")
    msgs = []

    class _Capture(logging.Handler):
        def emit(self, record):
            msgs.append(record.getMessage())

    handler = _Capture()
    prev = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield msgs
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev)


def _llm(by_model):
    llm = LLM("http://test", "key")
    llm.client = FakeClient(by_model)
    return llm


MSGS = [{"role": "user", "content": "hi"}]


def test_capacity_429_fails_over_after_one_attempt(fast):
    exc = Exception("Error code: 429 - infrastructure is at maximum capacity")
    llm = _llm({"glm": ("raise", exc)})
    with pytest.raises(Exception):
        llm.chat("sys", MSGS, "glm")
    assert llm.client.calls == ["glm"]  # one attempt, not eight


def test_chat_fails_over_to_healthy_model_on_429(fast):
    exc = Exception("Error code: 429 - too many requests")
    llm = _llm({"glm": ("raise", exc), "qwen": ("ok", "fallback answer")})
    out = _chat(llm, "sys", MSGS, "glm", ["qwen"], 256, 0.6)
    assert out == "fallback answer"
    assert llm.client.calls == ["glm", "qwen"]  # primary hit once, then over


def test_happy_path_uses_primary_once(fast):
    # Capacity cap must not perturb the common case ("prefer GLM if it is up").
    llm = _llm({"glm": ("ok", "primary answer")})
    out = _chat(llm, "sys", MSGS, "glm", ["qwen"], 256, 0.6)
    assert out == "primary answer"
    assert llm.client.calls == ["glm"]


def test_timeout_path_still_retries_three_times(fast):
    # Regression guard: the 3-attempt timeout cap (and the deliberate 90s GLM
    # read window behind it) is unchanged by the capacity fix.
    exc = Exception("request timed out")
    llm = _llm({"glm": ("raise", exc)})
    with pytest.raises(Exception):
        llm.chat("sys", MSGS, "glm")
    assert llm.client.calls == ["glm", "glm", "glm"]


# --- Empty-completion handling ----------------------------------------------
# Business rule: a 200 carrying blank content is not a usable turn (provider
# hiccup, or a reasoning model that emitted only hidden reasoning_content). It
# raises no exception, so it used to be stored as an empty turn (the "broken
# opening" / blank mid-debate turn on the site). It must be treated like any
# other transient failure: re-roll the same model once (empties are
# intermittent), then fall over to the next model, then label the game a void.


def test_chat_re_rolls_same_model_on_empty_then_returns_content(fast):
    # The common case: a one-off blank. One same-model re-roll lands content, so
    # we keep the primary's voice instead of falling over on the first blip.
    llm = _llm({"glm": ("seq", ["", "real opening"])})
    assert llm.chat("sys", MSGS, "glm") == "real opening"
    assert llm.client.calls == ["glm", "glm"]


def test_chat_gives_up_on_persistent_empty_after_cap(fast):
    # Whitespace-only counts as empty (it is .strip() that decides), and a model
    # that never produces content is abandoned after the cap, not looped forever.
    llm = _llm({"glm": ("ok", "   ")})
    assert llm.chat("sys", MSGS, "glm") == ""
    assert llm.client.calls == ["glm"] * engine._MAX_EMPTY_ATTEMPTS


def test_chat_fails_over_when_primary_only_returns_empty(fast):
    # Primary blank to exhaustion -> fall to the healthy fallback rather than
    # store an empty turn. The primary is tried (its empty cap) before the over.
    llm = _llm({"glm": ("ok", ""), "qwen": ("ok", "fallback answer")})
    out = _chat(llm, "sys", MSGS, "glm", ["qwen"], 256, 0.6)
    assert out == "fallback answer"
    assert llm.client.calls == ["glm"] * engine._MAX_EMPTY_ATTEMPTS + ["qwen"]


def test_chat_raises_when_every_model_returns_empty(fast):
    # All models blank -> _chat raises so the round loop labels the game a void
    # ("LLM error: empty completion ...") instead of recording a blank turn.
    llm = _llm({"glm": ("ok", ""), "qwen": ("ok", "  ")})
    with pytest.raises(Exception) as ei:
        _chat(llm, "sys", MSGS, "glm", ["qwen"], 256, 0.6)
    assert "empty completion" in str(ei.value)


def test_blank_completions_demote_the_primary(fast, breaker):
    # A primary that keeps returning blank content is degraded exactly like one
    # that keeps raising: each blank drops its health, so the adaptive selector
    # demotes it below the healthy fallback and stops paying the per-call empty
    # tax on it. (The health-ranking contract lives in test_adaptive_selector.py;
    # this asserts blanks feed it, not just raised exceptions.)
    llm = _llm({"glm": ("ok", ""), "qwen": ("ok", "fallback")})
    for _ in range(3):
        assert _chat(llm, "s", MSGS, "glm", ["qwen"], 256, 0.6) == "fallback"
    glm_calls = llm.client.calls.count("glm")
    assert _chat(llm, "s", MSGS, "glm", ["qwen"], 256, 0.6) == "fallback"
    assert llm.client.calls.count("glm") == glm_calls  # demoted: GLM no longer tried


# The debate-path circuit-breaker tests (open-after-threshold, recover-via-ping,
# probe-failure-grows-cooldown) were replaced when _chat moved from the 3-strike
# breaker to health-ranked failover: the debate path now demotes a flapping
# primary by reliability EWMA and recovers it by time-decay, covered in
# tests/test_adaptive_selector.py. The gate/probe/cooldown breaker itself is
# unchanged and still exercised on the JUDGE path (test_judge_breaker_* below).


# --- Preflight chain ---------------------------------------------------------
# Business rule: the epoch preflight must gate on the whole failover chain, not
# the primary alone. A 429 on the primary while a fallback is reachable must NOT
# skip the tournament (that was burning whole epochs during a primary-only
# capacity crunch); only a total outage (every model down) skips the round.


def test_ping_chain_starts_on_first_reachable_fallback(fast):
    # Primary and the first fallback are at capacity; the chain proceeds on the
    # first model that answers, so the epoch runs instead of being skipped.
    exc = Exception("Error code: 429 - infrastructure is at maximum capacity")
    llm = _llm({"glm": ("raise", exc), "glm51": ("raise", exc), "deepseek": ("ok", "pong")})
    ok, working, last = llm.ping_chain(["glm", "glm51", "deepseek"])
    assert (ok, working) == (True, "deepseek")
    assert llm.client.calls == ["glm", "glm51", "deepseek"]


def test_ping_chain_prefers_primary_when_up(fast):
    # If the primary answers, the chain stops there (one ping) and reports it as
    # the working model, so "prefer the primary if it is up" holds at preflight.
    llm = _llm({"glm": ("ok", "pong"), "deepseek": ("ok", "pong")})
    ok, working, last = llm.ping_chain(["glm", "deepseek"])
    assert (ok, working) == (True, "glm")
    assert llm.client.calls == ["glm"]


def test_ping_chain_fails_only_on_total_outage(fast):
    # Every model down -> preflight fails and the validator skips the epoch (the
    # deliberate, non-wasteful behavior); the last error is surfaced upward.
    exc = Exception("Error code: 429 - infrastructure is at maximum capacity")
    llm = _llm({"glm": ("raise", exc), "deepseek": ("raise", exc)})
    ok, working, last = llm.ping_chain(["glm", "deepseek"])
    assert ok is False
    assert working == ""
    assert "429" in last


def test_breaker_never_skips_without_a_fallback(fast, breaker):
    # With no fallback there is nowhere to skip to, so the breaker is bypassed
    # and the primary is tried on every call.
    exc = Exception("Error code: 429 - infrastructure is at maximum capacity")
    llm = _llm({"glm": ("raise", exc)})
    for _ in range(5):
        with pytest.raises(Exception):
            _chat(llm, "s", MSGS, "glm", [], 256, 0.6)
    assert llm.client.calls == ["glm"] * 5


# --- Judge-panel circuit breaker --------------------------------------------
# Business rule: the same per-model breaker also guards the judge path. A panel
# judge model (e.g. GLM) that keeps failing must not burn the read-timeout
# budget on every judged game; once its breaker opens, _judge_one skips it
# straight to the configured judge_fallback_model, and recovers by cheap ping.

JUDGE_CFG = {"game": {"judge_max_tokens": 64, "judge_fallback_model": "qwen"}}


def test_judge_breaker_skips_degraded_judge_model(fast, breaker):
    # Three consecutive judge failures open the model's breaker; the next judged
    # game must skip the judge model entirely and use the fallback.
    exc = Exception("Error code: 429 - infrastructure is at maximum capacity")
    llm = _llm({"glm": ("raise", exc), "qwen": ("ok", "CON\nstronger case")})
    for _ in range(3):
        assert _judge_one(llm, JUDGE_CFG, "glm", "t", ["think"]) == ("Con", "stronger case")
    assert llm.client.calls.count("glm") == 3      # three failed first-attempts
    # OPEN now: the GLM judge slot goes straight to the fallback, no GLM call.
    assert _judge_one(llm, JUDGE_CFG, "glm", "t", ["think"]) == ("Con", "stronger case")
    assert llm.client.calls.count("glm") == 3      # GLM judge was not tried again
    assert llm.client.calls[-1] == "qwen"


def test_judge_breaker_recovers_via_ping(fast, breaker):
    # Once the cooldown lapses the judge breaker half-opens and probes the judge
    # model with a one-token ping; a healthy reply closes it so GLM judges again.
    exc = Exception("Error code: 429 - infrastructure is at maximum capacity")
    llm = _llm({"glm": ("raise", exc), "qwen": ("ok", "CON\nfallback")})
    for _ in range(3):
        _judge_one(llm, JUDGE_CFG, "glm", "t", ["think"])         # open it
    llm.client.by_model["glm"] = ("ok", "PRO\nglm back")          # GLM judge recovers
    breaker.advance(31)                                           # past the 30s cooldown
    assert _judge_one(llm, JUDGE_CFG, "glm", "t", ["think"]) == ("Pro", "glm back")
    assert llm.client.token_budgets.count(1) == 1                # exactly one cheap ping


def test_judge_breaker_counts_heavy_failure_after_probe(fast, breaker):
    # Slow-but-pingable crunch: the 1-token recovery ping succeeds while the
    # full judge call still fails. The probe closes the breaker, but that heavy
    # failure must still be counted, otherwise a ping alone whitewashes a model
    # that cannot actually judge and the breaker never re-trips.
    exc = Exception("Error code: 429 - infrastructure is at maximum capacity")
    llm = _llm({"glm": ("raise", exc), "qwen": ("ok", "CON\nfb")})
    for _ in range(3):
        _judge_one(llm, JUDGE_CFG, "glm", "t", ["think"])         # open the breaker
    llm.client.ping_ok.add("glm")                                 # ping now succeeds...
    breaker.advance(31)                                           # half-open
    _judge_one(llm, JUDGE_CFG, "glm", "t", ["think"])             # probe: ping ok, heavy fails
    b = engine._breaker_for("glm")
    assert b._state == "closed"     # the ping recovered it
    assert b._fails == 1            # but the heavy failure on that probe was recorded


def test_judge_breaker_survives_a_raising_ping(fast, breaker, monkeypatch):
    # If ping() is ever made to raise (e.g. an overlaid LLM), the half-open
    # prober must still clear, not stick True forever and pin the model to its
    # fallback for the life of the process.
    exc = Exception("Error code: 429 - infrastructure is at maximum capacity")
    llm = _llm({"glm": ("raise", exc), "qwen": ("ok", "CON\nfb")})
    for _ in range(3):
        _judge_one(llm, JUDGE_CFG, "glm", "t", ["think"])         # open the breaker

    def _boom(*a, **k):
        raise RuntimeError("ping blew up")
    monkeypatch.setattr(llm, "ping", _boom)
    breaker.advance(31)                                           # half-open
    _judge_one(llm, JUDGE_CFG, "glm", "t", ["think"])             # probe: ping raises
    b = engine._breaker_for("glm")
    assert b._probing is False      # prober cleared despite the raise
    breaker.advance(61)             # past the re-grown cooldown
    assert b.gate() == "probe"      # it can probe again, not stuck on skip
