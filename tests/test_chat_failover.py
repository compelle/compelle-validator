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
        self.chat = self
        self.completions = self

    def create(self, model, messages, max_tokens, temperature):
        self.calls.append(model)
        self.token_budgets.append(max_tokens)
        kind, payload = self.by_model[model]
        if kind == "raise":
            raise payload
        return _resp(payload)


@pytest.fixture
def fast(monkeypatch):
    # Neutralize the global 1 req/s limiter and all backoff sleeps so the
    # retry/failover logic runs at full speed and deterministically.
    monkeypatch.setattr(engine, "_acquire_rate_token", lambda: None)
    monkeypatch.setattr(engine.time, "sleep", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _isolate_breaker():
    # Per-model breakers are process-wide state; clear the registry before each
    # test so neither it nor the pre-existing tests leak state across the suite.
    engine._breakers.clear()


@pytest.fixture
def breaker(monkeypatch):
    # Controllable monotonic clock so cooldown / half-open transitions are
    # deterministic. Breakers are created lazily by _breaker_for (threshold 3,
    # 30s base, 300s cap — the module defaults), so we just clear the registry
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


# --- Primary circuit breaker -------------------------------------------------
# Business rule: a primary (GLM) that keeps failing must stop taxing every call.
# After N consecutive failures the breaker opens and _chat skips the primary
# straight to the fallback; it auto-recovers by probing with a cheap ping and
# closes again when the primary answers. This keeps "prefer GLM if it is up"
# while removing the per-call timeout tax when it is down.


def test_breaker_opens_and_skips_primary_after_threshold(fast, breaker, engine_logs):
    # Three consecutive primary failures (the threshold) trip the breaker; the
    # next call must not touch the primary at all.
    exc = Exception("Error code: 429 - infrastructure is at maximum capacity")
    llm = _llm({"glm": ("raise", exc), "qwen": ("ok", "fallback")})
    for _ in range(3):
        assert _chat(llm, "s", MSGS, "glm", ["qwen"], 256, 0.6) == "fallback"
    assert llm.client.calls.count("glm") == 3
    # OPEN now: primary skipped, straight to the fallback.
    assert _chat(llm, "s", MSGS, "glm", ["qwen"], 256, 0.6) == "fallback"
    assert llm.client.calls.count("glm") == 3      # GLM was not tried again
    assert llm.client.calls[-1] == "qwen"
    # The transition must be observable: opening is a single logged event, not
    # a silent degrade (this is the whole point of the breaker on mainnet).
    assert any("circuit breaker OPEN" in m for m in engine_logs)


def test_breaker_recovers_via_cheap_ping(fast, breaker, engine_logs):
    # Once the cooldown lapses the breaker half-opens; the next call probes with
    # a one-token ping, and a healthy reply closes it so GLM is used again.
    exc = Exception("Error code: 429 - infrastructure is at maximum capacity")
    llm = _llm({"glm": ("raise", exc), "qwen": ("ok", "fallback")})
    for _ in range(3):
        _chat(llm, "s", MSGS, "glm", ["qwen"], 256, 0.6)       # open it
    llm.client.by_model["glm"] = ("ok", "primary back")        # GLM recovers
    breaker.advance(31)                                        # past the 30s cooldown
    assert _chat(llm, "s", MSGS, "glm", ["qwen"], 256, 0.6) == "primary back"
    assert llm.client.token_budgets.count(1) == 1              # exactly one cheap ping
    # Closed again: a later call uses GLM directly, no further ping.
    assert _chat(llm, "s", MSGS, "glm", ["qwen"], 256, 0.6) == "primary back"
    assert llm.client.token_budgets.count(1) == 1
    # The probe and the recovery are both logged so an operator can see the
    # breaker heal itself rather than infer it from the absence of warnings.
    assert any("HALF-OPEN" in m for m in engine_logs)
    assert any("circuit breaker CLOSED" in m for m in engine_logs)


def test_breaker_probe_failure_grows_cooldown(fast, breaker):
    # A failed recovery probe re-opens the breaker with a doubled cooldown, and
    # no further probe fires until that longer window elapses.
    exc = Exception("Error code: 429 - infrastructure is at maximum capacity")
    llm = _llm({"glm": ("raise", exc), "qwen": ("ok", "fallback")})
    for _ in range(3):
        _chat(llm, "s", MSGS, "glm", ["qwen"], 256, 0.6)       # open @ 30s cooldown
    breaker.advance(31)                                        # half-open
    _chat(llm, "s", MSGS, "glm", ["qwen"], 256, 0.6)           # probe fails -> 60s cooldown
    assert llm.client.token_budgets.count(1) == 1
    breaker.advance(40)                                        # still inside the 60s window
    _chat(llm, "s", MSGS, "glm", ["qwen"], 256, 0.6)           # still OPEN -> skip, no ping
    assert llm.client.token_budgets.count(1) == 1
    breaker.advance(25)                                        # past the 60s window
    _chat(llm, "s", MSGS, "glm", ["qwen"], 256, 0.6)           # half-open again -> ping
    assert llm.client.token_budgets.count(1) == 2


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
# budget on every judged game — once its breaker opens, _judge_one skips it
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
