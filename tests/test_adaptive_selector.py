"""Adaptive model selector: the debate path ranks its failover chain by live
model health and drops permanently-unavailable models.

The business rules these tests encode (the operator ask: "automatically and
adaptively decide to move to Qwen primary in cases such as this; if models are
no longer available it shouldn't even bother asking"):

  1. A 404 (model not found) is permanent. The model is dropped from the chain
     and never called or pinged again — not on the next turn, not at preflight.
  2. A primary that fails is demoted below a fallback that is answering, so the
     reliable model leads the very next call. We do not wait for three strikes.
  3. A demoted model recovers on its own: with no traffic its health decays back
     toward neutral, and once it out-ranks the fallback again it reclaims the
     lead — after a cheap one-token ping confirms it is actually back, so a still-
     broken model costs a ping, not a full read-timeout.
  4. When every model is healthy the chain keeps config order (the configured
     primary leads); the selector adds no churn to the common case.
  5. If the whole chain has 404'd, _chat raises cleanly so the round loop voids
     the game instead of looping over dead models.

The short-memory circuit breaker (gate/probe/cooldown) is unchanged and still
guards the JUDGE path; see test_chat_failover.py::test_judge_breaker_*.
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
    """Stands in for the OpenAI client. `by_model` maps a model id to ('raise',
    exc) or ('ok', text); every call is recorded in `calls`, and `token_budgets`
    records each call's max_tokens so a 1 marks a recovery ping. Models in
    `ping_ok` answer a 1-token ping even if their heavy calls fail."""

    def __init__(self, by_model):
        self.by_model = by_model
        self.calls = []
        self.token_budgets = []
        self.ping_ok = set()
        self.chat = self
        self.completions = self

    def create(self, model, messages, max_tokens, temperature):
        self.calls.append(model)
        self.token_budgets.append(max_tokens)
        if max_tokens == 1 and model in self.ping_ok:
            return _resp("pong")
        kind, payload = self.by_model[model]
        if kind == "raise":
            raise payload
        return _resp(payload)


@pytest.fixture
def fast(monkeypatch):
    # Neutralize the global 1 req/s limiter and all backoff sleeps so the
    # selector logic runs at full speed and deterministically.
    monkeypatch.setattr(engine, "_acquire_rate_token", lambda: None)
    monkeypatch.setattr(engine.time, "sleep", lambda *a, **k: None)


@pytest.fixture
def clock(monkeypatch):
    # Controllable monotonic clock so health decay is deterministic. `.advance`
    # moves time forward; the health half-life is read off engine constants.
    t = {"v": 1000.0}
    monkeypatch.setattr(engine.time, "monotonic", lambda: t["v"])

    class _Clock:
        def advance(self, seconds):
            t["v"] += seconds

    return _Clock()


@pytest.fixture(autouse=True)
def _isolate():
    # Per-model health units and the adaptive-lead log state are process-wide;
    # clear both before each test.
    engine._breakers.clear()
    engine._adaptive_lead.clear()


@pytest.fixture
def engine_logs():
    # Capture compelle.engine messages straight off its logger (caplog is
    # unreliable here: compelle.validator runs logging.basicConfig() at import).
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


# --- Rule 1: 404 is permanent ------------------------------------------------


def test_404_model_dropped_and_never_retried(fast, clock, engine_logs):
    exc = Exception("Error code: 404 - model not found: deepseek")
    llm = _llm({"deepseek": ("raise", exc), "qwen": ("ok", "answer")})
    # The configured primary 404s once, falls over to the fallback.
    assert _chat(llm, "s", MSGS, "deepseek", ["qwen"], 256, 0.6) == "answer"
    assert llm.client.calls == ["deepseek", "qwen"]
    assert engine._breaker_for("deepseek").dead is True
    assert any("dropping from the chain permanently" in m for m in engine_logs)
    # Every later call skips the dead model entirely — never called, never pinged.
    llm.client.calls.clear()
    llm.client.token_budgets.clear()
    for _ in range(3):
        assert _chat(llm, "s", MSGS, "deepseek", ["qwen"], 256, 0.6) == "answer"
    assert "deepseek" not in llm.client.calls
    assert llm.client.token_budgets.count(1) == 0  # no probe of a dead model


def test_ping_chain_skips_dead_models_at_preflight(fast, clock):
    # "shouldn't even bother asking" extends to the per-epoch preflight: a model
    # already known dead is not pinged; the chain proceeds on a live one.
    engine._breaker_for("deepseek").record(False, permanent=True)
    llm = _llm({"deepseek": ("ok", "pong"), "qwen": ("ok", "pong")})
    ok, working, _ = llm.ping_chain(["deepseek", "qwen"])
    assert (ok, working) == (True, "qwen")
    assert llm.client.calls == ["qwen"]  # the dead model was never pinged


# --- Rule 2: a flapping primary is demoted below a reliable fallback ---------


def test_flapping_primary_demoted_below_reliable_fallback(fast, clock, engine_logs):
    exc = Exception("Error code: 429 - infrastructure is at maximum capacity")
    llm = _llm({"glm": ("raise", exc), "qwen": ("ok", "qwen answer")})
    # Call 1: primary leads (config order), fails, falls over to the fallback.
    assert _chat(llm, "s", MSGS, "glm", ["qwen"], 256, 0.6) == "qwen answer"
    assert llm.client.calls == ["glm", "qwen"]
    # Call 2: the single failure already dropped glm's health below qwen's, so
    # qwen leads and glm is not retried — no waiting for three strikes.
    llm.client.calls.clear()
    assert _chat(llm, "s", MSGS, "glm", ["qwen"], 256, 0.6) == "qwen answer"
    assert llm.client.calls == ["qwen"]
    # The lead change is logged once, for the operator watching a brownout.
    assert any("adaptive selector" in m and "leading with qwen" in m for m in engine_logs)


# --- Rule 3: a demoted primary recovers by decay, verified by a cheap ping ---


def test_demoted_primary_recovers_after_decay_with_probe(fast, clock):
    exc = Exception("Error code: 429 - infrastructure is at maximum capacity")
    llm = _llm({"glm": ("raise", exc), "qwen": ("ok", "qwen answer")})
    _chat(llm, "s", MSGS, "glm", ["qwen"], 256, 0.6)  # glm fails -> demoted
    assert llm.client.calls[-1] == "qwen"
    # glm recovers upstream; let enough half-lives pass that its health decays
    # back above the fallback's.
    llm.client.by_model["glm"] = ("ok", "glm back")
    llm.client.ping_ok.add("glm")
    clock.advance(engine._HEALTH_DECAY_HALFLIFE * 8)
    llm.client.calls.clear()
    llm.client.token_budgets.clear()
    out = _chat(llm, "s", MSGS, "glm", ["qwen"], 256, 0.6)
    assert out == "glm back"                       # primary reclaims the lead
    assert llm.client.calls[0] == "glm"
    assert llm.client.token_budgets.count(1) == 1  # exactly one probationary ping


def test_demoted_primary_stays_down_if_probe_fails(fast, clock):
    # Same recovery path, but the model is still broken: the cheap probe fails,
    # so we fall back without ever paying a full read-timeout on the bad model.
    exc = Exception("Error code: 429 - infrastructure is at maximum capacity")
    llm = _llm({"glm": ("raise", exc), "qwen": ("ok", "qwen answer")})
    _chat(llm, "s", MSGS, "glm", ["qwen"], 256, 0.6)  # demote glm
    clock.advance(engine._HEALTH_DECAY_HALFLIFE * 8)  # health decays it back to lead
    llm.client.calls.clear()
    llm.client.token_budgets.clear()
    assert _chat(llm, "s", MSGS, "glm", ["qwen"], 256, 0.6) == "qwen answer"
    # The probationary ping fired (1-token) and failed; the heavy glm call (256
    # tokens) was never made — the fallback served instead.
    assert llm.client.token_budgets.count(1) == 1
    assert llm.client.token_budgets.count(256) == 1   # only qwen's real call
    assert llm.client.calls[-1] == "qwen"


# --- Rule 4: all-healthy chain keeps config order ----------------------------


def test_config_order_preserved_when_all_healthy(fast, clock):
    llm = _llm({"glm": ("ok", "primary"), "qwen": ("ok", "fb")})
    for _ in range(5):
        assert _chat(llm, "s", MSGS, "glm", ["qwen"], 256, 0.6) == "primary"
    assert set(llm.client.calls) == {"glm"}        # fallback never needed
    assert llm.client.token_budgets.count(1) == 0  # no spurious probes


# --- Rule 5: an all-dead chain raises cleanly --------------------------------


def test_all_dead_chain_raises_model_not_available(fast, clock):
    exc = Exception("Error code: 404 - model not found")
    llm = _llm({"glm": ("raise", exc), "qwen": ("raise", exc)})
    with pytest.raises(engine.ModelNotAvailableError):
        _chat(llm, "s", MSGS, "glm", ["qwen"], 256, 0.6)
    assert engine._breaker_for("glm").dead and engine._breaker_for("qwen").dead
    # Chain is now known-empty: raises immediately, calling no model.
    llm.client.calls.clear()
    with pytest.raises(engine.ModelNotAvailableError):
        _chat(llm, "s", MSGS, "glm", ["qwen"], 256, 0.6)
    assert llm.client.calls == []


# --- Judge path: dead-drop only, no reordering -------------------------------

JUDGE_CFG = {"game": {"judge_max_tokens": 64, "judge_fallback_model": "qwen"}}


def test_judge_routes_dead_model_to_fallback(fast, clock):
    exc = Exception("Error code: 404 - model not found: glm")
    llm = _llm({"glm": ("raise", exc), "qwen": ("ok", "CON\nbetter case")})
    # First judged game: glm 404s on attempt 0, is marked dead, attempt 1 uses
    # the fallback so the slot still casts a vote.
    assert _judge_one(llm, JUDGE_CFG, "glm", "t", ["think"]) == ("Con", "better case")
    assert engine._breaker_for("glm").dead is True
    # Next judged game: the dead model is routed straight to the fallback and is
    # never called again.
    llm.client.calls.clear()
    assert _judge_one(llm, JUDGE_CFG, "glm", "t", ["think"]) == ("Con", "better case")
    assert "glm" not in llm.client.calls
