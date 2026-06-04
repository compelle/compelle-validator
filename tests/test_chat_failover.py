"""Retry/failover behavior of LLM.chat and _chat under Chutes capacity errors.

The business rule these tests encode: when a model returns a capacity 429 it is
saturated *now*, so we must fail over to a healthy fallback on the first hit
rather than burning the 8-attempt backoff loop on it. The primary stays first
on every call, so a model that has capacity is still used (we "prefer GLM if it
is up"). Timeouts keep their separate, deliberate 3-attempt cap.
"""
import pytest

from compelle import engine
from compelle.engine import LLM, _chat


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
        self.chat = self
        self.completions = self

    def create(self, model, messages, max_tokens, temperature):
        self.calls.append(model)
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
