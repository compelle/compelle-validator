"""Thinking models get the chat_template_kwargs reasoning flag; others don't.

Business rule: DeepSeek-V3.2-TEE is a HYBRID model. Reasoning is OFF by default,
so a plain call returns a non-reasoned answer. To get the deliberate, traced
answer we want from debaters and the judge, we must send
extra_body={"chat_template_kwargs": {"thinking": True}}. The reasoning comes back
in a separate reasoning_content field, never in the answer, so nothing leaks.

Two invariants:
  1. A model listed in thinking_models gets the extra_body flag on every call.
  2. A model NOT listed gets a clean call with no extra_body (sending the flag to
     a non-hybrid model is wrong: it is either ignored or rejected, and it would
     mask a config typo by silently "working").
"""
import pytest

from compelle import engine
from compelle.engine import LLM


def _resp(content):
    msg = type("M", (), {"content": content})()
    choice = type("C", (), {"message": msg})()
    return type("R", (), {"choices": [choice]})()


class KwargRecordingClient:
    """OpenAI-client stand-in that records the kwargs of every create() call."""

    def __init__(self):
        self.calls = []  # one kwargs dict per create()
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _resp("an argument")


THINK_FLAG = {"chat_template_kwargs": {"thinking": True}}


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(engine, "_acquire_rate_token", lambda: None)
    monkeypatch.setattr(engine.time, "sleep", lambda *a, **k: None)
    engine._breakers.clear()


def _llm(thinking_models):
    llm = LLM("http://test", "key", thinking_models=thinking_models)
    llm.client = KwargRecordingClient()
    return llm


def test_thinking_model_gets_the_flag(fast):
    llm = _llm(["deepseek-ai/DeepSeek-V3.2-TEE"])
    llm.chat("sys", [{"role": "user", "content": "hi"}],
             model="deepseek-ai/DeepSeek-V3.2-TEE")
    assert llm.client.calls[0].get("extra_body") == THINK_FLAG


def test_non_thinking_model_gets_no_flag(fast):
    # GLM is a fallback but not in thinking_models: it must get a clean call.
    llm = _llm(["deepseek-ai/DeepSeek-V3.2-TEE"])
    llm.chat("sys", [{"role": "user", "content": "hi"}],
             model="zai-org/GLM-5.1-TEE")
    assert "extra_body" not in llm.client.calls[0]


def test_empty_thinking_models_never_sends_the_flag(fast):
    # Default construction (no thinking_models) must behave exactly as before:
    # no extra_body on any call, so existing models are untouched.
    llm = _llm([])
    llm.chat("sys", [{"role": "user", "content": "hi"}],
             model="deepseek-ai/DeepSeek-V3.2-TEE")
    assert "extra_body" not in llm.client.calls[0]
