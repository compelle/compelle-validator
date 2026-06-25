"""Pro's opening turn must be seeded with a kickoff user message.

Business rule: Pro speaks first, so its opening LLM call has no prior turns.
Many models (DeepSeek included) given only a system message and no user turn
return empty content about half the time, or parrot the missing scaffolding
("just begin...", "Just start speaking to your opponent"). That produced empty
or garbled Pro first turns on roughly half of all games. The fix seeds a single
user kickoff so the model always has a turn to answer.

Two invariants the fix must hold, both tested here:
  1. The opening is generated WITH a user turn (never system-only).
  2. The kickoff is internal: it never reaches the stored transcript, and the
     opponent's view of Pro's opening is Pro's real argument, not the kickoff.
"""
import copy

import pytest

from compelle import engine
from compelle.engine import LLM, play_game, _FIRST_TURN_KICKOFF


def _resp(content):
    msg = type("M", (), {"content": content})()
    choice = type("C", (), {"message": msg})()
    return type("R", (), {"choices": [choice]})()


class RecordingClient:
    """OpenAI-client stand-in that records the full message list of every call
    and returns a fixed, non-conceding reply."""

    def __init__(self, reply):
        self.reply = reply
        self.messages_seen = []  # one entry per create() call
        self.chat = self
        self.completions = self

    def create(self, model, messages, max_tokens, temperature):
        self.messages_seen.append(copy.deepcopy(messages))
        return _resp(self.reply)


CONFIG = {
    "game_prompt": "You are {side}. Motion: {topic}. Strategy: {strategy}.{context} Date: {date}",
    "thinking_tags": ["think"],
    "concession": {"symbol": "Δ", "min_length": 50},
    "game": {
        "model": "glm",
        "model_fallbacks": [],
        "max_tokens_per_turn": 256,
        "temperature": 0.6,
        "max_turns": 1,       # one Pro + one Con turn, then draw (no judge call)
        "allow_draws": True,
    },
}
TOPIC = {"motion": "Cats are better than dogs."}
PRO_REPLY = "Cats are independent and clean, which makes them the superior pet."


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(engine, "_acquire_rate_token", lambda: None)
    monkeypatch.setattr(engine.time, "sleep", lambda *a, **k: None)
    engine._breakers.clear()


def _run():
    llm = LLM("http://test", "key")
    llm.client = RecordingClient(PRO_REPLY)
    result = play_game(llm, CONFIG, TOPIC, "pro-strat", "con-strat")
    return llm.client, result


def test_pro_opening_is_never_called_system_only(fast):
    # The root cause was a system-only first call. The first model call (Pro's
    # opening) must contain a user turn, and that turn is the kickoff.
    client, _ = _run()
    first_call = client.messages_seen[0]
    roles = [m["role"] for m in first_call]
    assert "user" in roles, "Pro's opening was called with no user turn (the bug)"
    user_msgs = [m["content"] for m in first_call if m["role"] == "user"]
    assert user_msgs == [_FIRST_TURN_KICKOFF]


def test_kickoff_never_enters_the_transcript(fast):
    # Viewers must never see the seed. Rizzo's engine leaked exactly this string
    # into game.html; ours must not.
    _, result = _run()
    assert result.transcript[0]["speaker"] == "Pro"
    assert result.transcript[0]["text"] == PRO_REPLY
    assert all(_FIRST_TURN_KICKOFF not in t["text"] for t in result.transcript)


def test_opponent_sees_pros_real_opening_not_the_kickoff(fast):
    # Con's first turn must be a reply to Pro's actual argument, not to the
    # internal kickoff. Otherwise Con would be answering scaffolding.
    client, _ = _run()
    con_call = client.messages_seen[1]
    con_user_msgs = [m["content"] for m in con_call if m["role"] == "user"]
    assert con_user_msgs == [PRO_REPLY]
    assert _FIRST_TURN_KICKOFF not in con_user_msgs
