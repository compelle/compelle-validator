"""An LLM outage across the whole fallback chain must VOID, never draw.

Business rule: a game is a "draw" only when two debaters actually argued to a
tie (max turns with allow_draws, or an indecisive judge). When every model in
the fallback chain is down, no debate happened at all. Labelling that "draw"
poisons two places: the Elo ladder (a draw nudges both ratings) and the public
W/L stats (the backend counts draws as real ties). So infra failures carry
winner="void" instead, which run_tournament drops from Elo and the site
excludes from W/L.

Two invariants the relabel must hold, both tested here:
  1. A game whose every model call raises comes back winner="void" (not
     "draw"), with the reason naming the LLM error.
  2. Voided games leave every miner's Elo at its initial rating — a provider
     outage cannot move the ladder.

This guards the canary lesson directly: the epoch that voided 18 games during a
Chutes 500-storm must stay Elo-neutral, and a later code change that re-labels
those as draws (or drops the void from the skip condition) must break a test.
"""
import pytest

from compelle import engine
from compelle.engine import LLM, _play_round_lockstep, run_tournament


class RaisingClient:
    """OpenAI-client stand-in whose every completion call fails — models the
    entire fallback chain being unreachable (a provider 500-storm or outage)."""

    def __init__(self, exc):
        self.exc = exc
        self.chat = self
        self.completions = self

    def create(self, model, messages, max_tokens, temperature, **kwargs):
        raise self.exc


CONFIG = {
    "game_prompt": "You are {side}. Motion: {topic}. Strategy: {strategy}.{context} Date: {date}",
    "thinking_tags": ["think"],
    "concession": {"symbol": "Δ", "min_length": 50},
    "game": {
        "model": "glm",
        "model_fallbacks": ["deepseek", "minimax"],
        "max_tokens_per_turn": 256,
        "temperature": 0.6,
        "max_turns": 2,
        "allow_draws": True,
    },
    "elo": {"k_factor": 32, "initial_rating": 1000},
    "tournament": {"swiss_rounds": 1, "max_concurrent_games": 2},
    "topics": [{"motion": "Cats are better than dogs."}],
}
TOPIC = CONFIG["topics"][0]


@pytest.fixture
def fast(monkeypatch):
    # No real waiting: neutralize the rate token and every backoff sleep so the
    # fallback chain exhausts instantly, and clear circuit-breaker state so a
    # prior test's open breaker can't change this run's path.
    monkeypatch.setattr(engine, "_acquire_rate_token", lambda: None)
    monkeypatch.setattr(engine.time, "sleep", lambda *a, **k: None)
    engine._breakers.clear()


def _down_llm():
    llm = LLM("http://test", "key")
    llm.client = RaisingClient(RuntimeError("boom"))
    return llm


def test_total_outage_voids_not_draws(fast):
    # The tournament hot path: a round whose every model raises must return
    # voids, with the reason naming the LLM error — never a silent "draw".
    pairs = [("pro_hk", "con_hk")]
    strategies = {"pro_hk": "pro-strat", "con_hk": "con-strat"}
    completed = _play_round_lockstep(
        _down_llm(), CONFIG, pairs, TOPIC, strategies, workers=1,
    )
    assert completed, "round produced no game result"
    for _idx, _pro, _con, result in completed:
        assert result.winner == "void", (
            f"infra failure scored as {result.winner!r}, expected 'void'"
        )
        assert result.winner != "draw"
        assert "llm error" in (result.reason or "").lower()


def test_void_games_leave_elo_untouched(fast):
    # End to end through run_tournament's Elo-skip loop: an outage that voids
    # every game must leave the ladder exactly where it started. This is the
    # invariant that kept the 18-void canary epoch Elo-neutral.
    strategies = {"pro_hk": "pro-strat", "con_hk": "con-strat"}
    results, elo = run_tournament(
        _down_llm(), CONFIG, strategies, epoch_start_block=8_490_961,
    )
    assert results, "tournament produced no games"
    assert all(r.winner == "void" for _, _, r in results), (
        "an all-models-down tournament must void every game"
    )
    initial = CONFIG["elo"]["initial_rating"]
    for hk in strategies:
        assert elo.get(hk) == initial, (
            f"{hk} Elo moved to {elo.get(hk)} on a voided game; "
            f"infra failures must not touch the ladder"
        )
