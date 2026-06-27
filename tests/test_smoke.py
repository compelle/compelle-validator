import json

from compelle.engine import strip_thinking, resolve_strategy, MAX_STRATEGY_BYTES
from compelle.validator import (
    _block_from_filename, _validate_topics, _next_cycle_sleep,
    _validate_models, _apply_overrides, fetch_config,
)
from compelle import validator as _validator


def test_strip_thinking_removes_tagged_block():
    text = "before <think>scratch work here</think> after"
    assert strip_thinking(text, ["think"]) == "before  after"


def test_strip_thinking_multiple_tags():
    text = "a <think>x</think> b <reasoning>y</reasoning> c"
    assert strip_thinking(text, ["think", "reasoning"]) == "a  b  c"


def test_strip_thinking_case_insensitive():
    assert strip_thinking("x<THINK>hide</Think>y", ["think"]) == "xy"


def test_resolve_strategy_passthrough():
    assert resolve_strategy("be terse and rigorous") == "be terse and rigorous"


def test_resolve_strategy_oversize_direct_rejected():
    huge = "x" * (MAX_STRATEGY_BYTES + 1)
    assert resolve_strategy(huge) == ""


def test_resolve_strategy_bad_gist_format():
    assert resolve_strategy("gist:no-slash-here") == ""


def test_block_from_filename_valid():
    assert _block_from_filename("epoch_0000012345.json.gz") == 12345


def test_block_from_filename_invalid():
    assert _block_from_filename("not-an-epoch.txt") is None
    assert _block_from_filename("epoch_xxxx.json.gz") is None


def test_validate_topics_accepts_normal():
    assert _validate_topics([{"motion": "topic one"}, {"motion": "topic two"}]) is True


def test_validate_topics_accepts_full_object():
    assert _validate_topics([{"motion": "X is true", "context": "...", "framing": "direct",
                              "source": "polymarket", "source_url": "https://..."}]) is True


def test_validate_topics_rejects_non_list():
    assert _validate_topics("not a list") is False
    assert _validate_topics({"topics": "x"}) is False


def test_validate_topics_rejects_missing_motion():
    assert _validate_topics([{"context": "no motion field"}]) is False
    assert _validate_topics([{"motion": 42}]) is False
    assert _validate_topics([{}]) is False


def test_validate_topics_rejects_oversize_topic():
    assert _validate_topics([{"motion": "x" * 4001}]) is False


def test_validate_topics_rejects_too_many():
    assert _validate_topics([{"motion": "t"}] * 101) is False


def test_validate_topics_rejects_bad_framing():
    assert _validate_topics([{"motion": "X", "framing": "novel"}]) is False
    assert _validate_topics([{"motion": "X", "framing": "direct"}]) is True
    assert _validate_topics([{"motion": "X", "framing": "probability"}]) is True
    assert _validate_topics([{"motion": "X", "framing": "market_trajectory"}]) is True


def test_validate_topics_rejects_empty_list():
    assert _validate_topics([]) is False


def test_next_cycle_sleep_fills_to_tempo_when_tournament_short():
    # 17m tournament with 72m tempo: sleep enough to make the cycle = one tempo.
    assert _next_cycle_sleep(elapsed=17 * 60, epoch_seconds=4320,
                              sw_failed=False) == 4320 - 17 * 60


def test_next_cycle_sleep_zero_when_tournament_equals_tempo():
    assert _next_cycle_sleep(elapsed=4320, epoch_seconds=4320,
                              sw_failed=False) == 0.0


def test_next_cycle_sleep_clamps_to_zero_when_tournament_longer_than_tempo():
    # 90m tournament with 72m tempo: no sleep, start next immediately.
    assert _next_cycle_sleep(elapsed=90 * 60, epoch_seconds=4320,
                              sw_failed=False) == 0.0


def test_next_cycle_sleep_set_weights_failure_short_retry():
    # Failure path overrides the tempo math — short retry to recover.
    assert _next_cycle_sleep(elapsed=5 * 60, epoch_seconds=4320,
                              sw_failed=True) == 60.0
    # Even if the tournament ran long, failure still gets the short retry.
    assert _next_cycle_sleep(elapsed=100 * 60, epoch_seconds=4320,
                              sw_failed=True) == 60.0


# --- gist-rotatable model list ----------------------------------------------
# The model list is hot-editable from a hand-owned gist (separate from the
# auto-refreshed topics gist) without a code redeploy. These tests encode the
# rules: a malformed list is rejected whole (fail-closed), topics merge at the
# top level while model keys nest under cfg["game"], and the gist passes the
# model keys through validation untouched.


def test_validate_models_accepts_valid():
    assert _validate_models({"model": "glm", "model_fallbacks": ["qwen", "deepseek"]}) is True
    assert _validate_models({"model": "glm"}) is True
    assert _validate_models({"model_fallbacks": ["qwen"]}) is True


def test_validate_models_accepts_no_model_keys():
    # A topics-only gist carries no model keys; there is nothing to validate.
    assert _validate_models({"topics": [{"motion": "x"}]}) is True


def test_validate_models_rejects_empty_or_nonstring_model():
    assert _validate_models({"model": "   "}) is False
    assert _validate_models({"model": 42}) is False


def test_validate_models_rejects_bad_fallbacks():
    assert _validate_models({"model_fallbacks": "qwen"}) is False          # not a list
    assert _validate_models({"model_fallbacks": []}) is False              # no failover left
    assert _validate_models({"model_fallbacks": ["m"] * 13}) is False      # over MAX_FALLBACKS
    assert _validate_models({"model_fallbacks": ["qwen", ""]}) is False    # blank entry
    assert _validate_models({"model_fallbacks": ["qwen", 7]}) is False     # non-string entry


def test_apply_overrides_topics_go_top_level():
    cfg = {"game": {"model": "old"}, "topics": []}
    _apply_overrides(cfg, {"topics": [{"motion": "new"}]})
    assert cfg["topics"] == [{"motion": "new"}]
    assert cfg["game"]["model"] == "old"  # game block untouched


def test_apply_overrides_model_keys_nest_under_game():
    cfg = {"game": {"model": "old", "model_fallbacks": ["a"], "max_turns": 5}}
    _apply_overrides(cfg, {"model": "new", "model_fallbacks": ["b", "c"]})
    assert cfg["game"]["model"] == "new"
    assert cfg["game"]["model_fallbacks"] == ["b", "c"]
    assert cfg["game"]["max_turns"] == 5  # other game keys untouched


def test_apply_overrides_mixed_splits_correctly():
    cfg = {"game": {"model": "old"}, "topics": []}
    _apply_overrides(cfg, {"topics": [{"motion": "t"}], "model": "new"})
    assert cfg["topics"] == [{"motion": "t"}]
    assert cfg["game"]["model"] == "new"


def _fake_gist(monkeypatch, content, owner="compelle"):
    data = {
        "history": [{"version": "abc123def456"}],
        "owner": {"login": owner},
        "files": {"models.json": {"content": content}},
    }

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return data

    monkeypatch.setattr(_validator.requests, "get", lambda *a, **k: _Resp())


def test_fetch_config_passes_model_keys_through(monkeypatch):
    content = json.dumps({"min_block": 0, "model": "primary-x",
                          "model_fallbacks": ["fb-a", "fb-b"]})
    _fake_gist(monkeypatch, content)
    overrides, rev = fetch_config("gid", "compelle", 1000)
    assert overrides == {"model": "primary-x", "model_fallbacks": ["fb-a", "fb-b"]}
    assert rev == "abc123def456"


def test_fetch_config_rejects_bad_model_list_fail_closed(monkeypatch):
    # A malformed fallback list rejects the whole fetch so the loop keeps the
    # last good list rather than running on a broken chain.
    content = json.dumps({"min_block": 0, "model_fallbacks": "not-a-list"})
    _fake_gist(monkeypatch, content)
    overrides, _ = fetch_config("gid", "compelle", 1000)
    assert overrides is None


def test_fetch_config_models_gist_owner_guard(monkeypatch):
    content = json.dumps({"model_fallbacks": ["fb-a"]})
    _fake_gist(monkeypatch, content, owner="someone-else")
    overrides, _ = fetch_config("gid", "compelle", 1000)
    assert overrides is None
