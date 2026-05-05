import importlib
import os
import time

from compelle.engine import strip_thinking, resolve_strategy, MAX_STRATEGY_BYTES
from compelle.validator import _block_from_filename, _validate_topics, _heartbeat_sleep


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


def test_data_dir_env_override(monkeypatch):
    monkeypatch.setenv("COMPELLE_DATA_DIR", "/tmp/compelle-test-data")
    import compelle.validator as v
    importlib.reload(v)
    try:
        assert v.DATA_DIR == "/tmp/compelle-test-data"
        assert v.ELO_STATE_PATH.startswith("/tmp/compelle-test-data/")
        assert v.LAST_TEMPO_PATH.startswith("/tmp/compelle-test-data/")
        assert v.PUSHED_PATH.startswith("/tmp/compelle-test-data/")
    finally:
        monkeypatch.delenv("COMPELLE_DATA_DIR", raising=False)
        importlib.reload(v)


def test_heartbeat_sleep_pulses_progress():
    last_progress = [0.0]
    start = time.time()
    _heartbeat_sleep(last_progress, 0.2)
    assert time.time() - start >= 0.2
    assert last_progress[0] >= start  # got pulsed at least once


def test_config_has_model_fallbacks():
    """Empty fallbacks make a Chutes deprecation collapse every game to a draw."""
    import json
    here = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(here, "..", "compelle", "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    game = cfg["old_config"]["game"]
    assert len(game["model_fallbacks"]) >= 1, "need at least one debater fallback"
    assert len(game["judge_model_fallbacks"]) >= 1, "need at least one judge fallback"
