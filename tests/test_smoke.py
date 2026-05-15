from compelle.engine import strip_thinking, resolve_strategy, MAX_STRATEGY_BYTES
from compelle.validator import _block_from_filename, _validate_topics, _next_cycle_sleep


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
