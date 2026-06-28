"""KOTH crown-eligibility tests.

The business rule under test: taking the crown (100% of emission) requires a
GOOD intent verdict, not merely "not BAD". A PENDING (unresolved) miner still
competes and earns Elo, but must never be crowned king or advance the dethrone
streak — otherwise an attacker could stay deliberately unresolved to dodge a
lockout while out-Eloing the king. The caller passes koth_weights the GOOD-only
subset as `crown_eligible`; these tests pin that contract at the function.
"""
import compelle.validator as validator
from compelle.engine import Elo

KING = "5King"
GOOD = "5Good"
PENDING = "5Pend"


def _elo(ratings):
    e = Elo()
    e.ratings = dict(ratings)
    return e


def _run(monkeypatch, tmp_path, elo, crown_eligible, tempos, margin=200.0, hold=3):
    """Run koth_weights across consecutive tempos; return the last result."""
    monkeypatch.setattr(validator, "KOTH_STATE_PATH", str(tmp_path / "koth.json"))
    monkeypatch.setattr(validator, "_incumbent_king", lambda sub, netuid: KING)
    result = None
    for t in tempos:
        result = validator.koth_weights(None, 82, elo, crown_eligible, t,
                                        margin=margin, hold=hold)
    return result


def test_pending_with_top_elo_is_never_crowned(monkeypatch, tmp_path):
    # PENDING miner towers over the king but is absent from crown_eligible.
    # The only GOOD miner sits below the +200 line. King must keep the crown
    # forever, no matter how long the PENDING miner leads.
    elo = _elo({KING: 1500.0, PENDING: 5000.0, GOOD: 1550.0})
    crown_eligible = {KING: "k", GOOD: "g"}  # PENDING deliberately excluded
    out = _run(monkeypatch, tmp_path, elo, crown_eligible, range(10))
    assert out == {KING: 1.0}
    assert PENDING not in out


def test_pending_leader_does_not_shield_a_good_challenger(monkeypatch, tmp_path):
    # PENDING has the highest Elo, but a GOOD miner is also above the line.
    # Filtering the pool (not just rejecting the single top miner) means the
    # GOOD miner is the contender and can still dethrone; the PENDING leader
    # neither wins nor blocks.
    elo = _elo({KING: 1500.0, PENDING: 5000.0, GOOD: 1800.0})  # GOOD = king+300
    crown_eligible = {KING: "k", GOOD: "g"}
    out = _run(monkeypatch, tmp_path, elo, crown_eligible, range(3), hold=3)
    assert out == {GOOD: 1.0}


def test_good_challenger_dethrones_only_after_full_streak(monkeypatch, tmp_path):
    elo = _elo({KING: 1500.0, GOOD: 1800.0})  # +300, above the +200 line
    crown_eligible = {KING: "k", GOOD: "g"}
    # One epoch short of the hold: still the king.
    held_short = _run(monkeypatch, tmp_path, elo, crown_eligible, range(2), hold=3)
    assert held_short == {KING: 1.0}
    # The full streak (fresh state): crowned.
    crowned = _run(monkeypatch, tmp_path, elo, crown_eligible, range(3), hold=3)
    assert crowned == {GOOD: 1.0}


def test_streak_resets_if_challenger_drops_to_pending(monkeypatch, tmp_path):
    # GOOD builds a streak, then its verdict flips to PENDING (it leaves
    # crown_eligible). The streak must reset, so it cannot resume mid-climb.
    elo = _elo({KING: 1500.0, GOOD: 1800.0})
    monkeypatch.setattr(validator, "KOTH_STATE_PATH", str(tmp_path / "koth.json"))
    monkeypatch.setattr(validator, "_incumbent_king", lambda sub, netuid: KING)
    full = {KING: "k", GOOD: "g"}
    dropped = {KING: "k"}  # GOOD now PENDING -> out of the crown pool
    validator.koth_weights(None, 82, elo, full, 0, hold=3)      # streak 1
    validator.koth_weights(None, 82, elo, full, 1, hold=3)      # streak 2
    validator.koth_weights(None, 82, elo, dropped, 2, hold=3)   # reset
    validator.koth_weights(None, 82, elo, full, 3, hold=3)      # streak 1 again
    out = validator.koth_weights(None, 82, elo, full, 4, hold=3)  # streak 2 -> still king
    assert out == {KING: 1.0}
