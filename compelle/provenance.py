"""Opt-in verbose pre-debate provenance logging.

When COMPELLE_ENRICH_LOG is truthy the validator writes a structured JSONL
record for every miner before any debate runs (its on-chain commitment, the
resolved strategy text, and a content hash), plus one record per game at
pairing time, before the game is sent to the inference API. An operator or an
external auditor can then reconstruct the full path from on-chain submission
to resolved strategy to debate to verdict to Elo to weight, and verify any
strategy against its immutable gist revision by re-hashing it.

Off by default: no file is touched and the hooks are near-zero cost. The output
path defaults to data/provenance.jsonl (override with COMPELLE_ENRICH_LOG_PATH).
The file is size-capped at COMPELLE_ENRICH_LOG_MAXMB (default 500) with one
rotation generation so it cannot grow without bound.
"""

import hashlib
import json
import os
import time
from pathlib import Path


def enabled() -> bool:
    return os.environ.get("COMPELLE_ENRICH_LOG", "").strip().lower() in {
        "1", "true", "yes", "on"}


def _path() -> Path:
    return Path(os.environ.get("COMPELLE_ENRICH_LOG_PATH") or "data/provenance.jsonl")


def _max_bytes() -> int:
    try:
        return int(float(os.environ.get("COMPELLE_ENRICH_LOG_MAXMB", "500")) * 1024 * 1024)
    except ValueError:
        return 500 * 1024 * 1024


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def _append(obj: dict) -> None:
    # Logging must never break the tournament: swallow every IO error.
    try:
        p = _path()
        p.parent.mkdir(parents=True, exist_ok=True)
        cap = _max_bytes()
        if cap and p.exists() and p.stat().st_size >= cap:
            p.replace(p.with_name(p.name + ".1"))  # one rotation generation
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        pass


def log_miner(epoch_block: int, record, resolved_text: str) -> None:
    """One record per miner, emitted at strategy-resolve time (pre-debate)."""
    if not enabled():
        return
    # Build and write inside the guard: a missing attribute, a None commitment,
    # or any odd record must never propagate into the tournament loop.
    try:
        commitment = (record.commitment_text or "").strip()
        _append({
            "kind": "miner",
            "ts": int(time.time()),
            "epoch_block": int(epoch_block),
            "uid": record.uid,
            "hotkey": record.hotkey,
            "registration_block": record.registration_block,
            "commitment_block": record.commitment_block,
            "eligible": record.is_eligible,
            "placeholder": record.is_placeholder,
            "real": record.is_real,
            "intent_verdict": record.intent_verdict,
            "source": commitment if commitment.startswith("gist:") else "inline",
            "strategy_sha256": _sha(resolved_text),
            "strategy_bytes": len(resolved_text.encode("utf-8", "replace")),
            "strategy": resolved_text,
        })
    except Exception:
        pass


def log_game(epoch_block: int, round_num: int, idx: int, topic,
             pro_hk: str, con_hk: str, pro_text: str, con_text: str,
             pro_elo: float, con_elo: float) -> None:
    """One record per game, emitted at pairing time, before the inference call."""
    if not enabled():
        return
    try:
        _append({
            "kind": "game",
            "ts": int(time.time()),
            "epoch_block": int(epoch_block),
            "game_id": f"{int(epoch_block)}-r{round_num}-{idx}",
            "round": round_num,
            "topic": topic,
            "pro": {"hotkey": pro_hk, "strategy_sha256": _sha(pro_text),
                    "elo_in": round(float(pro_elo), 2)},
            "con": {"hotkey": con_hk, "strategy_sha256": _sha(con_text),
                    "elo_in": round(float(con_elo), 2)},
        })
    except Exception:
        pass
