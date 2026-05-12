from dataclasses import dataclass

import bittensor as bt


EPSILON_PLACEHOLDER = "epsilon"
VALIDATOR_DATA_PREFIX = "vdata:"
ELIGIBILITY_WINDOW_BLOCKS = 200  # 40 min @ 12s/block; allows for chain congestion / RPC failures during the registration sprint
EPS_BUDGET = 0.001
EPS_AGE_FLOOR = 50


@dataclass
class MinerRecord:
    hotkey: str
    uid: int
    registration_block: int
    commitment_block: int
    commitment_text: str
    # Set by compelle.intent_classifier.classify_records after the basic
    # fetch_records pass. Values: "GOOD", "BAD", "PENDING". Default PENDING
    # means classifier hasn't run yet (or is_real is False so it never will).
    intent_verdict: str = "PENDING"

    @property
    def is_eligible(self) -> bool:
        return 0 <= (self.commitment_block - self.registration_block) <= ELIGIBILITY_WINDOW_BLOCKS

    @property
    def is_placeholder(self) -> bool:
        return self.is_eligible and self.commitment_text.strip() == EPSILON_PLACEHOLDER

    @property
    def is_real(self) -> bool:
        """Real-miner basic criteria: eligible timing, non-epsilon, non-vdata.

        Does NOT include intent classification — see `is_real_and_clean` for that.
        Kept distinct so the classifier knows which records to evaluate.
        """
        return (
            self.is_eligible
            and not self.is_placeholder
            and not self.commitment_text.strip().startswith(VALIDATOR_DATA_PREFIX)
        )

    @property
    def is_real_and_clean(self) -> bool:
        """is_real AND intent classifier has not flagged the strategy as BAD.

        PENDING is treated as clean (i.e., a transient classifier failure does
        not lock out the miner — we retry next epoch).
        """
        return self.is_real and self.intent_verdict != "BAD"


def decode_commitment_info(info) -> str:
    """Decode a Commitments.CommitmentOf 'info' object to its UTF-8 text.

    Chain encodes commitments as one of two variant families:
      Raw0..Raw128 — fixed byte arrays for ≤128-byte commitments (small text, "epsilon")
      BigRaw       — bounded vec for 129..512-byte commitments (the strategy texts)

    The 'fields' value comes back in two equivalent shapes depending on the
    substrate-interface version used by the caller:
      A) [{"Raw7": "0x657073696c6f6e"}]               — flat dict, hex-string value
      B) [[{"Raw7": [[101, 112, 115, 105, 108, ...]]}]] — nested list, byte-tuple value
    We accept either. The value itself can be a hex string starting with '0x',
    a raw bytes/bytearray, or a nested list of ints. Any other shape returns "".
    """
    try:
        f0 = info["fields"][0]
        # Shape B: extra wrapping list around the variant dict
        if isinstance(f0, (list, tuple)) and len(f0) >= 1 and isinstance(f0[0], dict):
            variant = f0[0]
        elif isinstance(f0, dict):
            variant = f0
        else:
            return ""
        for key, value in variant.items():
            if not (key.startswith("Raw") or key == "BigRaw"):
                continue
            # Compat: bittensor 10.3 returns BigRaw payloads as triple-nested
            # tuples `(((b1, b2, ...),),)`, while 10.1/10.2 returns the older
            # `[[byte, byte, ...]]`. Loop the unwrap as long as we have a
            # single-element sequence whose only element is another sequence
            # (and not a string/bytes leaf). The supported version is pinned
            # in pyproject.toml; this loop is belt-and-suspenders so a future
            # version bump can't silently regress the lookup.
            while (isinstance(value, (list, tuple)) and len(value) == 1
                   and isinstance(value[0], (list, tuple))
                   and not isinstance(value[0], (str, bytes, bytearray))):
                value = value[0]
            # String form. Two cases:
            #  - Raw{N} hex string: "0x657073696c6f6e" → hex-decode then utf-8
            #  - BigRaw plain UTF-8 string (already decoded by SDK): return as-is
            if isinstance(value, str):
                if value.startswith("0x"):
                    try:
                        return bytes.fromhex(value[2:]).decode("utf-8", errors="replace")
                    except ValueError:
                        return ""
                return value
            # Bytes / bytearray
            if isinstance(value, (bytes, bytearray)):
                return bytes(value).decode("utf-8", errors="replace")
            # Iterable of ints
            if isinstance(value, (list, tuple)):
                try:
                    return bytes(value).decode("utf-8", errors="replace")
                except (TypeError, ValueError):
                    return ""
            return ""
    except (KeyError, TypeError, IndexError, ValueError, AttributeError):
        pass
    return ""


def _storage_key_to_ss58(subtensor: bt.Subtensor, storage_key) -> str | None:
    """Coerce a Commitments::CommitmentOf storage key to an ss58 string.

    Compat: bittensor 10.3 returns the storage_key as an ss58 string directly
    (which is what the validator runs on). 10.1/10.2 returned it as a raw
    32-byte tuple keyed by the public key — `str(storage_key)` then produced
    `"(46, 174, ...)"` which never matched any metagraph hotkey, silently
    zeroing out every commitment lookup. We pin to 10.3+ in pyproject.toml,
    but keep this helper so a fresh venv that lands on the older line still
    functions instead of failing silently.

    Returns None for shapes we can't decode (caller skips them).
    """
    skv = storage_key.value if hasattr(storage_key, "value") else storage_key
    if isinstance(skv, str):
        return skv
    if isinstance(skv, (tuple, list)):
        # One level of unwrap if the pubkey bytes are nested
        if len(skv) == 1 and isinstance(skv[0], (tuple, list)):
            skv = skv[0]
        try:
            return subtensor.substrate.ss58_encode(bytes(skv))
        except Exception:
            return None
    return None


def fetch_records(
    subtensor: bt.Subtensor,
    netuid: int,
    block: int | None = None,
    block_hash: str | None = None,
) -> dict[str, MinerRecord]:
    metagraph = subtensor.metagraph(netuid, block=block)
    raw = subtensor.substrate.query_map(
        module="Commitments", storage_function="CommitmentOf", params=[netuid],
        block_hash=block_hash,
    )
    commitments: dict[str, tuple[int, str]] = {}
    for storage_key, value in raw:
        hotkey = _storage_key_to_ss58(subtensor, storage_key)
        if hotkey is None:
            continue
        v = value.value if hasattr(value, "value") else value
        try:
            commitments[hotkey] = (int(v["block"]), decode_commitment_info(v.get("info")))
        except (KeyError, TypeError, ValueError):
            continue

    records: dict[str, MinerRecord] = {}
    for uid, hotkey in enumerate(metagraph.hotkeys):
        cb, ct = commitments.get(hotkey, (-1, ""))
        records[hotkey] = MinerRecord(
            hotkey=hotkey,
            uid=uid,
            registration_block=int(metagraph.block_at_registration[uid]),
            commitment_block=cb,
            commitment_text=ct,
        )
    return records


def assign_weights(
    records: dict[str, MinerRecord],
    real_weights: dict[str, float],
    epoch_start_block: int,
) -> dict[str, float]:
    ages = {
        r.hotkey: max(EPS_AGE_FLOOR, epoch_start_block - r.registration_block)
        for r in records.values() if r.is_placeholder
    }
    out: dict[str, float] = {}
    if ages:
        total = sum(ages.values())
        for hk, age in ages.items():
            out[hk] = EPS_BUDGET * age / total
        scale = 1.0 - EPS_BUDGET
    else:
        scale = 1.0
    for hk, w in real_weights.items():
        out[hk] = w * scale
    return out
