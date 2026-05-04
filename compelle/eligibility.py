from dataclasses import dataclass

import bittensor as bt


EPSILON_PLACEHOLDER = "epsilon"
VALIDATOR_DATA_PREFIX = "vdata:"
ELIGIBILITY_WINDOW_BLOCKS = 100  # 20 min @ 12s/block; gives buffer for slow commitment after registration
EPS_BUDGET = 0.001
EPS_AGE_FLOOR = 50


@dataclass
class MinerRecord:
    hotkey: str
    uid: int
    registration_block: int
    commitment_block: int
    commitment_text: str

    @property
    def is_eligible(self) -> bool:
        return 0 <= (self.commitment_block - self.registration_block) <= ELIGIBILITY_WINDOW_BLOCKS

    @property
    def is_placeholder(self) -> bool:
        return self.is_eligible and self.commitment_text.strip() == EPSILON_PLACEHOLDER

    @property
    def is_real(self) -> bool:
        return (
            self.is_eligible
            and not self.is_placeholder
            and not self.commitment_text.strip().startswith(VALIDATOR_DATA_PREFIX)
        )


def decode_commitment_info(info) -> str:
    try:
        variant = info["fields"][0][0]
        for key, value in variant.items():
            if not key.startswith("Raw"):
                continue
            if isinstance(value, tuple) and len(value) == 1 and isinstance(value[0], tuple):
                value = value[0]
            return bytes(value).decode("utf-8", errors="replace")
    except (KeyError, TypeError, IndexError, ValueError, AttributeError):
        pass
    return ""


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
        hotkey = storage_key.value if hasattr(storage_key, "value") else str(storage_key)
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
