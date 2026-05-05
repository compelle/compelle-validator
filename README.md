# Compelle Validator (SN82)

Reference validator for the Compelle subnet on Bittensor mainnet (netuid 82).

Compelle is an adversarial-persuasion subnet. Miners commit a written debate strategy on chain. Each epoch, the validator pairs miners in Pro/Con LLM debates over a shared topic, scores the outcomes via Elo, and sets weights on chain.

## What this validator does

Once per epoch:

1. Read the metagraph and all commitments for netuid 82.
2. Determine which miners are eligible to score this epoch.
3. Run a round-robin tournament across eligible miners.
4. Update Elo, derive weights, and call `set_weights`.
5. Write epoch results to `data/epoch_NNNN.json.gz`, sign and POST the file to the public aggregation endpoint (`push_url`), and sleep until the next epoch.

Eligibility and scoring logic lives in `compelle/eligibility.py` and `compelle/engine.py`. All validators run the same code path. Determinism is best-effort: matchup ordering and topic selection are seeded from the epoch's start block, but the LLM debater and judge calls have non-zero temperature and the underlying provider may produce non-bit-identical outputs across runs. Validators converge over many epochs via Yuma consensus rather than per-epoch agreement.

LLM inference is provided by [Chutes](https://chutes.ai/). The validator needs a funded Chutes API key.

## Requirements

- Python 3.12+
- A registered hotkey on netuid 82, staked enough to set weights
- A Chutes API key with credit balance
- Stable connection to a Bittensor mainnet endpoint (default: `wss://entrypoint-finney.opentensor.ai:443`)
- Modest CPU. No GPU. Inference is offloaded to Chutes.

## Install

```bash
git clone https://github.com/compelle/compelle-validator.git
cd compelle-validator
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
cp deploy/compelle.env.example .env
$EDITOR .env   # set CHUTES_API_KEY, BT_WALLET_NAME, BT_HOTKEY
```

## Run

```bash
set -a; . ./.env; set +a
python -m compelle.validator
```

For production, use the systemd unit:

```bash
sudo cp deploy/compelle-validator.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now compelle-validator
journalctl -u compelle-validator -f
```

## Configuration

`compelle/config.json` holds all game parameters: model identifiers, fallback chain, judge prompt, max turns, Elo k-factor, topic list, etc. These values are part of the consensus contract — every validator runs the same config.

### Topic schema

Each topic is a JSON object. Only `motion` is required.

```json
{
  "motion": "Bitcoin will hit $150k by Dec 31, 2026",
  "context": "Polymarket YES at 10%. Resolution: any 1m Binance candle high >= $150k. ([market](https://polymarket.com/event/...), [explainer](https://...))",
  "framing": "direct",
  "source": "polymarket",
  "source_url": "https://polymarket.com/event/..."
}
```

Fields:
- `motion` (required, string) — the debate proposition. Pro defends, Con denies.
- `context` (optional, markdown string) — background facts injected into both debaters' system prompts. Markdown links are preserved for downstream HTML rendering.
- `framing` (optional, default `"direct"`) — one of `direct`, `probability` (Pro argues market is mispriced), or `market_trajectory` (Pro argues price will cross threshold T by date).
- `source` (optional) — provenance hint. One of `polymarket | kalshi | grok | evergreen`.
- `source_url` (optional) — primary external URL for the topic, for "via X" attribution on downstream sites.

Topics flow into the validator from two places: the bundled `compelle/config.json` (the always-available fallback), and a remote gist pointed to by `config_gist_id` (refreshed each epoch, allows daily topic rotation without redeploying). The gist content must match the same schema.

Per-deployment values come from environment variables and override the file:

| Env var | Purpose |
|---|---|
| `CHUTES_API_KEY` | Chutes inference key (required) |
| `BT_WALLET_NAME` | Bittensor coldkey name |
| `BT_HOTKEY` | Bittensor hotkey name |
| `BT_NETUID` | Subnet netuid (defaults to 82) |
| `BT_NETWORK` | `finney` (default) or a custom endpoint |
| `CHUTES_BASE_URL` | Override for the Chutes API base URL |
| `COMPELLE_PUSH_URL` | Override the public aggregation endpoint (set to empty to disable) |

## Upgrading

### Manual (default)

When a release is announced, run:

```bash
sudo /opt/compelle-validator/deploy/upgrade.sh
```

The script fetches `origin/main`, shows the diff (commit hash + message), asks for confirmation, then `git reset --hard` + restart. Pin a specific tag instead with `upgrade.sh v1.2.3`.

### Watchtower (opt-in auto-update)

Auto-pull `origin/main` every 10 minutes and restart on change:

```bash
sudo systemctl enable --now compelle-watchtower.timer
```

Risks (operator must accept):
- We push a bad commit → all watchtower-enabled validators crash together
- We push a malicious commit (compromised github account) → validators run it
- Mitigation: keep an eye on `journalctl -u compelle-watchtower` after announcements

To disable: `sudo systemctl disable --now compelle-watchtower.timer`.

## License

MIT. See `LICENSE`.
