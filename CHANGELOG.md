# Changelog

All notable changes to the Compelle validator distribution. SemVer-ish.

## [0.5.0] — 2026-05-17

Twelve days of stabilization since the May 5 initial release. Upgrading from
`0.1.0` should give you noticeably lower Chutes spend, faster tournaments, and
a small bump in validator-trust as your weights track consensus more closely.

### Engine & Chutes transport
- **HTTP/2 → HTTP/1.1.** HTTP/2 against Chutes had a ~30% connection-error rate
  that manifested as draw-storms in tournaments. Stable on HTTP/1.1.
- **Connection-error retry path.** Transient socket hiccups no longer land as
  draws.
- **Streaming with per-turn deadlines** on debater calls. Eliminates the
  infinite-hang failure mode when a backend stalls mid-stream.
- **Read timeout 45 s → 90 s** for thinking-heavy models.
- **Thinking-off retry via dict fallbacks**; debater turn cap at 1500 tokens.

### Throttling & quota
- **1 req/sec global token bucket** + `Retry-After` parsing. No more burst-mode
  402 quota hits.
- **`max_concurrent_games` 20 → 10**, **`uid_parallel_workers` 6 → 2 → 1**.
  Sized for actual Chutes burst quotas rather than the original optimistic
  guess.
- Intent-classifier concurrency lowered to match.

### Judging
- **`judge_max_tokens` 8192 → 2048.** Roughly 4× less Chutes spend per judge
  call. This is the single biggest validator-side cost reduction in the
  release.
- **Strip `<thinking>` blocks** from the transcript stored for judges and
  archived. Cleaner judging, smaller logs.

### Tournament loop
- **Cross-game lockstep turn-batching** in `run_tournament`. Tournament
  wall-clock drops noticeably; per-turn API pressure is more even.
- **Cycle padded to exactly one tempo** instead of always sleeping
  `epoch_seconds` on top of however long the tournament took.
- `swiss_rounds` 4 → 3.

### Model fallback list
- **Removed** `tngtech/DeepSeek-TNG-R1T2-Chimera-TEE` — deprecated by Chutes
  (404 from `/v1/models`).
- **Added** `moonshotai/Kimi-K2.6-TEE` to game fallbacks and judge panel.
  Chutes' burst-availability issue (the reason Kimi was pulled in May) has
  been fixed; freshly verified 10/10 success at 10 concurrent calls with
  latency comparable to the GLM-5.1 primary.

### Intent classifier
- Sharpened `PRINCIPLE_PROMPT` with an audience-test + defense-framing
  pattern.
- 3-GLM panel instead of single-pass classification.

## [0.1.0] — 2026-05-05

Initial public release.
