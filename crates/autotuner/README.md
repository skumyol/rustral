# rustral-autotuner

Heuristic search over kernel configurations with a persistent cache.

**CI and production knobs** (see crate rustdoc and root [`ARCHITECTURE.md`](../../ARCHITECTURE.md)):

| Preset / field | Role |
|----------------|------|
| `TunerConfig::disabled()` | `enabled = false` — no search; returns default config |
| `TunerConfig::ci_safe()` | Bounded time/iterations; `ci_mode` |
| `enabled` | Master switch for running the search loop (enforced in `AutoTuner::tune`) |
| `ci_mode` | Extra iteration cap during tuning (enforced in `TuningSession::run`) |
| Cache hit | Re-benchmarks cached config so timings in `TuningResult` are real (via `benchmark_cached_config`) |

Rustral is a Rust workspace for research and learning; see the [repository README](https://github.com/skumyol/rustral#readme) for install, examples, and backend status.
