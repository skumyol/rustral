# NLP evaluation methodology

This document explains how Rustral's NLP examples train and evaluate. The goal is simple: a reader should be able to clone the repo, run the commands, and understand the numbers without guessing at hidden settings.

If a reported number disagrees with this file, treat the run as suspicious until the command, manifest, and dataset cache are checked.

## Tasks and corpora

| Task | Corpus | Source | Split | Tokenizer | Reported metric |
|---|---|---|---|---|---|
| Sentiment classification | SST-2 (binary) | HuggingFace `SetFit/sst2` mirror | `train.jsonl` / `dev.jsonl` | `rustral-data WordLevelTokenizer` (lowercase, whitespace, max_vocab=8192) | dev accuracy |
| Word-level language modelling | WikiText-2 raw v1 | `wikitext.smerity.com/wikitext-2-raw-v1.zip` | `wiki.{train,valid,test}.raw` | `rustral-data WordLevelTokenizer` (lowercase, whitespace, max_vocab=16384) | dev perplexity (`exp(mean cross-entropy nats)`) |

Tokenization is deliberately word-level. WikiText-2 is commonly reported that way, SST-2 works cleanly with it, and it keeps the dependency tree small. HuggingFace `tokenizers` can come later when a BPE or WordPiece baseline needs it. See `crates/data/src/tokenizer.rs`.

## Models

Both examples default to tiny architectures so they finish on CPU. Width, depth, and context length can be overridden on the CLI (`--seq-len`, `--d-model`, `--num-heads`, `--ffn-dim`, `--num-layers`, and for WikiText-2 `--block-size`) so you can trade accuracy for runtime. These are honest baselines, not leaderboard attempts.

The tape-trained transformer stacks use a **GELU** position-wise FFN (`TapeFeedForward`); topology diagrams below reflect that (older writeups may have said ReLU).

### SST-2 classifier (`crates/runtime/examples/sst2_classifier.rs`)

**Default topology:**

```
Embedding(V, 64) + PositionalEmbedding(32, 64)
  ↓ 2 × TransformerEncoderLayer (pre-LN, 4 heads, FFN=128, GELU in tape FFN)
  ↓ mean-pool over sequence
Linear(64 → 2)
```

- Sentence is encoded with the `WordLevelTokenizer`, padded / truncated to the configured sequence length (default 32).
- The transformer block is tape-trained (multi-head attention + layer norm + FFN) and writes a manifest with full provenance.

### WikiText-2 LM (`crates/runtime/examples/wikitext2_lm.rs`)

**Default topology:**

```
Embedding(V, 64) + PositionalEmbedding(32, 64)
  ↓ 2 × TransformerEncoderLayer (pre-LN, 4 heads, FFN=128, GELU in tape FFN) + causal mask
  ↓ take last position hidden state
Linear(64 → V)
```

- Sliding window of `block_size` consecutive tokens (default 32) predicts the next token.
- Trained on a capped subset of the train split and evaluated on a capped subset of validation windows for runtime practicality (caps are recorded in the manifest).
- Reported metric is dev perplexity, computed as `exp(mean cross-entropy nats)` over the evaluated windows.

## Hyperparameters (defaults)

| Hyperparameter | SST-2 | WikiText-2 |
|---|---|---|
| Seed (`--seed`) | `0xC0FFEE` | `0xC0FFEE` |
| Optimizer | Adam | Adam |
| Learning rate (`--lr`) | `5e-4` | `5e-4` |
| Batch size (`--batch`) | 32 | 32 |
| Epochs (`--epochs`) | 3 | 1 |
| Sequence/window length | 32 | 32 |
| Model dim (`d_model`) | 64 | 64 |
| Layers | 2 | 2 |
| Heads | 4 | 4 |
| FFN dim | 128 | 128 |
| Max vocab | 8192 | 16384 |

`--quick` modes, used by smoke tests, shrink the training set so the run finishes in seconds. SST-2 caps training at 256 examples. WikiText-2 caps training at 4 000 tokens (default inside `--quick`).

### Fast benchmark preset (recommended for CI and local timing)

[`scripts/eval/run_nlp_real.py`](scripts/eval/run_nlp_real.py) supports `--benchmark`: a tiny transformer (e.g. `d_model=32`, `seq_len` / block `16`, one layer, one epoch) and minimal WikiText-2 caps so end-to-end NLP checks finish in minutes on CPU. PyTorch mirrors the same idea via `--benchmark` on [`benchmarks/pytorch/sst2_baseline.py`](benchmarks/pytorch/sst2_baseline.py) and [`benchmarks/pytorch/wikitext2_baseline.py`](benchmarks/pytorch/wikitext2_baseline.py).

Without `--benchmark`, `run_nlp_real.py` still applies modest WikiText-2 caps by default (**8 000** train tokens, **400** train windows, **800** eval windows) so uncached runs stay tractable; SST-2 uses the full default architecture (see table above) unless you pass smaller flags to the example.

For heavier, slower runs, increase `--wikitext-*` caps and pass larger `--seq-len`, `--d-model`, etc. to the Rust examples explicitly.

### Pre-release full evaluation (paper profile)

Before a **release tag**, maintainers must capture **three-seed** paper-profile runs and **commit** the curated summaries under `benchmarks/runs/v<version>/nlp/`. This is **not** `--quick` or `--benchmark`: the Rust examples run with `--paper` (larger model, full SST-2 training split in the example, 200k WikiText-2 train tokens, shared vocab across seeds for fair aggregates). See [`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md).

One command (validates manifests after):

```bash
./scripts/eval/run_release_nlp_eval.sh 0.2.0
# Optional: ./scripts/eval/run_release_nlp_eval.sh 0.2.0 --pytorch
```

This wraps `python3 scripts/eval/run_nlp_real.py --paper --clean --curated-version <version> --seeds 0,1,2`. Replace `0.2.0` with the version directory you are about to tag.

Older curated JSON numbers in prose below may be stale; prefer the aggregates in `benchmarks/runs/<version>/nlp/*.json` for the commit you are on.

## Rustral vs PyTorch parity (v0.1.0)

Curated 3-seed snapshots live under `benchmarks/runs/v0.1.0/nlp/`:

- Rustral: `sst2.json`, `wikitext2.json`
- PyTorch: `sst2_pytorch.json`, `wikitext2_pytorch.json`

| Task | Metric | Where to read results |
|---|---|---|
| SST-2 | dev accuracy | `benchmarks/runs/v0.1.0/nlp/sst2.json` and `sst2_pytorch.json` |
| WikiText-2 | dev perplexity | `benchmarks/runs/v0.1.0/nlp/wikitext2.json` and `wikitext2_pytorch.json` |

Regenerate with `python3 scripts/eval/run_nlp_real.py --benchmark` (and the PyTorch baselines with `--benchmark`) when you need fresh tables; headline numbers depend on caps and model width recorded in each manifest.

## Reproducibility manifest

Every run writes a `manifest.json` next to its other artifacts capturing:

- `git_sha` (`git rev-parse HEAD` at run time)
- `seed`, all hyperparameters above
- `vocab_size`, `train_*`, `dev_*` counts
- `dataset_hash_fnv1a` of the concatenated raw corpus (FNV-1a so we don't grow a hash
  dependency for examples)
- Reported metric (`dev_accuracy` for SST-2, `dev_perplexity` for WikiText-2)
- `samples_per_sec` / `windows_per_sec`, `train_elapsed_sec`
- `tokenizer` description string and `dataset` description string
- `quick_mode` flag (true if `--quick` was passed)

When you cite a number from these examples, cite the manifest path too. Without the manifest, the number is just a number.

## Online vs offline runs

Both examples support online and offline runs through `rustral-data` environment variables:

- `RUSTRAL_CACHE_DIR=<path>`: overrides the dataset cache root (default `~/.cache/rustral`).
- `RUSTRAL_DATASET_OFFLINE=1`: refuse to network-fetch; require everything to already be
  in the cache. CI runs always set this.
- `RUSTRAL_DATASET_SKIP_CHECKSUM=1`: trust the cache without recomputing SHA-256.
  Useful when staging hand-curated splits (CI uses this for synthetic smoke fixtures).

## Pinning upstream artifacts

`crates/data/src/datasets/sst2.rs` and `crates/data/src/datasets/wikitext2.rs` declare
canonical URLs and a `*_SHA256` constant for each artifact. The hashes are real
SHA-256 values verified against the downloaded files; if upstream rotates a file
the loader fails loudly with `ChecksumMismatch` and a maintainer must re-pin deliberately.

| Artifact | URL | Verified SHA-256 |
|---|---|---|
| SST-2 train | `https://huggingface.co/datasets/SetFit/sst2/resolve/main/train.jsonl` | `7a4b1cfdd65be1dc48339404db86528bb2427e1d8772860ef838b76b8c38c4a8` |
| SST-2 dev | `https://huggingface.co/datasets/SetFit/sst2/resolve/main/dev.jsonl` | `573c3ed18d96aa0a79a6e5980a544b80543317a319f18bd4f1660c16b2f6b939` |
| WikiText-2 raw v1 zip | `https://wikitext.smerity.com/wikitext-2-raw-v1.zip` | `ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11` |

Updating a pinned hash is a small PR: run the example without `RUSTRAL_DATASET_SKIP_CHECKSUM`,
copy the `actual = <hash>` field reported by the `ChecksumMismatch` error into the
`*_SHA256` constant, and add a `CHANGELOG.md` line.

## Smoke tests

Both examples ship with `#[ignore]`-gated smoke tests under `crates/runtime/tests/`:

- `sst2_smoke.rs`: pre-stages a 10-line synthetic SST-2 TSV, runs the example with
  `--quick`, asserts manifest fields are present.
- `wikitext2_lm_smoke.rs`: pre-stages 2 000 synthetic tokens, runs `--quick`, asserts
  manifest fields are present.

Run them via:

```bash
cargo test -p rustral-runtime --features training -- --include-ignored sst2
cargo test -p rustral-runtime --features training -- --include-ignored wikitext2
```

CI runs these on every PR.

## Micro-benchmarks: frameworks and hardware (operator timing)

The harness in `scripts/bench/run_all.py` runs a **fixed workload list** (matmul, manual attention, conv2d, plus additional training-oriented kernels in the Rust binaries) across stacks that share the same **schema v2** (`benchmarks/SCHEMA.md`):

| Suite | Stack | Device class | When to cite |
|-------|--------|--------------|--------------|
| `rustral` | Rustral ndarray backend | CPU | Core comparisons vs other CPU frameworks |
| `candle` | Candle (Rust) | CPU | Another Rust ML baseline on CPU |
| `pytorch` | PyTorch | CPU | Industry baseline; requires `torch` in the active env |
| `pytorch-cuda` | PyTorch | NVIDIA GPU | Same workloads on CUDA; timings use `cuda.synchronize()` around each repeat |
| `rustral-cuda` | Rustral CUDA backend | NVIDIA GPU | Rustral on GPU (requires CUDA build of `rustral-bench`) |
| `rustral-metal` | Rustral Metal backend | Apple GPU | Rustral on Metal (macOS) |

**Academic / reporting practice:** treat these as **throughput-oriented micro-benchmarks**, not end-to-end training. Report **warmup and repeat counts** (`--warmup`, `--repeats`), **hardware model**, **OS**, and **library versions** (see `machine` in each suite JSON). Compare rows with the same **`name`** and **`params`** only. Do not compare CPU PyTorch latency to CUDA PyTorch without stating the device change explicitly.

**Orchestration:** `./scripts/bench/queue_all_benchmarks.sh` runs the CPU harnesses by default. Optional env flags:

- `RUN_PYTORCH_CUDA=1` — append `pytorch-cuda` to the PyTorch JSON (skipped automatically if no CUDA).
- `RUN_CUDA_BENCH=1` — run `rustral-cuda` to `benchmarks/results/queue-<stamp>-cuda.json`.
- `RUN_METAL_BENCH=1` — run `rustral-metal` to `queue-<stamp>-metal.json`.

`scripts/bench/comparative_report.py` merges the primary harness file with `--harness-extra` outputs so the paper draft table lists **CPU + GPU + PyTorch** in one place.

**Not in-tree (future):** JAX / TensorFlow / ONNX Runtime baselines would follow the same pattern (new suite emitting schema v2). Contributions welcome.

## Not yet covered

- Tokenization parity with HuggingFace `tokenizers` (BPE / WordPiece). Intentional gap;
  word-level is sufficient for the current baselines.
- Multi-seed reporting on SST-2 / WikiText-2. The examples already accept `--seed`; the
  per-release snapshots in `benchmarks/runs/<version>/` will eventually carry three-seed
  numbers (mean ± std) once a maintainer captures them.
- Larger tape-integrated transformer baselines on WikiText-2. The benchmark harness now has a forward-only 2-layer, 128-d transformer encoder timing, but full transformer train-step evaluation still needs tape support for attention layers.
