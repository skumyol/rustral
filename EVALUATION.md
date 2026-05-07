# NLP evaluation methodology

This document explains how Rustral's NLP examples train and evaluate. The goal is simple: a reader should be able to clone the repo, run the commands, and understand the numbers without guessing at hidden settings.

If a reported number disagrees with this file, treat the run as suspicious until the command, manifest, and dataset cache are checked.

## Tasks and corpora

| Task | Corpus | Source | Split | Tokenizer | Reported metric |
|---|---|---|---|---|---|
| Sentiment classification | SST-2 (binary) | GLUE mirror (`dl.fbaipublicfiles.com/glue/data/SST-2`) | `train.tsv` / `dev.tsv` | `rustral-data WordLevelTokenizer` (lowercase, whitespace, max_vocab=8192) | dev accuracy |
| Word-level language modelling | WikiText-2 raw v1 | `s3.amazonaws.com/research.metamind.io/wikitext` | `wiki.{train,valid,test}.raw` | `rustral-data WordLevelTokenizer` (lowercase, whitespace, max_vocab=16384) | dev perplexity (`exp(mean cross-entropy nats)`) |

Tokenization is deliberately word-level. WikiText-2 is commonly reported that way, SST-2 works cleanly with it, and it keeps the dependency tree small. HuggingFace `tokenizers` can come later when a BPE or WordPiece baseline needs it. See `crates/data/src/tokenizer.rs`.

## Models

Both examples use tiny architectures so they finish on CPU. These are honest baselines, not leaderboard attempts. The point is to make the evaluation path real and easy to inspect.

### SST-2 classifier (`crates/runtime/examples/sst2_classifier.rs`)

```
Embedding(V, 32)        # vocab_size up to 8192, dim 32
  ↓ reshape([1, 32 * 32])     # 32 token positions × 32 dim each
Linear(32 * 32 → 2)
```

- Sentence is encoded with the `WordLevelTokenizer`, padded / truncated to 32 tokens.
- Classifier is a **bag-of-positions linear model on learned embeddings**. Every token
  position contributes to the class logit. It is not a strong baseline, but it is a real
  one that trains fast and writes a useful manifest.

### WikiText-2 LM (`crates/runtime/examples/wikitext2_lm.rs`)

```
Embedding(V, 32)        # vocab_size up to 16384, dim 32
  ↓ reshape([1, 16 * 32])    # 16-token context window
Linear(16 * 32 → V)
```

- Sliding window of 16 consecutive tokens predicts the next token.
- Trained on a capped subset of the train split (default 50k tokens; `--quick` uses 4k).
- Reported metric is dev perplexity, computed as `exp(mean cross-entropy nats)` over
  every valid window in `wiki.valid.raw`.

## Hyperparameters (defaults)

| Hyperparameter | SST-2 | WikiText-2 |
|---|---|---|
| Seed (`--seed`) | `0xC0FFEE` | `0xC0FFEE` |
| Optimizer | Adam | Adam |
| Learning rate (`--lr`) | `3e-3` | `5e-3` |
| Batch size (`--batch`) | 32 | 32 |
| Epochs (`--epochs`) | 3 | 1 |
| Sequence/window length | 32 | 16 |
| Embedding dim | 32 | 32 |
| Max vocab | 8192 | 16384 |

`--quick` modes, used by smoke tests, shrink the training set so the run finishes in seconds. SST-2 caps training at 256 examples. WikiText-2 caps training at 4 000 tokens.

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
  Required while the upstream SHAs in `crates/data/src/datasets/{sst2,wikitext2}.rs` are
  still placeholder zeros. See the pinning section below.

## Pinning upstream artifacts

`crates/data/src/datasets/sst2.rs` and `crates/data/src/datasets/wikitext2.rs` declare
canonical URLs and a `*_SHA256` constant for each artifact. The hashes are currently
**placeholders** (all zeros) so first authoritative download verification is deferred to
the next time someone runs an online fetch on a clean cache; that operator must:

1. Run the example without `RUSTRAL_DATASET_SKIP_CHECKSUM`, observe the
   `ChecksumMismatch` error reporting `actual = <real_hash>`.
2. Replace the placeholder constant with the reported `actual` hash.
3. Open a small PR with the change and a note in `CHANGELOG.md`.

This keeps offline CI useful today without pretending we have verified a hash that nobody has checked yet.

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

## Not yet covered

- Tokenization parity with HuggingFace `tokenizers` (BPE / WordPiece). Intentional gap;
  word-level is sufficient for the current baselines.
- Multi-seed reporting on SST-2 / WikiText-2. The examples already accept `--seed`; the
  per-release snapshots in `benchmarks/runs/<version>/` will eventually carry three-seed
  numbers (mean ± std) once a maintainer captures them.
- Larger tape-integrated transformer baselines on WikiText-2. The benchmark harness now has a forward-only 2-layer, 128-d transformer encoder timing, but full transformer train-step evaluation still needs tape support for attention layers.
