# Experiments summary for EMNLP-style reporting

*Generated 2026-05-09T05:47:11Z UTC. Repository HEAD: `193331220960975cee82e158f645cd63ee11831c`.*

This document aggregates **micro-benchmarks** (operator timing, schema v2) and **NLP task** results (SST-2, WikiText-2) where available. Numbers are suitable as a **system paper** or **reproducibility appendix** draft; tighten claims after internal review.

---

## 1. Abstract (draft)

We report end-to-end and operator-level evaluation of Rustral, a Rust-first differentiable stack, against Candle (Rust) and PyTorch baselines on shared micro-workloads (matmul, attention, convolution) and on word-level SST-2 classification and WikiText-2 language modelling with matched tokenization and vocabulary. Timings include repeated runs with 95% confidence intervals where applicable; NLP aggregates report mean ± std over seeds. All artifacts are emitted as versioned JSON for independent verification.

## 2. Experimental setup

- **Micro-benchmarks**: `scripts/bench/run_all.py`, warmup + repeats as in each JSON; PyTorch CUDA uses device synchronization around timed regions.
- **NLP**: `scripts/eval/run_nlp_real.py`; `--benchmark` = small model for fast parity; curated manifests under `benchmarks/runs/<version>/nlp/` record hyperparameters and data caps.
- **Reproducibility**: See `EVALUATION.md`, `BENCHMARKS.md`, and per-run `manifest.json` / schema-v2 `machine` blocks.

## 3. Results

### Table 0. Harness environment (from JSON `machine` blocks)

| Source JSON | Suite | OS | Arch | Host | Extras |
|-------------|-------|----|------|------|--------|
| `benchmarks/results/emnlp_micro_20260509_051443.json` | rustral | linux | x86_64 | serkan3 | rustc=unknown |
| `benchmarks/results/emnlp_micro_20260509_051443.json` | candle | linux | x86_64 | serkan3 | rustc=unknown |
| `benchmarks/results/emnlp_micro_20260509_051443.json` | pytorch | linux | x86_64 | serkan3 | torch=2.5.1+cu121; py=3.12.3 |
| `benchmarks/results/emnlp_micro_20260509_051443.json` | pytorch-cuda | linux | x86_64 | serkan3 | torch=2.5.1+cu121; py=3.12.3; NVIDIA GeForce RTX 2080 Ti |
| `benchmarks/results/emnlp_rust_cuda_20260509_051443.json` | rustral | linux | x86_64 | serkan3 | rustc=unknown |

### Table A. Micro-benchmarks (wall-clock ms)

Rows share workload `name` and `params` across stacks. Mean and 95% CI on `runs_ms` where present.

| Workload | Params | Suite | Backend | Device | Mean (ms) | 95% CI | Std |
|----------|--------|-------|---------|--------|-----------|--------|-----|
| `matmul` | k=128 m=128 n=128 | rustral | ndarray-cpu | cpu | 0.096 | [0.088, 0.105] | 0.007 |
| `matmul` | k=256 m=256 n=256 | rustral | ndarray-cpu | cpu | 0.983 | [0.947, 1.018] | 0.028 |
| `matmul` | k=512 m=512 n=512 | rustral | ndarray-cpu | cpu | 5.772 | [5.431, 6.114] | 0.275 |
| `softmax_dim` | batch=32 features=64 | rustral | ndarray-cpu | cpu | 0.016 | [0.016, 0.016] | 0.000 |
| `softmax_dim` | batch=32 features=256 | rustral | ndarray-cpu | cpu | 0.063 | [0.063, 0.063] | 0.000 |
| `softmax_dim` | batch=64 features=1024 | rustral | ndarray-cpu | cpu | 0.535 | [0.473, 0.596] | 0.050 |
| `fused_linear_bias_gelu` | batch=32 in_dim=256 out_dim=1024 | rustral | ndarray-cpu | cpu | 1.240 | [1.052, 1.428] | 0.152 |
| `fused_linear_bias_gelu` | batch=32 in_dim=1024 out_dim=256 | rustral | ndarray-cpu | cpu | 0.705 | [0.660, 0.750] | 0.036 |
| `attention.small` | d_model=64 heads=4 seq_len=32 | rustral | ndarray-cpu | cpu | 0.028 | [0.016, 0.040] | 0.009 |
| `attention.medium` | d_model=256 heads=8 seq_len=128 | rustral | ndarray-cpu | cpu | 0.410 | [0.364, 0.457] | 0.038 |
| `conv2d.small` | batch=1 h=28 in_channels=1 kernel_h=5 kernel_w=5 out_channels=6 w=28 | rustral | ndarray-cpu | cpu | 0.797 | [0.000, 2.150] | 1.090 |
| `conv2d.medium` | batch=4 h=32 in_channels=16 kernel_h=3 kernel_w=3 out_channels=16 w=32 | rustral | ndarray-cpu | cpu | 7.516 | [3.326, 11.705] | 3.374 |
| `conv2d.large` | batch=8 h=64 in_channels=64 kernel_h=3 kernel_w=3 out_channels=64 w=64 | rustral | ndarray-cpu | cpu | 772.458 | [669.993, 874.924] | 82.523 |
| `lstm_forward` | config=small hidden_size=128 seq_len=10 | rustral | ndarray-cpu | cpu | 2.861 | [0.799, 4.923] | 1.661 |
| `lstm_forward` | config=medium hidden_size=256 seq_len=50 | rustral | ndarray-cpu | cpu | 30.736 | [30.184, 31.289] | 0.445 |
| `lstm_forward` | config=large hidden_size=512 seq_len=100 | rustral | ndarray-cpu | cpu | 254.059 | [225.573, 282.545] | 22.942 |
| `mlp_train_step` | batch=32 hidden=256 in_dim=128 optimizer=adam out_dim=64 | rustral | ndarray-cpu | cpu | 1.206 | [1.168, 1.244] | 0.031 |
| `optimizer_step.sgd` | profile=default slabs=8 total_params=10000000 | rustral | ndarray-cpu | cpu | 35.748 | [35.071, 36.426] | 0.546 |
| `optimizer_step.adam` | profile=default slabs=8 total_params=10000000 | rustral | ndarray-cpu | cpu | 262.547 | [229.265, 295.829] | 26.804 |
| `transformer_encoder.forward` | d_model=128 ff_dim=512 num_heads=4 num_layers=2 seq_len=128 vocab=1024 | rustral | ndarray-cpu | cpu | 34.589 | [24.716, 44.462] | 7.952 |
| `decoder.prefill` | d_model=128 num_heads=4 num_layers=2 seq_len=64 vocab=1024 | rustral | ndarray-cpu | cpu | 33.368 | [28.297, 38.438] | 4.084 |
| `decoder.decode_step.no_cache` | d_model=128 num_heads=4 num_layers=2 seq_len=128 vocab=1024 | rustral | ndarray-cpu | cpu | 55.022 | [32.350, 77.694] | 18.259 |
| `kv_cache.prefill` | head_dim=32 max_seq_len=256 num_heads=4 prefill_len=64 | rustral | ndarray-cpu | cpu | 0.042 | [0.032, 0.053] | 0.009 |
| `kv_cache.decode_step` | head_dim=32 max_seq_len=256 num_heads=4 token_len=1 | rustral | ndarray-cpu | cpu | 0.027 | [0.027, 0.028] | 0.000 |
| `model_io.save` | bytes=200004816 slab_size=1000000 slabs=50 total_params=50000000 | rustral | ndarray-cpu | cpu | 638.174 | [544.201, 732.148] | 75.683 |
| `model_io.load` | bytes=200004816 slab_size=1000000 slabs=50 total_params=50000000 | rustral | ndarray-cpu | cpu | 142.020 | [137.029, 147.011] | 4.019 |
| `matmul` | k=128 m=128 n=128 | candle | candle-cpu | cpu | 0.332 | [0.155, 0.509] | 0.142 |
| `matmul` | k=256 m=256 n=256 | candle | candle-cpu | cpu | 0.925 | [0.700, 1.150] | 0.181 |
| `matmul` | k=512 m=512 n=512 | candle | candle-cpu | cpu | 5.391 | [2.809, 7.972] | 2.079 |
| `attention.small` | d_model=64 heads=4 seq_len=32 | candle | candle-cpu | cpu | 0.027 | [0.020, 0.034] | 0.006 |
| `attention.medium` | d_model=256 heads=8 seq_len=128 | candle | candle-cpu | cpu | 0.768 | [0.274, 1.262] | 0.398 |
| `conv2d.small` | batch=1 h=28 in_channels=1 kernel_h=5 kernel_w=5 out_channels=6 w=28 | candle | candle-cpu | cpu | 0.157 | [0.120, 0.194] | 0.030 |
| `conv2d.medium` | batch=4 h=32 in_channels=16 kernel_h=3 kernel_w=3 out_channels=16 w=32 | candle | candle-cpu | cpu | 5.301 | [2.428, 8.173] | 2.314 |
| `conv2d.large` | batch=8 h=64 in_channels=64 kernel_h=3 kernel_w=3 out_channels=64 w=64 | candle | candle-cpu | cpu | 195.389 | [118.187, 272.590] | 62.176 |
| `matmul` | k=128 m=128 n=128 | pytorch | pytorch-cpu | cpu | 11.796 | [6.478, 17.113] | 4.283 |
| `matmul` | k=256 m=256 n=256 | pytorch | pytorch-cpu | cpu | 16.369 | [10.065, 22.673] | 5.077 |
| `matmul` | k=512 m=512 n=512 | pytorch | pytorch-cpu | cpu | 16.364 | [11.284, 21.443] | 4.091 |
| `attention.small` | d_model=64 heads=4 seq_len=32 | pytorch | pytorch-cpu | cpu | 12.318 | [4.939, 19.696] | 5.942 |
| `attention.medium` | d_model=256 heads=8 seq_len=128 | pytorch | pytorch-cpu | cpu | 43.304 | [33.088, 53.520] | 8.228 |
| `conv2d.small` | batch=1 h=28 in_channels=1 kernel_h=5 kernel_w=5 out_channels=6 w=28 | pytorch | pytorch-cpu | cpu | 22.819 | [17.810, 27.828] | 4.034 |
| `conv2d.medium` | batch=4 h=32 in_channels=16 kernel_h=3 kernel_w=3 out_channels=16 w=32 | pytorch | pytorch-cpu | cpu | 85.193 | [74.586, 95.801] | 8.543 |
| `conv2d.large` | batch=8 h=64 in_channels=64 kernel_h=3 kernel_w=3 out_channels=64 w=64 | pytorch | pytorch-cpu | cpu | 82.583 | [64.846, 100.320] | 14.285 |
| `matmul` | k=128 m=128 n=128 | pytorch-cuda | pytorch-cuda | cuda:0 | 0.079 | [0.047, 0.111] | 0.026 |
| `matmul` | k=256 m=256 n=256 | pytorch-cuda | pytorch-cuda | cuda:0 | 0.067 | [0.060, 0.074] | 0.006 |
| `matmul` | k=512 m=512 n=512 | pytorch-cuda | pytorch-cuda | cuda:0 | 0.085 | [0.077, 0.094] | 0.007 |
| `attention.small` | d_model=64 heads=4 seq_len=32 | pytorch-cuda | pytorch-cuda | cuda:0 | 0.272 | [0.210, 0.335] | 0.050 |
| `attention.medium` | d_model=256 heads=8 seq_len=128 | pytorch-cuda | pytorch-cuda | cuda:0 | 0.276 | [0.212, 0.339] | 0.051 |
| `conv2d.small` | batch=1 h=28 in_channels=1 kernel_h=5 kernel_w=5 out_channels=6 w=28 | pytorch-cuda | pytorch-cuda | cuda:0 | 0.122 | [0.015, 0.228] | 0.086 |
| `conv2d.medium` | batch=4 h=32 in_channels=16 kernel_h=3 kernel_w=3 out_channels=16 w=32 | pytorch-cuda | pytorch-cuda | cuda:0 | 0.135 | [0.040, 0.231] | 0.077 |
| `conv2d.large` | batch=8 h=64 in_channels=64 kernel_h=3 kernel_w=3 out_channels=64 w=64 | pytorch-cuda | pytorch-cuda | cuda:0 | 0.255 | [0.243, 0.267] | 0.010 |

*Source file: `benchmarks/results/emnlp_micro_20260509_051443.json`*

| `matmul` | k=128 m=128 n=128 | rustral | candle-cuda | cuda:0 | 0.032 | [0.006, 0.057] | 0.020 |
| `matmul` | k=256 m=256 n=256 | rustral | candle-cuda | cuda:0 | 0.029 | [0.024, 0.034] | 0.004 |
| `matmul` | k=512 m=512 n=512 | rustral | candle-cuda | cuda:0 | 0.062 | [0.046, 0.077] | 0.013 |
| `attention.small` | d_model=64 heads=4 seq_len=32 | rustral | candle-cuda | cuda:0 | 0.132 | [0.101, 0.163] | 0.025 |
| `attention.medium` | d_model=256 heads=8 seq_len=128 | rustral | candle-cuda | cuda:0 | 0.158 | [0.119, 0.197] | 0.032 |
| `conv2d.small` | batch=1 h=28 in_channels=1 kernel_h=5 kernel_w=5 out_channels=6 w=28 | rustral | candle-cuda | cuda:0 | 0.063 | [0.045, 0.080] | 0.014 |
| `conv2d.medium` | batch=4 h=32 in_channels=16 kernel_h=3 kernel_w=3 out_channels=16 w=32 | rustral | candle-cuda | cuda:0 | 0.062 | [0.060, 0.063] | 0.001 |
| `conv2d.large` | batch=8 h=64 in_channels=64 kernel_h=3 kernel_w=3 out_channels=64 w=64 | rustral | candle-cuda | cuda:0 | 2.539 | [2.319, 2.760] | 0.177 |

*Source file: `benchmarks/results/emnlp_rust_cuda_20260509_051443.json`*

## 4. NLP task results (curated aggregates)

*NLP directory: `benchmarks/runs/v0.1.0/nlp` — see manifests inside each JSON for exact model width, data caps, and tokenizer settings.*

### Table B. SST-2 (development accuracy)

| Stack | Mean ± Std | n | Curated file |
|-------|------------|---|--------------|
| **Rustral** | 0.5092 ± 0.0000 | 3 | `benchmarks/runs/v0.1.0/nlp/sst2.json` |
| **PyTorch** | 0.4897 ± 0.0200 | 3 | `benchmarks/runs/v0.1.0/nlp/sst2_pytorch.json` |

#### Per-seed SST-2

| Seed | Rustral | PyTorch | Δ |
|------|---------|---------|---|
| 0 | 0.509174 | 0.4988532066345215 | +0.0103 |
| 1 | 0.509174 | 0.4667431116104126 | +0.0424 |
| 2 | 0.509174 | 0.5034403800964355 | +0.0057 |

### Table C. WikiText-2 (development perplexity)

| Stack | Mean ± Std | n | Curated file |
|-------|------------|---|--------------|
| **Rustral** | 16181.6976 ± 149.1770 | 3 | `benchmarks/runs/v0.1.0/nlp/wikitext2.json` |
| **PyTorch** | 21516.0030 ± 3305.7062 | 3 | `benchmarks/runs/v0.1.0/nlp/wikitext2_pytorch.json` |

#### Per-seed WikiText-2

| Seed | Rustral | PyTorch | Δ |
|------|---------|---------|---|
| 0 | 16335.946289 | 20116.78396365468 | -3780.8377 |
| 1 | 16038.170898 | 19140.009683933826 | -3101.8388 |
| 2 | 16170.975586 | 25291.21534142461 | -9120.2398 |

## 5. Discussion and limitations (draft)

- **Scope**: Micro-benchmarks isolate operators; they do not replace end-to-end training efficiency studies.
- **Fairness**: Compare across rows with identical `name` and `params`; CPU vs GPU rows are not directly comparable without stating device.
- **NLP**: `--benchmark` uses small models and caps; paper-scale runs use `--paper` and larger caps (see `EVALUATION.md`).
- **Variance**: CI width reflects run-to-run noise; for publication, report hardware, library versions, and fixed seeds from JSON.

## 6. Checklist before submission

- [ ] Attach or cite exact JSON paths and git SHA.
- [ ] Confirm shared vocabulary paths for Rustral vs PyTorch NLP rows.
- [ ] State train/eval caps for WikiText-2.
- [ ] Run `python3 scripts/bench/validate_schema.py` on all harness JSON used in tables.

---

## 7. EMNLP-oriented paper draft (system / reproducibility track)

### Suggested title

*Rustral: A Rust-Centric Differentiable Stack with Reproducible NLP and Operator Benchmarks*

### Authors

*[Author list and affiliations — placeholder]*

### Abstract (submission length, ~200 words)

We present Rustral, an open-source differentiable programming stack implemented primarily in Rust, with explicit attention to reproducible evaluation. We release a unified JSON schema for micro-benchmarks that compares Rustral's ndarray CPU backend, a Candle-based Rust baseline, and PyTorch on CPU and CUDA over matched operator workloads (matrix multiply, multi-head attention-style computation, and convolution). Timings report repeated wall-clock measurements with 95% confidence intervals. Complementing operator data, we evaluate word-level SST-2 sentiment classification and WikiText-2 language modeling with shared vocabularies and documented data caps, including PyTorch mirrors for stack parity. All experiments are driven by scripts that pin dataset checksums, emit per-run manifests, and integrate with continuous integration for schema validation. Our goal is not leaderboard placement on these small baselines, but transparent, citeable numbers suitable for systems-oriented NLP venues such as EMNLP. We discuss limitations—including word-level tokenization, modest model scale in default presets, and the gap between forward-only and full tape training for some layers—and outline a roadmap toward larger models and additional frameworks.

### 1. Introduction

Systems papers at EMNLP increasingly require evidence that a new implementation is not only correct but also competitive and reproducible. Rustral targets researchers who want native Rust performance and safety while retaining familiar autodiff ergonomics. This draft grounds claims in **paired experiments**: the same workload names and tensor shapes across backends, and matched NLP tasks against PyTorch with shared vocabulary files.

### 2. Method: tasks and metrics

**Micro-benchmarks.** We follow schema v2 (`benchmarks/SCHEMA.md`): each sample records `runs_ms`, aggregates, optional 95% CIs, and machine metadata. GPU timings synchronize the device before and after each timed region.

**NLP.** SST-2 accuracy is measured on the development split; WikiText-2 perplexity is `exp(mean cross-entropy in nats)` over capped evaluation windows, as documented in `EVALUATION.md`.

### 3. Experiments (summary)

Table 0 summarizes hardware and software captured in JSON. Table A lists operator timings across stacks. Tables B–C summarize NLP aggregates; per-seed rows support variance reporting.

### 4. Analysis (draft bullets)

- Compare **within device class** (CPU vs CPU, GPU vs GPU) when discussing speedups.
- Use CIs to avoid over-interpreting single runs.
- For NLP, verify that `vocab.txt` paths match between Rustral and PyTorch rows before claiming parity.

### 5. Ethics and reproducibility

Public datasets (SST-2, WikiText-2) are accessed via pinned URLs and hashes in `rustral-data`. No additional annotation or crowd-sourcing is introduced. Full commands to regenerate this document are given in `BENCHMARKS.md`.

### 6. References (placeholder)

- Socher et al., SST / sentiment (cite standard SST-2 reference used by the community).
- Merity et al., WikiText-2.
- Paszke et al., PyTorch.

---

## 8. Input artifacts (this run)

- `benchmarks/results/emnlp_micro_20260509_051443.json`
- `benchmarks/results/emnlp_rust_cuda_20260509_051443.json`
- `benchmarks/runs/v0.1.0/nlp/` (NLP curated JSONs)

### Regenerating this report

```bash
STAMP=$(date +%Y%m%d_%H%M%S)
venv/bin/python scripts/bench/run_all.py --repeats 5 --warmup 1 \
  --suite rustral --suite candle --suite pytorch --suite pytorch-cuda \
  --out benchmarks/results/emnlp_micro_${STAMP}.json
venv/bin/python scripts/bench/run_all.py --repeats 5 --warmup 1 \
  --suite rustral-cuda --out benchmarks/results/emnlp_rust_cuda_${STAMP}.json
venv/bin/python scripts/bench/build_emnlp_report.py \
  --micro benchmarks/results/emnlp_micro_${STAMP}.json \
  --rust-cuda benchmarks/results/emnlp_rust_cuda_${STAMP}.json \
  --nlp-dir benchmarks/runs/v0.1.0/nlp \
  --out docs/emnlp_experiments_report.md
cp docs/emnlp_experiments_report.md docs/emnlp_experiments_report.txt
```

Optional: full paper-profile NLP + queue — `RUN_NLP_PAPER=1 ./scripts/bench/queue_all_benchmarks.sh`; Apple GPU — `RUN_METAL_BENCH=1`. After new NLP JSONs exist, point `--nlp-dir` at `benchmarks/runs/v<version>/nlp/`.
