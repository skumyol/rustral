# Rustral comparative analysis

Generated `2026-05-09T07:18:47Z` (UTC).
NLP aggregates read from `benchmarks/runs/v0.1.0/nlp/` (requested `0.2.0`, fallback `0.1.0` if needed).

> Use this as a starting draft for the paper: add statistical tests, confidence intervals,
> and ablations as needed. Verify that Rustral and PyTorch runs share vocabulary and
> hyperparameters before claiming parity.

## SST-2 (dev accuracy)

| Stack | Metric | Mean ± Std | n | Notes |
|-------|--------|------------|---|-------|
| **Rustral** | ? | 0.5092 | 3 | curated `benchmarks/runs/v0.1.0/nlp/sst2.json` |
| **PyTorch** | ? | 0.4897 ± 0.0200 | 3 | curated `benchmarks/runs/v0.1.0/nlp/sst2_pytorch.json` |

### Per-seed (headline metric)

| Seed | Rustral | PyTorch | Δ |
|------|---------|---------|---|
| 0 | 0.509174 | 0.4988532066345215 | +0.0103 |
| 1 | 0.509174 | 0.4667431116104126 | +0.0424 |
| 2 | 0.509174 | 0.5034403800964355 | +0.0057 |

### Rustral run provenance (seed 0 excerpt)

- `task`: sst2_classifier
- `model_type`: transformer_encoder
- `seq_len`: 16
- `d_model`: 32
- `num_heads`: 2
- `ffn_dim`: 64
- `num_layers`: 1
- `vocab_size`: 1735
- `total_params`: 64642
- `epochs`: 1
- `batch_size`: 32
- `learning_rate`: 0.0005
- `train_examples`: 256
- `quick_mode`: True

## WikiText-2 (dev perplexity)

| Stack | Metric | Mean ± Std | n | Notes |
|-------|--------|------------|---|-------|
| **Rustral** | ? | 16181.6976 ± 149.1770 | 3 | curated `benchmarks/runs/v0.1.0/nlp/wikitext2.json` |
| **PyTorch** | ? | 21516.0030 ± 3305.7062 | 3 | curated `benchmarks/runs/v0.1.0/nlp/wikitext2_pytorch.json` |

### Per-seed (headline metric)

| Seed | Rustral | PyTorch | Δ |
|------|---------|---------|---|
| 0 | 16335.946289 | 20116.78396365468 | -3780.8377 |
| 1 | 16038.170898 | 19140.009683933826 | -3101.8388 |
| 2 | 16170.975586 | 25291.21534142461 | -9120.2398 |

### Rustral run provenance (seed 0 excerpt)

- `task`: wikitext2_word_lm
- `model_type`: transformer_lm
- `block_size`: 16
- `d_model`: 32
- `num_heads`: 2
- `ffn_dim`: 64
- `num_layers`: 1
- `vocab_size`: 16384
- `total_params`: 1074016
- `epochs`: 1
- `batch_size`: 32
- `learning_rate`: 0.0005
- `train_tokens_used`: 4000
- `quick_mode`: True

## Micro-benchmarks (schema v2 harness)

### Methodology (operator timing)

- **Wall-clock ms** per repeat; **CPU** timings use `time.perf_counter()` around the op.
- **PyTorch CUDA**: `torch.cuda.synchronize()` before/after each timed region so async launch does not skew means (standard practice for academic GPU micro-benchmarks).
- **Rustral `rustral-cuda` / `rustral-metal`**: same schema; compare only within the same device class (CPU vs CPU, GPU vs GPU).
- **Frameworks**: ndarray CPU and Candle CPU vs PyTorch CPU/CUDA — apples-to-apples on `name` + `params`, not across different op fusion or autograd paths.

| Workload | Suite | Backend | Device | Mean (ms) | 95% CI | Std (ms) | JSON |
|----------|-------|---------|--------|-----------|--------|----------|------|
| matmul | rustral | ndarray-cpu | cpu | 0.264 | — | 0.000 | `benchmarks/results/queue-20260509_071825.json` |
| matmul | rustral | ndarray-cpu | cpu | 0.952 | — | 0.000 | `benchmarks/results/queue-20260509_071825.json` |
| matmul | rustral | ndarray-cpu | cpu | 6.495 | — | 0.000 | `benchmarks/results/queue-20260509_071825.json` |
| softmax_dim | rustral | ndarray-cpu | cpu | 0.064 | — | 0.000 | `benchmarks/results/queue-20260509_071825.json` |
| softmax_dim | rustral | ndarray-cpu | cpu | 0.075 | — | 0.000 | `benchmarks/results/queue-20260509_071825.json` |
| softmax_dim | rustral | ndarray-cpu | cpu | 0.746 | — | 0.000 | `benchmarks/results/queue-20260509_071825.json` |
| fused_linear_bias_gelu | rustral | ndarray-cpu | cpu | 2.982 | — | 0.000 | `benchmarks/results/queue-20260509_071825.json` |
| fused_linear_bias_gelu | rustral | ndarray-cpu | cpu | 1.214 | — | 0.000 | `benchmarks/results/queue-20260509_071825.json` |
| attention.small | rustral | ndarray-cpu | cpu | 0.050 | — | 0.000 | `benchmarks/results/queue-20260509_071825.json` |
| attention.medium | rustral | ndarray-cpu | cpu | 0.522 | — | 0.000 | `benchmarks/results/queue-20260509_071825.json` |
| conv2d.small | rustral | ndarray-cpu | cpu | 0.351 | — | 0.000 | `benchmarks/results/queue-20260509_071825.json` |
| conv2d.medium | rustral | ndarray-cpu | cpu | 2.359 | — | 0.000 | `benchmarks/results/queue-20260509_071825.json` |
| conv2d.large | rustral | ndarray-cpu | cpu | 313.438 | — | 0.000 | `benchmarks/results/queue-20260509_071825.json` |
| lstm_forward | rustral | ndarray-cpu | cpu | 1.570 | — | 0.000 | `benchmarks/results/queue-20260509_071825.json` |
| lstm_forward | rustral | ndarray-cpu | cpu | 28.171 | — | 0.000 | `benchmarks/results/queue-20260509_071825.json` |
| lstm_forward | rustral | ndarray-cpu | cpu | 227.415 | — | 0.000 | `benchmarks/results/queue-20260509_071825.json` |
| mlp_train_step | rustral | ndarray-cpu | cpu | 1.156 | — | 0.000 | `benchmarks/results/queue-20260509_071825.json` |
| optimizer_step.sgd | rustral | ndarray-cpu | cpu | 60.809 | — | 0.000 | `benchmarks/results/queue-20260509_071825.json` |
| optimizer_step.adam | rustral | ndarray-cpu | cpu | 413.523 | — | 0.000 | `benchmarks/results/queue-20260509_071825.json` |
| transformer_encoder.forward | rustral | ndarray-cpu | cpu | 9.888 | — | 0.000 | `benchmarks/results/queue-20260509_071825.json` |
| decoder.prefill | rustral | ndarray-cpu | cpu | 11.596 | — | 0.000 | `benchmarks/results/queue-20260509_071825.json` |
| decoder.decode_step.no_cache | rustral | ndarray-cpu | cpu | 18.253 | — | 0.000 | `benchmarks/results/queue-20260509_071825.json` |
| kv_cache.prefill | rustral | ndarray-cpu | cpu | 0.072 | — | 0.000 | `benchmarks/results/queue-20260509_071825.json` |
| kv_cache.decode_step | rustral | ndarray-cpu | cpu | 0.059 | — | 0.000 | `benchmarks/results/queue-20260509_071825.json` |
| model_io.save | rustral | ndarray-cpu | cpu | 630.129 | — | 0.000 | `benchmarks/results/queue-20260509_071825.json` |
| model_io.load | rustral | ndarray-cpu | cpu | 425.130 | — | 0.000 | `benchmarks/results/queue-20260509_071825.json` |
| matmul | candle | candle-cpu | cpu | 6.358 | — | 0.000 | `benchmarks/results/queue-20260509_071825.json` |
| matmul | candle | candle-cpu | cpu | 0.612 | — | 0.000 | `benchmarks/results/queue-20260509_071825.json` |
| matmul | candle | candle-cpu | cpu | 1.740 | — | 0.000 | `benchmarks/results/queue-20260509_071825.json` |
| attention.small | candle | candle-cpu | cpu | 0.158 | — | 0.000 | `benchmarks/results/queue-20260509_071825.json` |
| attention.medium | candle | candle-cpu | cpu | 0.640 | — | 0.000 | `benchmarks/results/queue-20260509_071825.json` |
| conv2d.small | candle | candle-cpu | cpu | 0.210 | — | 0.000 | `benchmarks/results/queue-20260509_071825.json` |
| conv2d.medium | candle | candle-cpu | cpu | 1.583 | — | 0.000 | `benchmarks/results/queue-20260509_071825.json` |
| conv2d.large | candle | candle-cpu | cpu | 57.084 | — | 0.000 | `benchmarks/results/queue-20260509_071825.json` |
| matmul | pytorch | pytorch-cpu | cpu | 6.736 | — | 0.000 | `benchmarks/results/queue-20260509_071825-pytorch.json` |
| matmul | pytorch | pytorch-cpu | cpu | 1.658 | — | 0.000 | `benchmarks/results/queue-20260509_071825-pytorch.json` |
| matmul | pytorch | pytorch-cpu | cpu | 4.339 | — | 0.000 | `benchmarks/results/queue-20260509_071825-pytorch.json` |
| attention.small | pytorch | pytorch-cpu | cpu | 2.987 | — | 0.000 | `benchmarks/results/queue-20260509_071825-pytorch.json` |
| attention.medium | pytorch | pytorch-cpu | cpu | 5.658 | — | 0.000 | `benchmarks/results/queue-20260509_071825-pytorch.json` |
| conv2d.small | pytorch | pytorch-cpu | cpu | 2.882 | — | 0.000 | `benchmarks/results/queue-20260509_071825-pytorch.json` |
| conv2d.medium | pytorch | pytorch-cpu | cpu | 0.828 | — | 0.000 | `benchmarks/results/queue-20260509_071825-pytorch.json` |
| conv2d.large | pytorch | pytorch-cpu | cpu | 15.832 | — | 0.000 | `benchmarks/results/queue-20260509_071825-pytorch.json` |
| matmul | jax | jax-cpu | cpu | 0.236 | — | 0.000 | `benchmarks/results/queue-20260509_071825-extra.json` |
| matmul | jax | jax-cpu | cpu | 0.330 | — | 0.000 | `benchmarks/results/queue-20260509_071825-extra.json` |
| matmul | jax | jax-cpu | cpu | 1.422 | — | 0.000 | `benchmarks/results/queue-20260509_071825-extra.json` |
| attention.small | jax | jax-cpu | cpu | 0.122 | — | 0.000 | `benchmarks/results/queue-20260509_071825-extra.json` |
| attention.medium | jax | jax-cpu | cpu | 0.558 | — | 0.000 | `benchmarks/results/queue-20260509_071825-extra.json` |
| conv2d.small | jax | jax-cpu | cpu | 0.147 | — | 0.000 | `benchmarks/results/queue-20260509_071825-extra.json` |
| conv2d.medium | jax | jax-cpu | cpu | 0.654 | — | 0.000 | `benchmarks/results/queue-20260509_071825-extra.json` |
| conv2d.large | jax | jax-cpu | cpu | 7.309 | — | 0.000 | `benchmarks/results/queue-20260509_071825-extra.json` |
| matmul | tensorflow | tensorflow-cpu | cpu | 1.118 | — | 0.000 | `benchmarks/results/queue-20260509_071825-extra.json` |
| matmul | tensorflow | tensorflow-cpu | cpu | 0.876 | — | 0.000 | `benchmarks/results/queue-20260509_071825-extra.json` |
| matmul | tensorflow | tensorflow-cpu | cpu | 2.015 | — | 0.000 | `benchmarks/results/queue-20260509_071825-extra.json` |
| attention.small | tensorflow | tensorflow-cpu | cpu | 0.631 | — | 0.000 | `benchmarks/results/queue-20260509_071825-extra.json` |
| attention.medium | tensorflow | tensorflow-cpu | cpu | 1.177 | — | 0.000 | `benchmarks/results/queue-20260509_071825-extra.json` |
| conv2d.small | tensorflow | tensorflow-cpu | cpu | 0.566 | — | 0.000 | `benchmarks/results/queue-20260509_071825-extra.json` |
| conv2d.medium | tensorflow | tensorflow-cpu | cpu | 1.328 | — | 0.000 | `benchmarks/results/queue-20260509_071825-extra.json` |
| conv2d.large | tensorflow | tensorflow-cpu | cpu | 12.477 | — | 0.000 | `benchmarks/results/queue-20260509_071825-extra.json` |
| matmul | onnxruntime | onnxruntime-cpu | cpu | 0.656 | — | 0.000 | `benchmarks/results/queue-20260509_071825-extra.json` |
| matmul | onnxruntime | onnxruntime-cpu | cpu | 0.317 | — | 0.000 | `benchmarks/results/queue-20260509_071825-extra.json` |
| matmul | onnxruntime | onnxruntime-cpu | cpu | 1.114 | — | 0.000 | `benchmarks/results/queue-20260509_071825-extra.json` |
| attention.small | onnxruntime | onnxruntime-cpu | cpu | 0.163 | — | 0.000 | `benchmarks/results/queue-20260509_071825-extra.json` |
| attention.medium | onnxruntime | onnxruntime-cpu | cpu | 0.218 | — | 0.000 | `benchmarks/results/queue-20260509_071825-extra.json` |
| conv2d.small | onnxruntime | onnxruntime-cpu | cpu | 0.178 | — | 0.000 | `benchmarks/results/queue-20260509_071825-extra.json` |
| conv2d.medium | onnxruntime | onnxruntime-cpu | cpu | 0.359 | — | 0.000 | `benchmarks/results/queue-20260509_071825-extra.json` |
| conv2d.large | onnxruntime | onnxruntime-cpu | cpu | 10.876 | — | 0.000 | `benchmarks/results/queue-20260509_071825-extra.json` |

## Checklist for the paper

- [ ] Same tokenizer vocabulary file for Rustral and PyTorch rows being compared.
- [ ] Same train token cap / eval window cap documented (WikiText-2).
- [ ] Hardware (CPU/GPU), batch size, and wall-clock noted.
- [ ] Micro-benchmark table: compare same `name`+`params` only; GPU rows use synchronized timing.
- [ ] Learning-rate schedule: document if warmup differs between stacks.
