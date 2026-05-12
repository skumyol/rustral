# rustral-inference-server

Minimal **HTTP JSON** server that loads a Rustral Safetensors artifact (`save_model` / `save_model_to_path`) and runs forward inference for a **single `Linear`** layer (default: 1×1 with bias, matching `tape_train_demo`).

## Build

```bash
cargo build -p rustral-inference-server --release
```

## Create an artifact on disk

```bash
cargo run -p rustral-runtime --features training --example save_linear_artifact -- target/tiny_linear.safetensors
```

(If you omit the path, it defaults to `tiny_linear.safetensors` in the current directory.)

## Run

```bash
./target/release/rustral-inference-server \
  --artifact target/tiny_linear.safetensors \
  --bind 127.0.0.1:8080 \
  --in-features 1 \
  --out-features 1 \
  --bias
```

## Llama-class HTTP server (local SafeTensors)

The **`rustral-llama-server`** binary (same crate; default features) loads a **local** model directory (`config.json`, `tokenizer.json`, sharded or single `model.safetensors`) and serves greedy generation with a **fresh [`LlamaDecodeCache`](../../nn/src/llama.rs) per request** (KV prefill + per-token decode).

```bash
cargo build -p rustral-inference-server --release --bin rustral-llama-server
./target/release/rustral-llama-server --model-dir /path/to/snapshot --bind 127.0.0.1:8081
# JSON: POST /v1/generate  body: {"prompt":"Hello","max_new_tokens":16}
# SSE:  POST /v1/generate/stream (events: token_id lines, final done + text)
```

## Production notes

See [`DEPLOYMENT.md`](DEPLOYMENT.md) for nginx, Kubernetes probes, Prometheus `/metrics`, Docker, and graceful shutdown.

## Call

Health:

```bash
curl -s http://127.0.0.1:8080/health
```

Readiness (MVP same as health):

```bash
curl -s http://127.0.0.1:8080/ready
```

Metrics (Prometheus text):

```bash
curl -s http://127.0.0.1:8080/metrics
```

Infer (batch of row vectors):

```bash
curl -s -X POST http://127.0.0.1:8080/v1/infer \
  -H 'content-type: application/json' \
  -d '{"input":[[0.25],[0.5]]}'
```

Response shape: `output` is `[batch, out_features]` flattened row-major in nested JSON arrays.

## Limits

- Architecture is fixed to one `Linear` for this MVP; artifact keys must match `lin.weight` / `lin.bias` when bias is enabled.
- For multi-layer models, extend this binary or use `InferencePool` in your own service.
