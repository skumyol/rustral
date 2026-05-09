# Deploying `rustral-inference-server`

This document covers **integration-style** production deployment: reverse proxy, containers, health checks, and metrics. Rustral does not ship a first-party Triton plugin; use this HTTP service directly or behind your platform’s ingress.

## Health and readiness

| Path | Purpose |
|------|---------|
| `GET /health` | Liveness: process is up (returns `ok`). |
| `GET /ready` | Readiness: model loaded and server accepting traffic (MVP: same as liveness). |
| `GET /v1/metadata` | JSON with `in_features`, `out_features`, artifact path. |
| `GET /metrics` | Prometheus text exposition (counters + latency sum). |

Kubernetes example:

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8080
readinessProbe:
  httpGet:
    path: /ready
    port: 8080
```

## Metrics

Scrape `GET /metrics` with Prometheus or a compatible collector. Counters:

- `rustral_infer_requests_total` — completed inference requests.
- `rustral_infer_errors_total` — requests that returned an error response.
- `rustral_infer_latency_ns_sum` — sum of server-side inference latency (nanoseconds); divide by requests for a coarse average (not a histogram).

For full request latency including JSON parsing, place instrumentation at your reverse proxy or service mesh.

## Reverse proxy (nginx)

```nginx
upstream rustral_infer {
    server 127.0.0.1:8080;
    keepalive 32;
}

server {
    listen 80;
    client_max_body_size 1m;

    location / {
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header Connection "";
        proxy_pass http://rustral_infer;
    }
}
```

Tune `client_max_body_size` to your largest JSON payload. For TLS, terminate at nginx or your cloud load balancer.

## Envoy (minimal cluster)

Point a cluster to the server address and port; use `/ready` for health checks and `/metrics` if your control plane scrapes Envoy admin or a sidecar.

## Autoscaling signals

Use **CPU** and **request rate** from the proxy or ingress as primary signals. If you scrape `/metrics`, **error rate** (`rustral_infer_errors_total` delta) and **latency trend** (from proxy histograms) complement CPU.

## Graceful shutdown

The server handles **SIGINT** / **SIGTERM** (via Tokio’s `ctrl_c` handler on Unix and graceful shutdown on Windows where supported). Under Kubernetes, set `terminationGracePeriodSeconds` so in-flight requests can finish after the endpoint is removed from the service.

## Docker

See [`Dockerfile`](Dockerfile) and [`docker-compose.yml`](docker-compose.yml). Mount your Safetensors artifact or bake it into an image layer for reproducible demos.

## Triton / TorchServe

To run inside NVIDIA Triton, use a **Python backend** or **HTTP backend** that forwards to this service (sidecar or network call). Maintaining a C++ Triton backend is out of scope for the core Rustral repo.
