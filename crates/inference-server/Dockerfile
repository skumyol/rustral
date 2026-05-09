# Build stage: compile the inference server from the workspace root.
# Pinned to workspace MSRV (root Cargo.toml `rust-version`); image verified on Docker Hub.
FROM rust:1.75-bookworm AS builder
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
RUN cargo build --release -p rustral-inference-server

FROM debian:bookworm-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
COPY --from=builder /app/target/release/rustral-inference-server /usr/local/bin/
EXPOSE 8080
# Override at run time, e.g.:
# docker run -v /path/to/model.safetensors:/model/model.safetensors:ro \
#   rustral-inference-server --artifact /model/model.safetensors --bind 0.0.0.0:8080
ENTRYPOINT ["/usr/local/bin/rustral-inference-server"]
CMD ["--artifact", "/model/model.safetensors", "--bind", "0.0.0.0:8080", "--in-features", "1", "--out-features", "1", "--bias"]
