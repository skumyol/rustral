//! HTTP JSON inference MVP: one `Linear` layer loaded from `save_model` / `save_model_to_path` artifacts.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use rustral_core::{Backend, ForwardCtx, Mode, Module, NamedParameters};
use rustral_ndarray_backend::CpuBackend;
use rustral_nn::{Linear, LinearBuilder};
use rustral_runtime::load_model_from_path;
use serde::{Deserialize, Serialize};
use tower_http::trace::TraceLayer;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "rustral-inference-server")]
struct Args {
    /// Path to a Safetensors file from `save_model_to_path`.
    #[arg(long)]
    artifact: PathBuf,

    /// Listen address, e.g. 127.0.0.1:8080
    #[arg(long, default_value = "127.0.0.1:8080")]
    bind: SocketAddr,

    /// `Linear` in_features (must match artifact).
    #[arg(long, default_value = "1")]
    in_features: usize,

    /// `Linear` out_features (must match artifact).
    #[arg(long, default_value = "1")]
    out_features: usize,

    /// Whether the layer has bias (must match artifact).
    #[arg(long, default_value = "true")]
    bias: bool,
}

#[derive(Clone)]
struct TinyModel {
    lin: Linear<CpuBackend>,
}

impl NamedParameters<CpuBackend> for TinyModel {
    fn visit_parameters(&self, f: &mut dyn FnMut(&str, &rustral_core::Parameter<CpuBackend>)) {
        self.lin.visit_parameters(f);
    }

    fn visit_parameters_mut(&mut self, f: &mut dyn FnMut(&str, &mut rustral_core::Parameter<CpuBackend>)) {
        self.lin.visit_parameters_mut(f);
    }
}

struct AppState {
    backend: CpuBackend,
    model: TinyModel,
    in_features: usize,
    out_features: usize,
    artifact_path: String,
    infer_total: AtomicU64,
    infer_errors: AtomicU64,
    infer_latency_ns_sum: AtomicU64,
}

#[derive(Deserialize)]
struct InferRequest {
    /// Row-major batch: `[batch][in_features]`.
    input: Vec<Vec<f32>>,
}

#[derive(Serialize)]
struct InferResponse {
    output: Vec<Vec<f32>>,
}

#[derive(Serialize)]
struct MetadataResponse {
    in_features: usize,
    out_features: usize,
    artifact: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();

    let args = Args::parse();
    let backend = CpuBackend::default();
    let lin = LinearBuilder::new(args.in_features, args.out_features)
        .with_bias(args.bias)
        .seed(0)
        .build(&backend)?;
    let mut model = TinyModel { lin };
    load_model_from_path(&args.artifact, &mut model, &backend)?;

    let state = Arc::new(AppState {
        backend,
        model,
        in_features: args.in_features,
        out_features: args.out_features,
        artifact_path: args.artifact.display().to_string(),
        infer_total: AtomicU64::new(0),
        infer_errors: AtomicU64::new(0),
        infer_latency_ns_sum: AtomicU64::new(0),
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/ready", get(ready))
        .route("/v1/metadata", get(metadata))
        .route("/v1/infer", post(infer))
        .route("/metrics", get(metrics))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    tracing::info!("listening on http://{}", args.bind);
    let listener = tokio::net::TcpListener::bind(args.bind).await?;

    let shutdown = async {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{signal, SignalKind};
            let mut sigint = signal(SignalKind::interrupt()).expect("sigint");
            let mut sigterm = signal(SignalKind::terminate()).expect("sigterm");
            tokio::select! {
                _ = sigint.recv() => {},
                _ = sigterm.recv() => {},
            }
        }
        #[cfg(not(unix))]
        {
            let _ = tokio::signal::ctrl_c().await;
        }
        tracing::info!("graceful shutdown");
    };

    axum::serve(listener, app).with_graceful_shutdown(shutdown).await?;
    Ok(())
}

async fn health() -> &'static str {
    "ok"
}

async fn ready() -> &'static str {
    "ok"
}

async fn metadata(State(state): State<Arc<AppState>>) -> Json<MetadataResponse> {
    Json(MetadataResponse {
        in_features: state.in_features,
        out_features: state.out_features,
        artifact: state.artifact_path.clone(),
    })
}

async fn metrics(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let n = state.infer_total.load(Ordering::Relaxed);
    let e = state.infer_errors.load(Ordering::Relaxed);
    let sum = state.infer_latency_ns_sum.load(Ordering::Relaxed);
    let body = format!(
        "# HELP rustral_infer_requests_total Completed inference requests\n\
# TYPE rustral_infer_requests_total counter\n\
rustral_infer_requests_total {n}\n\
# HELP rustral_infer_errors_total Failed inference requests\n\
# TYPE rustral_infer_errors_total counter\n\
rustral_infer_errors_total {e}\n\
# HELP rustral_infer_latency_ns_sum Sum of inference latency in nanoseconds\n\
# TYPE rustral_infer_latency_ns_sum counter\n\
rustral_infer_latency_ns_sum {sum}\n"
    );
    ([(header::CONTENT_TYPE, "text/plain; version=0.0.4")], body)
}

async fn infer(
    State(state): State<Arc<AppState>>,
    Json(req): Json<InferRequest>,
) -> Result<Json<InferResponse>, (StatusCode, String)> {
    let t0 = Instant::now();
    let res = infer_inner(&state, req);
    let elapsed = t0.elapsed().as_nanos() as u64;
    match &res {
        Ok(_) => {
            state.infer_total.fetch_add(1, Ordering::Relaxed);
            state.infer_latency_ns_sum.fetch_add(elapsed, Ordering::Relaxed);
        }
        Err(_) => {
            state.infer_errors.fetch_add(1, Ordering::Relaxed);
        }
    }
    res
}

fn infer_inner(state: &AppState, req: InferRequest) -> Result<Json<InferResponse>, (StatusCode, String)> {
    if req.input.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "empty batch".into()));
    }
    for row in &req.input {
        if row.len() != state.in_features {
            return Err((
                StatusCode::BAD_REQUEST,
                format!("expected row length {}, got {}", state.in_features, row.len()),
            ));
        }
    }

    let batch = req.input.len();
    let flat: Vec<f32> = req.input.iter().flatten().copied().collect();
    let x = state
        .backend
        .tensor_from_vec(flat, &[batch, state.in_features])
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("tensor_from_vec: {e:?}")))?;

    let mut ctx = ForwardCtx::new(&state.backend, Mode::Inference);
    let y = state
        .model
        .lin
        .forward(x, &mut ctx)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("forward: {e:?}")))?;

    let flat_out = state
        .backend
        .ops()
        .tensor_to_vec(&y)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("tensor_to_vec: {e:?}")))?;

    if flat_out.len() != batch * state.out_features {
        return Err((StatusCode::INTERNAL_SERVER_ERROR, "unexpected output length".into()));
    }

    let mut output = Vec::with_capacity(batch);
    let stride = state.out_features;
    for i in 0..batch {
        let start = i * stride;
        output.push(flat_out[start..start + stride].to_vec());
    }

    Ok(Json(InferResponse { output }))
}
