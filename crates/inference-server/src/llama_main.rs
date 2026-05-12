//! HTTP JSON server for **Llama**-class causal LM: local SafeTensors directory, per-request
//! [`LlamaDecodeCache`](rustral_nn::LlamaDecodeCache) (prefill + greedy decode steps).
//!
//! ```bash
//! cargo run -p rustral-inference-server --bin rustral-llama-server -- \
//!   --model-dir /path/to/snapshot --bind 127.0.0.1:8081
//! ```

use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use rustral_core::{ForwardCtx, Mode};
use rustral_io::load_meta_state_dict_from_paths;
use rustral_llm::{HfLlamaConfig, LlamaCausalLm, TokenizerHandle};
use rustral_nn::LlamaDecodeCache;
use serde::{Deserialize, Serialize};
use tower_http::trace::TraceLayer;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "rustral-llama-server")]
struct Args {
    #[arg(long)]
    model_dir: PathBuf,

    #[arg(long, default_value = "127.0.0.1:8081")]
    bind: SocketBuf,
}

/// Newtype so clap can parse `SocketAddr` without adding `value_parser` boilerplate.
#[derive(Clone, Debug)]
struct SocketBuf(SocketAddr);

impl std::str::FromStr for SocketBuf {
    type Err = std::net::AddrParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.parse().map(SocketBuf)
    }
}

struct AppState {
    model: Arc<LlamaCausalLm>,
    tokenizer: Arc<TokenizerHandle>,
}

#[derive(Deserialize)]
struct GenerateRequest {
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_new_tokens: usize,
}

fn default_max_tokens() -> usize {
    32
}

#[derive(Serialize)]
struct GenerateResponse {
    token_ids: Vec<usize>,
    text: String,
}

fn llama_error(e: rustral_llm::LlmError) -> (StatusCode, String) {
    (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
}

fn generate_blocking(
    state: &AppState,
    prompt: &str,
    max_new_tokens: usize,
) -> Result<GenerateResponse, rustral_llm::LlmError> {
    let prompt_ids_u32 = state.tokenizer.encode(prompt)?;
    let mut prompt_ids: Vec<usize> = prompt_ids_u32.iter().map(|&x| x as usize).collect();

    let mut ctx = ForwardCtx::new(state.model.backend(), Mode::Inference);
    let cfg = state.model.model().config();
    let mut cache = LlamaDecodeCache::new(cfg, 1, cfg.max_seq_len);

    if max_new_tokens == 0 {
        let out_u32: Vec<u32> = prompt_ids.iter().map(|&x| x as u32).collect();
        let text = state.tokenizer.decode(&out_u32)?;
        return Ok(GenerateResponse { token_ids: prompt_ids, text });
    }

    let first = state.model.greedy_first_token_after_prefill(&mut ctx, prompt_ids.clone(), &mut cache)?;
    prompt_ids.push(first);
    for _ in 1..max_new_tokens {
        let prev = *prompt_ids.last().expect("non-empty");
        let next = state.model.greedy_step_from_cache(&mut ctx, &mut cache, prev)?;
        prompt_ids.push(next);
    }

    let out_u32: Vec<u32> = prompt_ids.iter().map(|&x| x as u32).collect();
    let text = state.tokenizer.decode(&out_u32)?;
    Ok(GenerateResponse { token_ids: prompt_ids, text })
}

async fn generate(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, (StatusCode, String)> {
    let st = state.clone();
    let prompt = req.prompt.clone();
    let max = req.max_new_tokens;
    let res = tokio::task::spawn_blocking(move || generate_blocking(&st, &prompt, max))
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    res.map_err(llama_error).map(Json)
}

type SseStream = Sse<tokio_stream::wrappers::ReceiverStream<Result<Event, Infallible>>>;

async fn generate_stream(State(state): State<Arc<AppState>>, Json(req): Json<GenerateRequest>) -> Result<SseStream, (StatusCode, String)> {
    let st = state.clone();
    let prompt = req.prompt.clone();
    let max = req.max_new_tokens;

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(64);

    tokio::task::spawn_blocking(move || {
        let run = || -> Result<(), rustral_llm::LlmError> {
            let prompt_ids_u32 = st.tokenizer.encode(&prompt)?;
            let mut ids: Vec<usize> = prompt_ids_u32.iter().map(|&x| x as usize).collect();

            let mut ctx = ForwardCtx::new(st.model.backend(), Mode::Inference);
            let cfg = st.model.model().config();
            let mut cache = LlamaDecodeCache::new(cfg, 1, cfg.max_seq_len);

            if max == 0 {
                let _ = tx.blocking_send(Ok(Event::default().data(
                    serde_json::json!({ "done": true, "token_ids": ids }).to_string(),
                )));
                return Ok(());
            }

            let first = st.model.greedy_first_token_after_prefill(&mut ctx, ids.clone(), &mut cache)?;
            ids.push(first);
            let _ = tx.blocking_send(Ok(
                Event::default().data(serde_json::json!({ "token_id": first }).to_string()),
            ));

            for _ in 1..max {
                let prev = *ids.last().expect("non-empty");
                let next = st.model.greedy_step_from_cache(&mut ctx, &mut cache, prev)?;
                ids.push(next);
                let _ = tx.blocking_send(Ok(
                    Event::default().data(serde_json::json!({ "token_id": next }).to_string()),
                ));
            }

            let out_u32: Vec<u32> = ids.iter().map(|&x| x as u32).collect();
            let text = st.tokenizer.decode(&out_u32)?;
            let _ = tx.blocking_send(Ok(
                Event::default().data(serde_json::json!({ "done": true, "text": text }).to_string()),
            ));
            Ok(())
        };

        if let Err(e) = run() {
            let _ = tx.blocking_send(Ok(
                Event::default()
                    .event("error")
                    .data(serde_json::json!({ "message": e.to_string() }).to_string()),
            ));
        }
    });

    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

async fn health() -> &'static str {
    "ok"
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();

    let args = Args::parse();
    let snap = rustral_hf::scan_local_model_dir(&args.model_dir).map_err(|e| anyhow::anyhow!("{e}"))?;
    let cfg_path = snap
        .files
        .config_json
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("missing config.json under model dir"))?;
    let cfg = HfLlamaConfig::from_json_file(cfg_path).map_err(|e| anyhow::anyhow!("{e}"))?;

    if snap.files.safetensors_files.is_empty() {
        anyhow::bail!("no safetensors shards under model dir");
    }
    let meta = load_meta_state_dict_from_paths(&snap.files.safetensors_files).map_err(|e| anyhow::anyhow!("{e}"))?;
    let (model, rep) = LlamaCausalLm::from_hf_meta(&cfg, &meta, 0).map_err(|e| anyhow::anyhow!("{e}"))?;
    tracing::info!(loaded = rep.loaded_rustral_keys.len(), "Llama weights loaded");

    let tok_path = snap
        .files
        .tokenizer_json
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("missing tokenizer.json"))?;
    let tokenizer = TokenizerHandle::from_file(tok_path).map_err(|e| anyhow::anyhow!("{e}"))?;

    let state = Arc::new(AppState {
        model: Arc::new(model),
        tokenizer: Arc::new(tokenizer),
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/generate", post(generate))
        .route("/v1/generate/stream", post(generate_stream))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    tracing::info!("llama server listening on http://{}", args.bind.0);
    let listener = tokio::net::TcpListener::bind(args.bind.0).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
