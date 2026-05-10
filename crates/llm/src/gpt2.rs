use std::path::Path;

use anyhow::Context;
use rustral_core::{ForwardCtx, Mode};
use rustral_ndarray_backend::CpuBackend;
use rustral_nn::TransformerDecoderConfig;

use crate::LlmError;

/// Minimal subset of Hugging Face GPT-2 config fields we care about.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct HfGpt2Config {
    pub vocab_size: usize,
    pub n_positions: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    #[serde(default)]
    pub resid_pdrop: Option<f32>,
}

impl HfGpt2Config {
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, LlmError> {
        let bytes = std::fs::read(path.as_ref())
            .with_context(|| format!("read config json at {}", path.as_ref().display()))?;
        let cfg: Self = serde_json::from_slice(&bytes).with_context(|| "parse gpt2 config json".to_string())?;
        Ok(cfg)
    }
}

/// A small, config-driven GPT-style decoder backed by `rustral-nn::TransformerDecoder`.
///
/// This is a stepping stone:
/// - Now: build a model that matches HF config *shape* and can run greedy generation.
/// - Next: map HF weights into stable `NamedParameters` keys and load them.
pub struct Gpt2Decoder {
    backend: CpuBackend,
    model: rustral_nn::TransformerDecoder<CpuBackend>,
}

impl Gpt2Decoder {
    pub fn new_random(cfg: &HfGpt2Config, seed: u64) -> Result<Self, LlmError> {
        let backend = CpuBackend::default();
        let mut dec_cfg =
            TransformerDecoderConfig::new(cfg.n_embd, cfg.n_head, cfg.n_layer, cfg.n_embd * 4).with_max_seq_len(cfg.n_positions);
        if let Some(p) = cfg.resid_pdrop {
            dec_cfg.dropout = p;
        }
        let model = rustral_nn::TransformerDecoder::new(&backend, dec_cfg, cfg.vocab_size, seed)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(Self { backend, model })
    }

    pub fn generate_greedy(&self, mut input_ids: Vec<usize>, max_new_tokens: usize) -> Result<Vec<usize>, LlmError> {
        let mut ctx = ForwardCtx::new(&self.backend, Mode::Inference);
        for _ in 0..max_new_tokens {
            let next = self
                .model
                .generate_token(input_ids.clone(), &mut ctx)
                .map_err(|e| anyhow::anyhow!("{e}"))? as usize;
            input_ids.push(next);
        }
        Ok(input_ids)
    }
}

