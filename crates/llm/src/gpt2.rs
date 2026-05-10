use std::path::Path;

use anyhow::Context;
use rustral_core::{ForwardCtx, Mode};
use rustral_io::MetaStateDict;
use rustral_ndarray_backend::CpuBackend;
use rustral_nn::TransformerDecoderConfig;

use crate::LlmError;

pub mod hf_weights;

pub use hf_weights::{
    build_gpt2_flat_map, detect_gpt2_state_dict_prefix, load_hf_gpt2_weights_into_decoder, Gpt2WeightLoadReport,
};

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
    /// Borrow the underlying decoder (e.g. for tests or advanced loading).
    pub fn decoder(&self) -> &rustral_nn::TransformerDecoder<CpuBackend> {
        &self.model
    }

    /// Mutable reference to the underlying decoder.
    pub fn decoder_mut(&mut self) -> &mut rustral_nn::TransformerDecoder<CpuBackend> {
        &mut self.model
    }

    /// Load compatible Hugging Face GPT-2 tensors from a merged [`MetaStateDict`] (see `gpt2::hf_weights`).
    ///
    /// Attention weights are **not** mapped yet; they remain at initialization values.
    pub fn load_hf_weights_from_meta(&mut self, meta: &MetaStateDict, cfg: &HfGpt2Config) -> Result<Gpt2WeightLoadReport, LlmError> {
        hf_weights::load_hf_gpt2_weights_into_decoder(&mut self.model, &self.backend, meta, cfg)
    }

    /// Build a decoder and load compatible checkpoint tensors in one step.
    pub fn from_hf_meta(cfg: &HfGpt2Config, meta: &MetaStateDict, seed: u64) -> Result<(Self, Gpt2WeightLoadReport), LlmError> {
        let mut dec = Self::new_random(cfg, seed)?;
        let report = dec.load_hf_weights_from_meta(meta, cfg)?;
        Ok((dec, report))
    }

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

