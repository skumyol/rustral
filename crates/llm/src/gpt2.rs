use std::path::Path;
use std::time::Instant;

use anyhow::Context;
use rustral_core::{ForwardCtx, Mode};
use rustral_io::MetaStateDict;
use rustral_ndarray_backend::CpuBackend;
use rustral_nn::TransformerDecoderConfig;

use crate::causal_lm::CausalLm;
use crate::LlmError;

pub mod hf_weights;

pub use hf_weights::{
    build_gpt2_flat_map, detect_gpt2_state_dict_prefix, load_hf_gpt2_weights_into_decoder,
    Gpt2WeightLoadReport,
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
        let cfg: Self =
            serde_json::from_slice(&bytes).with_context(|| "parse gpt2 config json".to_string())?;
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

/// Wall-clock timings for greedy decoding (`decode_wall_ms` covers all `max_new_tokens` steps).
#[derive(Clone, Debug, serde::Serialize)]
pub struct GreedyDecodeTiming {
    /// Milliseconds for the **first** new token (single forward over the prompt).
    pub first_token_ms: f64,
    /// Milliseconds from the start of decoding until the last new token is appended.
    pub decode_wall_ms: f64,
}

impl Gpt2Decoder {
    /// Reference to the CPU backend used by this decoder (use this when building a [`ForwardCtx`] for [`crate::CausalLm`]).
    pub fn backend(&self) -> &CpuBackend {
        &self.backend
    }

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
    pub fn load_hf_weights_from_meta(
        &mut self,
        meta: &MetaStateDict,
        cfg: &HfGpt2Config,
    ) -> Result<Gpt2WeightLoadReport, LlmError> {
        hf_weights::load_hf_gpt2_weights_into_decoder(&mut self.model, &self.backend, meta, cfg)
    }

    /// Build a decoder and load compatible checkpoint tensors in one step.
    pub fn from_hf_meta(
        cfg: &HfGpt2Config,
        meta: &MetaStateDict,
        seed: u64,
    ) -> Result<(Self, Gpt2WeightLoadReport), LlmError> {
        let mut dec = Self::new_random(cfg, seed)?;
        let report = dec.load_hf_weights_from_meta(meta, cfg)?;
        Ok((dec, report))
    }

    pub fn new_random(cfg: &HfGpt2Config, seed: u64) -> Result<Self, LlmError> {
        let backend = CpuBackend::default();
        let mut dec_cfg = TransformerDecoderConfig::new(cfg.n_embd, cfg.n_head, cfg.n_layer, cfg.n_embd * 4)
            .with_max_seq_len(cfg.n_positions);
        if let Some(p) = cfg.resid_pdrop {
            dec_cfg.dropout = p;
        }
        let model = rustral_nn::TransformerDecoder::new(&backend, dec_cfg, cfg.vocab_size, seed)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(Self { backend, model })
    }

    fn greedy_steps(
        &self,
        ctx: &mut ForwardCtx<'_, CpuBackend>,
        mut input_ids: Vec<usize>,
        steps: usize,
    ) -> Result<Vec<usize>, LlmError> {
        for _ in 0..steps {
            let next = self
                .model
                .generate_token(input_ids.clone(), ctx)
                .map_err(|e| anyhow::anyhow!("{e}"))? as usize;
            input_ids.push(next);
        }
        Ok(input_ids)
    }

    /// Greedy generation with a fresh inference [`ForwardCtx`], plus timing suitable for CLI metrics.
    pub fn generate_greedy_timed(
        &self,
        mut input_ids: Vec<usize>,
        max_new_tokens: usize,
    ) -> Result<(Vec<usize>, GreedyDecodeTiming), LlmError> {
        let mut ctx = ForwardCtx::new(&self.backend, Mode::Inference);
        let decode_start = Instant::now();
        if max_new_tokens == 0 {
            return Ok((
                input_ids,
                GreedyDecodeTiming {
                    first_token_ms: 0.0,
                    decode_wall_ms: 0.0,
                },
            ));
        }

        let t_first = Instant::now();
        let next = self
            .model
            .generate_token(input_ids.clone(), &mut ctx)
            .map_err(|e| anyhow::anyhow!("{e}"))? as usize;
        let first_token_ms = t_first.elapsed().as_secs_f64() * 1000.0;
        input_ids.push(next);
        input_ids = self.greedy_steps(&mut ctx, input_ids, max_new_tokens.saturating_sub(1))?;
        let decode_wall_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
        Ok((
            input_ids,
            GreedyDecodeTiming {
                first_token_ms,
                decode_wall_ms,
            },
        ))
    }

    /// Greedy generation with a fresh inference [`ForwardCtx`].
    ///
    /// For profiling or custom [`rustral_core::ShapePolicy`], use [`CausalLm::generate_greedy`] with your own context.
    pub fn generate_greedy(
        &self,
        input_ids: Vec<usize>,
        max_new_tokens: usize,
    ) -> Result<Vec<usize>, LlmError> {
        let mut ctx = ForwardCtx::new(&self.backend, Mode::Inference);
        self.greedy_steps(&mut ctx, input_ids, max_new_tokens)
    }
}

impl CausalLm<CpuBackend> for Gpt2Decoder {
    fn generate_greedy(
        &self,
        ctx: &mut ForwardCtx<'_, CpuBackend>,
        input_ids: Vec<usize>,
        max_new_tokens: usize,
    ) -> Result<Vec<usize>, LlmError> {
        self.greedy_steps(ctx, input_ids, max_new_tokens)
    }
}

#[cfg(test)]
mod greedy_timing_tests {
    use super::*;

    #[test]
    fn greedy_timed_extends_by_max_new_tokens() {
        let cfg = HfGpt2Config {
            vocab_size: 64,
            n_positions: 32,
            n_embd: 16,
            n_layer: 1,
            n_head: 2,
            resid_pdrop: Some(0.0),
        };
        let m = Gpt2Decoder::new_random(&cfg, 42).expect("decoder");
        let (ids, t) = m.generate_greedy_timed(vec![0usize], 4).expect("timed");
        assert_eq!(ids.len(), 5);
        assert!(t.decode_wall_ms >= t.first_token_ms);
        assert!(t.first_token_ms >= 0.0);
    }
}
