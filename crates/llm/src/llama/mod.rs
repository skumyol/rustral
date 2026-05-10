//! Llama-family causal LM adapter (`rustral_nn::LlamaDecoder` + Hugging Face weight loading).

mod hf_weights;

pub use hf_weights::{
    build_llama_flat_map, detect_llama_state_dict_root, load_hf_llama_weights_into_decoder, LlamaWeightLoadReport,
};

use std::path::Path;
use std::time::Instant;

use anyhow::Context;
use rustral_core::{ForwardCtx, Mode};
use rustral_io::MetaStateDict;
use rustral_ndarray_backend::CpuBackend;
use rustral_nn::LlamaDecoderConfig;

use crate::causal_lm::CausalLm;
use crate::LlmError;

/// Minimal Hugging Face `config.json` subset for Llama / Llama-2-class models.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct HfLlamaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,
}

fn default_rms_eps() -> f32 {
    1e-5
}

fn default_rope_theta() -> f32 {
    10_000.0
}

impl HfLlamaConfig {
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, LlmError> {
        let bytes = std::fs::read(path.as_ref())
            .with_context(|| format!("read Llama config json at {}", path.as_ref().display()))?;
        let cfg: Self = serde_json::from_slice(&bytes).with_context(|| "parse Llama config json".to_string())?;
        Ok(cfg)
    }

    /// Return an error when the checkpoint uses GQA/MQA (not implemented in [`LlamaDecoder`] yet).
    pub fn validate_supported(&self) -> Result<(), LlmError> {
        if let Some(kv) = self.num_key_value_heads {
            if kv != self.num_attention_heads {
                return Err(LlmError::InvalidArg(format!(
                    "num_key_value_heads ({kv}) != num_attention_heads ({}) — GQA/MQA Llama not supported in this decoder yet",
                    self.num_attention_heads
                )));
            }
        }
        Ok(())
    }

    pub fn to_decoder_config(&self) -> LlamaDecoderConfig {
        let mut c = LlamaDecoderConfig::new(
            self.hidden_size,
            self.num_attention_heads,
            self.num_hidden_layers,
            self.intermediate_size,
        )
        .with_rms_eps(self.rms_norm_eps)
        .with_rope_theta(self.rope_theta);
        if let Some(m) = self.max_position_embeddings {
            c = c.with_max_seq_len(m);
        }
        c
    }
}

/// CPU Llama-shaped causal LM with optional HF weight load.
pub struct LlamaCausalLm {
    backend: CpuBackend,
    model: rustral_nn::LlamaDecoder<CpuBackend>,
}

impl LlamaCausalLm {
    pub fn backend(&self) -> &CpuBackend {
        &self.backend
    }

    pub fn model(&self) -> &rustral_nn::LlamaDecoder<CpuBackend> {
        &self.model
    }

    pub fn model_mut(&mut self) -> &mut rustral_nn::LlamaDecoder<CpuBackend> {
        &mut self.model
    }

    pub fn new_random(cfg: &HfLlamaConfig, seed: u64) -> Result<Self, LlmError> {
        cfg.validate_supported()?;
        let backend = CpuBackend::default();
        let dec_cfg = cfg.to_decoder_config();
        let model = rustral_nn::LlamaDecoder::new(&backend, dec_cfg, cfg.vocab_size, seed)
            .map_err(|e| LlmError::InvalidArg(format!("LlamaDecoder::new: {e:?}")))?;
        Ok(Self { backend, model })
    }

    pub fn load_hf_weights_from_meta(
        &mut self,
        meta: &MetaStateDict,
        cfg: &HfLlamaConfig,
    ) -> Result<LlamaWeightLoadReport, LlmError> {
        hf_weights::load_hf_llama_weights_into_decoder(&mut self.model, &self.backend, meta, cfg)
    }

    pub fn from_hf_meta(cfg: &HfLlamaConfig, meta: &MetaStateDict, seed: u64) -> Result<(Self, LlamaWeightLoadReport), LlmError> {
        let mut m = Self::new_random(cfg, seed)?;
        let report = m.load_hf_weights_from_meta(meta, cfg)?;
        Ok((m, report))
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

    pub fn generate_greedy_timed(
        &self,
        mut input_ids: Vec<usize>,
        max_new_tokens: usize,
    ) -> Result<(Vec<usize>, crate::gpt2::GreedyDecodeTiming), LlmError> {
        let mut ctx = ForwardCtx::new(&self.backend, Mode::Inference);
        let decode_start = Instant::now();
        if max_new_tokens == 0 {
            return Ok((
                input_ids,
                crate::gpt2::GreedyDecodeTiming {
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
            crate::gpt2::GreedyDecodeTiming {
                first_token_ms,
                decode_wall_ms,
            },
        ))
    }

    pub fn generate_greedy(&self, input_ids: Vec<usize>, max_new_tokens: usize) -> Result<Vec<usize>, LlmError> {
        let mut ctx = ForwardCtx::new(&self.backend, Mode::Inference);
        self.greedy_steps(&mut ctx, input_ids, max_new_tokens)
    }
}

impl CausalLm<CpuBackend> for LlamaCausalLm {
    fn generate_greedy(
        &self,
        ctx: &mut ForwardCtx<'_, CpuBackend>,
        input_ids: Vec<usize>,
        max_new_tokens: usize,
    ) -> Result<Vec<usize>, LlmError> {
        self.greedy_steps(ctx, input_ids, max_new_tokens)
    }
}
