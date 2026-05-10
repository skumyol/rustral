//! LLaMA-style causal decoder (reference CPU implementation).
//!
//! Uses **RMSNorm**, **RoPE**, **SwiGLU** MLP, and [`crate::attention::sdpa_multi_head_f32`] for attention.
//! Intended for correctness alignment with Hugging Face Llama checkpoints when paired with a weight
//! loader (future); eager paths use plain **f32** host loops, not fused GPU kernels.

use rustral_core::{
    Backend, ForwardCtx, Module, NamedParameters, Parameter, ParameterRef, Result, Trainable,
};
use serde::{Deserialize, Serialize};

use crate::attention::{idx_bshd_fixed, sdpa_multi_head_f32};
use crate::{
    Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig,
};

/// Decoder hyperparameters (Llama-family shaped).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LlamaDecoderConfig {
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_seq_len: usize,
}

impl LlamaDecoderConfig {
    pub fn new(d_model: usize, num_heads: usize, num_layers: usize, intermediate_size: usize) -> Self {
        Self {
            d_model,
            num_heads,
            num_layers,
            intermediate_size,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            max_seq_len: 2048,
        }
    }

    pub fn with_rope_theta(mut self, theta: f32) -> Self {
        self.rope_theta = theta;
        self
    }

    pub fn with_max_seq_len(mut self, max_seq_len: usize) -> Self {
        self.max_seq_len = max_seq_len;
        self
    }

    pub fn with_rms_eps(mut self, eps: f32) -> Self {
        self.rms_norm_eps = eps;
        self
    }

    fn head_dim(&self) -> usize {
        self.d_model / self.num_heads
    }
}

/// Precompute cos/sin rows for RoPE (HF-style duplicated freqs).
fn rope_cos_sin(seq_len: usize, head_dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut inv_freq = vec![0f32; half];
    for i in 0..half {
        inv_freq[i] = 1.0 / theta.powf((2 * i) as f32 / head_dim as f32);
    }

    let mut cos = vec![0f32; seq_len * head_dim];
    let mut sin = vec![0f32; seq_len * head_dim];

    for s in 0..seq_len {
        let pos = s as f32;
        for j in 0..half {
            let angle = pos * inv_freq[j];
            let c = angle.cos();
            let si = angle.sin();
            cos[s * head_dim + 2 * j] = c;
            cos[s * head_dim + 2 * j + 1] = c;
            sin[s * head_dim + 2 * j] = si;
            sin[s * head_dim + 2 * j + 1] = si;
        }
    }
    (cos, sin)
}

fn rotate_half_inplace(scratch: &mut [f32], x: &[f32]) {
    let d = x.len();
    let half = d / 2;
    for i in 0..half {
        scratch[i] = -x[half + i];
        scratch[half + i] = x[i];
    }
}

fn apply_rope_head(vec: &mut [f32], cos: &[f32], sin: &[f32]) {
    let d = vec.len();
    let mut rh = vec![0f32; d];
    rotate_half_inplace(&mut rh, vec);
    let mut out = vec![0f32; d];
    for i in 0..d {
        out[i] = vec[i] * cos[i] + rh[i] * sin[i];
    }
    vec.copy_from_slice(&out);
}

fn apply_rope_qk(
    q: &mut [f32],
    k: &mut [f32],
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    cos: &[f32],
    sin: &[f32],
) {
    for b in 0..batch {
        for s in 0..seq_len {
            let c_off = s * head_dim;
            let cos_s = &cos[c_off..c_off + head_dim];
            let sin_s = &sin[c_off..c_off + head_dim];
            for h in 0..num_heads {
                let base = idx_bshd_fixed(b, s, h, 0, seq_len, num_heads, head_dim);
                apply_rope_head(&mut q[base..base + head_dim], cos_s, sin_s);
                apply_rope_head(&mut k[base..base + head_dim], cos_s, sin_s);
            }
        }
    }
}

/// Multi-head self-attention with RoPE (Llama naming: `o_proj`).
pub struct LlamaAttention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    o_proj: Linear<B>,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
    rope_theta: f32,
}

impl<B: Backend> LlamaAttention<B>
where
    B::Tensor: Clone,
{
    pub fn new(backend: &B, cfg: &LlamaDecoderConfig, seed: u64) -> Result<Self> {
        let d = cfg.d_model;
        let hd = cfg.head_dim();
        assert_eq!(d % cfg.num_heads, 0);
        let q_proj = Linear::new(backend, LinearConfig::new(d, d).with_bias(false))?;
        let k_proj = Linear::new(backend, LinearConfig::new(d, d).with_bias(false))?;
        let v_proj = Linear::new(backend, LinearConfig::new(d, d).with_bias(false))?;
        let o_proj = Linear::new(backend, LinearConfig::new(d, d).with_bias(false))?;
        let scale = 1.0 / (hd as f32).sqrt();
        let _ = seed;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: cfg.num_heads,
            head_dim: hd,
            scale,
            rope_theta: cfg.rope_theta,
        })
    }

    pub fn forward(&self, input: B::Tensor, ctx: &mut ForwardCtx<B>) -> Result<B::Tensor> {
        let ops = ctx.backend().ops();
        let shape = ops.shape(&input);
        if shape.len() != 3 {
            return Err(rustral_core::CoreError::InvalidShape {
                shape,
                reason: "LlamaAttention expects [batch, seq, d_model]".into(),
            });
        }
        let batch = shape[0];
        let seq_len = shape[1];
        let d_model = shape[2];
        if d_model != self.q_proj.config().in_dim {
            return Err(rustral_core::CoreError::ShapeMismatch {
                expected: vec![self.q_proj.config().in_dim],
                actual: vec![d_model],
            });
        }

        let flat = ops.reshape(&input, &[batch * seq_len, d_model])?;
        let q_f = self.q_proj.forward(flat.clone(), ctx)?;
        let k_f = self.k_proj.forward(flat.clone(), ctx)?;
        let v_f = self.v_proj.forward(flat, ctx)?;

        let mut q_vec = ops.tensor_to_vec(&q_f)?;
        let mut k_vec = ops.tensor_to_vec(&k_f)?;
        let v_vec = ops.tensor_to_vec(&v_f)?;

        let (cos, sin) = rope_cos_sin(seq_len, self.head_dim, self.rope_theta);
        apply_rope_qk(
            &mut q_vec,
            &mut k_vec,
            batch,
            seq_len,
            self.num_heads,
            self.head_dim,
            &cos,
            &sin,
        );

        let merged = sdpa_multi_head_f32(
            &q_vec,
            &k_vec,
            &v_vec,
            batch,
            seq_len,
            self.num_heads,
            self.head_dim,
            self.scale,
            true,
        );

        let attn_flat = ops.tensor_from_vec(merged, &[batch * seq_len, d_model])?;
        let out = self.o_proj.forward(attn_flat, ctx)?;
        ops.reshape(&out, &[batch, seq_len, d_model])
    }
}

impl<B: Backend> NamedParameters<B> for LlamaAttention<B> {
    fn visit_parameters(&self, f: &mut dyn FnMut(&str, &Parameter<B>)) {
        self.q_proj.visit_parameters(&mut |n, p| f(&format!("q_proj.{n}"), p));
        self.k_proj.visit_parameters(&mut |n, p| f(&format!("k_proj.{n}"), p));
        self.v_proj.visit_parameters(&mut |n, p| f(&format!("v_proj.{n}"), p));
        self.o_proj.visit_parameters(&mut |n, p| f(&format!("o_proj.{n}"), p));
    }

    fn visit_parameters_mut(&mut self, f: &mut dyn FnMut(&str, &mut Parameter<B>)) {
        self.q_proj.visit_parameters_mut(&mut |n, p| f(&format!("q_proj.{n}"), p));
        self.k_proj.visit_parameters_mut(&mut |n, p| f(&format!("k_proj.{n}"), p));
        self.v_proj.visit_parameters_mut(&mut |n, p| f(&format!("v_proj.{n}"), p));
        self.o_proj.visit_parameters_mut(&mut |n, p| f(&format!("o_proj.{n}"), p));
    }
}

impl<B: Backend> Trainable<B> for LlamaAttention<B> {
    fn parameters(&self) -> Vec<ParameterRef> {
        let mut v = self.q_proj.parameters();
        v.extend(self.k_proj.parameters());
        v.extend(self.v_proj.parameters());
        v.extend(self.o_proj.parameters());
        v
    }
}

/// SwiGLU feed-forward (`down(silu(gate(x)) ⊙ up(x))`).
pub struct LlamaMlp<B: Backend> {
    gate_proj: Linear<B>,
    up_proj: Linear<B>,
    down_proj: Linear<B>,
}

impl<B: Backend> LlamaMlp<B>
where
    B::Tensor: Clone,
{
    pub fn new(backend: &B, cfg: &LlamaDecoderConfig, seed: u64) -> Result<Self> {
        let _ = seed;
        let d = cfg.d_model;
        let inter = cfg.intermediate_size;
        let gate_proj = Linear::new(backend, LinearConfig::new(d, inter).with_bias(false))?;
        let up_proj = Linear::new(backend, LinearConfig::new(d, inter).with_bias(false))?;
        let down_proj = Linear::new(backend, LinearConfig::new(inter, d).with_bias(false))?;
        Ok(Self { gate_proj, up_proj, down_proj })
    }

    pub fn forward(&self, input: B::Tensor, ctx: &mut ForwardCtx<B>) -> Result<B::Tensor> {
        let ops = ctx.backend().ops();
        let shape = ops.shape(&input);
        if shape.len() != 3 {
            return Err(rustral_core::CoreError::InvalidShape {
                shape,
                reason: "LlamaMlp expects [batch, seq, d_model]".into(),
            });
        }
        let batch = shape[0];
        let seq_len = shape[1];
        let d_model = shape[2];
        let flat = ops.reshape(&input, &[batch * seq_len, d_model])?;
        let gate = self.gate_proj.forward(flat.clone(), ctx)?;
        let up = self.up_proj.forward(flat, ctx)?;
        let gate_sigmoid = ops.sigmoid(&gate)?;
        let gate_silu = ops.mul(&gate, &gate_sigmoid)?;
        let activated = ops.mul(&gate_silu, &up)?;
        let out = self.down_proj.forward(activated, ctx)?;
        ops.reshape(&out, &[batch, seq_len, d_model])
    }
}

impl<B: Backend> NamedParameters<B> for LlamaMlp<B> {
    fn visit_parameters(&self, f: &mut dyn FnMut(&str, &Parameter<B>)) {
        self.gate_proj.visit_parameters(&mut |n, p| f(&format!("gate_proj.{n}"), p));
        self.up_proj.visit_parameters(&mut |n, p| f(&format!("up_proj.{n}"), p));
        self.down_proj.visit_parameters(&mut |n, p| f(&format!("down_proj.{n}"), p));
    }

    fn visit_parameters_mut(&mut self, f: &mut dyn FnMut(&str, &mut Parameter<B>)) {
        self.gate_proj.visit_parameters_mut(&mut |n, p| f(&format!("gate_proj.{n}"), p));
        self.up_proj.visit_parameters_mut(&mut |n, p| f(&format!("up_proj.{n}"), p));
        self.down_proj.visit_parameters_mut(&mut |n, p| f(&format!("down_proj.{n}"), p));
    }
}

impl<B: Backend> Trainable<B> for LlamaMlp<B> {
    fn parameters(&self) -> Vec<ParameterRef> {
        let mut v = self.gate_proj.parameters();
        v.extend(self.up_proj.parameters());
        v.extend(self.down_proj.parameters());
        v
    }
}

/// One LLaMA decoder block (pre-RMSNorm layout).
pub struct LlamaDecoderLayer<B: Backend> {
    input_layernorm: RmsNorm<B>,
    self_attn: LlamaAttention<B>,
    post_attention_layernorm: RmsNorm<B>,
    mlp: LlamaMlp<B>,
}

impl<B: Backend> LlamaDecoderLayer<B>
where
    B::Tensor: Clone,
{
    pub fn new(backend: &B, cfg: &LlamaDecoderConfig, seed: u64) -> Result<Self> {
        let input_layernorm =
            RmsNorm::new(backend, RmsNormConfig::new(cfg.d_model).with_eps(cfg.rms_norm_eps), seed)?;
        let self_attn = LlamaAttention::new(backend, cfg, seed.wrapping_add(1))?;
        let post_attention_layernorm = RmsNorm::new(
            backend,
            RmsNormConfig::new(cfg.d_model).with_eps(cfg.rms_norm_eps),
            seed.wrapping_add(2),
        )?;
        let mlp = LlamaMlp::new(backend, cfg, seed.wrapping_add(3))?;
        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        })
    }

    pub fn forward(&self, input: B::Tensor, ctx: &mut ForwardCtx<B>) -> Result<B::Tensor> {
        let ops = ctx.backend().ops();
        let normed = self.input_layernorm.forward(input.clone(), ctx)?;
        let attn = self.self_attn.forward(normed, ctx)?;
        let hidden = ops.add(&input, &attn)?;
        let normed_mlp = self.post_attention_layernorm.forward(hidden.clone(), ctx)?;
        let mlp_out = self.mlp.forward(normed_mlp, ctx)?;
        ops.add(&hidden, &mlp_out)
    }
}

impl<B: Backend> NamedParameters<B> for LlamaDecoderLayer<B> {
    fn visit_parameters(&self, f: &mut dyn FnMut(&str, &Parameter<B>)) {
        self.input_layernorm.visit_parameters(&mut |n, p| f(&format!("input_layernorm.{n}"), p));
        self.self_attn.visit_parameters(&mut |n, p| f(&format!("self_attn.{n}"), p));
        self.post_attention_layernorm
            .visit_parameters(&mut |n, p| f(&format!("post_attention_layernorm.{n}"), p));
        self.mlp.visit_parameters(&mut |n, p| f(&format!("mlp.{n}"), p));
    }

    fn visit_parameters_mut(&mut self, f: &mut dyn FnMut(&str, &mut Parameter<B>)) {
        self.input_layernorm.visit_parameters_mut(&mut |n, p| f(&format!("input_layernorm.{n}"), p));
        self.self_attn.visit_parameters_mut(&mut |n, p| f(&format!("self_attn.{n}"), p));
        self.post_attention_layernorm
            .visit_parameters_mut(&mut |n, p| f(&format!("post_attention_layernorm.{n}"), p));
        self.mlp.visit_parameters_mut(&mut |n, p| f(&format!("mlp.{n}"), p));
    }
}

impl<B: Backend> Trainable<B> for LlamaDecoderLayer<B> {
    fn parameters(&self) -> Vec<ParameterRef> {
        let mut p = self.input_layernorm.parameters();
        p.extend(self.self_attn.parameters());
        p.extend(self.post_attention_layernorm.parameters());
        p.extend(self.mlp.parameters());
        p
    }
}

/// Causal LM backbone: token embedding → N × [`LlamaDecoderLayer`] → RMSNorm → `lm_head`.
pub struct LlamaDecoder<B: Backend> {
    token_embedding: Embedding<B>,
    layers: Vec<LlamaDecoderLayer<B>>,
    norm: RmsNorm<B>,
    lm_head: Linear<B>,
    config: LlamaDecoderConfig,
    vocab_size: usize,
}

impl<B: Backend> LlamaDecoder<B>
where
    B::Tensor: Clone,
{
    pub fn new(backend: &B, config: LlamaDecoderConfig, vocab_size: usize, seed: u64) -> Result<Self> {
        assert_eq!(
            config.d_model % config.num_heads,
            0,
            "d_model ({}) must be divisible by num_heads ({})",
            config.d_model,
            config.num_heads
        );
        let token_embedding =
            Embedding::new(backend, EmbeddingConfig::new(vocab_size, config.d_model), seed)?;
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            layers.push(LlamaDecoderLayer::new(backend, &config, seed.wrapping_add((i as u64).wrapping_mul(16)))?);
        }
        let norm =
            RmsNorm::new(backend, RmsNormConfig::new(config.d_model).with_eps(config.rms_norm_eps), seed.wrapping_add(9000))?;
        let lm_head =
            Linear::new(backend, LinearConfig::new(config.d_model, vocab_size).with_bias(false))?;
        Ok(Self {
            token_embedding,
            layers,
            norm,
            lm_head,
            config,
            vocab_size,
        })
    }

    pub fn config(&self) -> &LlamaDecoderConfig {
        &self.config
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Token IDs → logits `[batch, seq_len, vocab_size]` (batch 1 when `len <= max_seq_len`).
    pub fn forward(&self, input: Vec<usize>, ctx: &mut ForwardCtx<B>) -> Result<B::Tensor> {
        let ops = ctx.backend().ops();
        let embedded = self.token_embedding.forward(input, ctx)?;
        let embed_shape = ops.shape(&embedded);
        let total_tokens = embed_shape[0];
        let d_model = embed_shape[1];

        let hidden = if total_tokens <= self.config.max_seq_len {
            ops.reshape(&embedded, &[1, total_tokens, d_model])?
        } else {
            let batch_size = total_tokens / self.config.max_seq_len;
            let seq_len = self.config.max_seq_len;
            ops.reshape(&embedded, &[batch_size, seq_len, d_model])?
        };

        let mut hidden = hidden;
        for layer in &self.layers {
            hidden = layer.forward(hidden, ctx)?;
        }
        hidden = self.norm.forward(hidden, ctx)?;

        let hidden_shape = ops.shape(&hidden);
        let batch = hidden_shape[0];
        let seq_len = hidden_shape[1];
        let flat = ops.reshape(&hidden, &[batch * seq_len, d_model])?;
        let logits_flat = self.lm_head.forward(flat, ctx)?;
        ops.reshape(&logits_flat, &[batch, seq_len, self.vocab_size])
    }

    /// Greedy next-token step: run forward on `prefix` and take argmax of the last position logits.
    pub fn generate_token(&self, prefix: Vec<usize>, ctx: &mut ForwardCtx<B>) -> Result<u32> {
        let logits = self.forward(prefix, ctx)?;
        let ops = ctx.backend().ops();
        let shape = ops.shape(&logits);
        let vocab_size = shape[2];

        let flat = ops.tensor_to_vec(&logits)?;
        let offset = (shape[0] - 1) * shape[1] * vocab_size;
        let last_logits = &flat[offset..offset + vocab_size];

        let (idx, _) = last_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap_or((0, &0.0));

        Ok(idx as u32)
    }
}

impl<B: Backend> Module<B> for LlamaDecoder<B>
where
    B::Tensor: Clone,
{
    type Input = Vec<usize>;
    type Output = B::Tensor;

    fn forward(&self, input: Self::Input, ctx: &mut ForwardCtx<B>) -> Result<Self::Output> {
        LlamaDecoder::forward(self, input, ctx)
    }
}

impl<B: Backend> Trainable<B> for LlamaDecoder<B> {
    fn parameters(&self) -> Vec<ParameterRef> {
        let mut p = self.token_embedding.parameters();
        for layer in &self.layers {
            p.extend(layer.parameters());
        }
        p.extend(self.norm.parameters());
        p.extend(self.lm_head.parameters());
        p
    }
}

impl<B: Backend> NamedParameters<B> for LlamaDecoder<B> {
    fn visit_parameters(&self, f: &mut dyn FnMut(&str, &Parameter<B>)) {
        self.token_embedding.visit_parameters(&mut |n, p| {
            let full = format!("token_embedding.{n}");
            f(&full, p);
        });
        for (i, layer) in self.layers.iter().enumerate() {
            layer.visit_parameters(&mut |n, p| {
                let full = format!("layers.{i}.{n}");
                f(&full, p);
            });
        }
        self.norm.visit_parameters(&mut |n, p| {
            let full = format!("norm.{n}");
            f(&full, p);
        });
        self.lm_head.visit_parameters(&mut |n, p| {
            let full = format!("lm_head.{n}");
            f(&full, p);
        });
    }

    fn visit_parameters_mut(&mut self, f: &mut dyn FnMut(&str, &mut Parameter<B>)) {
        self.token_embedding.visit_parameters_mut(&mut |n, p| {
            let full = format!("token_embedding.{n}");
            f(&full, p);
        });
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.visit_parameters_mut(&mut |n, p| {
                let full = format!("layers.{i}.{n}");
                f(&full, p);
            });
        }
        self.norm.visit_parameters_mut(&mut |n, p| {
            let full = format!("norm.{n}");
            f(&full, p);
        });
        self.lm_head.visit_parameters_mut(&mut |n, p| {
            let full = format!("lm_head.{n}");
            f(&full, p);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustral_core::Mode;
    use rustral_ndarray_backend::CpuBackend;

    #[test]
    fn llama_decoder_forward_smoke() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let cfg = LlamaDecoderConfig::new(32, 4, 2, 128).with_max_seq_len(64);
        let vocab = 50usize;
        let model = LlamaDecoder::new(&backend, cfg, vocab, 42).expect("model");
        let toks = vec![1usize, 2, 3];
        let logits = model.forward(toks, &mut ctx).expect("forward");
        let sh = backend.ops().shape(&logits);
        assert_eq!(sh, &[1, 3, vocab]);
    }

    #[test]
    fn rope_cos_sin_has_expected_length() {
        let (c, s) = rope_cos_sin(8, 16, 10_000.0);
        assert_eq!(c.len(), 8 * 16);
        assert_eq!(s.len(), 8 * 16);
    }

    /// Causal LM invariant: logits at position `s` depend only on tokens `0..=s`; incremental prefix forwards must match one-shot forward.
    #[test]
    fn causal_prefix_last_logits_match_full_forward() {
        let backend = CpuBackend::default();
        let cfg = LlamaDecoderConfig::new(32, 4, 1, 48).with_max_seq_len(128);
        let vocab = 60usize;
        let model = LlamaDecoder::new(&backend, cfg, vocab, 99_877).expect("model");
        let tokens: Vec<usize> = vec![5, 12, 3, 40, 8, 1];
        let ops = backend.ops();

        let mut ctx_full = ForwardCtx::new(&backend, Mode::Inference);
        let full_logits = model.forward(tokens.clone(), &mut ctx_full).expect("full");
        let shape = ops.shape(&full_logits);
        assert_eq!(shape, &[1, tokens.len(), vocab]);
        let full_flat = ops.tensor_to_vec(&full_logits).expect("vec");
        let sl = vocab;

        for s in 0..tokens.len() {
            let prefix = tokens[..=s].to_vec();
            let mut ctx_p = ForwardCtx::new(&backend, Mode::Inference);
            let step_logits = model.forward(prefix, &mut ctx_p).expect("prefix");
            let step_flat = ops.tensor_to_vec(&step_logits).expect("step vec");
            let seq_len = ops.shape(&step_logits)[1];
            let offset_last = (seq_len - 1) * sl;
            for i in 0..sl {
                let a = step_flat[offset_last + i];
                let b = full_flat[s * sl + i];
                assert!(
                    (a - b).abs() < 1e-4,
                    "mismatch at seq pos {s} vocab dim {i}: {a} vs {b}"
                );
            }
        }
    }
}
