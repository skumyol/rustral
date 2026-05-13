//! Attention mechanisms for neural networks.
//!
//! Provides SelfAttention, MultiHeadAttention, and TransformerEncoderBlock
//! implementations for modern NLP and vision architectures.

use rustral_core::{
    Backend, ForwardCtx, Module, NamedParameters, Parameter, ParameterRef, Result, Trainable,
};
use serde::{Deserialize, Serialize};

#[cfg(feature = "autodiff")]
use crate::tape::TapeModule;
#[cfg(feature = "autodiff")]
use rustral_autodiff::{Tape, TensorId};

/// Configuration for self-attention.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SelfAttentionConfig {
    /// Input/output dimension (d_model).
    pub d_model: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Head dimension (d_k = d_model / num_heads).
    pub head_dim: usize,
    /// Dropout probability for attention weights.
    pub dropout: f32,
}

impl SelfAttentionConfig {
    /// Create a new SelfAttention configuration.
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        assert_eq!(
            d_model % num_heads,
            0,
            "d_model ({}) must be divisible by num_heads ({})",
            d_model,
            num_heads
        );
        Self { d_model, num_heads, head_dim: d_model / num_heads, dropout: 0.0 }
    }

    /// Set dropout probability.
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }
}

/// Single-head self-attention.
pub struct SelfAttention<B: Backend> {
    config: SelfAttentionConfig,
    q_proj: Parameter<B>,
    k_proj: Parameter<B>,
    v_proj: Parameter<B>,
    out_proj: Parameter<B>,
    scale: f32,
}

impl<B: Backend> SelfAttention<B> {
    pub fn new(backend: &B, config: SelfAttentionConfig, seed: u64) -> Result<Self> {
        let d_k = config.head_dim;
        let q_proj = backend.normal_parameter("q_proj", &[config.d_model, d_k], seed, 0.02)?;
        let k_proj =
            backend.normal_parameter("k_proj", &[config.d_model, d_k], seed.wrapping_add(1), 0.02)?;
        let v_proj =
            backend.normal_parameter("v_proj", &[config.d_model, d_k], seed.wrapping_add(2), 0.02)?;
        let out_proj =
            backend.normal_parameter("out_proj", &[d_k, config.d_model], seed.wrapping_add(3), 0.02)?;
        let scale = 1.0 / (config.head_dim as f32).sqrt();
        Ok(Self { config, q_proj, k_proj, v_proj, out_proj, scale })
    }

    pub fn config(&self) -> &SelfAttentionConfig {
        &self.config
    }
}

impl<B: Backend> NamedParameters<B> for SelfAttention<B> {
    fn visit_parameters(&self, f: &mut dyn FnMut(&str, &Parameter<B>)) {
        f("q_proj", &self.q_proj);
        f("k_proj", &self.k_proj);
        f("v_proj", &self.v_proj);
        f("out_proj", &self.out_proj);
    }
    fn visit_parameters_mut(&mut self, f: &mut dyn FnMut(&str, &mut Parameter<B>)) {
        f("q_proj", &mut self.q_proj);
        f("k_proj", &mut self.k_proj);
        f("v_proj", &mut self.v_proj);
        f("out_proj", &mut self.out_proj);
    }
}

impl<B: Backend> Trainable<B> for SelfAttention<B> {
    fn parameters(&self) -> Vec<ParameterRef> {
        vec![
            ParameterRef { id: self.q_proj.id() },
            ParameterRef { id: self.k_proj.id() },
            ParameterRef { id: self.v_proj.id() },
            ParameterRef { id: self.out_proj.id() },
        ]
    }
}

impl<B: Backend> Module<B> for SelfAttention<B> {
    type Input = B::Tensor;
    type Output = B::Tensor;
    fn forward(&self, input: Self::Input, ctx: &mut ForwardCtx<B>) -> Result<Self::Output> {
        let ops = ctx.backend().ops();
        let input_shape = ops.shape(&input);
        let batch = input_shape[0];
        let seq_len = input_shape[1];
        let flat_input = ops.reshape(&input, &[batch * seq_len, self.config.d_model])?;
        let q = ops.matmul(&flat_input, self.q_proj.tensor())?;
        let k = ops.matmul(&flat_input, self.k_proj.tensor())?;
        let v = ops.matmul(&flat_input, self.v_proj.tensor())?;

        let q = ops.reshape(&q, &[batch * seq_len, self.config.head_dim])?;
        let k = ops.reshape(&k, &[batch * seq_len, self.config.head_dim])?;
        let v = ops.reshape(&v, &[batch * seq_len, self.config.head_dim])?;

        let kt = ops.transpose(&k)?;
        let scores = ops.matmul(&q, &kt)?;
        let scaled = ops.mul_scalar(&scores, self.scale)?;
        let weights = ops.softmax(&scaled)?;
        let attn_out = ops.matmul(&weights, &v)?;

        let output = ops.matmul(&attn_out, self.out_proj.tensor())?;
        ops.reshape(&output, &[batch, seq_len, self.config.d_model])
    }
}

/// Multi-head self-attention.
pub struct MultiHeadAttention<B: Backend> {
    config: SelfAttentionConfig,
    q_proj: Parameter<B>,
    k_proj: Parameter<B>,
    v_proj: Parameter<B>,
    out_proj: Parameter<B>,
    scale: f32,
}

impl<B: Backend> MultiHeadAttention<B> {
    pub fn new(backend: &B, config: SelfAttentionConfig, seed: u64) -> Result<Self> {
        let d_model = config.d_model;
        let q_proj = backend.normal_parameter("q_proj", &[d_model, d_model], seed, 0.02)?;
        let k_proj = backend.normal_parameter("k_proj", &[d_model, d_model], seed.wrapping_add(1), 0.02)?;
        let v_proj = backend.normal_parameter("v_proj", &[d_model, d_model], seed.wrapping_add(2), 0.02)?;
        let out_proj =
            backend.normal_parameter("out_proj", &[d_model, d_model], seed.wrapping_add(3), 0.02)?;
        let scale = 1.0 / (config.head_dim as f32).sqrt();
        Ok(Self { config, q_proj, k_proj, v_proj, out_proj, scale })
    }

    pub fn config(&self) -> &SelfAttentionConfig {
        &self.config
    }
}

impl<B: Backend> NamedParameters<B> for MultiHeadAttention<B> {
    fn visit_parameters(&self, f: &mut dyn FnMut(&str, &Parameter<B>)) {
        f("q_proj", &self.q_proj);
        f("k_proj", &self.k_proj);
        f("v_proj", &self.v_proj);
        f("out_proj", &self.out_proj);
    }
    fn visit_parameters_mut(&mut self, f: &mut dyn FnMut(&str, &mut Parameter<B>)) {
        f("q_proj", &mut self.q_proj);
        f("k_proj", &mut self.k_proj);
        f("v_proj", &mut self.v_proj);
        f("out_proj", &mut self.out_proj);
    }
}

impl<B: Backend> Trainable<B> for MultiHeadAttention<B> {
    fn parameters(&self) -> Vec<ParameterRef> {
        vec![
            ParameterRef { id: self.q_proj.id() },
            ParameterRef { id: self.k_proj.id() },
            ParameterRef { id: self.v_proj.id() },
            ParameterRef { id: self.out_proj.id() },
        ]
    }
}

impl<B: Backend> Module<B> for MultiHeadAttention<B> {
    type Input = B::Tensor;
    type Output = B::Tensor;

    fn forward(&self, input: Self::Input, ctx: &mut ForwardCtx<B>) -> Result<Self::Output> {
        let ops = ctx.backend().ops();
        let input_shape = ops.shape(&input);
        let batch = input_shape[0];
        let seq_len = input_shape[1];
        let h = self.config.num_heads;
        let d_k = self.config.head_dim;

        let flat_input = ops.reshape(&input, &[batch * seq_len, self.config.d_model])?;
        let q = ops.matmul(&flat_input, self.q_proj.tensor())?;
        let k = ops.matmul(&flat_input, self.k_proj.tensor())?;
        let v = ops.matmul(&flat_input, self.v_proj.tensor())?;

        let q = ops.reshape(&q, &[batch, seq_len, h, d_k])?;
        let k = ops.reshape(&k, &[batch, seq_len, h, d_k])?;
        let v = ops.reshape(&v, &[batch, seq_len, h, d_k])?;

        let q = ops.transpose_axes(&q, 1, 2)?;
        let k = ops.transpose_axes(&k, 1, 2)?;
        let v = ops.transpose_axes(&v, 1, 2)?;

        let q = ops.reshape(&q, &[batch * h, seq_len, d_k])?;
        let k = ops.reshape(&k, &[batch * h, seq_len, d_k])?;
        let v = ops.reshape(&v, &[batch * h, seq_len, d_k])?;

        let mut k_t_results = Vec::with_capacity(batch * h);
        for i in 0..(batch * h) {
            let ki = ops.slice(&k, i, i + 1)?;
            let ki_2d = ops.reshape(&ki, &[seq_len, d_k])?;
            let ki_t = ops.transpose(&ki_2d)?;
            k_t_results.push(ops.reshape(&ki_t, &[1, d_k, seq_len])?);
        }
        let k_t_refs: Vec<&B::Tensor> = k_t_results.iter().collect();
        let k_t = ops.concat(&k_t_refs, 0)?;

        let scores = ops.matmul_batched(&q, &k_t)?;
        let scaled = ops.mul_scalar(&scores, self.scale)?;
        let weights = ops.softmax_dim(&scaled, 2)?;
        let attn_out = ops.matmul_batched(&weights, &v)?;

        let attn_out = ops.reshape(&attn_out, &[batch, h, seq_len, d_k])?;
        let attn_out = ops.transpose_axes(&attn_out, 1, 2)?;
        let attn_out = ops.reshape(&attn_out, &[batch * seq_len, self.config.d_model])?;

        let output = ops.matmul(&attn_out, self.out_proj.tensor())?;
        ops.reshape(&output, &[batch, seq_len, self.config.d_model])
    }
}

/// Placeholder for Flash Attention.
pub struct FlashAttention<B: Backend> {
    inner: MultiHeadAttention<B>,
}
impl<B: Backend> FlashAttention<B> {
    pub fn new(backend: &B, config: SelfAttentionConfig, seed: u64) -> Result<Self> {
        Ok(Self { inner: MultiHeadAttention::new(backend, config, seed)? })
    }
}
impl<B: Backend> Module<B> for FlashAttention<B> {
    type Input = B::Tensor;
    type Output = B::Tensor;
    fn forward(&self, input: Self::Input, ctx: &mut ForwardCtx<B>) -> Result<Self::Output> {
        self.inner.forward(input, ctx)
    }
}

impl<B: Backend> Trainable<B> for FlashAttention<B> {
    fn parameters(&self) -> Vec<ParameterRef> {
        self.inner.parameters()
    }
}

impl<B: Backend> NamedParameters<B> for FlashAttention<B> {
    fn visit_parameters(&self, f: &mut dyn FnMut(&str, &Parameter<B>)) {
        self.inner.visit_parameters(f);
    }
    fn visit_parameters_mut(&mut self, f: &mut dyn FnMut(&str, &mut Parameter<B>)) {
        self.inner.visit_parameters_mut(f);
    }
}

pub fn causal_mask<B: Backend>(backend: &B, seq_len: usize) -> Result<B::Tensor> {
    let ops = backend.ops();
    let mut mask_values = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask_values[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }
    ops.tensor_from_vec(mask_values, &[seq_len, seq_len])
}

pub struct TransformerEncoderBlock<B: Backend> {
    self_attn: MultiHeadAttention<B>,
    norm1: crate::normalization::LayerNorm<B>,
    ff1: crate::Linear<B>,
    ff2: crate::Linear<B>,
    norm2: crate::normalization::LayerNorm<B>,
}

impl<B: Backend> TransformerEncoderBlock<B> {
    pub fn new(
        self_attn: MultiHeadAttention<B>,
        norm1: crate::normalization::LayerNorm<B>,
        ff1: crate::Linear<B>,
        ff2: crate::Linear<B>,
        norm2: crate::normalization::LayerNorm<B>,
    ) -> Self {
        Self { self_attn, norm1, ff1, ff2, norm2 }
    }
}

impl<B: Backend> Module<B> for TransformerEncoderBlock<B> {
    type Input = B::Tensor;
    type Output = B::Tensor;
    fn forward(&self, input: Self::Input, ctx: &mut ForwardCtx<B>) -> Result<Self::Output> {
        let ops = ctx.backend().ops();
        let normed = self.norm1.forward(input.clone(), ctx)?;
        let attn_out = self.self_attn.forward(normed, ctx)?;
        let x = ops.add(&input, &attn_out)?;
        let normed = self.norm2.forward(x.clone(), ctx)?;
        let ff_out = self.ff1.forward(normed, ctx)?;
        let ff_out = ops.gelu(&ff_out)?;
        let ff_out = self.ff2.forward(ff_out, ctx)?;
        ops.add(&x, &ff_out)
    }
}

impl<B: Backend> Trainable<B> for TransformerEncoderBlock<B> {
    fn parameters(&self) -> Vec<ParameterRef> {
        let mut params = self.self_attn.parameters();
        params.extend(self.norm1.parameters());
        params.extend(self.ff1.parameters());
        params.extend(self.ff2.parameters());
        params.extend(self.norm2.parameters());
        params
    }
}

impl<B: Backend> NamedParameters<B> for TransformerEncoderBlock<B> {
    fn visit_parameters(&self, f: &mut dyn FnMut(&str, &Parameter<B>)) {
        self.self_attn.visit_parameters(&mut |name, p| f(&format!("self_attn.{name}"), p));
        self.norm1.visit_parameters(&mut |name, p| f(&format!("norm1.{name}"), p));
        self.ff1.visit_parameters(&mut |name, p| f(&format!("ff1.{name}"), p));
        self.ff2.visit_parameters(&mut |name, p| f(&format!("ff2.{name}"), p));
        self.norm2.visit_parameters(&mut |name, p| f(&format!("norm2.{name}"), p));
    }
    fn visit_parameters_mut(&mut self, f: &mut dyn FnMut(&str, &mut Parameter<B>)) {
        self.self_attn.visit_parameters_mut(&mut |name, p| f(&format!("self_attn.{name}"), p));
        self.norm1.visit_parameters_mut(&mut |name, p| f(&format!("norm1.{name}"), p));
        self.ff1.visit_parameters_mut(&mut |name, p| f(&format!("ff1.{name}"), p));
        self.ff2.visit_parameters_mut(&mut |name, p| f(&format!("ff2.{name}"), p));
        self.norm2.visit_parameters_mut(&mut |name, p| f(&format!("norm2.{name}"), p));
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FlashAttentionConfig {
    pub attention: SelfAttentionConfig,
    pub block_size: usize,
}
impl FlashAttentionConfig {
    pub fn from_attention(attention: SelfAttentionConfig) -> Self {
        Self { attention, block_size: 128 }
    }
    pub fn with_block_size(mut self, size: usize) -> Self {
        self.block_size = size;
        self
    }
}
pub struct AttentionMemoryStats {
    pub seq_len: usize,
    pub batch_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub standard_memory_bytes: usize,
    pub flash_memory_bytes: usize,
    pub reduction_factor: f32,
}
impl AttentionMemoryStats {
    pub fn calculate(seq_len: usize, batch_size: usize, num_heads: usize, head_dim: usize) -> Self {
        let attn_matrix_size = seq_len * seq_len * batch_size * num_heads * 4;
        let flash_size = seq_len * batch_size * num_heads * head_dim * 4;
        let standard_total = attn_matrix_size + flash_size;
        let flash_total = flash_size * 2;
        Self {
            seq_len,
            batch_size,
            num_heads,
            head_dim,
            standard_memory_bytes: standard_total,
            flash_memory_bytes: flash_total,
            reduction_factor: standard_total as f32 / flash_total.max(1) as f32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustral_core::{ForwardCtx, Mode};
    use rustral_ndarray_backend::CpuBackend;
    #[test]
    fn test_multi_head_attention_forward() {
        let backend = CpuBackend::default();
        let config = SelfAttentionConfig::new(16, 4);
        let mha = MultiHeadAttention::new(&backend, config, 42).unwrap();
        let input = backend.tensor_from_vec(vec![0.1f32; 32], &[2, 1, 16]).unwrap();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let out = mha.forward(input, &mut ctx).unwrap();
        assert_eq!(backend.ops().shape(&out), &[2, 1, 16]);
    }
}
