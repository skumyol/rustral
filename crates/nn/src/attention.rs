//! Attention mechanisms for neural networks.

use rustral_core::{
    Backend, ForwardCtx, Module, NamedParameters, Parameter, ParameterRef, Result, Trainable,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SelfAttentionConfig {
    pub d_model: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub dropout: f32,
}

impl SelfAttentionConfig {
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        assert_eq!(d_model % num_heads, 0);
        Self { d_model, num_heads, head_dim: d_model / num_heads, dropout: 0.0 }
    }
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }
}

/// Simplified single-head implementation for internal use.
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
        let k_proj = backend.normal_parameter("k_proj", &[config.d_model, d_k], seed + 1, 0.02)?;
        let v_proj = backend.normal_parameter("v_proj", &[config.d_model, d_k], seed + 2, 0.02)?;
        let out_proj = backend.normal_parameter("out_proj", &[d_k, config.d_model], seed + 3, 0.02)?;
        let scale = 1.0 / (config.head_dim as f32).sqrt();
        Ok(Self { config, q_proj, k_proj, v_proj, out_proj, scale })
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
        let shape = ops.shape(&input);
        let (batch, seq_len) = (shape[0], shape[1]);
        let flat = ops.reshape(&input, &[batch * seq_len, self.config.d_model])?;
        let q = ops.matmul(&flat, self.q_proj.tensor())?;
        let k = ops.matmul(&flat, self.k_proj.tensor())?;
        let v = ops.matmul(&flat, self.v_proj.tensor())?;
        let scores = ops.matmul(&q, &ops.transpose(&k)?)?;
        let attn = ops.softmax(&ops.mul_scalar(&scores, self.scale)?)?;
        let out = ops.matmul(&attn, &v)?;
        let projected = ops.matmul(&out, self.out_proj.tensor())?;
        ops.reshape(&projected, &[batch, seq_len, self.config.d_model])
    }
}

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
        let k_proj = backend.normal_parameter("k_proj", &[d_model, d_model], seed + 1, 0.02)?;
        let v_proj = backend.normal_parameter("v_proj", &[d_model, d_model], seed + 2, 0.02)?;
        let out_proj = backend.normal_parameter("out_proj", &[d_model, d_model], seed + 3, 0.02)?;
        let scale = 1.0 / (config.head_dim as f32).sqrt();
        Ok(Self { config, q_proj, k_proj, v_proj, out_proj, scale })
    }
    pub fn config(&self) -> &SelfAttentionConfig {
        &self.config
    }
}

impl<B: Backend> Module<B> for MultiHeadAttention<B> {
    type Input = B::Tensor;
    type Output = B::Tensor;
    fn forward(&self, input: Self::Input, ctx: &mut ForwardCtx<B>) -> Result<Self::Output> {
        let ops = ctx.backend().ops();
        let shape = ops.shape(&input);
        let (batch, seq_len, d_model) = (shape[0], shape[1], shape[2]);
        let (h, d_k) = (self.config.num_heads, self.config.head_dim);
        let flat = ops.reshape(&input, &[batch * seq_len, d_model])?;

        let q = ops.reshape(&ops.matmul(&flat, self.q_proj.tensor())?, &[batch, seq_len, h, d_k])?;
        let k = ops.reshape(&ops.matmul(&flat, self.k_proj.tensor())?, &[batch, seq_len, h, d_k])?;
        let v = ops.reshape(&ops.matmul(&flat, self.v_proj.tensor())?, &[batch, seq_len, h, d_k])?;

        // [B, S, H, D] -> [B, H, S, D] -> [B*H, S, D]
        let q = ops.reshape(&ops.transpose_axes(&q, 1, 2)?, &[batch * h, seq_len, d_k])?;
        let k = ops.reshape(&ops.transpose_axes(&k, 1, 2)?, &[batch * h, seq_len, d_k])?;
        let v = ops.reshape(&ops.transpose_axes(&v, 1, 2)?, &[batch * h, seq_len, d_k])?;

        // [B*H, S, D] -> [B*H, D, S] for K^T
        let k_t = ops.transpose_axes(&k, 1, 2)?;

        // Parallel dot product over all heads: [B*H, S, D] x [B*H, D, S] -> [B*H, S, S]
        let scores = ops.matmul_batched(&q, &k_t)?;
        let attn = ops.softmax_dim(&ops.mul_scalar(&scores, self.scale)?, 2)?;

        // [B*H, S, S] x [B*H, S, D] -> [B*H, S, D]
        let attn_out = ops.matmul_batched(&attn, &v)?;

        // [B*H, S, D] -> [B, H, S, D] -> [B, S, H, D]
        let attn_out = ops.transpose_axes(&ops.reshape(&attn_out, &[batch, h, seq_len, d_k])?, 1, 2)?;
        let output =
            ops.matmul(&ops.reshape(&attn_out, &[batch * seq_len, d_model])?, self.out_proj.tensor())?;
        ops.reshape(&output, &[batch, seq_len, d_model])
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

pub struct FlashAttention<B: Backend> {
    inner: MultiHeadAttention<B>,
}
impl<B: Backend> FlashAttention<B> {
    pub fn new(b: &B, c: SelfAttentionConfig, s: u64) -> Result<Self> {
        Ok(Self { inner: MultiHeadAttention::new(b, c, s)? })
    }
}
impl<B: Backend> Module<B> for FlashAttention<B> {
    type Input = B::Tensor;
    type Output = B::Tensor;
    fn forward(&self, i: Self::Input, ctx: &mut ForwardCtx<B>) -> Result<Self::Output> {
        self.inner.forward(i, ctx)
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
    let mut v = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            v[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }
    backend.ops().tensor_from_vec(v, &[seq_len, seq_len])
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
        let ff_out = self.ff2.forward(ops.gelu(&self.ff1.forward(normed, ctx)?)?, ctx)?;
        ops.add(&x, &ff_out)
    }
}

impl<B: Backend> Trainable<B> for TransformerEncoderBlock<B> {
    fn parameters(&self) -> Vec<ParameterRef> {
        let mut p = self.self_attn.parameters();
        p.extend(self.norm1.parameters());
        p.extend(self.ff1.parameters());
        p.extend(self.ff2.parameters());
        p.extend(self.norm2.parameters());
        p
    }
}

impl<B: Backend> NamedParameters<B> for TransformerEncoderBlock<B> {
    fn visit_parameters(&self, f: &mut dyn FnMut(&str, &Parameter<B>)) {
        self.self_attn.visit_parameters(&mut |n, p| f(&format!("self_attn.{n}"), p));
        self.norm1.visit_parameters(&mut |n, p| f(&format!("norm1.{n}"), p));
        self.ff1.visit_parameters(&mut |n, p| f(&format!("ff1.{n}"), p));
        self.ff2.visit_parameters(&mut |n, p| f(&format!("ff2.{n}"), p));
        self.norm2.visit_parameters(&mut |n, p| f(&format!("norm2.{n}"), p));
    }
    fn visit_parameters_mut(&mut self, f: &mut dyn FnMut(&str, &mut Parameter<B>)) {
        self.self_attn.visit_parameters_mut(&mut |n, p| f(&format!("self_attn.{n}"), p));
        self.norm1.visit_parameters_mut(&mut |n, p| f(&format!("norm1.{n}"), p));
        self.ff1.visit_parameters_mut(&mut |n, p| f(&format!("ff1.{n}"), p));
        self.ff2.visit_parameters_mut(&mut |n, p| f(&format!("ff2.{n}"), p));
        self.norm2.visit_parameters_mut(&mut |n, p| f(&format!("norm2.{n}"), p));
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FlashAttentionConfig {
    pub attention: SelfAttentionConfig,
    pub block_size: usize,
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
        let standard = seq_len * seq_len * batch_size * num_heads * 4;
        let flash = seq_len * batch_size * num_heads * head_dim * 4;
        Self {
            seq_len,
            batch_size,
            num_heads,
            head_dim,
            standard_memory_bytes: standard + flash,
            flash_memory_bytes: flash * 2,
            reduction_factor: (standard + flash) as f32 / (flash * 2) as f32,
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
