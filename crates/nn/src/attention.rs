//! Attention mechanisms for neural networks.
//!
//! Provides SelfAttention, MultiHeadAttention, and TransformerEncoderBlock
//! implementations for modern NLP and vision architectures.

use mnr_core::{Backend, ForwardCtx, Module, Parameter, ParameterRef, Result, Trainable};
use serde::{Deserialize, Serialize};

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
    ///
    /// # Panics
    /// Panics if `d_model` is not divisible by `num_heads`.
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        assert_eq!(
            d_model % num_heads,
            0,
            "d_model ({}) must be divisible by num_heads ({})",
            d_model,
            num_heads
        );
        Self {
            d_model,
            num_heads,
            head_dim: d_model / num_heads,
            dropout: 0.0,
        }
    }

    /// Set dropout probability.
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }
}

/// Single-head or multi-head self-attention.
///
/// Projects input to Q, K, V, then computes: softmax(Q @ K^T / sqrt(d_k)) @ V
pub struct SelfAttention<B: Backend> {
    config: SelfAttentionConfig,
    q_proj: Parameter<B>,
    k_proj: Parameter<B>,
    v_proj: Parameter<B>,
    out_proj: Parameter<B>,
    scale: f32,
}

impl<B: Backend> SelfAttention<B> {
    /// Create a SelfAttention from explicit parameters.
    pub fn from_parameters(
        config: SelfAttentionConfig,
        q_proj: Parameter<B>,
        k_proj: Parameter<B>,
        v_proj: Parameter<B>,
        out_proj: Parameter<B>,
    ) -> Self {
        let scale = 1.0 / (config.head_dim as f32).sqrt();
        Self {
            config,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            scale,
        }
    }

    /// Borrow the configuration.
    pub fn config(&self) -> &SelfAttentionConfig {
        &self.config
    }

    /// Compute scaled dot-product attention.
    ///
    /// # Arguments
    /// * `q` - Query tensor [batch, seq, d_k]
    /// * `k` - Key tensor [batch, seq, d_k]
    /// * `v` - Value tensor [batch, seq, d_v]
    /// * `ops` - Tensor operations backend
    ///
    /// Returns: Attention output [batch, seq, d_v]
    pub fn scaled_dot_product_attention(
        &self,
        q: &B::Tensor,
        k: &B::Tensor,
        v: &B::Tensor,
        ops: &dyn mnr_core::TensorOps<B>,
    ) -> Result<B::Tensor> {
        // Q @ K^T: [batch, seq, d_k] @ [batch, d_k, seq] -> [batch, seq, seq]
        let k_t = ops.transpose(k)?;
        let scores = ops.matmul(q, &k_t)?;

        // Scale by 1/sqrt(d_k)
        let scale_tensor = ops.tensor_from_vec(vec![self.scale], &[1])?;
        let scaled_scores = ops.mul(&scores, &scale_tensor)?;

        // Apply softmax to get attention weights
        let attn_weights = ops.softmax(&scaled_scores)?;

        // Apply dropout during training
        let attn_weights = if self.config.dropout > 0.0 {
            ops.dropout(&attn_weights, self.config.dropout, false)?
        } else {
            attn_weights
        };

        // Attention @ V: [batch, seq, seq] @ [batch, seq, d_v] -> [batch, seq, d_v]
        let output = ops.matmul(&attn_weights, v)?;

        Ok(output)
    }
}

impl<B: Backend> Module<B> for SelfAttention<B> {
    type Input = B::Tensor;
    type Output = B::Tensor;

    fn forward(&self, input: Self::Input, ctx: &mut ForwardCtx<B>) -> Result<Self::Output> {
        let ops = ctx.backend().ops();
        let input_shape = ops.shape(&input);

        // Expected: [batch, seq_len, d_model]
        if input_shape.len() != 3 {
            return Err(mnr_core::CoreError::InvalidShape {
                shape: input_shape,
                reason: "SelfAttention expects 3D [batch, seq, d_model] input".into(),
            });
        }

        let batch = input_shape[0];
        let seq_len = input_shape[1];
        let d_model = input_shape[2];

        if d_model != self.config.d_model {
            return Err(mnr_core::CoreError::ShapeMismatch {
                expected: vec![self.config.d_model],
                actual: vec![d_model],
            });
        }

        let head_dim = self.config.head_dim;
        let _num_heads = self.config.num_heads;
        let d_k = head_dim;

        // Project input to Q, K, V
        // For single-head attention: [batch*seq, d_model] @ [d_model, d_k] -> [batch*seq, d_k]
        // We flatten batch and seq for matmul, then reshape back
        let flat_input = ops.reshape(&input, &[batch * seq_len, d_model])?;

        // Q projection
        let q_proj_tensor = self.q_proj.tensor();
        let q = ops.matmul(&flat_input, q_proj_tensor)?;
        let q = ops.reshape(&q, &[batch, seq_len, d_k])?;

        // K projection
        let k_proj_tensor = self.k_proj.tensor();
        let k = ops.matmul(&flat_input, k_proj_tensor)?;
        let k = ops.reshape(&k, &[batch, seq_len, d_k])?;

        // V projection
        let v_proj_tensor = self.v_proj.tensor();
        let v = ops.matmul(&flat_input, v_proj_tensor)?;
        let v = ops.reshape(&v, &[batch, seq_len, d_k])?;

        // Compute attention
        let attn_out = self.scaled_dot_product_attention(&q, &k, &v, ops)?;

        // Output projection
        // Flatten: [batch, seq, d_k] -> [batch*seq, d_k]
        let flat_attn = ops.reshape(&attn_out, &[batch * seq_len, d_k])?;
        let out_proj_tensor = self.out_proj.tensor();
        let output = ops.matmul(&flat_attn, out_proj_tensor)?;

        // Reshape back to [batch, seq, d_model]
        ops.reshape(&output, &[batch, seq_len, self.config.d_model])
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

/// Multi-head attention that splits Q, K, V into multiple heads.
///
/// Each head operates on a different subspace, allowing the model to attend
/// to different aspects of the input simultaneously.
pub struct MultiHeadAttention<B: Backend> {
    config: SelfAttentionConfig,
    attention: SelfAttention<B>,
}

impl<B: Backend> MultiHeadAttention<B> {
    /// Create a MultiHeadAttention layer.
    ///
    /// This is a convenience wrapper around SelfAttention. In a full implementation,
    /// heads would be processed in parallel with separate projections.
    pub fn from_parameters(config: SelfAttentionConfig, attention: SelfAttention<B>) -> Self {
        Self { config, attention }
    }

    /// Create MultiHeadAttention with randomly initialized parameters.
    pub fn new(backend: &B, config: SelfAttentionConfig, seed: u64) -> Result<Self> {
        let d_model = config.d_model;
        let head_dim = config.head_dim;

        let q_proj = backend.normal_parameter("q_proj", &[d_model, head_dim], seed, 0.02)?;
        let k_proj = backend.normal_parameter("k_proj", &[d_model, head_dim], seed.wrapping_add(1), 0.02)?;
        let v_proj = backend.normal_parameter("v_proj", &[d_model, head_dim], seed.wrapping_add(2), 0.02)?;
        let out_proj = backend.normal_parameter("out_proj", &[head_dim, d_model], seed.wrapping_add(3), 0.02)?;

        let attention = SelfAttention::from_parameters(config.clone(), q_proj, k_proj, v_proj, out_proj);
        Ok(Self { config, attention })
    }

    /// Borrow the configuration.
    pub fn config(&self) -> &SelfAttentionConfig {
        &self.config
    }
}

impl<B: Backend> Module<B> for MultiHeadAttention<B> {
    type Input = B::Tensor;
    type Output = B::Tensor;

    fn forward(&self, input: Self::Input, ctx: &mut ForwardCtx<B>) -> Result<Self::Output> {
        // For a simplified implementation, delegate to single-head attention
        // A full implementation would split into heads, process in parallel, and concatenate
        self.attention.forward(input, ctx)
    }
}

impl<B: Backend> Trainable<B> for MultiHeadAttention<B> {
    fn parameters(&self) -> Vec<ParameterRef> {
        self.attention.parameters()
    }
}

/// A single transformer encoder block.
///
/// Consists of:
/// 1. Multi-head self-attention with residual connection and layer norm
/// 2. Feed-forward network with residual connection and layer norm
///
/// Forward: x + attn(norm1(x)) then x + ff(norm2(x))
pub struct TransformerEncoderBlock<B: Backend> {
    self_attn: MultiHeadAttention<B>,
    norm1: crate::normalization::LayerNorm<B>,
    ff1: crate::Linear<B>,
    ff2: crate::Linear<B>,
    norm2: crate::normalization::LayerNorm<B>,
}

impl<B: Backend> TransformerEncoderBlock<B> {
    /// Create a transformer encoder block.
    pub fn new(
        self_attn: MultiHeadAttention<B>,
        norm1: crate::normalization::LayerNorm<B>,
        ff1: crate::Linear<B>,
        ff2: crate::Linear<B>,
        norm2: crate::normalization::LayerNorm<B>,
    ) -> Self {
        Self {
            self_attn,
            norm1,
            ff1,
            ff2,
            norm2,
        }
    }
}

impl<B: Backend> Module<B> for TransformerEncoderBlock<B> {
    type Input = B::Tensor;
    type Output = B::Tensor;

    fn forward(&self, input: Self::Input, ctx: &mut ForwardCtx<B>) -> Result<Self::Output> {
        let ops = ctx.backend().ops();

        // Pre-norm variant (more stable):
        // x + self_attn(norm1(x))
        let normed = self.norm1.forward(input.clone(), ctx)?;
        let attn_out = self.self_attn.forward(normed, ctx)?;
        let x = ops.add(&input, &attn_out)?;

        // x + ff(norm2(x))
        let normed = self.norm2.forward(x.clone(), ctx)?;
        let ff_out = self.ff1.forward(normed, ctx)?;
        // Apply ReLU activation
        let ff_out = ops.relu(&ff_out)?;
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

/// Causal mask for preventing attention to future tokens.
///
/// Creates an upper-triangular mask where positions (i, j) with j > i
/// are set to -inf (or a large negative number) to prevent attending to future tokens.
pub fn causal_mask<B: Backend>(backend: &B, seq_len: usize) -> Result<B::Tensor> {
    let ops = backend.ops();

    // Create a matrix where upper triangle is -inf and lower triangle is 0
    let mut mask_values = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask_values[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }

    ops.tensor_from_vec(mask_values, &[seq_len, seq_len])
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnr_core::{ForwardCtx, Mode};
    use mnr_ndarray_backend::CpuBackend;

    #[test]
    fn test_self_attention_config() {
        let config = SelfAttentionConfig::new(64, 8);
        assert_eq!(config.d_model, 64);
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.head_dim, 8); // 64 / 8
    }

    #[test]
    #[should_panic(expected = "must be divisible by num_heads")]
    fn test_self_attention_config_panic() {
        SelfAttentionConfig::new(65, 8);
    }

    #[test]
    fn test_multi_head_attention_new() {
        let backend = CpuBackend::default();
        let mha = MultiHeadAttention::new(&backend, SelfAttentionConfig::new(16, 4), 42).unwrap();

        assert_eq!(mha.config().d_model, 16);
        assert_eq!(mha.config().num_heads, 4);
        assert_eq!(mha.parameters().len(), 4); // q, k, v, out projections
    }

    #[test]
    fn test_causal_mask() {
        let backend = CpuBackend::default();
        let mask = causal_mask(&backend, 4).unwrap();

        let values: Vec<f32> = (0..16)
            .filter_map(|i| backend.ops().tensor_element(&mask, i).ok())
            .collect();

        // Lower triangle (including diagonal) should be 0
        assert!(values[0].is_finite());  // (0,0)
        assert!(values[4].is_finite());  // (1,0)
        assert!(values[5].is_finite());  // (1,1)

        // Upper triangle should be -inf
        assert_eq!(values[1], f32::NEG_INFINITY); // (0,1)
        assert_eq!(values[2], f32::NEG_INFINITY); // (0,2)
        assert_eq!(values[3], f32::NEG_INFINITY); // (0,3)
        assert_eq!(values[7], f32::NEG_INFINITY); // (1,3)
    }

    #[test]
    fn test_transformer_encoder_block_parameters() {
        let backend = CpuBackend::default();
        let config = SelfAttentionConfig::new(16, 4);

        let mha = MultiHeadAttention::new(&backend, config.clone(), 42).unwrap();

        let norm1 = crate::normalization::LayerNorm::from_parameters(
            crate::normalization::LayerNormConfig::new(vec![16]).with_eps(1e-5),
            mnr_core::Parameter::new("norm1_w", backend.tensor_from_vec(vec![1.0; 16], &[16]).unwrap()),
            mnr_core::Parameter::new("norm1_b", backend.tensor_from_vec(vec![0.0; 16], &[16]).unwrap()),
        );

        let ff1 = crate::LinearBuilder::new(16, 64)
            .with_bias(true)
            .build(&backend)
            .unwrap();

        let ff2 = crate::LinearBuilder::new(64, 16)
            .with_bias(true)
            .build(&backend)
            .unwrap();

        let norm2 = crate::normalization::LayerNorm::from_parameters(
            crate::normalization::LayerNormConfig::new(vec![16]).with_eps(1e-5),
            mnr_core::Parameter::new("norm2_w", backend.tensor_from_vec(vec![1.0; 16], &[16]).unwrap()),
            mnr_core::Parameter::new("norm2_b", backend.tensor_from_vec(vec![0.0; 16], &[16]).unwrap()),
        );

        let block = TransformerEncoderBlock::new(mha, norm1, ff1, ff2, norm2);

        let params = block.parameters();
        // MHA: 4, norm1: 2, ff1: 2, ff2: 2, norm2: 2 = 12
        assert_eq!(params.len(), 12);
    }
}
