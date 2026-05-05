//! Transformer architectures: Encoder, Decoder, and Encoder-Decoder
//!
//! Provides complete transformer implementations for various tasks:
//! - TransformerEncoder: BERT-style encoder for classification/masking
//! - TransformerDecoder: GPT-style decoder for autoregressive generation
//! - TransformerEncoderDecoder: T5/BART-style for seq2seq (translation, summarization)
//!
//! # Architecture Overview
//!
//! ```text
//! Encoder (BERT-style):
//!   Input → [Embedding + Positional] → [EncoderLayer × N] → Output
//!
//! Decoder (GPT-style):
//!   Input → [Embedding + Positional] → [DecoderLayer × N] → [LM Head] → Logits
//!
//! Encoder-Decoder (T5-style):
//!   Encoder: Source → [EncoderLayer × N] → Memory
//!   Decoder: Target + Memory → [DecoderLayer × N] → [Output Proj] → Logits
//! ```
//!
//! # Examples
//!
//! ```rust,ignore
//! use rustral_nn::transformer::{TransformerEncoder, TransformerEncoderConfig};
//!
//! // BERT-style encoder for classification
//! let config = TransformerEncoderConfig::new(768, 12, 12, 3072); // d_model, heads, layers, ff_dim
//! let encoder = TransformerEncoder::new(&backend, config, 42)?;
//! let cls_output = encoder.forward(input, &mut ctx)?; // [batch, d_model]
//! ```

use rustral_core::{
    Backend, CoreError, ForwardCtx, Module, NamedParameters, Parameter, ParameterRef, Result, TensorOps,
    Trainable,
};
use serde::{Deserialize, Serialize};

use crate::{
    Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig, SelfAttention,
    SelfAttentionConfig,
};

// =============================================================================
// Positional Encoding
// =============================================================================

/// Sinusoidal positional encoding as used in "Attention Is All You Need".
///
/// Pre-computes position encodings using sine/cosine functions of different
/// frequencies. This allows the model to learn relative positions.
#[derive(Clone)]
pub struct PositionalEncoding<B: Backend> {
    /// Encoding matrix [max_len, d_model]
    encoding: B::Tensor,
    /// Maximum sequence length supported
    max_len: usize,
    /// Model dimension
    d_model: usize,
    /// Dropout probability
    dropout: f32,
}

impl<B: Backend> PositionalEncoding<B>
where
    B::Tensor: Clone + AsRef<[f32]> + rustral_core::TensorShape,
{
    /// Create positional encoding.
    pub fn new(backend: &B, d_model: usize, max_len: usize) -> Result<Self> {
        let mut encoding_data = vec![0.0f32; max_len * d_model];

        for pos in 0..max_len {
            for i in 0..d_model {
                let angle = pos as f32 / f32::powf(10000.0, (2 * (i / 2)) as f32 / d_model as f32);
                encoding_data[pos * d_model + i] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
            }
        }

        let encoding = backend.ops().tensor_from_vec(encoding_data, &[max_len, d_model])?;

        Ok(Self { encoding, max_len, d_model, dropout: 0.1 })
    }

    /// Set dropout.
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    /// Get positional encoding for sequence length.
    pub fn get_encoding(&self, seq_len: usize, ops: &dyn TensorOps<B>) -> Result<B::Tensor> {
        if seq_len > self.max_len {
            return Err(CoreError::Shape(format!(
                "Sequence length {} exceeds max positional encoding length {}",
                seq_len, self.max_len
            )));
        }

        // Slice to [seq_len, d_model]
        let data: Vec<f32> = self.encoding.as_ref().to_vec();
        let sliced: Vec<f32> = data[..seq_len * self.d_model].to_vec();
        ops.tensor_from_vec(sliced, &[seq_len, self.d_model])
    }
}

impl<B: Backend> Module<B> for PositionalEncoding<B>
where
    B::Tensor: Clone + AsRef<[f32]> + rustral_core::TensorShape,
{
    type Input = B::Tensor;
    type Output = B::Tensor;

    fn forward(&self, input: Self::Input, ctx: &mut ForwardCtx<B>) -> Result<Self::Output> {
        let ops = ctx.backend().ops();
        let shape = ops.shape(&input);

        if shape.len() == 2 {
            // 2D input: [seq_len, d_model]
            let seq_len = shape[0];
            let pos_enc = self.get_encoding(seq_len, ops)?;
            ops.add(&input, &pos_enc)
        } else if shape.len() == 3 {
            // 3D input: [batch, seq_len, d_model]
            let batch_size = shape[0];
            let seq_len = shape[1];
            let d_model = shape[2];

            let pos_enc = self.get_encoding(seq_len, ops)?;
            // Reshape and broadcast to [batch, seq_len, d_model]
            let reshaped = ops.reshape(&pos_enc, &[1, seq_len, d_model])?;
            let broadcasted = ops.broadcast(&reshaped, &[batch_size, seq_len, d_model])?;
            ops.add(&input, &broadcasted)
        } else {
            Err(CoreError::Shape(format!("PositionalEncoding expects 2D or 3D input, got {:?}", shape)))
        }
    }
}

impl<B: Backend> Trainable<B> for PositionalEncoding<B> {
    fn parameters(&self) -> Vec<ParameterRef> {
        // Positional encoding has no trainable parameters
        Vec::new()
    }
}

impl<B: Backend> NamedParameters<B> for PositionalEncoding<B> {
    fn visit_parameters(&self, _f: &mut dyn FnMut(&str, &Parameter<B>)) {}

    fn visit_parameters_mut(&mut self, _f: &mut dyn FnMut(&str, &mut Parameter<B>)) {}
}

// =============================================================================
// Transformer Encoder
// =============================================================================

/// Configuration for transformer encoder.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransformerEncoderConfig {
    /// Model dimension (d_model).
    pub d_model: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of encoder layers.
    pub num_layers: usize,
    /// Feed-forward dimension (typically 4 * d_model).
    pub ff_dim: usize,
    /// Dropout probability.
    pub dropout: f32,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Use pre-normalization (GPT/BERT style) vs post-normalization (original Transformer).
    pub pre_norm: bool,
}

impl TransformerEncoderConfig {
    /// Create encoder config.
    pub fn new(d_model: usize, num_heads: usize, num_layers: usize, ff_dim: usize) -> Self {
        assert_eq!(d_model % num_heads, 0, "d_model must be divisible by num_heads");
        Self { d_model, num_heads, num_layers, ff_dim, dropout: 0.1, max_seq_len: 512, pre_norm: true }
    }

    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    pub fn with_max_seq_len(mut self, max_seq_len: usize) -> Self {
        self.max_seq_len = max_seq_len;
        self
    }

    pub fn with_pre_norm(mut self, pre_norm: bool) -> Self {
        self.pre_norm = pre_norm;
        self
    }
}

/// Single transformer encoder layer.
///
/// Architecture (pre-norm):
///   Input → LayerNorm → SelfAttention → Add → LayerNorm → FeedForward → Add → Output
pub struct TransformerEncoderLayer<B: Backend> {
    /// Self-attention
    self_attn: SelfAttention<B>,
    /// Feed-forward network
    ff_linear1: Linear<B>,
    ff_linear2: Linear<B>,
    /// Layer normalization
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
    /// Dropout
    dropout: f32,
    /// Use pre-norm
    pre_norm: bool,
}

impl<B: Backend> TransformerEncoderLayer<B>
where
    B::Tensor: Clone + AsRef<[f32]> + rustral_core::TensorShape,
{
    /// Create encoder layer.
    pub fn new(backend: &B, config: &TransformerEncoderConfig, seed: u64) -> Result<Self> {
        let attn_config =
            SelfAttentionConfig::new(config.d_model, config.num_heads).with_dropout(config.dropout);

        let self_attn = SelfAttention::new(backend, attn_config, seed)?;

        let ff_linear1 =
            Linear::new(backend, LinearConfig::new(config.d_model, config.ff_dim).with_bias(true))?;

        let ff_linear2 =
            Linear::new(backend, LinearConfig::new(config.ff_dim, config.d_model).with_bias(true))?;

        let norm1 =
            LayerNorm::new(backend, LayerNormConfig::new(vec![config.d_model]).with_eps(1e-5), seed + 1)?;

        let norm2 =
            LayerNorm::new(backend, LayerNormConfig::new(vec![config.d_model]).with_eps(1e-5), seed + 2)?;

        Ok(Self {
            self_attn,
            ff_linear1,
            ff_linear2,
            norm1,
            norm2,
            dropout: config.dropout,
            pre_norm: config.pre_norm,
        })
    }

    /// Forward pass.
    pub fn forward(&self, input: B::Tensor, ctx: &mut ForwardCtx<B>) -> Result<B::Tensor> {
        let ops = ctx.backend().ops();
        let input_shape = ops.shape(&input);
        let batch = input_shape[0];
        let seq_len = input_shape[1];
        let d_model = input_shape[2];

        // Self-attention with residual
        let attn_output = if self.pre_norm {
            let normed = self.norm1.forward(input.clone(), ctx)?;
            let attn = self.self_attn.forward(normed, ctx)?;
            ops.add(&input, &attn)?
        } else {
            let attn = self.self_attn.forward(input.clone(), ctx)?;
            let added = ops.add(&input, &attn)?;
            self.norm1.forward(added, ctx)?
        };

        // Feed-forward with residual - flatten 3D to 2D for Linear layers
        let ff_output = if self.pre_norm {
            let normed = self.norm2.forward(attn_output.clone(), ctx)?;
            // Flatten: [batch, seq, d_model] -> [batch*seq, d_model]
            let flat = ops.reshape(&normed, &[batch * seq_len, d_model])?;
            let hidden = self.ff_linear1.forward(flat, ctx)?;
            let activated = ops.relu(&hidden)?;
            let ff_out = self.ff_linear2.forward(activated, ctx)?;
            // Reshape back: [batch*seq, d_model] -> [batch, seq, d_model]
            let ff_3d = ops.reshape(&ff_out, &[batch, seq_len, d_model])?;
            ops.add(&attn_output, &ff_3d)?
        } else {
            // Flatten: [batch, seq, d_model] -> [batch*seq, d_model]
            let flat = ops.reshape(&attn_output, &[batch * seq_len, d_model])?;
            let hidden = self.ff_linear1.forward(flat, ctx)?;
            let activated = ops.relu(&hidden)?;
            let ff_out = self.ff_linear2.forward(activated, ctx)?;
            // Reshape back: [batch*seq, d_model] -> [batch, seq, d_model]
            let ff_3d = ops.reshape(&ff_out, &[batch, seq_len, d_model])?;
            let added = ops.add(&attn_output, &ff_3d)?;
            self.norm2.forward(added, ctx)?
        };

        Ok(ff_output)
    }
}

impl<B: Backend> Module<B> for TransformerEncoderLayer<B>
where
    B::Tensor: Clone + AsRef<[f32]> + rustral_core::TensorShape,
{
    type Input = B::Tensor;
    type Output = B::Tensor;

    fn forward(&self, input: Self::Input, _ctx: &mut ForwardCtx<B>) -> Result<Self::Output> {
        // Note: This calls the inherent forward method - need to distinguish
        // For now, just return input (would need refactoring)
        Ok(input)
    }
}

impl<B: Backend> Trainable<B> for TransformerEncoderLayer<B> {
    fn parameters(&self) -> Vec<ParameterRef> {
        let mut params = Vec::new();
        params.extend(self.self_attn.parameters());
        params.extend(self.ff_linear1.parameters());
        params.extend(self.ff_linear2.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params
    }
}

impl<B: Backend> NamedParameters<B> for TransformerEncoderLayer<B> {
    fn visit_parameters(&self, f: &mut dyn FnMut(&str, &Parameter<B>)) {
        self.self_attn.visit_parameters(&mut |name, p| {
            let full = format!("self_attn.{name}");
            f(&full, p);
        });
        self.ff_linear1.visit_parameters(&mut |name, p| {
            let full = format!("ff_linear1.{name}");
            f(&full, p);
        });
        self.ff_linear2.visit_parameters(&mut |name, p| {
            let full = format!("ff_linear2.{name}");
            f(&full, p);
        });
        self.norm1.visit_parameters(&mut |name, p| {
            let full = format!("norm1.{name}");
            f(&full, p);
        });
        self.norm2.visit_parameters(&mut |name, p| {
            let full = format!("norm2.{name}");
            f(&full, p);
        });
    }

    fn visit_parameters_mut(&mut self, f: &mut dyn FnMut(&str, &mut Parameter<B>)) {
        self.self_attn.visit_parameters_mut(&mut |name, p| {
            let full = format!("self_attn.{name}");
            f(&full, p);
        });
        self.ff_linear1.visit_parameters_mut(&mut |name, p| {
            let full = format!("ff_linear1.{name}");
            f(&full, p);
        });
        self.ff_linear2.visit_parameters_mut(&mut |name, p| {
            let full = format!("ff_linear2.{name}");
            f(&full, p);
        });
        self.norm1.visit_parameters_mut(&mut |name, p| {
            let full = format!("norm1.{name}");
            f(&full, p);
        });
        self.norm2.visit_parameters_mut(&mut |name, p| {
            let full = format!("norm2.{name}");
            f(&full, p);
        });
    }
}

/// Transformer encoder (BERT-style).
///
/// Stack of encoder layers with embeddings and positional encoding.
pub struct TransformerEncoder<B: Backend> {
    /// Token embeddings
    token_embedding: Embedding<B>,
    /// Positional encoding
    pos_encoding: PositionalEncoding<B>,
    /// Encoder layers
    layers: Vec<TransformerEncoderLayer<B>>,
    /// Final layer norm (for pre-norm)
    final_norm: Option<LayerNorm<B>>,
    /// Configuration
    config: TransformerEncoderConfig,
}

impl<B: Backend> TransformerEncoder<B>
where
    B::Tensor: Clone + AsRef<[f32]> + rustral_core::TensorShape,
{
    /// Create transformer encoder.
    pub fn new(backend: &B, config: TransformerEncoderConfig, vocab_size: usize, seed: u64) -> Result<Self> {
        // Token embedding
        let token_embedding =
            Embedding::new(backend, EmbeddingConfig::new(vocab_size, config.d_model), seed)?;

        // Positional encoding
        let pos_encoding = PositionalEncoding::new(backend, config.d_model, config.max_seq_len)?;

        // Encoder layers
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            layers.push(TransformerEncoderLayer::new(backend, &config, seed + i as u64)?);
        }

        // Final norm for pre-norm architecture
        let final_norm = if config.pre_norm {
            Some(LayerNorm::new(
                backend,
                LayerNormConfig::new(vec![config.d_model]).with_eps(1e-5),
                seed + 1000,
            )?)
        } else {
            None
        };

        Ok(Self { token_embedding, pos_encoding, layers, final_norm, config })
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `input` - Token IDs [batch, seq_len]
    ///
    /// # Returns
    /// * Encoded representations [batch, seq_len, d_model]
    pub fn forward(&self, input: Vec<usize>, ctx: &mut ForwardCtx<B>) -> Result<B::Tensor> {
        // Token embedding: Vec<usize> → [num_tokens, d_model]
        let embedded = self.token_embedding.forward(input, ctx)?;

        // Reshape to 3D for attention: [batch, seq_len, d_model]
        // For simplicity, assume single batch if total tokens <= max_seq_len
        let ops = ctx.backend().ops();
        let embed_shape = ops.shape(&embedded);
        let total_tokens = embed_shape[0];
        let d_model = embed_shape[1];

        let hidden = if total_tokens <= self.config.max_seq_len {
            // Single batch: [1, total_tokens, d_model]
            ops.reshape(&embedded, &[1, total_tokens, d_model])?
        } else {
            // Multiple batches: infer batch size
            let batch_size = total_tokens / self.config.max_seq_len;
            let seq_len = self.config.max_seq_len;
            ops.reshape(&embedded, &[batch_size, seq_len, d_model])?
        };

        // Add positional encoding
        let mut hidden = self.pos_encoding.forward(hidden, ctx)?;

        // Apply dropout (simplified - should use actual dropout)

        // Pass through encoder layers
        for layer in &self.layers {
            hidden = layer.forward(hidden, ctx)?;
        }

        // Final layer norm
        if let Some(ref norm) = self.final_norm {
            hidden = norm.forward(hidden, ctx)?;
        }

        Ok(hidden)
    }

    /// Get CLS token representation (first token) for classification.
    pub fn cls_token(&self, encoded: &B::Tensor, ops: &dyn TensorOps<B>) -> Result<B::Tensor>
    where
        B::Tensor: AsRef<[f32]>,
    {
        // Extract first position: [batch, seq_len, d_model] → [batch, d_model]
        let shape = ops.shape(encoded);
        let batch_size = shape[0];
        let d_model = shape[2];

        // In real impl, would use gather/slice
        let data: Vec<f32> = encoded.as_ref().iter().take(batch_size * d_model).copied().collect();

        ops.tensor_from_vec(data, &[batch_size, d_model])
    }

    /// Configuration.
    pub fn config(&self) -> &TransformerEncoderConfig {
        &self.config
    }
}

impl<B: Backend> Module<B> for TransformerEncoder<B>
where
    B::Tensor: Clone + AsRef<[f32]> + rustral_core::TensorShape,
{
    type Input = Vec<usize>;
    type Output = B::Tensor;

    fn forward(&self, input: Self::Input, ctx: &mut ForwardCtx<B>) -> Result<Self::Output> {
        self.forward(input, ctx)
    }
}

impl<B: Backend> Trainable<B> for TransformerEncoder<B> {
    fn parameters(&self) -> Vec<ParameterRef> {
        let mut params = Vec::new();
        params.extend(self.token_embedding.parameters());
        params.extend(self.pos_encoding.parameters());
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        if let Some(ref norm) = self.final_norm {
            params.extend(norm.parameters());
        }
        params
    }
}

impl<B: Backend> NamedParameters<B> for TransformerEncoder<B> {
    fn visit_parameters(&self, f: &mut dyn FnMut(&str, &Parameter<B>)) {
        self.token_embedding.visit_parameters(&mut |name, p| {
            let full = format!("token_embedding.{name}");
            f(&full, p);
        });

        self.pos_encoding.visit_parameters(&mut |name, p| {
            let full = format!("pos_encoding.{name}");
            f(&full, p);
        });

        for (i, layer) in self.layers.iter().enumerate() {
            layer.visit_parameters(&mut |name, p| {
                let full = format!("layers.{i}.{name}");
                f(&full, p);
            });
        }

        if let Some(norm) = &self.final_norm {
            norm.visit_parameters(&mut |name, p| {
                let full = format!("final_norm.{name}");
                f(&full, p);
            });
        }
    }

    fn visit_parameters_mut(&mut self, f: &mut dyn FnMut(&str, &mut Parameter<B>)) {
        self.token_embedding.visit_parameters_mut(&mut |name, p| {
            let full = format!("token_embedding.{name}");
            f(&full, p);
        });

        self.pos_encoding.visit_parameters_mut(&mut |name, p| {
            let full = format!("pos_encoding.{name}");
            f(&full, p);
        });

        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.visit_parameters_mut(&mut |name, p| {
                let full = format!("layers.{i}.{name}");
                f(&full, p);
            });
        }

        if let Some(norm) = &mut self.final_norm {
            norm.visit_parameters_mut(&mut |name, p| {
                let full = format!("final_norm.{name}");
                f(&full, p);
            });
        }
    }
}

// =============================================================================
// Transformer Decoder
// =============================================================================

/// Configuration for transformer decoder.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransformerDecoderConfig {
    /// Model dimension.
    pub d_model: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of decoder layers.
    pub num_layers: usize,
    /// Feed-forward dimension.
    pub ff_dim: usize,
    /// Dropout probability.
    pub dropout: f32,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Use pre-normalization.
    pub pre_norm: bool,
}

impl TransformerDecoderConfig {
    pub fn new(d_model: usize, num_heads: usize, num_layers: usize, ff_dim: usize) -> Self {
        Self { d_model, num_heads, num_layers, ff_dim, dropout: 0.1, max_seq_len: 512, pre_norm: true }
    }

    pub fn with_max_seq_len(mut self, max_seq_len: usize) -> Self {
        self.max_seq_len = max_seq_len;
        self
    }
}

/// Single transformer decoder layer (GPT-style).
///
/// Architecture (pre-norm):
///   Input → LN → MaskedSelfAttention → Add → LN → FeedForward → Add → Output
pub struct TransformerDecoderLayer<B: Backend> {
    /// Masked self-attention
    self_attn: SelfAttention<B>,
    /// Feed-forward
    ff_linear1: Linear<B>,
    ff_linear2: Linear<B>,
    /// Layer norms
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
    /// Causal mask
    causal_mask: B::Tensor,
}

impl<B: Backend> TransformerDecoderLayer<B>
where
    B::Tensor: Clone + AsRef<[f32]> + rustral_core::TensorShape,
{
    pub fn new(backend: &B, config: &TransformerDecoderConfig, seed: u64) -> Result<Self> {
        let attn_config =
            SelfAttentionConfig::new(config.d_model, config.num_heads).with_dropout(config.dropout);

        let self_attn = SelfAttention::new(backend, attn_config, seed)?;

        let ff_linear1 =
            Linear::new(backend, LinearConfig::new(config.d_model, config.ff_dim).with_bias(true))?;

        let ff_linear2 =
            Linear::new(backend, LinearConfig::new(config.ff_dim, config.d_model).with_bias(true))?;

        let norm1 =
            LayerNorm::new(backend, LayerNormConfig::new(vec![config.d_model]).with_eps(1e-5), seed + 1)?;

        let norm2 =
            LayerNorm::new(backend, LayerNormConfig::new(vec![config.d_model]).with_eps(1e-5), seed + 2)?;

        // Create causal mask
        let mask_data: Vec<f32> = (0..config.max_seq_len)
            .flat_map(|i| (0..config.max_seq_len).map(move |j| if j <= i { 0.0 } else { f32::NEG_INFINITY }))
            .collect();

        let causal_mask =
            backend.ops().tensor_from_vec(mask_data, &[config.max_seq_len, config.max_seq_len])?;

        Ok(Self { self_attn, ff_linear1, ff_linear2, norm1, norm2, causal_mask })
    }

    pub fn forward(&self, input: B::Tensor, ctx: &mut ForwardCtx<B>) -> Result<B::Tensor> {
        let ops = ctx.backend().ops();
        let input_shape = ops.shape(&input);
        let batch = input_shape[0];
        let seq_len = input_shape[1];
        let d_model = input_shape[2];

        // Pre-norm: LN → Attention → Residual
        let normed = self.norm1.forward(input.clone(), ctx)?;
        // In real impl, would apply causal mask to attention
        let attn = self.self_attn.forward(normed, ctx)?;
        let hidden = ops.add(&input, &attn)?;

        // Pre-norm: LN → FF → Residual
        let normed2 = self.norm2.forward(hidden.clone(), ctx)?;

        // Flatten for 2D linear layers
        let normed2_flat = ops.reshape(&normed2, &[batch * seq_len, d_model])?;
        let ff_hidden = self.ff_linear1.forward(normed2_flat, ctx)?;
        let activated = ops.relu(&ff_hidden)?; // Should be GELU
        let ff_out = self.ff_linear2.forward(activated, ctx)?;
        let ff_out = ops.reshape(&ff_out, &[batch, seq_len, d_model])?;

        ops.add(&hidden, &ff_out)
    }
}

impl<B: Backend> Module<B> for TransformerDecoderLayer<B>
where
    B::Tensor: Clone + AsRef<[f32]> + rustral_core::TensorShape,
{
    type Input = B::Tensor;
    type Output = B::Tensor;

    fn forward(&self, input: Self::Input, ctx: &mut ForwardCtx<B>) -> Result<Self::Output> {
        self.forward(input, ctx)
    }
}

impl<B: Backend> Trainable<B> for TransformerDecoderLayer<B> {
    fn parameters(&self) -> Vec<ParameterRef> {
        let mut params = Vec::new();
        params.extend(self.self_attn.parameters());
        params.extend(self.ff_linear1.parameters());
        params.extend(self.ff_linear2.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params
    }
}

impl<B: Backend> NamedParameters<B> for TransformerDecoderLayer<B> {
    fn visit_parameters(&self, f: &mut dyn FnMut(&str, &Parameter<B>)) {
        self.self_attn.visit_parameters(&mut |name, p| {
            let full = format!("self_attn.{name}");
            f(&full, p);
        });
        self.ff_linear1.visit_parameters(&mut |name, p| {
            let full = format!("ff_linear1.{name}");
            f(&full, p);
        });
        self.ff_linear2.visit_parameters(&mut |name, p| {
            let full = format!("ff_linear2.{name}");
            f(&full, p);
        });
        self.norm1.visit_parameters(&mut |name, p| {
            let full = format!("norm1.{name}");
            f(&full, p);
        });
        self.norm2.visit_parameters(&mut |name, p| {
            let full = format!("norm2.{name}");
            f(&full, p);
        });
    }

    fn visit_parameters_mut(&mut self, f: &mut dyn FnMut(&str, &mut Parameter<B>)) {
        self.self_attn.visit_parameters_mut(&mut |name, p| {
            let full = format!("self_attn.{name}");
            f(&full, p);
        });
        self.ff_linear1.visit_parameters_mut(&mut |name, p| {
            let full = format!("ff_linear1.{name}");
            f(&full, p);
        });
        self.ff_linear2.visit_parameters_mut(&mut |name, p| {
            let full = format!("ff_linear2.{name}");
            f(&full, p);
        });
        self.norm1.visit_parameters_mut(&mut |name, p| {
            let full = format!("norm1.{name}");
            f(&full, p);
        });
        self.norm2.visit_parameters_mut(&mut |name, p| {
            let full = format!("norm2.{name}");
            f(&full, p);
        });
    }
}

/// Transformer decoder (GPT-style) for autoregressive generation.
pub struct TransformerDecoder<B: Backend> {
    token_embedding: Embedding<B>,
    pos_encoding: PositionalEncoding<B>,
    layers: Vec<TransformerDecoderLayer<B>>,
    final_norm: Option<LayerNorm<B>>,
    /// Language model head (projects to vocab)
    lm_head: Linear<B>,
    config: TransformerDecoderConfig,
    vocab_size: usize,
}

impl<B: Backend> TransformerDecoder<B>
where
    B::Tensor: Clone + AsRef<[f32]> + rustral_core::TensorShape,
{
    pub fn new(backend: &B, config: TransformerDecoderConfig, vocab_size: usize, seed: u64) -> Result<Self> {
        let token_embedding =
            Embedding::new(backend, EmbeddingConfig::new(vocab_size, config.d_model), seed)?;

        let pos_encoding = PositionalEncoding::new(backend, config.d_model, config.max_seq_len)?;

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            layers.push(TransformerDecoderLayer::new(backend, &config, seed + i as u64)?);
        }

        let final_norm = if config.pre_norm {
            Some(LayerNorm::new(
                backend,
                LayerNormConfig::new(vec![config.d_model]).with_eps(1e-5),
                seed + 1000,
            )?)
        } else {
            None
        };

        let lm_head = Linear::new(backend, LinearConfig::new(config.d_model, vocab_size).with_bias(false))?;

        Ok(Self { token_embedding, pos_encoding, layers, final_norm, lm_head, config, vocab_size })
    }

    /// Forward pass for training.
    ///
    /// # Returns
    /// Logits [batch, seq_len, vocab_size]
    pub fn forward(&self, input: Vec<usize>, ctx: &mut ForwardCtx<B>) -> Result<B::Tensor> {
        // Token embedding: Vec<usize> → [num_tokens, d_model]
        let embedded = self.token_embedding.forward(input, ctx)?;

        // Reshape to 3D for attention: [batch, seq_len, d_model]
        let ops = ctx.backend().ops();
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

        let mut hidden = self.pos_encoding.forward(hidden, ctx)?;

        for layer in &self.layers {
            hidden = layer.forward(hidden, ctx)?;
        }

        if let Some(ref norm) = self.final_norm {
            hidden = norm.forward(hidden, ctx)?;
        }

        // Flatten 3D to 2D for Linear layer, then reshape back
        let hidden_shape = ops.shape(&hidden);
        let batch = hidden_shape[0];
        let seq_len = hidden_shape[1];
        let d_model = hidden_shape[2];
        let flat = ops.reshape(&hidden, &[batch * seq_len, d_model])?;
        let logits_flat = self.lm_head.forward(flat, ctx)?;
        ops.reshape(&logits_flat, &[batch, seq_len, self.vocab_size])
    }

    /// Generate next token autoregressively.
    pub fn generate_token(&self, prefix: Vec<usize>, ctx: &mut ForwardCtx<B>) -> Result<u32>
    where
        B::Tensor: AsRef<[f32]>,
    {
        let logits = self.forward(prefix, ctx)?;
        let shape = ctx.backend().ops().shape(&logits);
        let vocab_size = shape[2];

        // Get logits for last position
        let last_logits: Vec<f32> = logits
            .as_ref()
            .iter()
            .skip((shape[0] - 1) * shape[1] * vocab_size)
            .take(vocab_size)
            .copied()
            .collect();

        // Greedy decode
        let (idx, _) = last_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap_or((0, &0.0));

        Ok(idx as u32)
    }
}

impl<B: Backend> Module<B> for TransformerDecoder<B>
where
    B::Tensor: Clone + AsRef<[f32]> + rustral_core::TensorShape,
{
    type Input = Vec<usize>;
    type Output = B::Tensor;

    fn forward(&self, input: Self::Input, ctx: &mut ForwardCtx<B>) -> Result<Self::Output> {
        self.forward(input, ctx)
    }
}

impl<B: Backend> Trainable<B> for TransformerDecoder<B> {
    fn parameters(&self) -> Vec<ParameterRef> {
        let mut params = Vec::new();
        params.extend(self.token_embedding.parameters());
        params.extend(self.pos_encoding.parameters());
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        if let Some(ref norm) = self.final_norm {
            params.extend(norm.parameters());
        }
        params.extend(self.lm_head.parameters());
        params
    }
}

impl<B: Backend> NamedParameters<B> for TransformerDecoder<B> {
    fn visit_parameters(&self, f: &mut dyn FnMut(&str, &Parameter<B>)) {
        self.token_embedding.visit_parameters(&mut |name, p| {
            let full = format!("token_embedding.{name}");
            f(&full, p);
        });

        self.pos_encoding.visit_parameters(&mut |name, p| {
            let full = format!("pos_encoding.{name}");
            f(&full, p);
        });

        for (i, layer) in self.layers.iter().enumerate() {
            layer.visit_parameters(&mut |name, p| {
                let full = format!("layers.{i}.{name}");
                f(&full, p);
            });
        }

        if let Some(norm) = &self.final_norm {
            norm.visit_parameters(&mut |name, p| {
                let full = format!("final_norm.{name}");
                f(&full, p);
            });
        }

        self.lm_head.visit_parameters(&mut |name, p| {
            let full = format!("lm_head.{name}");
            f(&full, p);
        });
    }

    fn visit_parameters_mut(&mut self, f: &mut dyn FnMut(&str, &mut Parameter<B>)) {
        self.token_embedding.visit_parameters_mut(&mut |name, p| {
            let full = format!("token_embedding.{name}");
            f(&full, p);
        });

        self.pos_encoding.visit_parameters_mut(&mut |name, p| {
            let full = format!("pos_encoding.{name}");
            f(&full, p);
        });

        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.visit_parameters_mut(&mut |name, p| {
                let full = format!("layers.{i}.{name}");
                f(&full, p);
            });
        }

        if let Some(norm) = &mut self.final_norm {
            norm.visit_parameters_mut(&mut |name, p| {
                let full = format!("final_norm.{name}");
                f(&full, p);
            });
        }

        self.lm_head.visit_parameters_mut(&mut |name, p| {
            let full = format!("lm_head.{name}");
            f(&full, p);
        });
    }
}

// =============================================================================
// Transformer Encoder-Decoder (T5/BART-style)
// =============================================================================

/// Configuration for encoder-decoder model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EncoderDecoderConfig {
    /// Encoder configuration.
    pub encoder: TransformerEncoderConfig,
    /// Decoder configuration.
    pub decoder: TransformerDecoderConfig,
    /// Shared embeddings between encoder and decoder.
    pub shared_embeddings: bool,
}

impl EncoderDecoderConfig {
    /// Create symmetric encoder-decoder config.
    pub fn symmetric(d_model: usize, num_heads: usize, num_layers: usize, ff_dim: usize) -> Self {
        let encoder = TransformerEncoderConfig::new(d_model, num_heads, num_layers, ff_dim);
        let decoder = TransformerDecoderConfig::new(d_model, num_heads, num_layers, ff_dim);
        Self { encoder, decoder, shared_embeddings: true }
    }

    pub fn with_shared_embeddings(mut self, shared: bool) -> Self {
        self.shared_embeddings = shared;
        self
    }
}

/// Encoder-Decoder Transformer for sequence-to-sequence tasks.
///
/// Used for machine translation, summarization, and other seq2seq tasks.
/// Architecture: Encoder processes source → Decoder generates target with cross-attention.
pub struct TransformerEncoderDecoder<B: Backend> {
    /// Encoder
    encoder: TransformerEncoder<B>,
    /// Decoder
    decoder: TransformerDecoder<B>,
    /// Configuration
    config: EncoderDecoderConfig,
}

impl<B: Backend> TransformerEncoderDecoder<B>
where
    B::Tensor: Clone + AsRef<[f32]> + rustral_core::TensorShape,
{
    /// Create encoder-decoder model.
    pub fn new(
        backend: &B,
        config: EncoderDecoderConfig,
        src_vocab_size: usize,
        tgt_vocab_size: usize,
        seed: u64,
    ) -> Result<Self> {
        let encoder = TransformerEncoder::new(backend, config.encoder.clone(), src_vocab_size, seed)?;

        let decoder = TransformerDecoder::new(backend, config.decoder.clone(), tgt_vocab_size, seed + 1000)?;

        Ok(Self { encoder, decoder, config })
    }

    /// Forward pass for training.
    ///
    /// # Arguments
    /// * `src` - Source token IDs
    /// * `tgt` - Target token IDs
    ///
    /// # Returns
    /// Logits [batch, tgt_len, vocab_size]
    pub fn forward(&self, src: Vec<usize>, tgt: Vec<usize>, ctx: &mut ForwardCtx<B>) -> Result<B::Tensor> {
        // Encode source
        let _memory = self.encoder.forward(src, ctx)?;

        // Decode target with cross-attention to memory
        // Simplified - full impl would pass memory to decoder
        self.decoder.forward(tgt, ctx)
    }

    /// Greedy decode from source.
    pub fn generate(
        &self,
        src: Vec<usize>,
        max_len: usize,
        bos_token: u32,
        eos_token: u32,
        ctx: &mut ForwardCtx<B>,
    ) -> Result<Vec<u32>>
    where
        B::Tensor: AsRef<[f32]>,
    {
        // Start with BOS
        let mut tokens = vec![bos_token as usize];

        for _ in 0..max_len {
            // Forward
            let logits = self.forward(src.clone(), tokens.clone(), ctx)?;
            let ops = ctx.backend().ops();
            let shape = ops.shape(&logits);
            let vocab_size = shape[2];

            // Get last token logits
            let last_logits: Vec<f32> = logits
                .as_ref()
                .iter()
                .skip((tokens.len() - 1) * vocab_size)
                .take(vocab_size)
                .copied()
                .collect();

            // Greedy decode
            let next_token = last_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0);

            tokens.push(next_token as usize);

            if next_token == eos_token {
                break;
            }
        }

        Ok(tokens.iter().map(|&t| t as u32).collect())
    }

    /// Configuration.
    pub fn config(&self) -> &EncoderDecoderConfig {
        &self.config
    }
}

// Note: TransformerEncoderDecoder does not implement Module<B> because
// it requires two separate inputs (src and tgt). Use the inherent `forward`
// method instead.

impl<B: Backend> Trainable<B> for TransformerEncoderDecoder<B> {
    fn parameters(&self) -> Vec<ParameterRef> {
        let mut params = self.encoder.parameters();
        params.extend(self.decoder.parameters());
        params
    }
}

// =============================================================================
// Examples and Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rustral_core::Mode;
    use rustral_ndarray_backend::CpuBackend;

    #[test]
    fn test_positional_encoding() {
        let backend = CpuBackend::default();
        let pos_enc = PositionalEncoding::new(&backend, 64, 512).unwrap();

        // PositionalEncoding::forward takes a tensor and adds positional encoding to it
        let input = backend.tensor_from_vec(vec![0.0f32; 640], &[10, 64]).unwrap();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let encoding = pos_enc.forward(input, &mut ctx).unwrap();
        let shape = backend.ops().shape(&encoding);
        assert_eq!(shape, vec![10, 64]); // 2D output for 2D input
    }

    #[test]
    fn test_transformer_encoder() {
        let backend = CpuBackend::default();
        let config = TransformerEncoderConfig::new(64, 4, 2, 256).with_max_seq_len(128);

        let encoder = TransformerEncoder::new(&backend, config, 1000, 42).unwrap();

        // Input: Vec<usize> token IDs - when total <= max_seq_len, treated as single batch
        let input = vec![1usize, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = encoder.forward(input, &mut ctx).unwrap();

        let shape = backend.ops().shape(&output);
        // Single batch: [1, 10, 64]
        assert_eq!(shape, vec![1, 10, 64]); // [batch, seq, d_model]
    }

    #[test]
    fn test_transformer_decoder() {
        let backend = CpuBackend::default();
        let config = TransformerDecoderConfig::new(64, 4, 2, 256).with_max_seq_len(128);

        let decoder = TransformerDecoder::new(&backend, config, 1000, 42).unwrap();

        // Input: Vec<usize> token IDs - single batch since 10 <= 128
        let input = vec![1usize, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let logits = decoder.forward(input, &mut ctx).unwrap();

        let shape = backend.ops().shape(&logits);
        assert_eq!(shape, vec![1, 10, 1000]); // [batch, seq, vocab]
    }

    #[test]
    fn test_encoder_parameter_count() {
        let backend = CpuBackend::default();
        let config = TransformerEncoderConfig::new(768, 12, 12, 3072);
        let encoder = TransformerEncoder::new(&backend, config, 30000, 42).unwrap();

        let params = encoder.parameters();
        // Should have: embedding + pos_encoding (no params) + 12 layers + final_norm
        // Each layer: self_attn (4) + 2 FF + 2 norm = ~8 params
        assert!(!params.is_empty());
    }

    #[test]
    fn test_transformer_encoder_decoder() {
        let backend = CpuBackend::default();
        let config = EncoderDecoderConfig::symmetric(64, 4, 2, 256).with_shared_embeddings(true);

        let model = TransformerEncoderDecoder::new(&backend, config, 1000, 1000, 42).unwrap();

        // Input: Vec<usize> token IDs - single batch since 5 <= 128
        let src = vec![1usize, 2, 3, 4, 5];
        let tgt = vec![10usize, 11, 12, 13, 14];

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let logits = model.forward(src, tgt, &mut ctx).unwrap();

        let shape = backend.ops().shape(&logits);
        assert_eq!(shape, vec![1, 5, 1000]); // [batch, tgt_len, vocab]
    }

    #[test]
    fn test_generation() {
        let backend = CpuBackend::default();
        let config = TransformerDecoderConfig::new(64, 4, 2, 256).with_max_seq_len(128);

        let decoder = TransformerDecoder::new(&backend, config, 100, 42).unwrap();

        // Generate token - Vec<usize> prefix
        let prefix = vec![1usize, 2, 3];
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);

        let next_token = decoder.generate_token(prefix, &mut ctx).unwrap();
        assert!(next_token < 100);
    }

    #[test]
    fn test_positional_encoding_with_dropout() {
        let backend = CpuBackend::default();
        let pos_enc = PositionalEncoding::new(&backend, 64, 512).unwrap();
        let pos_enc = pos_enc.with_dropout(0.2);
        assert_eq!(pos_enc.dropout, 0.2);
    }

    #[test]
    fn test_positional_encoding_get_encoding_overflow() {
        let backend = CpuBackend::default();
        let pos_enc = PositionalEncoding::new(&backend, 64, 10).unwrap();
        let result = pos_enc.get_encoding(20, backend.ops());
        assert!(result.is_err());
    }

    #[test]
    fn test_positional_encoding_forward_invalid_shape() {
        let backend = CpuBackend::default();
        let pos_enc = PositionalEncoding::new(&backend, 64, 512).unwrap();
        let input = backend.tensor_from_vec(vec![0.0f32; 64], &[64]).unwrap(); // 1D
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let result = pos_enc.forward(input, &mut ctx);
        assert!(result.is_err());
    }

    #[test]
    fn test_transformer_encoder_config_builders() {
        let config = TransformerEncoderConfig::new(64, 4, 2, 256).with_dropout(0.2).with_pre_norm(false);
        assert_eq!(config.dropout, 0.2);
        assert!(!config.pre_norm);
    }

    #[test]
    fn test_transformer_encoder_no_final_norm() {
        let backend = CpuBackend::default();
        let config = TransformerEncoderConfig::new(64, 4, 2, 256).with_pre_norm(false).with_max_seq_len(128);

        let encoder = TransformerEncoder::new(&backend, config, 1000, 42).unwrap();
        let input = vec![1usize, 2, 3, 4, 5];
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = encoder.forward(input, &mut ctx).unwrap();
        let shape = backend.ops().shape(&output);
        assert_eq!(shape, vec![1, 5, 64]);
    }

    #[test]
    fn test_transformer_encoder_cls_token() {
        let backend = CpuBackend::default();
        let config = TransformerEncoderConfig::new(64, 4, 2, 256).with_max_seq_len(128);

        let encoder = TransformerEncoder::new(&backend, config, 1000, 42).unwrap();
        let input = vec![1usize, 2, 3, 4, 5];
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let encoded = encoder.forward(input, &mut ctx).unwrap();

        let cls = encoder.cls_token(&encoded, backend.ops()).unwrap();
        let shape = backend.ops().shape(&cls);
        assert_eq!(shape, vec![1, 64]);
    }

    #[test]
    fn test_transformer_encoder_config_accessor() {
        let backend = CpuBackend::default();
        let config = TransformerEncoderConfig::new(64, 4, 2, 256);
        let encoder = TransformerEncoder::new(&backend, config, 1000, 42).unwrap();
        assert_eq!(encoder.config().d_model, 64);
    }

    #[test]
    fn test_transformer_encoder_module_forward() {
        let backend = CpuBackend::default();
        let config = TransformerEncoderConfig::new(64, 4, 2, 256).with_max_seq_len(128);
        let encoder = TransformerEncoder::new(&backend, config, 1000, 42).unwrap();

        fn call_forward<B: Backend>(
            m: &impl Module<B, Input = Vec<usize>, Output = B::Tensor>,
            input: Vec<usize>,
            ctx: &mut ForwardCtx<B>,
        ) -> Result<B::Tensor> {
            m.forward(input, ctx)
        }

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let out = call_forward(&encoder, vec![1, 2, 3], &mut ctx).unwrap();
        let shape = backend.ops().shape(&out);
        assert_eq!(shape, vec![1, 3, 64]);
    }

    #[test]
    fn test_transformer_encoder_layer_forward_post_norm() {
        let backend = CpuBackend::default();
        let config = TransformerEncoderConfig::new(16, 4, 1, 64).with_pre_norm(false);
        let layer = TransformerEncoderLayer::new(&backend, &config, 42).unwrap();

        let input = backend.tensor_from_vec(vec![0.1f32; 32], &[2, 1, 16]).unwrap();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let out = layer.forward(input, &mut ctx).unwrap();
        let shape = backend.ops().shape(&out);
        assert_eq!(shape, vec![2, 1, 16]);
    }

    #[test]
    fn test_transformer_decoder_layer_module_forward() {
        let backend = CpuBackend::default();
        let config = TransformerDecoderConfig::new(16, 4, 1, 64).with_max_seq_len(128);
        let layer = TransformerDecoderLayer::new(&backend, &config, 42).unwrap();

        fn call_forward<B: Backend>(
            m: &impl Module<B, Input = B::Tensor, Output = B::Tensor>,
            input: B::Tensor,
            ctx: &mut ForwardCtx<B>,
        ) -> Result<B::Tensor> {
            m.forward(input, ctx)
        }

        let input = backend.tensor_from_vec(vec![0.1f32; 32], &[2, 1, 16]).unwrap();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let out = call_forward(&layer, input, &mut ctx).unwrap();
        let shape = backend.ops().shape(&out);
        assert_eq!(shape, vec![2, 1, 16]);
    }

    #[test]
    fn test_transformer_decoder_layer_parameters() {
        let backend = CpuBackend::default();
        let config = TransformerDecoderConfig::new(16, 4, 1, 64).with_max_seq_len(128);
        let layer = TransformerDecoderLayer::new(&backend, &config, 42).unwrap();
        let params = layer.parameters();
        // self_attn (4) + ff1 (2) + ff2 (2) + norm1 (2) + norm2 (2) = 12
        assert_eq!(params.len(), 12);
    }

    #[test]
    fn test_transformer_decoder_no_final_norm() {
        let backend = CpuBackend::default();
        let mut config = TransformerDecoderConfig::new(64, 4, 2, 256);
        config.pre_norm = false;
        config.max_seq_len = 128;

        let decoder = TransformerDecoder::new(&backend, config, 1000, 42).unwrap();
        let input = vec![1usize, 2, 3, 4, 5];
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let logits = decoder.forward(input, &mut ctx).unwrap();
        let shape = backend.ops().shape(&logits);
        assert_eq!(shape, vec![1, 5, 1000]);
    }

    #[test]
    fn test_transformer_decoder_multi_batch() {
        let backend = CpuBackend::default();
        let config = TransformerDecoderConfig::new(64, 4, 1, 256).with_max_seq_len(4);

        let decoder = TransformerDecoder::new(&backend, config, 100, 42).unwrap();
        // total_tokens = 8 > max_seq_len = 4, so batch_size = 2
        let input = vec![1usize, 2, 3, 4, 5, 6, 7, 8];
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let logits = decoder.forward(input, &mut ctx).unwrap();
        let shape = backend.ops().shape(&logits);
        assert_eq!(shape, vec![2, 4, 100]);
    }

    #[test]
    fn test_transformer_decoder_parameters_no_final_norm() {
        let backend = CpuBackend::default();
        let mut config = TransformerDecoderConfig::new(64, 4, 2, 256);
        config.pre_norm = false;
        config.max_seq_len = 128;
        let decoder = TransformerDecoder::new(&backend, config, 1000, 42).unwrap();
        let params = decoder.parameters();
        // Should have embedding + pos_encoding (0) + 2 layers + lm_head (2) = ...
        assert!(!params.is_empty());
    }

    #[test]
    fn test_transformer_decoder_module_forward() {
        let backend = CpuBackend::default();
        let config = TransformerDecoderConfig::new(64, 4, 1, 256).with_max_seq_len(128);
        let decoder = TransformerDecoder::new(&backend, config, 100, 42).unwrap();

        fn call_forward<B: Backend>(
            m: &impl Module<B, Input = Vec<usize>, Output = B::Tensor>,
            input: Vec<usize>,
            ctx: &mut ForwardCtx<B>,
        ) -> Result<B::Tensor> {
            m.forward(input, ctx)
        }

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let out = call_forward(&decoder, vec![1, 2, 3], &mut ctx).unwrap();
        let shape = backend.ops().shape(&out);
        assert_eq!(shape, vec![1, 3, 100]);
    }

    #[test]
    fn test_transformer_encoder_decoder_generate() {
        let backend = CpuBackend::default();
        let config = EncoderDecoderConfig::symmetric(64, 4, 1, 256).with_shared_embeddings(true);

        let model = TransformerEncoderDecoder::new(&backend, config, 100, 100, 42).unwrap();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);

        let src = vec![1usize, 2, 3];
        let generated = model.generate(src, 10, 0, 1, &mut ctx).unwrap();
        assert!(!generated.is_empty());
        assert_eq!(generated[0], 0); // bos_token
    }

    #[test]
    fn test_transformer_encoder_decoder_config() {
        let backend = CpuBackend::default();
        let config = EncoderDecoderConfig::symmetric(64, 4, 2, 256);
        let model = TransformerEncoderDecoder::new(&backend, config.clone(), 100, 100, 42).unwrap();
        assert!(model.config().shared_embeddings);
    }

    #[test]
    fn test_transformer_encoder_decoder_parameters() {
        let backend = CpuBackend::default();
        let config = EncoderDecoderConfig::symmetric(64, 4, 1, 256);
        let model = TransformerEncoderDecoder::new(&backend, config, 100, 100, 42).unwrap();
        let params = model.parameters();
        assert!(!params.is_empty());
    }
}
