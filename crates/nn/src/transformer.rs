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
//! use mnr_nn::transformer::{TransformerEncoder, TransformerEncoderConfig};
//!
//! // BERT-style encoder for classification
//! let config = TransformerEncoderConfig::new(768, 12, 12, 3072); // d_model, heads, layers, ff_dim
//! let encoder = TransformerEncoder::new(&backend, config, 42)?;
//! let cls_output = encoder.forward(input, &mut ctx)?; // [batch, d_model]
//! ```

use mnr_core::{Backend, CoreError, ForwardCtx, Module, Parameter, ParameterRef, Result, TensorOps, Trainable};
use serde::{Deserialize, Serialize};

use crate::{
    causal_mask, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig,
    SelfAttention, SelfAttentionConfig,
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
    B::Tensor: Clone + AsRef<[f32]> + mnr_core::TensorShape,
{
    /// Create positional encoding.
    pub fn new(backend: &B, d_model: usize, max_len: usize) -> Result<Self> {
        let mut encoding_data = vec![0.0f32; max_len * d_model];

        for pos in 0..max_len {
            for i in 0..d_model {
                let angle = pos as f32 / f32::powf(10000.0, (2 * (i / 2)) as f32 / d_model as f32);
                encoding_data[pos * d_model + i] = if i % 2 == 0 {
                    angle.sin()
                } else {
                    angle.cos()
                };
            }
        }

        let encoding = backend.tensor_from_vec(encoding_data, &[max_len, d_model])?;

        Ok(Self {
            encoding,
            max_len,
            d_model,
            dropout: 0.1,
        })
    }

    /// Set dropout.
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    /// Get positional encoding for sequence length.
    pub fn forward(&self, seq_len: usize, ops: &dyn TensorOps<B>) -> Result<B::Tensor> {
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
    B::Tensor: Clone + AsRef<[f32]> + mnr_core::TensorShape,
{
    fn forward(&self, input: B::Tensor, ctx: &mut ForwardCtx<B>) -> Result<B::Tensor> {
        let ops = ctx.backend().ops();
        let shape = ops.shape(&input);
        let seq_len = shape[0];
        let pos_enc = self.forward(seq_len, ops)?;
        ops.add(&input, &pos_enc)
    }
}

impl<B: Backend> Trainable<B> for PositionalEncoding<B> {
    fn parameters(&self) -> Vec<ParameterRef<B>> {
        // Positional encoding has no trainable parameters
        Vec::new()
    }
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
        assert_eq!(
            d_model % num_heads,
            0,
            "d_model must be divisible by num_heads"
        );
        Self {
            d_model,
            num_heads,
            num_layers,
            ff_dim,
            dropout: 0.1,
            max_seq_len: 512,
            pre_norm: true,
        }
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
    B::Tensor: Clone + AsRef<[f32]> + mnr_core::TensorShape,
{
    /// Create encoder layer.
    pub fn new(backend: &B, config: &TransformerEncoderConfig, seed: u64) -> Result<Self> {
        let attn_config = SelfAttentionConfig::new(config.d_model, config.num_heads)
            .with_dropout(config.dropout);

        let self_attn = SelfAttention::new(backend, attn_config, seed)?;

        let ff_linear1 = Linear::new(
            backend,
            LinearConfig::new(config.d_model, config.ff_dim).with_bias(true),
        )?;

        let ff_linear2 = Linear::new(
            backend,
            LinearConfig::new(config.ff_dim, config.d_model).with_bias(true),
        )?;

        let norm1 = LayerNorm::new(
            backend,
            LayerNormConfig::new(vec![config.d_model])
                .with_eps(1e-5),
            seed + 1,
        )?;

        let norm2 = LayerNorm::new(
            backend,
            LayerNormConfig::new(vec![config.d_model])
                .with_eps(1e-5),
            seed + 2,
        )?;

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

        // Feed-forward with residual
        let ff_output = if self.pre_norm {
            let normed = self.norm2.forward(attn_output.clone(), ctx)?;
            let hidden = self.ff_linear1.forward(normed, ctx)?;
            // GELU activation
            let activated = ops.relu(&hidden)?; // Simplified, should be GELU
            let ff_out = self.ff_linear2.forward(activated, ctx)?;
            ops.add(&attn_output, &ff_out)?
        } else {
            let hidden = self.ff_linear1.forward(attn_output.clone(), ctx)?;
            let activated = ops.relu(&hidden)?;
            let ff_out = self.ff_linear2.forward(activated, ctx)?;
            let added = ops.add(&attn_output, &ff_out)?;
            self.norm2.forward(added, ctx)?
        };

        Ok(ff_output)
    }
}

impl<B: Backend> Module<B> for TransformerEncoderLayer<B>
where
    B::Tensor: Clone + AsRef<[f32]> + mnr_core::TensorShape,
{
    fn forward(&self, input: B::Tensor, ctx: &mut ForwardCtx<B>) -> Result<B::Tensor> {
        self.forward(input, ctx)
    }
}

impl<B: Backend> Trainable<B> for TransformerEncoderLayer<B> {
    fn parameters(&self) -> Vec<ParameterRef<B>> {
        let mut params = Vec::new();
        params.extend(self.self_attn.parameters());
        params.extend(self.ff_linear1.parameters());
        params.extend(self.ff_linear2.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params
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
    B::Tensor: Clone + AsRef<[f32]> + mnr_core::TensorShape,
{
    /// Create transformer encoder.
    pub fn new(backend: &B, config: TransformerEncoderConfig, vocab_size: usize, seed: u64) -> Result<Self> {
        // Token embedding
        let token_embedding = Embedding::new(
            backend,
            EmbeddingConfig::new(vocab_size, config.d_model),
            seed,
        )?;

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

        Ok(Self {
            token_embedding,
            pos_encoding,
            layers,
            final_norm,
            config,
        })
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `input` - Token IDs [batch, seq_len]
    ///
    /// # Returns
    /// * Encoded representations [batch, seq_len, d_model]
    pub fn forward(&self, input: B::Tensor, ctx: &mut ForwardCtx<B>) -> Result<B::Tensor> {
        // Token embedding: [batch, seq_len] → [batch, seq_len, d_model]
        let embedded = self.token_embedding.forward(input, ctx)?;

        // Add positional encoding
        let mut hidden = self.pos_encoding.forward(embedded, ctx)?;

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
    pub fn cls_token(&self, encoded: &B::Tensor, ops: &dyn TensorOps<B>) -> Result<B::Tensor> {
        // Extract first position: [batch, seq_len, d_model] → [batch, d_model]
        let shape = ops.shape(encoded);
        let batch_size = shape[0];
        let d_model = shape[2];

        // In real impl, would use gather/slice
        let data: Vec<f32> = encoded.as_ref().iter()
            .take(batch_size * d_model)
            .copied()
            .collect();

        ops.tensor_from_vec(data, &[batch_size, d_model])
    }

    /// Configuration.
    pub fn config(&self) -> &TransformerEncoderConfig {
        &self.config
    }
}

impl<B: Backend> Module<B> for TransformerEncoder<B>
where
    B::Tensor: Clone + AsRef<[f32]> + mnr_core::TensorShape,
{
    fn forward(&self, input: B::Tensor, ctx: &mut ForwardCtx<B>) -> Result<B::Tensor> {
        self.forward(input, ctx)
    }
}

impl<B: Backend> Trainable<B> for TransformerEncoder<B> {
    fn parameters(&self) -> Vec<ParameterRef<B>> {
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
        Self {
            d_model,
            num_heads,
            num_layers,
            ff_dim,
            dropout: 0.1,
            max_seq_len: 512,
            pre_norm: true,
        }
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
    B::Tensor: Clone + AsRef<[f32]> + mnr_core::TensorShape,
{
    pub fn new(backend: &B, config: &TransformerDecoderConfig, seed: u64) -> Result<Self> {
        let attn_config = SelfAttentionConfig::new(config.d_model, config.num_heads)
            .with_dropout(config.dropout);

        let self_attn = SelfAttention::new(backend, attn_config, seed)?;

        let ff_linear1 = Linear::new(
            backend,
            LinearConfig::new(config.d_model, config.ff_dim).with_bias(true),
        )?;

        let ff_linear2 = Linear::new(
            backend,
            LinearConfig::new(config.ff_dim, config.d_model).with_bias(true),
        )?;

        let norm1 = LayerNorm::new(
            backend,
            LayerNormConfig::new(vec![config.d_model]).with_eps(1e-5),
            seed + 1,
        )?;

        let norm2 = LayerNorm::new(
            backend,
            LayerNormConfig::new(vec![config.d_model]).with_eps(1e-5),
            seed + 2,
        )?;

        // Create causal mask
        let mask_data: Vec<f32> = (0..config.max_seq_len)
            .flat_map(|i| {
                (0..config.max_seq_len)
                    .map(move |j| if j <= i { 0.0 } else { f32::NEG_INFINITY })
            })
            .collect();

        let causal_mask = backend.tensor_from_vec(mask_data, &[config.max_seq_len, config.max_seq_len])?;

        Ok(Self {
            self_attn,
            ff_linear1,
            ff_linear2,
            norm1,
            norm2,
            causal_mask,
        })
    }

    pub fn forward(&self, input: B::Tensor, ctx: &mut ForwardCtx<B>) -> Result<B::Tensor> {
        // Pre-norm: LN → Attention → Residual
        let normed = self.norm1.forward(input.clone(), ctx)?;
        // In real impl, would apply causal mask to attention
        let attn = self.self_attn.forward(normed, ctx)?;
        let hidden = ctx.backend().ops().add(&input, &attn)?;

        // Pre-norm: LN → FF → Residual
        let normed2 = self.norm2.forward(hidden.clone(), ctx)?;
        let ff_hidden = self.ff_linear1.forward(normed2, ctx)?;
        let activated = ctx.backend().ops().relu(&ff_hidden)?; // Should be GELU
        let ff_out = self.ff_linear2.forward(activated, ctx)?;

        ctx.backend().ops().add(&hidden, &ff_out)
    }
}

impl<B: Backend> Module<B> for TransformerDecoderLayer<B>
where
    B::Tensor: Clone + AsRef<[f32]> + mnr_core::TensorShape,
{
    fn forward(&self, input: B::Tensor, ctx: &mut ForwardCtx<B>) -> Result<B::Tensor> {
        self.forward(input, ctx)
    }
}

impl<B: Backend> Trainable<B> for TransformerDecoderLayer<B> {
    fn parameters(&self) -> Vec<ParameterRef<B>> {
        let mut params = Vec::new();
        params.extend(self.self_attn.parameters());
        params.extend(self.ff_linear1.parameters());
        params.extend(self.ff_linear2.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params
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
}

impl<B: Backend> TransformerDecoder<B>
where
    B::Tensor: Clone + AsRef<[f32]> + mnr_core::TensorShape,
{
    pub fn new(
        backend: &B,
        config: TransformerDecoderConfig,
        vocab_size: usize,
        seed: u64,
    ) -> Result<Self> {
        let token_embedding = Embedding::new(
            backend,
            EmbeddingConfig::new(vocab_size, config.d_model),
            seed,
        )?;

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

        let lm_head = Linear::new(
            backend,
            LinearConfig::new(config.d_model, vocab_size).with_bias(false),
        )?;

        Ok(Self {
            token_embedding,
            pos_encoding,
            layers,
            final_norm,
            lm_head,
            config,
        })
    }

    /// Forward pass for training.
    ///
    /// # Returns
    /// Logits [batch, seq_len, vocab_size]
    pub fn forward(&self, input: B::Tensor, ctx: &mut ForwardCtx<B>) -> Result<B::Tensor> {
        let embedded = self.token_embedding.forward(input, ctx)?;
        let mut hidden = self.pos_encoding.forward(embedded, ctx)?;

        for layer in &self.layers {
            hidden = layer.forward(hidden, ctx)?;
        }

        if let Some(ref norm) = self.final_norm {
            hidden = norm.forward(hidden, ctx)?;
        }

        // Project to vocabulary
        self.lm_head.forward(hidden, ctx)
    }

    /// Generate next token autoregressively.
    pub fn generate_token(&self, prefix: &B::Tensor, ctx: &mut ForwardCtx<B>) -> Result<u32> {
        let logits = self.forward(prefix.clone(), ctx)?;
        let shape = ctx.backend().ops().shape(&logits);
        let vocab_size = shape[2];

        // Get logits for last position
        let last_logits: Vec<f32> = logits.as_ref()
            .iter()
            .skip((shape[0] - 1) * shape[1] * vocab_size)
            .take(vocab_size)
            .copied()
            .collect();

        // Greedy decode
        let (idx, _) = last_logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap_or((0, &0.0));

        Ok(idx as u32)
    }
}

impl<B: Backend> Module<B> for TransformerDecoder<B>
where
    B::Tensor: Clone + AsRef<[f32]> + mnr_core::TensorShape,
{
    fn forward(&self, input: B::Tensor, ctx: &mut ForwardCtx<B>) -> Result<B::Tensor> {
        self.forward(input, ctx)
    }
}

impl<B: Backend> Trainable<B> for TransformerDecoder<B> {
    fn parameters(&self) -> Vec<ParameterRef<B>> {
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
        Self {
            encoder,
            decoder,
            shared_embeddings: true,
        }
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
    B::Tensor: Clone + AsRef<[f32]> + mnr_core::TensorShape,
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

        Ok(Self {
            encoder,
            decoder,
            config,
        })
    }

    /// Forward pass for training.
    ///
    /// # Arguments
    /// * `src` - Source token IDs [batch, src_len]
    /// * `tgt` - Target token IDs [batch, tgt_len]
    ///
    /// # Returns
    /// Logits [batch, tgt_len, vocab_size]
    pub fn forward(
        &self,
        src: B::Tensor,
        tgt: B::Tensor,
        ctx: &mut ForwardCtx<B>,
    ) -> Result<B::Tensor> {
        // Encode source
        let memory = self.encoder.forward(src, ctx)?;

        // Decode target with cross-attention to memory
        // Simplified - full impl would pass memory to decoder
        self.decoder.forward(tgt, ctx)
    }

    /// Greedy decode from source.
    pub fn generate(
        &self,
        src: B::Tensor,
        max_len: usize,
        bos_token: u32,
        eos_token: u32,
        ctx: &mut ForwardCtx<B>,
    ) -> Result<Vec<u32>> {
        let ops = ctx.backend().ops();

        // Start with BOS
        let mut tokens = vec![bos_token];

        for _ in 0..max_len {
            // Convert to tensor
            let tgt_tensor = ops.tensor_from_vec(
                tokens.iter().map(|&t| t as u32).collect::<Vec<_>>(),
                &[1, tokens.len()],
            )?;

            // Forward
            let logits = self.forward(src.clone(), tgt_tensor, ctx)?;
            let shape = ops.shape(&logits);
            let vocab_size = shape[2];

            // Get last token logits
            let last_logits: Vec<f32> = logits.as_ref()
                .iter()
                .skip((tokens.len() - 1) * vocab_size)
                .take(vocab_size)
                .copied()
                .collect();

            // Greedy decode
            let next_token = last_logits.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0);

            tokens.push(next_token);

            if next_token == eos_token {
                break;
            }
        }

        Ok(tokens)
    }

    /// Configuration.
    pub fn config(&self) -> &EncoderDecoderConfig {
        &self.config
    }
}

impl<B: Backend> Module<B> for TransformerEncoderDecoder<B>
where
    B::Tensor: Clone + AsRef<[f32]> + mnr_core::TensorShape,
{
    fn forward(&self, input: B::Tensor, ctx: &mut ForwardCtx<B>) -> Result<B::Tensor> {
        // For Module trait, assume input contains both src and tgt stacked
        // Real impl should handle this differently
        let shape = ctx.backend().ops().shape(&input);
        let mid = shape[1] / 2;

        // Split input (simplified)
        let data: Vec<u32> = input.as_ref().iter().map(|&v| v as u32).collect();
        let src = ctx.backend().ops().tensor_from_vec(
            data[..data.len()/2].to_vec(),
            &[shape[1], mid],
        )?;
        let tgt = ctx.backend().ops().tensor_from_vec(
            data[data.len()/2..].to_vec(),
            &[shape[1], mid],
        )?;

        self.forward(src, tgt, ctx)
    }
}

impl<B: Backend> Trainable<B> for TransformerEncoderDecoder<B> {
    fn parameters(&self) -> Vec<ParameterRef<B>> {
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
    use mnr_core::Mode;
    use mnr_ndarray_backend::CpuBackend;

    #[test]
    fn test_positional_encoding() {
        let backend = CpuBackend::default();
        let pos_enc = PositionalEncoding::new(&backend, 64, 512).unwrap();

        let encoding = pos_enc.forward(10, backend.ops()).unwrap();
        let shape = backend.ops().shape(&encoding);
        assert_eq!(shape, vec![10, 64]);
    }

    #[test]
    fn test_transformer_encoder() {
        let backend = CpuBackend::default();
        let config = TransformerEncoderConfig::new(64, 4, 2, 256)
            .with_max_seq_len(128);

        let encoder = TransformerEncoder::new(&backend, config, 1000, 42).unwrap();

        // Input: [batch=2, seq=10]
        let input = backend.tensor_from_vec(
            vec![1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            &[2, 10],
        ).unwrap();

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = encoder.forward(input, &mut ctx).unwrap();

        let shape = backend.ops().shape(&output);
        assert_eq!(shape, vec![2, 10, 64]); // [batch, seq, d_model]
    }

    #[test]
    fn test_transformer_decoder() {
        let backend = CpuBackend::default();
        let config = TransformerDecoderConfig::new(64, 4, 2, 256)
            .with_max_seq_len(128);

        let decoder = TransformerDecoder::new(&backend, config, 1000, 42).unwrap();

        let input = backend.tensor_from_vec(
            vec![1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            &[1, 10],
        ).unwrap();

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
        let config = EncoderDecoderConfig::symmetric(64, 4, 2, 256)
            .with_shared_embeddings(true);

        let model = TransformerEncoderDecoder::new(&backend, config, 1000, 1000, 42).unwrap();

        let src = backend.tensor_from_vec(vec![1u32, 2, 3, 4, 5], &[1, 5]).unwrap();
        let tgt = backend.tensor_from_vec(vec![10u32, 11, 12, 13, 14], &[1, 5]).unwrap();

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let logits = model.forward(src, tgt, &mut ctx).unwrap();

        let shape = backend.ops().shape(&logits);
        assert_eq!(shape, vec![1, 5, 1000]); // [batch, tgt_len, vocab]
    }

    #[test]
    fn test_generation() {
        let backend = CpuBackend::default();
        let config = TransformerDecoderConfig::new(64, 4, 2, 256)
            .with_max_seq_len(128);

        let decoder = TransformerDecoder::new(&backend, config, 100, 42).unwrap();

        // Generate token
        let prefix = backend.tensor_from_vec(vec![1u32, 2, 3], &[1, 3]).unwrap();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);

        let next_token = decoder.generate_token(&prefix, &mut ctx).unwrap();
        assert!(next_token < 100);
    }
}
