//! Reusable neural-network modules built over the core backend traits.
#![allow(dead_code)]
//!
//! # Module Overview
//!
//! ## Transformer Architectures
//!
//! Complete transformer implementations:
//! - [`TransformerEncoder`](transformer::TransformerEncoder): BERT-style bidirectional encoder
//! - [`TransformerDecoder`](transformer::TransformerDecoder): GPT-style autoregressive decoder
//! - [`TransformerEncoderDecoder`](transformer::TransformerEncoderDecoder): T5/BART-style seq2seq
//!
//! See `TRANSFORMERS.md` for detailed documentation and examples.

mod attention;
mod continuous_batching;
mod conv;
mod dropout;
mod embedding;
mod expert_choice;
mod kv_cache;
mod linear;
mod loss;
mod lstm;
mod moe;
mod multi_readout;
mod normalization;
mod quantization;
mod readout;
mod sequential;
mod shared_expert;
mod transformer;

pub use attention::{
    causal_mask, FlashAttention, MultiHeadAttention, SelfAttention, SelfAttentionConfig,
    TransformerEncoderBlock,
};
pub use attention::{AttentionMemoryStats, FlashAttentionConfig};
pub use continuous_batching::{
    Batch, Request, RequestPriority, RequestState, Sampler, Scheduler, SchedulerStats, SchedulingPolicy,
    ServingEngine,
};
pub use conv::{global_max_pool2d, max_pool2d, Conv2d, Conv2dConfig};
pub use dropout::{Dropout, DropoutConfig};
pub use embedding::{Embedding, EmbeddingConfig};
pub use expert_choice::{
    ExpertAssignments, ExpertChoiceConfig, ExpertChoiceRouter, ExpertChoiceStats, RoutingComparison,
    TokenAssignment,
};
pub use kv_cache::{
    BatchedCache, CacheConfig, CacheMemoryStats, CacheQuantization, KVCache, PagedCache, SlidingWindowCache,
};
pub use linear::{Linear, LinearBuilder, LinearConfig};
pub use loss::{BCEWithLogitsLoss, CrossEntropyLoss, MSELoss};
pub use lstm::{
    BidirectionalOutput, BidirectionalRnn, GruCell, GruConfig, GruState, LstmCell, LstmConfig, LstmState,
    RnnCell, StackedLstm,
};
pub use moe::{
    Expert, ExpertLayer, ExpertParallel, GatingOutput, MoEConfig, MoEOutput, MoEStats, TopKGating,
};
pub use multi_readout::{BinaryPrediction, BinaryReadout, LabelBinaryPrediction, MultiReadout};
pub use normalization::{BatchNorm, BatchNormConfig, LayerNorm, LayerNormConfig};
pub use quantization::{
    quantize_model, DynamicQuantizer, GPTQLinear, QATTrainer, QuantConfig, QuantParams, QuantizationScheme,
    QuantizationStats, QuantizedLinear,
};
pub use readout::{Readout, ReadoutConfig};
pub use sequential::{chain, Sequential2};
pub use shared_expert::{
    HybridRouting, SharedAndRoutedConfig, SharedExpertConfig, SharedExpertLayer, SharedExpertStats,
};
pub use transformer::{
    EncoderDecoderConfig, PositionalEncoding, TransformerDecoder, TransformerDecoderConfig,
    TransformerDecoderLayer, TransformerEncoder, TransformerEncoderConfig, TransformerEncoderDecoder,
    TransformerEncoderLayer,
};
