//! Reusable neural-network modules built over the core backend traits.
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
mod shared_expert;
mod multi_readout;
mod normalization;
mod quantization;
mod readout;
mod sequential;
mod transformer;

pub use attention::{causal_mask, FlashAttention, MultiHeadAttention, SelfAttention, SelfAttentionConfig, TransformerEncoderBlock};
pub use attention::{AttentionMemoryStats, FlashAttentionConfig};
pub use transformer::{
    EncoderDecoderConfig, PositionalEncoding, TransformerDecoder,
    TransformerDecoderConfig, TransformerDecoderLayer, TransformerEncoder,
    TransformerEncoderConfig, TransformerEncoderDecoder, TransformerEncoderLayer,
};
pub use continuous_batching::{
    Batch, Request, RequestPriority, RequestState, Sampler, Scheduler,
    SchedulingPolicy, SchedulerStats, ServingEngine,
};
pub use conv::{Conv2d, Conv2dConfig, global_max_pool2d, max_pool2d};
pub use dropout::{Dropout, DropoutConfig};
pub use embedding::{Embedding, EmbeddingConfig};
pub use kv_cache::{
    BatchedCache, CacheConfig, CacheMemoryStats, CacheQuantization,
    KVCache, PagedCache, SlidingWindowCache,
};
pub use linear::{Linear, LinearBuilder, LinearConfig};
pub use loss::{MSELoss, CrossEntropyLoss, BCEWithLogitsLoss};
pub use lstm::{BidirectionalOutput, BidirectionalRnn, GruCell, GruConfig, GruState, LstmCell, LstmConfig, LstmState, RnnCell, StackedLstm};
pub use expert_choice::{
    ExpertChoiceConfig, ExpertChoiceRouter, ExpertChoiceStats,
    ExpertAssignments, TokenAssignment, RoutingComparison,
};
pub use moe::{Expert, ExpertLayer, ExpertParallel, GatingOutput, MoEConfig, MoEOutput, MoEStats, TopKGating};
pub use shared_expert::{
    SharedExpertLayer, SharedExpertConfig, SharedExpertStats,
    HybridRouting, SharedAndRoutedConfig,
};
pub use multi_readout::{BinaryPrediction, BinaryReadout, LabelBinaryPrediction, MultiReadout};
pub use normalization::{BatchNorm, BatchNormConfig, LayerNorm, LayerNormConfig};
pub use quantization::{
    DynamicQuantizer, GPTQLinear, QATTrainer, QuantConfig, QuantizationScheme,
    QuantizationStats, QuantizedLinear, QuantParams, quantize_model,
};
pub use readout::{Readout, ReadoutConfig};
pub use sequential::{Sequential2, chain};
