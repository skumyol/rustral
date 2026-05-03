//! Reusable neural-network modules built over the core backend traits.

mod attention;
mod conv;
mod dropout;
mod embedding;
mod linear;
mod loss;
mod lstm;
mod multi_readout;
mod normalization;
mod readout;
mod sequential;

pub use attention::{causal_mask, MultiHeadAttention, SelfAttention, SelfAttentionConfig, TransformerEncoderBlock};
pub use conv::{Conv2d, Conv2dConfig, global_max_pool2d, max_pool2d};
pub use dropout::{Dropout, DropoutConfig};
pub use embedding::{Embedding, EmbeddingConfig};
pub use linear::{Linear, LinearBuilder, LinearConfig};
pub use loss::{MSELoss, CrossEntropyLoss, BCEWithLogitsLoss};
pub use lstm::{BidirectionalOutput, BidirectionalRnn, GruCell, GruConfig, GruState, LstmCell, LstmConfig, LstmState, RnnCell, StackedLstm};
pub use multi_readout::{BinaryPrediction, BinaryReadout, LabelBinaryPrediction, MultiReadout};
pub use normalization::{BatchNorm, BatchNormConfig, LayerNorm, LayerNormConfig};
pub use readout::{Readout, ReadoutConfig};
pub use sequential::{Sequential2, chain};
