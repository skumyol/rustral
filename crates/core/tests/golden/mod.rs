//! Golden tests for cross-backend numerical comparisons.
//!
//! These tests compare forward and backward passes across different backends
//! to ensure numerical stability and correctness of optimizations.

mod golden_transformer_layer;
mod golden_attention;
mod golden_mlp;

pub use golden_transformer_layer::*;
pub use golden_attention::*;
pub use golden_mlp::*;
