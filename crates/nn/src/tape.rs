//! Tape-aware forward helpers for `rustral-nn` layers.
//!
//! This is feature-gated behind `rustral-nn/autodiff` to avoid introducing an autodiff dependency
//! for pure inference users.

use rustral_autodiff::{Tape, TensorId};
use rustral_core::{Backend, ForwardCtx, Result};

use crate::Linear;

/// A module that can execute its forward pass while recording into an autodiff [`Tape`].
///
/// The output is a `TensorId` inside the tape.
pub trait TapeModule<B: Backend> {
    fn forward_tape(&self, input: TensorId, tape: &mut Tape<B>, ctx: &mut ForwardCtx<B>) -> Result<TensorId>;
}

impl<B: Backend> TapeModule<B> for Linear<B>
where
    B::Tensor: Clone,
{
    fn forward_tape(&self, input: TensorId, tape: &mut Tape<B>, ctx: &mut ForwardCtx<B>) -> Result<TensorId> {
        let w_id = tape.watch_parameter(self.weight());
        let w_t = tape.transpose_tape(w_id, ctx)?;
        let out = tape.matmul(input, w_t, ctx)?;

        // Bias support will require an `add_row_vector_tape` op (needs batch-wise reduce for bias grad).
        // For now, disallow bias to avoid silently incorrect gradients.
        if self.config().bias {
            return Err(rustral_core::CoreError::InvalidArgument(
                "Linear::forward_tape currently requires bias=false".to_string(),
            ));
        }

        Ok(out)
    }
}

