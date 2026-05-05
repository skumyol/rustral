//! Tape-aware forward helpers for `rustral-nn` layers.
//!
//! This is feature-gated behind `rustral-nn/autodiff` to avoid introducing an autodiff dependency
//! for pure inference users.

use rustral_autodiff::{Tape, TensorId};
use rustral_core::{Backend, ForwardCtx, Result};

use crate::{Embedding, Linear};

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

        if let Some(bias) = self.bias() {
            let b_id = tape.watch_parameter(bias);
            tape.add_row_vector_tape(out, b_id, ctx)
        } else {
            Ok(out)
        }
    }
}

impl<B: Backend> TapeModule<B> for Embedding<B>
where
    B::Tensor: Clone,
{
    fn forward_tape(&self, input: TensorId, tape: &mut Tape<B>, ctx: &mut ForwardCtx<B>) -> Result<TensorId> {
        tape.gather_rows_tape(self.table(), input, ctx)
    }
}
