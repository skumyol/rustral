//! Generic training loop for Tape-aware models.
//!
//! This is a small, correctness-first trainer meant to replace ad-hoc training loops in examples
//! once core layers implement tape-forward.

use rustral_autodiff::{GradExtFromStore, Tape, TensorId};
use rustral_core::{Backend, ForwardCtx, Mode, Parameter, Result};
use rustral_optim::{Gradient, Optimizer};

/// Basic training configuration for [`TapeTrainer`].
#[derive(Clone, Debug)]
pub struct TapeTrainerConfig {
    pub epochs: usize,
    pub learning_rate: f32,
}

impl Default for TapeTrainerConfig {
    fn default() -> Self {
        Self { epochs: 1, learning_rate: 1e-3 }
    }
}

/// A generic trainer operating on a single process (no distribution).
pub struct TapeTrainer<B: Backend, O: Optimizer<B>> {
    pub config: TapeTrainerConfig,
    pub optimizer: O,
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend, O: Optimizer<B>> TapeTrainer<B, O>
where
    B::Tensor: Clone,
{
    pub fn new(config: TapeTrainerConfig, optimizer: O) -> Self {
        Self { config, optimizer, _phantom: std::marker::PhantomData }
    }

    /// Train a single-parameter model where the model forward is provided as a closure that writes into the tape.
    ///
    /// This is intentionally minimal and will be generalized as more layers become tape-aware.
    pub fn train<D, F>(
        &mut self,
        backend: &B,
        params: &mut [Parameter<B>],
        data: &[D],
        mut forward_and_loss: F,
    ) -> anyhow::Result<()>
    where
        F: FnMut(&D, &mut Tape<B>, &mut ForwardCtx<B>) -> Result<TensorId>,
    {
        let ops = backend.ops();
        for _epoch in 0..self.config.epochs {
            for sample in data {
                let mut ctx = ForwardCtx::new(backend, Mode::Train);
                let mut tape = Tape::<B>::new();

                // Register parameters for gradient extraction.
                for p in params.iter() {
                    tape.watch_parameter(p);
                }

                let loss_id = forward_and_loss(sample, &mut tape, &mut ctx)?;

                let param_map = tape.param_map().clone();
                let make_ones = |data: Vec<f32>, shape: &[usize]| ops.tensor_from_vec(data, shape);
                let grads_store = tape.backward(loss_id, make_ones, ops)?;

                let mut grads = Vec::with_capacity(params.len());
                for p in params.iter() {
                    let Some(g) = p.gradient_from_store(&grads_store, &param_map) else {
                        continue;
                    };
                    grads.push(Gradient { param_id: p.id(), tensor: g.clone() });
                }

                self.optimizer.step(params, &grads, &mut ctx).map_err(|e| anyhow::anyhow!("{:?}", e))?;
            }
        }
        Ok(())
    }
}
