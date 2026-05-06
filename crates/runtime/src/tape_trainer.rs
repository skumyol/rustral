//! Generic training loop for Tape-aware models.
//!
//! This is a small, correctness-first trainer meant to replace ad-hoc training loops in examples
//! once core layers implement tape-forward.

use std::collections::HashMap;
use std::time::Instant;

use rustral_autodiff::{GradExtFromStore, Tape, TensorId};
use rustral_core::{Backend, ForwardCtx, Mode, NamedParameters, Parameter, ParameterId, Result};
use rustral_optim::{Gradient, Optimizer};

use crate::EpochStats;

/// Basic training configuration for [`TapeTrainer`].
#[derive(Clone, Debug)]
pub struct TapeTrainerConfig {
    pub epochs: usize,
    pub learning_rate: f32,
    pub batch_size: usize,
}

impl Default for TapeTrainerConfig {
    fn default() -> Self {
        Self { epochs: 1, learning_rate: 1e-3, batch_size: 1 }
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

    /// Train a model by consuming `NamedParameters` directly (no caller-managed parameter slices).
    ///
    /// This is the first step toward the north-star API. It remains intentionally minimal:
    /// callers provide a closure that builds the forward pass and returns a scalar loss `TensorId`.
    ///
    /// Design note: we currently clone parameters into a temporary `Vec<Parameter<B>>` to reuse the
    /// existing `Optimizer::step` API, then write updated tensors back into the model by `ParameterId`.
    pub fn train_model<M, D, F>(
        &mut self,
        backend: &B,
        model: &mut M,
        data: &[D],
        mut forward_and_loss: F,
    ) -> anyhow::Result<Vec<EpochStats>>
    where
        M: NamedParameters<B>,
        F: FnMut(&mut M, &D, &mut Tape<B>, &mut ForwardCtx<B>) -> Result<TensorId>,
    {
        let ops = backend.ops();
        if self.config.batch_size == 0 {
            anyhow::bail!("batch_size must be non-zero");
        }

        let mut stats = Vec::with_capacity(self.config.epochs);
        for epoch in 0..self.config.epochs {
            let start = Instant::now();
            let mut losses = Vec::new();

            for batch in data.chunks(self.config.batch_size) {
                let mut ctx = ForwardCtx::new(backend, Mode::Train);
                let mut tape = Tape::<B>::new();

                // Register parameters for gradient extraction.
                model.visit_parameters(&mut |_name, p| {
                    tape.watch_parameter(p);
                });

                let mut batch_loss: Option<TensorId> = None;
                for sample in batch {
                    let l = forward_and_loss(model, sample, &mut tape, &mut ctx)?;
                    batch_loss = Some(match batch_loss {
                        Some(acc) => tape.add(acc, l, &mut ctx)?,
                        None => l,
                    });
                }
                let Some(batch_loss) = batch_loss else { continue };

                let inv_batch = 1.0 / batch.len() as f32;
                let batch_loss = tape.mul_scalar(batch_loss, inv_batch, &mut ctx)?;

                let loss_tensor = tape
                    .value(batch_loss)
                    .ok_or_else(|| anyhow::anyhow!("internal error: loss tensor missing from tape"))?;
                let loss_value = ops.tensor_to_vec(loss_tensor)?.first().copied().unwrap_or(0.0);
                losses.push(loss_value);

                let param_map = tape.param_map().clone();
                let make_ones = |data: Vec<f32>, shape: &[usize]| ops.tensor_from_vec(data, shape);
                let grads_store = tape.backward(batch_loss, make_ones, ops)?;

                // Collect gradients.
                let mut grads = Vec::new();
                model.visit_parameters(&mut |_name, p| {
                    if let Some(g) = p.gradient_from_store(&grads_store, &param_map) {
                        grads.push(Gradient { param_id: p.id(), tensor: g.clone() });
                    }
                });

                // Clone parameters into a temporary vector for the optimizer step.
                let mut params_vec: Vec<Parameter<B>> = Vec::new();
                model.visit_parameters(&mut |_name, p| params_vec.push(p.clone()));

                self.optimizer
                    .step(&mut params_vec, &grads, &mut ctx)
                    .map_err(|e| anyhow::anyhow!("{:?}", e))?;

                // Write back updated tensors by id.
                let mut updated: HashMap<ParameterId, B::Tensor> = HashMap::with_capacity(params_vec.len());
                for p in params_vec {
                    updated.insert(p.id(), p.into_tensor());
                }

                model.visit_parameters_mut(&mut |_name, p| {
                    if let Some(t) = updated.get(&p.id()) {
                        *p = p.clone().with_tensor(t.clone());
                    }
                });
            }

            let mean_loss =
                if losses.is_empty() { 0.0 } else { losses.iter().sum::<f32>() / losses.len() as f32 };
            stats.push(EpochStats { epoch, examples: data.len(), mean_loss, elapsed: start.elapsed() });
        }

        Ok(stats)
    }
}
