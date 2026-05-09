//! Generic training loop for Tape-aware models.
//!
//! This is a small, correctness-first trainer meant to replace ad-hoc training loops in examples
//! once core layers implement tape-forward.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rustral_autodiff::{GradExtFromStore, Tape, TensorId};
use rustral_core::{Backend, ForwardCtx, Mode, NamedParameters, OperationProfiler, Parameter, ParameterId, Result, TensorPool};
use rustral_optim::{Gradient, Optimizer};

use crate::EpochStats;

fn operation_profiler_from_env() -> Option<Arc<Mutex<OperationProfiler>>> {
    if std::env::var("RUSTRAL_PROFILE").as_deref() != Ok("1") {
        return None;
    }
    let ci = std::env::var("CI").is_ok() || std::env::var("RUSTRAL_PROFILE_CI").as_deref() == Ok("1");
    let p = if ci { OperationProfiler::new_ci_safe() } else { OperationProfiler::new() };
    Some(Arc::new(Mutex::new(p)))
}

fn finish_training_profiler(profiler: &Arc<Mutex<OperationProfiler>>) {
    if let Ok(p) = profiler.lock() {
        p.print_report();
        if let Ok(path) = std::env::var("RUSTRAL_PROFILE_EXPORT_JSON") {
            let limit = std::env::var("RUSTRAL_PROFILE_SNAPSHOT_LIMIT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(64usize);
            if let Err(e) = p.export_regression_report(&path, limit) {
                eprintln!("RUSTRAL_PROFILE_EXPORT_JSON: failed to write {path}: {e}");
            }
        }
    }
}

/// A supervised model that can run forward + loss on a tape.
///
/// This is the building block for a high-level `fit(...)` API: the trainer owns the tape,
/// batching, backprop, optimizer step, and metrics.
pub trait SupervisedTapeModel<B: Backend, X, Y>: NamedParameters<B> {
    fn forward_tape(&mut self, input: X, tape: &mut Tape<B>, ctx: &mut ForwardCtx<B>) -> Result<TensorId>;

    fn loss_tape(
        &mut self,
        logits: TensorId,
        target: Y,
        tape: &mut Tape<B>,
        ctx: &mut ForwardCtx<B>,
    ) -> Result<TensorId>;
}

#[derive(Clone, Debug)]
pub struct ThroughputStats {
    pub epoch: usize,
    pub examples: usize,
    pub batches: usize,
    pub elapsed: std::time::Duration,
    pub examples_per_sec: f64,
    pub batches_per_sec: f64,
}

#[derive(Clone, Debug)]
pub struct TrainingReport {
    pub epochs: Vec<EpochStats>,
    /// Accuracy per epoch when classification targets are usable; otherwise None.
    pub accuracy: Option<Vec<f32>>,
    /// Throughput per epoch (examples/sec, batches/sec).
    pub throughput: Vec<ThroughputStats>,
}

/// Basic training configuration for [`TapeTrainer`].
#[derive(Clone, Debug)]
pub struct TapeTrainerConfig {
    pub epochs: usize,
    pub learning_rate: f32,
    pub batch_size: usize,
    pub shuffle: bool,
    pub seed: u64,
}

impl Default for TapeTrainerConfig {
    fn default() -> Self {
        Self { epochs: 1, learning_rate: 1e-3, batch_size: 1, shuffle: true, seed: 0 }
    }
}

/// A generic trainer operating on a single process (no distribution).
pub struct TapeTrainer<B: Backend, O: Optimizer<B>> {
    pub config: TapeTrainerConfig,
    pub optimizer: O,
    pub tensor_pool: Option<Arc<Mutex<TensorPool<B>>>>,
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend, O: Optimizer<B>> TapeTrainer<B, O>
where
    B::Tensor: Clone,
{
    pub fn new(config: TapeTrainerConfig, optimizer: O) -> Self {
        Self { config, optimizer, tensor_pool: None, _phantom: std::marker::PhantomData }
    }

    /// Set an optional tensor pool for memory management during training.
    pub fn with_tensor_pool(mut self, pool: TensorPool<B>) -> Self {
        self.tensor_pool = Some(Arc::new(Mutex::new(pool)));
        self
    }

    /// Provide a shared tensor pool owned by the caller/runtime.
    pub fn with_shared_tensor_pool(mut self, pool: Arc<Mutex<TensorPool<B>>>) -> Self {
        self.tensor_pool = Some(pool);
        self
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

        let profiler = operation_profiler_from_env();

        let mut stats = Vec::with_capacity(self.config.epochs);
        for epoch in 0..self.config.epochs {
            let start = Instant::now();
            let mut losses = Vec::new();

            for batch in data.chunks(self.config.batch_size) {
                let mut ctx = ForwardCtx::new(backend, Mode::Train);
                if let Some(ref p) = profiler {
                    ctx.set_profiler(Some(p.clone()));
                }
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

                // Call tensor pool begin_step after optimizer step for memory management
                if let Some(ref pool) = self.tensor_pool {
                    if let Ok(mut p) = pool.lock() {
                        p.begin_step();
                    }
                }

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

        if let Some(ref p) = profiler {
            finish_training_profiler(p);
        }

        Ok(stats)
    }

    /// Train a supervised model without requiring caller-provided tape code.
    ///
    /// This is a minimal-but-structured `fit`:
    /// - the model defines `forward_tape` and `loss_tape`
    /// - the trainer owns tape, backprop, optimizer step, batching, shuffling, and metrics
    pub fn fit<M, X, Y>(
        &mut self,
        backend: &B,
        model: &mut M,
        train: &[(X, Y)],
    ) -> anyhow::Result<TrainingReport>
    where
        M: SupervisedTapeModel<B, X, Y>,
        X: Clone,
        Y: Clone,
    {
        self.fit_inner(backend, model, train, None::<fn(&Y) -> usize>)
    }

    /// Supervised fit with built-in accuracy (classification targets).
    pub fn fit_classification<M, X, Y>(
        &mut self,
        backend: &B,
        model: &mut M,
        train: &[(X, Y)],
    ) -> anyhow::Result<TrainingReport>
    where
        M: SupervisedTapeModel<B, X, Y>,
        X: Clone,
        Y: Copy + Into<usize>,
    {
        self.fit_inner(backend, model, train, Some(|y: &Y| (*y).into()))
    }

    fn fit_inner<M, X, Y>(
        &mut self,
        backend: &B,
        model: &mut M,
        train: &[(X, Y)],
        target_to_class: Option<fn(&Y) -> usize>,
    ) -> anyhow::Result<TrainingReport>
    where
        M: SupervisedTapeModel<B, X, Y>,
        X: Clone,
        Y: Clone,
    {
        // Native TUI: auto-init the global dashboard on first training call.
        #[cfg(feature = "tui")]
        crate::tui_hook::init_global_dashboard();

        use rand::seq::SliceRandom;
        use rand::SeedableRng;

        let ops = backend.ops();
        if self.config.batch_size == 0 {
            anyhow::bail!("batch_size must be non-zero");
        }

        let mut indices: Vec<usize> = (0..train.len()).collect();
        let mut acc_hist: Vec<f32> = Vec::with_capacity(self.config.epochs);
        let mut epochs: Vec<EpochStats> = Vec::with_capacity(self.config.epochs);
        let mut throughput: Vec<ThroughputStats> = Vec::with_capacity(self.config.epochs);

        let profiler = operation_profiler_from_env();

        for epoch in 0..self.config.epochs {
            let start = Instant::now();
            let mut rng =
                rand::rngs::StdRng::seed_from_u64(self.config.seed ^ (epoch as u64).wrapping_mul(0x9E37));
            if self.config.shuffle {
                indices.shuffle(&mut rng);
            }

            let mut losses = Vec::new();
            let mut correct: usize = 0;
            let mut total: usize = 0;

            let mut batch_count: usize = 0;
            for batch_idx in indices.chunks(self.config.batch_size) {
                batch_count += 1;
                let mut ctx = ForwardCtx::new(backend, Mode::Train);
                if let Some(ref p) = profiler {
                    ctx.set_profiler(Some(p.clone()));
                }
                let mut tape = Tape::<B>::new();

                model.visit_parameters(&mut |_name, p| {
                    tape.watch_parameter(p);
                });

                let mut batch_loss: Option<TensorId> = None;

                for &i in batch_idx {
                    let (ref x, ref y) = train[i];
                    let logits = model.forward_tape(x.clone(), &mut tape, &mut ctx)?;
                    let loss = model.loss_tape(logits, y.clone(), &mut tape, &mut ctx)?;

                    if let Some(target_to_class) = target_to_class {
                        if let Some(t) = tape.value(logits) {
                            if let Ok(v) = ops.tensor_to_vec(t) {
                                if !v.is_empty() {
                                    let pred = v
                                        .iter()
                                        .enumerate()
                                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                                        .map(|(idx, _)| idx)
                                        .unwrap_or(0);
                                    let truth = target_to_class(y);
                                    total += 1;
                                    if pred == truth {
                                        correct += 1;
                                    }
                                }
                            }
                        }
                    }

                    batch_loss = Some(match batch_loss {
                        Some(acc) => tape.add(acc, loss, &mut ctx)?,
                        None => loss,
                    });
                }

                let Some(batch_loss) = batch_loss else { continue };
                let inv_batch = 1.0 / batch_idx.len() as f32;
                let batch_loss = tape.mul_scalar(batch_loss, inv_batch, &mut ctx)?;

                let loss_tensor = tape
                    .value(batch_loss)
                    .ok_or_else(|| anyhow::anyhow!("internal error: loss tensor missing from tape"))?;
                let loss_value = ops.tensor_to_vec(loss_tensor)?.first().copied().unwrap_or(0.0);
                losses.push(loss_value);

                let param_map = tape.param_map().clone();
                let make_ones = |data: Vec<f32>, shape: &[usize]| ops.tensor_from_vec(data, shape);
                let grads_store = tape.backward(batch_loss, make_ones, ops)?;

                let mut grads = Vec::new();
                model.visit_parameters(&mut |_name, p| {
                    if let Some(g) = p.gradient_from_store(&grads_store, &param_map) {
                        grads.push(Gradient { param_id: p.id(), tensor: g.clone() });
                    }
                });

                let mut params_vec: Vec<Parameter<B>> = Vec::new();
                model.visit_parameters(&mut |_name, p| params_vec.push(p.clone()));
                self.optimizer
                    .step(&mut params_vec, &grads, &mut ctx)
                    .map_err(|e| anyhow::anyhow!("{:?}", e))?;

                // Call tensor pool begin_step after optimizer step for memory management
                if let Some(ref pool) = self.tensor_pool {
                    if let Ok(mut p) = pool.lock() {
                        p.begin_step();
                    }
                }

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
            let elapsed = start.elapsed();
            epochs.push(EpochStats { epoch, examples: train.len(), mean_loss, elapsed });
            let examples_per_sec =
                if elapsed.as_secs_f64() > 0.0 { train.len() as f64 / elapsed.as_secs_f64() } else { 0.0 };
            let batches_per_sec =
                if elapsed.as_secs_f64() > 0.0 { batch_count as f64 / elapsed.as_secs_f64() } else { 0.0 };
            throughput.push(ThroughputStats {
                epoch,
                examples: train.len(),
                batches: batch_count,
                elapsed,
                examples_per_sec,
                batches_per_sec,
            });

            // Native TUI: push epoch snapshot to the global dashboard.
            #[cfg(feature = "tui")]
            if let Some(dash) = crate::tui_hook::dashboard() {
                let mut db = dash.lock().unwrap();
                db.set_total_epochs(self.config.epochs as u64);
                db.set_epoch(epoch as u64);
                db.set_step(epoch as u64);
                db.record_loss(mean_loss as f64);
                if target_to_class.is_some() && total > 0 {
                    let acc_val = correct as f32 / total as f32;
                    db.record_accuracy(acc_val as f64);
                }
            }

            if target_to_class.is_some() && total > 0 {
                acc_hist.push(correct as f32 / total as f32);
            }
        }

        if let Some(ref p) = profiler {
            finish_training_profiler(p);
        }

        Ok(TrainingReport { epochs, accuracy: if acc_hist.is_empty() { None } else { Some(acc_hist) }, throughput })
    }
}
