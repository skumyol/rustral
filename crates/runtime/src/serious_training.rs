//! End-to-end training utilities (tape autodiff + Adam + checkpoints).
//!
//! Enable with the `training` feature on `rustral-runtime`.

use std::collections::HashMap;

use rustral_autodiff::{GradExtFromStore, Tape};
use rustral_core::{Backend, CoreError, ForwardCtx, Mode, TensorOps};
use rustral_io::{load_parameters, save_state_dict};
use rustral_optim::{Adam, Gradient, Optimizer};

/// Hyperparameters for the bundled synthetic classification demo.
#[derive(Clone, Debug)]
pub struct SeriousTrainingConfig {
    /// Pseudo-random seed for dataset + init when applicable.
    pub seed: u64,
    pub epochs: usize,
    /// Samples in the synthetic dataset.
    pub dataset_size: usize,
    pub input_dim: usize,
    pub num_classes: usize,
    pub learning_rate: f32,
    /// Optional checkpoint directory; if set, saves mid-training and reloads to verify I/O.
    pub checkpoint_dir: Option<std::path::PathBuf>,
}

impl Default for SeriousTrainingConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            epochs: 400,
            dataset_size: 128,
            input_dim: 4,
            num_classes: 3,
            learning_rate: 0.05,
            checkpoint_dir: None,
        }
    }
}

/// Result summary after [`train_synthetic_classification`].
#[derive(Clone, Debug)]
pub struct SeriousTrainingOutcome {
    pub loss_start: f32,
    pub loss_end: f32,
    pub checkpoint_roundtrip_ok: bool,
}

/// Train a tiny softmax classifier on synthetic data using tape autodiff + Adam.
///
/// Weight tensor layout is `[input_dim, num_classes]` so `logits = x @ W` with `x` shaped `[1, input_dim]`.
///
/// # Type bounds
///
/// Matches typical Rustral backends used for training (`CpuBackend`, `CandleBackend`, …).
pub fn train_synthetic_classification<B>(
    backend: &B,
    config: SeriousTrainingConfig,
) -> anyhow::Result<SeriousTrainingOutcome>
where
    B: Backend,
    B::Tensor: Clone,
{
    let SeriousTrainingConfig {
        seed,
        epochs,
        dataset_size,
        input_dim,
        num_classes,
        learning_rate,
        checkpoint_dir,
    } = config;

    if dataset_size == 0 || input_dim == 0 || num_classes == 0 {
        anyhow::bail!("dataset_size, input_dim, and num_classes must be non-zero");
    }

    let data = build_synthetic_dataset(seed, dataset_size, input_dim, num_classes);

    let mut w = backend
        .normal_parameter("classifier.weight", &[input_dim, num_classes], seed, 0.15)
        .map_err(|e| anyhow::anyhow!("{:?}", e))?;

    let mut adam = Adam::<B>::new(learning_rate);
    let ops = backend.ops();

    let mut loss_start = None::<f32>;
    let mut loss_end = 0.0f32;
    let mut checkpoint_roundtrip_ok = false;

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0f32;

        for (x_vec, label) in &data {
            let mut ctx = ForwardCtx::new(backend, Mode::Train);
            let mut tape = Tape::<B>::new();

            let x_tensor = ops
                .tensor_from_vec(x_vec.clone(), &[1, input_dim])
                .map_err(|e| anyhow::anyhow!("{:?}", e))?;
            let x_id = tape.watch(x_tensor);

            let w_id = tape.watch_parameter(&w);

            let logits_id = tape.matmul(x_id, w_id, &mut ctx).map_err(|e| anyhow::anyhow!("{:?}", e))?;

            let mut one_hot = vec![0.0f32; num_classes];
            one_hot[*label] = 1.0;
            let target_tensor =
                ops.tensor_from_vec(one_hot, &[1, num_classes]).map_err(|e| anyhow::anyhow!("{:?}", e))?;
            let target_id = tape.watch(target_tensor);

            let loss_id = tape
                .cross_entropy_loss(logits_id, target_id, &mut ctx)
                .map_err(|e| anyhow::anyhow!("{:?}", e))?;

            let loss_scalar = ops
                .tensor_element(tape.value(loss_id).ok_or_else(|| anyhow::anyhow!("missing loss"))?, 0)
                .map_err(|e| anyhow::anyhow!("{:?}", e))?;
            epoch_loss += loss_scalar;

            let param_map = tape.param_map().clone();
            let make_ones = |data: Vec<f32>, shape: &[usize]| ops.tensor_from_vec(data, shape);
            let grads_store =
                tape.backward(loss_id, make_ones, ops).map_err(|e| anyhow::anyhow!("{:?}", e))?;

            let mut grads = Vec::new();
            let g = w
                .gradient_from_store(&grads_store, &param_map)
                .ok_or_else(|| anyhow::anyhow!("missing gradient for weight"))?;
            grads.push(Gradient { param_id: w.id(), tensor: g.clone() });

            adam.step(std::slice::from_mut(&mut w), &grads, &mut ctx)
                .map_err(|e| anyhow::anyhow!("{:?}", e))?;
        }

        let mean = epoch_loss / dataset_size as f32;
        if epoch == 0 {
            loss_start = Some(mean);
        }
        loss_end = mean;

        // Mid-run checkpoint roundtrip once.
        if epoch == epochs / 2 {
            if let Some(dir) = &checkpoint_dir {
                std::fs::create_dir_all(dir).map_err(|e| anyhow::anyhow!(e))?;
                let path = dir.join("mid_train.safetensors");
                let flat_w = flatten_tensor(ops, w.tensor()).map_err(|e| anyhow::anyhow!("{:?}", e))?;
                let mut dict = HashMap::new();
                dict.insert("classifier.weight".to_string(), flat_w);
                let bytes = save_state_dict(&dict).map_err(|e| anyhow::anyhow!("{:?}", e))?;
                std::fs::write(&path, bytes).map_err(|e| anyhow::anyhow!(e))?;

                let loaded_map = load_parameters(&std::fs::read(&path).map_err(|e| anyhow::anyhow!(e))?)
                    .map_err(|e| anyhow::anyhow!("{:?}", e))?;
                let flat =
                    loaded_map.get("classifier.weight").ok_or_else(|| anyhow::anyhow!("missing key"))?;
                let shape = ops.shape(w.tensor());
                let n: usize = shape.iter().product();
                if flat.len() != n {
                    anyhow::bail!("checkpoint shape mismatch");
                }
                let t = ops.tensor_from_vec(flat.clone(), &shape).map_err(|e| anyhow::anyhow!("{:?}", e))?;
                w = w.with_tensor(t);

                checkpoint_roundtrip_ok = true;
            }
        }
    }

    Ok(SeriousTrainingOutcome {
        loss_start: loss_start.unwrap_or(loss_end),
        loss_end,
        checkpoint_roundtrip_ok,
    })
}

fn build_synthetic_dataset(
    seed: u64,
    n: usize,
    input_dim: usize,
    num_classes: usize,
) -> Vec<(Vec<f32>, usize)> {
    let mut out = Vec::with_capacity(n);
    let mut s = seed;
    for _ in 0..n {
        let mut x = Vec::with_capacity(input_dim);
        for _ in 0..input_dim {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            x.push((s as f32) / (u64::MAX as f32) * 2.0 - 1.0);
        }
        let sum: f32 = x.iter().sum();
        let label = (sum.abs() as usize) % num_classes;
        out.push((x, label));
    }
    out
}

fn flatten_tensor<B: Backend>(ops: &dyn TensorOps<B>, t: &B::Tensor) -> Result<Vec<f32>, CoreError> {
    // Prefer a single bulk readback when supported (GPU: one transfer; CPU: often a copy).
    ops.tensor_to_vec(t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustral_ndarray_backend::CpuBackend;

    #[test]
    fn synthetic_training_cpu_loss_decreases() {
        let backend = CpuBackend::default();
        let cfg = SeriousTrainingConfig { epochs: 200, dataset_size: 64, ..Default::default() };
        let out = train_synthetic_classification(&backend, cfg).unwrap();
        assert!(out.loss_end < out.loss_start, "expected {} < {}", out.loss_end, out.loss_start);
    }
}
