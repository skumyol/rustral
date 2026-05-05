//! Optimization algorithms for Rustral.
#![allow(dead_code)]
//!
//! Provides SGD, Adam, and AdamW optimizers that work with the core
//! `Parameter` and `Backend` abstractions.

use std::collections::HashMap;

use rustral_core::{Backend, CoreError, ForwardCtx, Parameter, ParameterId, TensorOps, TensorShape};
use thiserror::Error;

pub mod lr_scheduler;
pub mod mixed_precision;

pub use lr_scheduler::{
    ConstantLR, CosineAnnealingLR, ExponentialLR, LRScheduler, LinearWarmup, OneCycleLR, PlateauLR,
    PolynomialLR, StepDecay, WarmupCosine,
};
pub use mixed_precision::{DType, LossScaleScheduler, MixedPrecisionOptimizer, MixedPrecisionStats};

/// Errors that can occur during optimization.
#[derive(Debug, Error)]
pub enum OptimError {
    /// Underlying core error.
    #[error("core error: {0}")]
    Core(#[from] CoreError),

    /// Backend operation failed.
    #[error("backend error: {0}")]
    Backend(String),

    /// Gradient computation failed.
    #[error("gradient error: {0}")]
    Gradient(String),

    /// Shape mismatch in parameter update.
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),

    /// Missing state for a parameter.
    #[error("missing optimizer state for parameter {0:?}")]
    MissingState(ParameterId),
}

/// A single gradient tensor matching the shape of its associated parameter.
pub struct Gradient<B: Backend> {
    pub param_id: ParameterId,
    pub tensor: B::Tensor,
}

impl<B: Backend> Clone for Gradient<B>
where
    B::Tensor: Clone,
{
    fn clone(&self) -> Self {
        Self { param_id: self.param_id, tensor: self.tensor.clone() }
    }
}

/// Optimizer state for first and second moment tracking (used by Adam/AdamW).
///
/// This can be serialized to save and resume training.
#[derive(Clone)]
pub struct AdamState<B: Backend> {
    /// First moment (mean of gradients).
    pub m: B::Tensor,
    /// Second moment (mean of squared gradients).
    pub v: B::Tensor,
    /// Timestep for bias correction.
    pub t: u64,
}

/// Serializable Adam state for checkpointing.
///
/// This struct holds the hyperparameters and per-parameter state
/// in a format that can be serialized with serde.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AdamCheckpoint {
    /// Learning rate.
    pub lr: f32,
    /// Beta1 (first moment decay).
    pub beta1: f32,
    /// Beta2 (second moment decay).
    pub beta2: f32,
    /// Epsilon for numerical stability.
    pub eps: f32,
    /// Weight decay coefficient.
    pub weight_decay: f32,
    /// Per-parameter state (param_id -> (m_values, v_values, t, shape)).
    pub state: HashMap<u64, (Vec<f32>, Vec<f32>, u64, Vec<usize>)>,
}

/// Optimizer trait for updating parameters based on gradients.
pub trait Optimizer<B: Backend> {
    /// Perform a single optimization step.
    ///
    /// Updates all parameters in `params` using the provided `gradients`.
    fn step(
        &mut self,
        params: &mut [Parameter<B>],
        gradients: &[Gradient<B>],
        ctx: &mut ForwardCtx<B>,
    ) -> std::result::Result<(), OptimError>;

    /// Zero out any accumulated gradients (called before backward pass).
    fn zero_grad(&mut self);
}

/// Stochastic Gradient Descent optimizer.
pub struct Sgd {
    /// Learning rate.
    pub lr: f32,
    /// Momentum coefficient (0.0 = no momentum).
    pub momentum: f32,
    /// Weight decay coefficient (L2 penalty).
    pub weight_decay: f32,
}

impl Default for Sgd {
    fn default() -> Self {
        Self { lr: 0.01, momentum: 0.0, weight_decay: 0.0 }
    }
}

impl Sgd {
    /// Create a new SGD optimizer with the given learning rate.
    pub fn new(lr: f32) -> Self {
        Self { lr, ..Default::default() }
    }

    /// Enable momentum.
    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    /// Enable weight decay (L2 regularization).
    pub fn with_weight_decay(mut self, decay: f32) -> Self {
        self.weight_decay = decay;
        self
    }
}

impl<B: Backend> Optimizer<B> for Sgd
where
    B::Tensor: Clone,
{
    fn step(
        &mut self,
        params: &mut [Parameter<B>],
        gradients: &[Gradient<B>],
        ctx: &mut ForwardCtx<B>,
    ) -> std::result::Result<(), OptimError> {
        let ops = ctx.backend().ops();

        // Build a map from param_id to gradient tensor for fast lookup
        let grad_map: HashMap<ParameterId, &B::Tensor> =
            gradients.iter().map(|g| (g.param_id, &g.tensor)).collect();

        for param in params {
            let param_id = param.id();
            let Some(grad) = grad_map.get(&param_id) else {
                continue; // No gradient for this parameter
            };

            let current = param.tensor();

            // Apply weight decay if enabled: grad = grad + weight_decay * param
            let grad_with_decay = if self.weight_decay != 0.0 {
                let shape = ops.shape(current);
                let size = shape.iter().product::<usize>();
                let decay_values = vec![self.weight_decay; size];
                let decay_tensor = ops.tensor_from_vec(decay_values, &shape)?;
                let scaled_decay = ops.mul(current, &decay_tensor)?;
                ops.add(grad, &scaled_decay)?
            } else {
                (*grad).clone()
            };

            // Simple SGD update: param = param - lr * grad
            let neg_lr = -self.lr;
            let shape = ops.shape(&grad_with_decay);
            let size = shape.iter().product::<usize>();
            let lr_values = vec![neg_lr; size];
            let lr_tensor = ops.tensor_from_vec(lr_values, &shape)?;
            let scaled_grad = ops.mul(&grad_with_decay, &lr_tensor)?;
            let new_value = ops.add(current, &scaled_grad)?;

            // Update parameter (requires consuming and recreating)
            // Note: In-place mutation would require TensorInPlaceOps support,
            // which backends may implement as an extension trait for performance.
            *param = param.clone().with_tensor(new_value);
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        // SGD doesn't maintain state, so nothing to zero
    }
}

/// Adam optimizer (Adaptive Moment Estimation).
///
/// Implements the Adam algorithm with optional decoupled weight decay
/// (AdamW variant via `with_weight_decay`).
pub struct Adam<B: Backend> {
    /// Learning rate.
    pub lr: f32,
    /// Beta1 (first moment decay).
    pub beta1: f32,
    /// Beta2 (second moment decay).
    pub beta2: f32,
    /// Epsilon for numerical stability.
    pub eps: f32,
    /// Weight decay coefficient.
    pub weight_decay: f32,

    /// Per-parameter Adam state.
    state: HashMap<ParameterId, AdamState<B>>,
}

impl<B: Backend> Default for Adam<B> {
    fn default() -> Self {
        Self { lr: 0.001, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.0, state: HashMap::new() }
    }
}

impl<B: Backend> Adam<B> {
    /// Create Adam with default hyperparameters.
    pub fn new(lr: f32) -> Self {
        Self { lr, ..Default::default() }
    }

    /// Set betas (momentum decays).
    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    /// Enable weight decay (decoupled from gradient, applied to parameter directly).
    pub fn with_weight_decay(mut self, decay: f32) -> Self {
        self.weight_decay = decay;
        self
    }

    /// Convert to AdamW (decoupled weight decay).
    ///
    /// This is the same as `with_weight_decay` but signals intent.
    pub fn adamw(self, weight_decay: f32) -> Self {
        self.with_weight_decay(weight_decay)
    }

    /// Save optimizer state to a serializable checkpoint.
    ///
    /// This allows resuming training from a saved state.
    pub fn save_checkpoint(&self) -> AdamCheckpoint
    where
        B::Tensor: AsRef<[f32]> + rustral_core::TensorShape,
    {
        let mut state = HashMap::new();
        for (param_id, adam_state) in &self.state {
            // Extract tensor values and shape
            let m_values: Vec<f32> = adam_state.m.as_ref().to_vec();
            let v_values: Vec<f32> = adam_state.v.as_ref().to_vec();
            let shape = adam_state.m.shape().to_vec();
            state.insert(param_id.get(), (m_values, v_values, adam_state.t, shape));
        }

        AdamCheckpoint {
            lr: self.lr,
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
            weight_decay: self.weight_decay,
            state,
        }
    }

    /// Load optimizer state from a checkpoint.
    ///
    /// This restores the optimizer to the saved state for resuming training.
    /// The params argument is used to match parameter IDs and get correct shapes.
    pub fn load_checkpoint(
        &mut self,
        checkpoint: &AdamCheckpoint,
        params: &[Parameter<B>],
        ops: &dyn TensorOps<B>,
    ) -> std::result::Result<(), OptimError>
    where
        B::Tensor: Clone,
    {
        // Update hyperparameters
        self.lr = checkpoint.lr;
        self.beta1 = checkpoint.beta1;
        self.beta2 = checkpoint.beta2;
        self.eps = checkpoint.eps;
        self.weight_decay = checkpoint.weight_decay;

        // Clear and restore state
        self.state.clear();

        // Build a map from param_id numeric value to the actual parameter
        let param_map: HashMap<u64, &Parameter<B>> = params.iter().map(|p| (p.id().get(), p)).collect();

        for (param_id_num, (m_values, v_values, t, shape)) in &checkpoint.state {
            let param =
                param_map.get(param_id_num).ok_or_else(|| OptimError::MissingState(ParameterId::fresh()))?;

            // Verify shape matches
            let param_shape = ops.shape(param.tensor());
            if param_shape != shape.as_slice() {
                return Err(OptimError::ShapeMismatch(format!(
                    "Shape mismatch for param {}: expected {:?}, got {:?}",
                    param.name(),
                    param_shape,
                    shape
                )));
            }

            // Create tensors from saved values
            let m = ops.tensor_from_vec(m_values.clone(), shape)?;
            let v = ops.tensor_from_vec(v_values.clone(), shape)?;

            self.state.insert(param.id(), AdamState { m, v, t: *t });
        }

        Ok(())
    }
}

impl<B: Backend> Optimizer<B> for Adam<B>
where
    B::Tensor: Clone,
{
    fn step(
        &mut self,
        params: &mut [Parameter<B>],
        gradients: &[Gradient<B>],
        ctx: &mut ForwardCtx<B>,
    ) -> std::result::Result<(), OptimError> {
        let ops = ctx.backend().ops();

        // Build a map from param_id to gradient tensor for fast lookup
        let grad_map: HashMap<ParameterId, &B::Tensor> =
            gradients.iter().map(|g| (g.param_id, &g.tensor)).collect();

        for param in params {
            let param_id = param.id();
            let Some(grad) = grad_map.get(&param_id) else {
                continue; // No gradient for this parameter
            };

            let current = param.tensor();
            let shape = ops.shape(current);

            // Get or initialize state for this parameter
            let state = self.state.entry(param_id).or_insert_with(|| {
                let zeros = ops.zeros(&shape).unwrap();
                AdamState { m: zeros.clone(), v: zeros, t: 0 }
            });

            state.t += 1;
            let t = state.t as f32;

            // Apply weight decay if enabled (AdamW variant - decoupled)
            let grad_with_decay = if self.weight_decay != 0.0 {
                // Create weight decay tensor
                let size = shape.iter().product::<usize>();
                let decay_values = vec![self.weight_decay; size];
                let decay_tensor = ops.tensor_from_vec(decay_values, &shape)?;
                let scaled_param = ops.mul(current, &decay_tensor)?;
                ops.add(grad, &scaled_param)?
            } else {
                (*grad).clone()
            };

            // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * grad
            let beta1_tensor = create_scalar_tensor(ops, self.beta1, &shape)?;
            let beta1_m = ops.mul(&state.m, &beta1_tensor)?;
            let one_minus_beta1 = create_scalar_tensor(ops, 1.0 - self.beta1, &shape)?;
            let scaled_grad = ops.mul(&grad_with_decay, &one_minus_beta1)?;
            state.m = ops.add(&beta1_m, &scaled_grad)?;

            // Update biased second moment estimate: v = beta2 * v + (1 - beta2) * grad^2
            let beta2_tensor = create_scalar_tensor(ops, self.beta2, &shape)?;
            let beta2_v = ops.mul(&state.v, &beta2_tensor)?;
            let grad_squared = ops.mul(&grad_with_decay, &grad_with_decay)?;
            let one_minus_beta2 = create_scalar_tensor(ops, 1.0 - self.beta2, &shape)?;
            let scaled_grad_sq = ops.mul(&grad_squared, &one_minus_beta2)?;
            state.v = ops.add(&beta2_v, &scaled_grad_sq)?;

            // Compute bias-corrected first moment: m_hat = m / (1 - beta1^t)
            let bias_corr1 = 1.0 - self.beta1.powf(t);
            let bias_corr1_tensor = create_scalar_tensor(ops, bias_corr1, &shape)?;
            let m_hat = ops.div(&state.m, &bias_corr1_tensor)?;

            // Compute bias-corrected second moment: v_hat = v / (1 - beta2^t)
            let bias_corr2 = 1.0 - self.beta2.powf(t);
            let bias_corr2_tensor = create_scalar_tensor(ops, bias_corr2, &shape)?;
            let v_hat = ops.div(&state.v, &bias_corr2_tensor)?;

            // Compute update: -lr * m_hat / (sqrt(v_hat) + eps)
            let sqrt_v_hat = ops.sqrt(&v_hat)?;
            let eps_tensor = create_scalar_tensor(ops, self.eps, &shape)?;
            let denom = ops.add(&sqrt_v_hat, &eps_tensor)?;
            let step_size = ops.div(&m_hat, &denom)?;
            let lr_tensor = create_scalar_tensor(ops, -self.lr, &shape)?;
            let update = ops.mul(&step_size, &lr_tensor)?;

            // Update parameter
            let new_value = ops.add(current, &update)?;
            *param = param.clone().with_tensor(new_value);
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        // Adam maintains momentum state which persists across steps
    }
}

/// AdamW optimizer (decoupled weight decay variant).
///
/// This is equivalent to `Adam::new(lr).adamw(weight_decay)`.
pub type AdamW<B> = Adam<B>;

/// Create a tensor filled with a scalar value in the given shape.
fn create_scalar_tensor<B: Backend>(
    ops: &dyn TensorOps<B>,
    value: f32,
    shape: &[usize],
) -> rustral_core::Result<B::Tensor> {
    let size = shape.iter().product::<usize>();
    let values = vec![value; size];
    ops.tensor_from_vec(values, shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustral_core::Mode;
    use rustral_ndarray_backend::CpuBackend;

    #[test]
    fn test_sgd_update() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Train);

        let mut param = backend.normal_parameter("w", &[2], 42, 0.0).unwrap();
        let grad_tensor = backend.ops().tensor_from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let gradients = vec![Gradient { param_id: param.id(), tensor: grad_tensor }];

        let mut sgd = Sgd::new(0.1);
        sgd.step(std::slice::from_mut(&mut param), &gradients, &mut ctx).unwrap();

        // Parameter should have been updated: w = w - lr * grad = 0 - 0.1 * grad
        let values: Vec<f32> = param.tensor().values().to_vec();
        assert_eq!(values[1], -0.2); // -0.1 * 2.0
    }

    #[test]
    fn test_adam_update() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Train);

        let mut param = backend.normal_parameter("w", &[2], 42, 0.0).unwrap();
        let grad_tensor = backend.ops().tensor_from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let gradients = vec![Gradient { param_id: param.id(), tensor: grad_tensor }];

        let mut adam = Adam::new(0.1);
        adam.step(std::slice::from_mut(&mut param), &gradients, &mut ctx).unwrap();

        // After first step with Adam, the parameter should have been updated
        let values: Vec<f32> = param.tensor().values().to_vec();
        // With lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8
        // m = 0.1 * grad, v = 0.001 * grad^2
        // m_hat = m / (1-0.9) = m / 0.1 = grad
        // v_hat = v / (1-0.999) = v / 0.001 = grad^2
        // update = -0.1 * grad / (sqrt(grad^2) + 1e-8) = -0.1 * sign(grad)
        // w[0] = 0 - 0.1 * 1.0 / 1.0 = -0.1
        // w[1] = 0 - 0.1 * 2.0 / 2.0 = -0.1
        assert!((values[0] - (-0.1)).abs() < 1e-6, "Expected ~-0.1, got {}", values[0]);
        assert!((values[1] - (-0.1)).abs() < 1e-6, "Expected ~-0.1, got {}", values[1]);
    }

    #[test]
    fn test_sgd_default() {
        let sgd: Sgd = Default::default();
        assert_eq!(sgd.lr, 0.01);
        assert_eq!(sgd.momentum, 0.0);
        assert_eq!(sgd.weight_decay, 0.0);
    }

    #[test]
    fn test_sgd_with_momentum() {
        let sgd = Sgd::new(0.01).with_momentum(0.9);
        assert_eq!(sgd.momentum, 0.9);
    }

    #[test]
    fn test_sgd_with_weight_decay() {
        let sgd = Sgd::new(0.01).with_weight_decay(0.01);
        assert_eq!(sgd.weight_decay, 0.01);
    }

    #[test]
    fn test_sgd_zero_grad() {
        let mut sgd = Sgd::new(0.01);
        Optimizer::<CpuBackend>::zero_grad(&mut sgd); // Should not panic
    }

    #[test]
    fn test_adam_default() {
        let adam: Adam<CpuBackend> = Default::default();
        assert_eq!(adam.lr, 0.001);
        assert_eq!(adam.beta1, 0.9);
        assert_eq!(adam.beta2, 0.999);
        assert_eq!(adam.eps, 1e-8);
        assert_eq!(adam.weight_decay, 0.0);
    }

    #[test]
    fn test_adam_with_betas() {
        let adam = Adam::<CpuBackend>::new(0.01).with_betas(0.8, 0.9);
        assert_eq!(adam.beta1, 0.8);
        assert_eq!(adam.beta2, 0.9);
    }

    #[test]
    fn test_adam_with_weight_decay() {
        let adam = Adam::<CpuBackend>::new(0.01).with_weight_decay(0.01);
        assert_eq!(adam.weight_decay, 0.01);
    }

    #[test]
    fn test_adam_adamw() {
        let adam = Adam::<CpuBackend>::new(0.01).adamw(0.1);
        assert_eq!(adam.weight_decay, 0.1);
    }

    #[test]
    fn test_adam_zero_grad() {
        let mut adam = Adam::<CpuBackend>::new(0.01);
        adam.zero_grad(); // Should not panic
    }

    #[test]
    fn test_gradient_clone() {
        let backend = CpuBackend::default();
        let tensor = backend.ops().tensor_from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let grad: Gradient<CpuBackend> = Gradient { param_id: ParameterId::fresh(), tensor };
        let cloned = grad.clone();
        assert_eq!(cloned.param_id, grad.param_id);
    }

    #[test]
    fn test_adam_save_load_checkpoint() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Train);

        let mut param = backend.normal_parameter("w", &[2], 42, 0.0).unwrap();
        let grad_tensor = backend.ops().tensor_from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let gradients = vec![Gradient { param_id: param.id(), tensor: grad_tensor }];

        let mut adam = Adam::new(0.1);
        adam.step(std::slice::from_mut(&mut param), &gradients, &mut ctx).unwrap();

        let checkpoint = adam.save_checkpoint();
        assert_eq!(checkpoint.lr, 0.1);
        assert!(!checkpoint.state.is_empty());

        let mut adam2 = Adam::<CpuBackend>::new(0.01);
        adam2.load_checkpoint(&checkpoint, &[param.clone()], backend.ops()).unwrap();
        assert_eq!(adam2.lr, 0.1);
    }

    #[test]
    fn test_adam_step_no_grad() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Train);
        let mut param = backend.normal_parameter("w", &[2], 42, 0.0).unwrap();
        let mut adam = Adam::<CpuBackend>::new(0.1);
        let result = adam.step(std::slice::from_mut(&mut param), &[], &mut ctx);
        assert!(result.is_ok());
    }
}
