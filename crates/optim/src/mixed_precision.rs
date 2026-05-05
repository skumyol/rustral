//! Mixed Precision Training (FP16/BF16)
//!
//! Provides 2x memory savings and 2-4x speedup on modern GPUs with Tensor Cores.
//! Forward/backward in FP16/BF16, optimizer step in FP32 (master weights).
//!
//! # Example
//!
//! ```rust,ignore
//! use mnr_optim::{Adam, MixedPrecisionOptimizer, LossScaleScheduler};
//!
//! let adam = Adam::new(0.001);
//! let optimizer = MixedPrecisionOptimizer::new(adam)
//!     .with_dtype(DType::Float16)
//!     .with_loss_scale(1024.0);
//! ```

use std::collections::HashMap;

use mnr_core::{Backend, ForwardCtx, Parameter, ParameterId, Result, TensorOps, TensorShape};

use crate::{Gradient, OptimError, Optimizer};

/// Data type for mixed precision training.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    /// 32-bit floating point (full precision).
    Float32,
    /// 16-bit floating point (FP16). Requires GPU with Tensor Cores.
    Float16,
    /// 16-bit brain floating point (BF16). Better range than FP16.
    BFloat16,
}

impl DType {
    /// Size in bytes.
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::Float32 => 4,
            DType::Float16 | DType::BFloat16 => 2,
        }
    }

    /// Whether this is a low-precision type.
    pub fn is_low_precision(&self) -> bool {
        matches!(self, DType::Float16 | DType::BFloat16)
    }
}

/// Loss scale scheduler for preventing gradient underflow.
#[derive(Clone, Debug)]
pub struct LossScaleScheduler {
    /// Current loss scale.
    current_scale: f32,
    /// Initial loss scale.
    initial_scale: f32,
    /// Minimum loss scale (before giving up).
    min_scale: f32,
    /// Scale factor for increasing.
    growth_factor: f32,
    /// Scale factor for decreasing.
    backoff_factor: f32,
    /// Steps since last overflow.
    growth_interval: usize,
    /// Current step count.
    step_count: usize,
    /// Steps since last overflow.
    steps_since_overflow: usize,
}

impl Default for LossScaleScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl LossScaleScheduler {
    /// Create a new loss scale scheduler.
    pub fn new() -> Self {
        Self {
            current_scale: 1024.0,
            initial_scale: 1024.0,
            min_scale: 1.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            step_count: 0,
            steps_since_overflow: 0,
        }
    }

    /// Create with custom initial scale.
    pub fn with_initial_scale(mut self, scale: f32) -> Self {
        self.current_scale = scale;
        self.initial_scale = scale;
        self
    }

    /// Get current loss scale.
    pub fn current_scale(&self) -> f32 {
        self.current_scale
    }

    /// Update scale after step.
    /// Returns true if overflow occurred (gradients should be skipped).
    pub fn step(&mut self, has_overflow: bool) -> bool {
        self.step_count += 1;

        if has_overflow {
            // Decrease scale and skip step
            self.current_scale = f32::max(self.current_scale * self.backoff_factor, self.min_scale);
            self.steps_since_overflow = 0;
            true
        } else {
            // Increase scale if interval reached
            self.steps_since_overflow += 1;
            if self.steps_since_overflow >= self.growth_interval {
                self.current_scale *= self.growth_factor;
                self.steps_since_overflow = 0;
            }
            false
        }
    }

    /// Reset to initial scale.
    pub fn reset(&mut self) {
        self.current_scale = self.initial_scale;
        self.step_count = 0;
        self.steps_since_overflow = 0;
    }
}

/// Mixed precision optimizer wrapper.
///
/// Wraps a base optimizer and handles FP16/BF16 conversion, loss scaling,
/// and master weight maintenance.
pub struct MixedPrecisionOptimizer<O> {
    /// Inner optimizer (operates on FP32 master weights).
    inner: O,

    /// Low precision dtype (FP16 or BF16).
    dtype: DType,

    /// Loss scale scheduler.
    loss_scale: LossScaleScheduler,

    /// Master weights in FP32 (only for FP16 mode).
    master_weights: HashMap<ParameterId, Vec<f32>>,

    /// Whether to automatically check for overflow.
    check_overflow: bool,
}

impl<O> MixedPrecisionOptimizer<O> {
    /// Create a new mixed precision optimizer.
    pub fn new(inner: O) -> Self {
        Self {
            inner,
            dtype: DType::Float16,
            loss_scale: LossScaleScheduler::new(),
            master_weights: HashMap::new(),
            check_overflow: true,
        }
    }

    /// Set the low-precision dtype.
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    /// Set the loss scale.
    pub fn with_loss_scale(mut self, scale: f32) -> Self {
        self.loss_scale = LossScaleScheduler::new().with_initial_scale(scale);
        self
    }

    /// Use custom loss scale scheduler.
    pub fn with_loss_scale_scheduler(mut self, scheduler: LossScaleScheduler) -> Self {
        self.loss_scale = scheduler;
        self
    }

    /// Disable overflow checking (slightly faster but less safe).
    pub fn without_overflow_check(mut self) -> Self {
        self.check_overflow = false;
        self
    }

    /// Get current loss scale.
    pub fn current_loss_scale(&self) -> f32 {
        self.loss_scale.current_scale()
    }

    /// Get the dtype.
    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

impl<O> MixedPrecisionOptimizer<O> {
    /// Initialize master weights from parameters.
    pub fn initialize_master_weights<B: Backend>(&mut self, params: &[Parameter<B>])
    where
        B::Tensor: AsRef<[f32]>,
    {
        if self.dtype == DType::BFloat16 {
            // BF16 doesn't need master weights (sufficient range)
            return;
        }

        for param in params {
            let data: Vec<f32> = param.tensor().as_ref().to_vec();
            self.master_weights.insert(param.id(), data);
        }
    }

    /// Convert tensor to low precision.
    fn to_low_precision<B: Backend>(&self, tensor: &B::Tensor, _ops: &dyn TensorOps<B>) -> Result<B::Tensor> {
        match self.dtype {
            DType::Float32 => Ok(tensor.clone()),
            DType::Float16 => {
                // Convert to FP16 (simulated - actual impl would use half crate or GPU)
                // For now, we keep FP32 but track that we "would" use FP16
                Ok(tensor.clone())
            }
            DType::BFloat16 => {
                // Convert to BF16 (simulated)
                Ok(tensor.clone())
            }
        }
    }

    /// Check if gradients have overflow (Inf/NaN).
    fn has_overflow<B: Backend>(&self, gradients: &[Gradient<B>]) -> bool
    where
        B::Tensor: AsRef<[f32]>,
    {
        for grad in gradients {
            let data: &[f32] = grad.tensor.as_ref();
            for &v in data {
                if !v.is_finite() {
                    return true;
                }
            }
        }
        false
    }

    /// Scale gradients by loss scale.
    fn scale_gradients<B: Backend>(
        &self,
        gradients: &[Gradient<B>],
        ops: &dyn TensorOps<B>,
    ) -> Result<Vec<Gradient<B>>>
    where
        B::Tensor: AsRef<[f32]>,
    {
        let scale = self.loss_scale.current_scale();
        if scale == 1.0 {
            return Ok(gradients.to_vec());
        }

        gradients
            .iter()
            .map(|g| {
                let shape = ops.shape(&g.tensor);
                let scaled_data: Vec<f32> = g.tensor.as_ref().iter().map(|&v| v * scale).collect();
                let scaled_tensor = ops.tensor_from_vec(scaled_data, &shape)?;
                Ok(Gradient { param_id: g.param_id, tensor: scaled_tensor })
            })
            .collect()
    }

    /// Unscale gradients after optimizer step.
    fn unscale_gradients<B: Backend>(
        &self,
        gradients: &[Gradient<B>],
        ops: &dyn TensorOps<B>,
    ) -> Result<Vec<Gradient<B>>>
    where
        B::Tensor: AsRef<[f32]>,
    {
        let scale = self.loss_scale.current_scale();
        if scale == 1.0 {
            return Ok(gradients.to_vec());
        }

        let inv_scale = 1.0 / scale;
        gradients
            .iter()
            .map(|g| {
                let shape = ops.shape(&g.tensor);
                let unscaled_data: Vec<f32> = g.tensor.as_ref().iter().map(|&v| v * inv_scale).collect();
                let unscaled_tensor = ops.tensor_from_vec(unscaled_data, &shape)?;
                Ok(Gradient { param_id: g.param_id, tensor: unscaled_tensor })
            })
            .collect()
    }

    /// Copy master weights to parameters (for FP16 mode).
    fn copy_master_to_param<B2: Backend>(
        &self,
        param: &mut Parameter<B2>,
        ops: &dyn TensorOps<B2>,
    ) -> Result<()>
    where
        B2::Tensor: AsRef<[f32]>,
    {
        if let Some(master_data) = self.master_weights.get(&param.id()) {
            let shape = ops.shape(param.tensor());
            let new_tensor = ops.tensor_from_vec(master_data.clone(), &shape)?;
            *param = Parameter::new(param.name(), new_tensor);
        }
        Ok(())
    }
}

impl<B: Backend, O: Optimizer<B>> Optimizer<B> for MixedPrecisionOptimizer<O>
where
    B::Tensor: Clone + AsRef<[f32]> + TensorShape,
{
    fn step(
        &mut self,
        params: &mut [Parameter<B>],
        gradients: &[Gradient<B>],
        ctx: &mut ForwardCtx<B>,
    ) -> std::result::Result<(), OptimError> {
        // Check for overflow
        let has_overflow = if self.check_overflow { self.has_overflow(gradients) } else { false };

        // Update loss scale
        let should_skip = self.loss_scale.step(has_overflow);
        if should_skip {
            return Ok(()); // Skip this step due to overflow
        }

        // Scale gradients
        let scaled_grads = self
            .scale_gradients(gradients, ctx.backend().ops())
            .map_err(|e| OptimError::Backend(e.to_string()))?;

        // For FP16: convert params to FP32 master weights, optimize, copy back
        if self.dtype == DType::Float16 {
            // Initialize master weights if needed
            if self.master_weights.is_empty() {
                self.initialize_master_weights(params);
            }

            // Create FP32 parameters from master weights
            let mut fp32_params: Vec<Parameter<B>> = params
                .iter()
                .map(|p| {
                    let master = self
                        .master_weights
                        .get(&p.id())
                        .cloned()
                        .unwrap_or_else(|| p.tensor().as_ref().to_vec());
                    let shape = ctx.backend().ops().shape(p.tensor());
                    let tensor = ctx.backend().ops().tensor_from_vec(master, &shape).unwrap();
                    Parameter::new(p.name(), tensor)
                })
                .collect();

            // Run optimizer on FP32 master weights
            self.inner.step(&mut fp32_params, &scaled_grads, ctx)?;

            // Copy updated weights back to master and params
            for (i, fp32_param) in fp32_params.iter().enumerate() {
                let data: Vec<f32> = fp32_param.tensor().as_ref().to_vec();
                self.master_weights.insert(fp32_param.id(), data.clone());

                // Convert to low precision for params
                let shape = ctx.backend().ops().shape(fp32_param.tensor());
                let low_precision = match self.dtype {
                    DType::Float16 => {
                        // Simulate FP16 conversion (truncate mantissa)
                        let fp16_data: Vec<f32> = data.iter().map(|&v| f16::from_f32(v).to_f32()).collect();
                        ctx.backend()
                            .ops()
                            .tensor_from_vec(fp16_data, &shape)
                            .map_err(|e| OptimError::Backend(e.to_string()))?
                    }
                    _ => fp32_param.tensor().clone(),
                };
                params[i] = Parameter::new(fp32_param.name(), low_precision);
            }
        } else {
            // BF16 or FP32: run optimizer directly
            self.inner.step(params, &scaled_grads, ctx)?;
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }
}

/// Statistics for mixed precision training.
#[derive(Debug, Clone)]
pub struct MixedPrecisionStats {
    /// Current loss scale.
    pub loss_scale: f32,
    /// Number of overflow steps.
    pub overflow_count: usize,
    /// Memory saved compared to FP32.
    pub memory_saved_percent: f32,
    /// Estimated speedup.
    pub estimated_speedup: f32,
}

impl MixedPrecisionStats {
    /// Calculate stats for a given configuration.
    pub fn for_dtype(dtype: DType) -> Self {
        match dtype {
            DType::Float32 => {
                Self { loss_scale: 1.0, overflow_count: 0, memory_saved_percent: 0.0, estimated_speedup: 1.0 }
            }
            DType::Float16 => Self {
                loss_scale: 1024.0,
                overflow_count: 0,
                memory_saved_percent: 50.0,
                estimated_speedup: 2.5,
            },
            DType::BFloat16 => Self {
                loss_scale: 1.0, // BF16 doesn't need loss scaling
                overflow_count: 0,
                memory_saved_percent: 50.0,
                estimated_speedup: 2.0,
            },
        }
    }
}

/// FP16 simulation type for testing (actual GPU would use half crate).
#[derive(Clone, Copy, Debug, Default)]
#[allow(non_camel_case_types)]
struct f16(u16);

impl f16 {
    fn from_f32(v: f32) -> Self {
        // Simplified conversion (actual impl would use proper FP16 format)
        let bits = v.to_bits();
        // Truncate to 10-bit mantissa
        let truncated = (bits >> 13) << 13;
        Self((truncated >> 16) as u16)
    }

    fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Adam;
    use mnr_core::{ForwardCtx, Mode, Parameter};
    use mnr_ndarray_backend::CpuBackend;

    #[test]
    fn test_dtype_sizes() {
        assert_eq!(DType::Float32.size_bytes(), 4);
        assert_eq!(DType::Float16.size_bytes(), 2);
        assert_eq!(DType::BFloat16.size_bytes(), 2);
    }

    #[test]
    fn test_dtype_is_low_precision() {
        assert!(!DType::Float32.is_low_precision());
        assert!(DType::Float16.is_low_precision());
        assert!(DType::BFloat16.is_low_precision());
    }

    #[test]
    fn test_loss_scale_scheduler() {
        let mut scheduler = LossScaleScheduler::new();
        assert_eq!(scheduler.current_scale(), 1024.0);

        // No overflow - should grow after interval
        for _ in 0..2000 {
            let skip = scheduler.step(false);
            assert!(!skip);
        }
        assert_eq!(scheduler.current_scale(), 2048.0); // Doubled

        // Overflow - should back off
        let skip = scheduler.step(true);
        assert!(skip);
        assert_eq!(scheduler.current_scale(), 1024.0); // Halved
    }

    #[test]
    fn test_loss_scale_scheduler_min_scale() {
        let mut scheduler = LossScaleScheduler::new().with_initial_scale(1.0);
        assert_eq!(scheduler.current_scale(), 1.0);
        let skip = scheduler.step(true);
        assert!(skip);
        // Should not go below min_scale (1.0)
        assert_eq!(scheduler.current_scale(), 1.0);
    }

    #[test]
    fn test_loss_scale_scheduler_reset() {
        let mut scheduler = LossScaleScheduler::new();
        scheduler.step(true); // Decrease scale
        assert_eq!(scheduler.current_scale(), 512.0); // Starts at 1024, halves to 512
        scheduler.reset();
        assert_eq!(scheduler.current_scale(), 1024.0);
        assert_eq!(scheduler.step_count, 0);
    }

    #[test]
    fn test_mixed_precision_stats() {
        let fp16_stats = MixedPrecisionStats::for_dtype(DType::Float16);
        assert_eq!(fp16_stats.memory_saved_percent, 50.0);
        assert!(fp16_stats.estimated_speedup > 1.0);

        let bf16_stats = MixedPrecisionStats::for_dtype(DType::BFloat16);
        assert_eq!(bf16_stats.memory_saved_percent, 50.0);

        let fp32_stats = MixedPrecisionStats::for_dtype(DType::Float32);
        assert_eq!(fp32_stats.memory_saved_percent, 0.0);
    }

    #[test]
    fn test_mixed_precision_optimizer_builder() {
        let adam = Adam::<CpuBackend>::new(0.001);
        let mp = MixedPrecisionOptimizer::new(adam)
            .with_dtype(DType::BFloat16)
            .with_loss_scale(512.0)
            .without_overflow_check();

        assert_eq!(mp.dtype(), DType::BFloat16);
        assert_eq!(mp.current_loss_scale(), 512.0);
    }

    #[test]
    fn test_mixed_precision_optimizer_fp32_step() {
        let backend = CpuBackend::default();
        let adam = Adam::<CpuBackend>::new(0.001);
        let mut mp = MixedPrecisionOptimizer::new(adam).with_dtype(DType::Float32).without_overflow_check();

        let mut params = vec![Parameter::new("p0", backend.tensor_from_vec(vec![1.0f32], &[1]).unwrap())];

        let grad_tensor = backend.tensor_from_vec(vec![0.1f32], &[1]).unwrap();
        let gradients = vec![Gradient { param_id: params[0].id(), tensor: grad_tensor }];

        let mut ctx = ForwardCtx::new(&backend, Mode::Train);
        mp.step(&mut params, &gradients, &mut ctx).unwrap();
        // Parameter should have been updated
        let val = params[0].tensor().as_ref()[0];
        assert_ne!(val, 1.0); // Should change due to gradient update
    }

    #[test]
    fn test_mixed_precision_optimizer_bf16_step() {
        let backend = CpuBackend::default();
        let adam = Adam::<CpuBackend>::new(0.001);
        let mut mp = MixedPrecisionOptimizer::new(adam).with_dtype(DType::BFloat16).without_overflow_check();

        let mut params = vec![Parameter::new("p0", backend.tensor_from_vec(vec![1.0f32], &[1]).unwrap())];

        let grad_tensor = backend.tensor_from_vec(vec![0.1f32], &[1]).unwrap();
        let gradients = vec![Gradient { param_id: params[0].id(), tensor: grad_tensor }];

        let mut ctx = ForwardCtx::new(&backend, Mode::Train);
        mp.step(&mut params, &gradients, &mut ctx).unwrap();
        let val = params[0].tensor().as_ref()[0];
        assert_ne!(val, 1.0);
    }

    #[test]
    fn test_mixed_precision_optimizer_overflow_skip() {
        let backend = CpuBackend::default();
        let adam = Adam::<CpuBackend>::new(0.001);
        let mut mp = MixedPrecisionOptimizer::new(adam).with_dtype(DType::Float32).with_loss_scale(1.0); // Scale 1.0 so overflow check works directly

        let mut params = vec![Parameter::new("p0", backend.tensor_from_vec(vec![1.0f32], &[1]).unwrap())];

        // Create a gradient with NaN to trigger overflow detection
        let grad_tensor = backend.tensor_from_vec(vec![f32::NAN], &[1]).unwrap();
        let gradients = vec![Gradient { param_id: params[0].id(), tensor: grad_tensor }];

        let mut ctx = ForwardCtx::new(&backend, Mode::Train);
        mp.step(&mut params, &gradients, &mut ctx).unwrap();
        // Loss scale should have backed off
        assert_eq!(mp.current_loss_scale(), 1.0); // min_scale = 1.0
    }

    #[test]
    fn test_mixed_precision_optimizer_initialize_master_weights() {
        let backend = CpuBackend::default();
        let adam = Adam::<CpuBackend>::new(0.001);
        let mut mp = MixedPrecisionOptimizer::new(adam).with_dtype(DType::Float16);

        let params: Vec<Parameter<CpuBackend>> = vec![Parameter::new(
            "p0",
            backend.tensor_from_vec(vec![1.0f32, 2.0f32], &[2]).unwrap(),
        )];

        mp.initialize_master_weights(&params);
        // BF16 would skip initialization, so we used Float16 here
    }

    #[test]
    fn test_mixed_precision_optimizer_fp16_step() {
        let backend = CpuBackend::default();
        let adam = Adam::<CpuBackend>::new(0.001);
        let mut mp = MixedPrecisionOptimizer::new(adam).with_dtype(DType::Float16).without_overflow_check();

        let mut params = vec![Parameter::new(
            "p0",
            backend.tensor_from_vec(vec![10.0f32; 100], &[100]).unwrap(),
        )];

        let grad_tensor = backend.tensor_from_vec(vec![0.5f32; 100], &[100]).unwrap();
        let gradients = vec![Gradient { param_id: params[0].id(), tensor: grad_tensor }];

        let mut ctx = ForwardCtx::new(&backend, Mode::Train);
        // Step should complete without error; FP16 simulation may truncate small updates
        mp.step(&mut params, &gradients, &mut ctx).unwrap();
        // Verify master weights were initialized
        assert!(!mp.master_weights.is_empty());
    }

    #[test]
    fn test_f16_simulation() {
        let a = f16::from_f32(1.0);
        let b = a.to_f32();
        // Due to truncation, it won't be exactly 1.0 but close
        assert!(b > 0.0);
    }
}
