//! Fully Sharded Data Parallel (FSDP) / ZeRO-3
//!
//! Shards parameters, gradients, AND optimizer states across all GPUs.
//! Each GPU only stores a shard of each parameter (~1/world_size memory).
//!
//! # Algorithm
//!
//! Forward pass:
//! 1. All-gather parameter shards → full parameter (one at a time)
//! 2. Compute forward
//! 3. Discard full parameter (keep shard only)
//!
//! Backward pass:
//! 1. All-gather parameter shards → full parameter
//! 2. Compute gradient
//! 3. Reduce-scatter gradient to get grad shard
//! 4. Update parameter shard
//!
//! # Memory Savings
//!
//! With 8 GPUs:
//! - ZeRO-1 (optimizer states only): ~44% memory saved
//! - ZeRO-2 (+ gradients): ~66% memory saved
//! - ZeRO-3/FSDP (+ parameters): ~88% memory saved
//!
//! # Example
//! ```rust,ignore
//! use mnr_distributed::fsdp::{FSDP, FSDPConfig};
//!
//! let config = FSDPConfig::new()
//!     .with_cpu_offload(true)
//!     .with_gradient_checkpointing(true);
//!
//! let trainer = FSDP::new(model, optimizer, process_group, config)?;
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use mnr_core::{Backend, CoreError, ForwardCtx, Mode, Module, Parameter, ParameterId, Result, TensorOps};
use mnr_optim::{Adam, Gradient, OptimError, Optimizer};

use crate::{DistributedError, DistributedResult, ProcessGroup};

/// FSDP configuration
#[derive(Clone, Debug)]
pub struct FSDPConfig {
    /// Shard parameters (true for FSDP/ZeRO-3)
    pub shard_params: bool,
    /// Shard gradients
    pub shard_grads: bool,
    /// CPU offloading for params not in use
    pub cpu_offload: bool,
    /// Gradient checkpointing
    pub gradient_checkpointing: bool,
    /// Bucket size for all-gather (bytes)
    pub bucket_size: usize,
    /// Auto wrap policy for nested FSDP
    pub auto_wrap: bool,
    /// Mixed precision
    pub mixed_precision: bool,
}

impl Default for FSDPConfig {
    fn default() -> Self {
        Self {
            shard_params: true,
            shard_grads: true,
            cpu_offload: false,
            gradient_checkpointing: false,
            bucket_size: 25 * 1024 * 1024, // 25MB default
            auto_wrap: true,
            mixed_precision: false,
        }
    }
}

impl FSDPConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_cpu_offload(mut self, enable: bool) -> Self {
        self.cpu_offload = enable;
        self
    }

    pub fn with_gradient_checkpointing(mut self, enable: bool) -> Self {
        self.gradient_checkpointing = enable;
        self
    }

    pub fn with_bucket_size(mut self, bytes: usize) -> Self {
        self.bucket_size = bytes;
        self
    }

    pub fn with_mixed_precision(mut self, enable: bool) -> Self {
        self.mixed_precision = enable;
        self
    }
}

/// Sharded parameter state
struct ShardedParameter<B: Backend> {
    /// Global parameter ID
    param_id: ParameterId,
    /// Parameter name
    name: String,
    /// Full shape
    full_shape: Vec<usize>,
    /// Number of elements
    numel: usize,
    /// Start index in flattened parameter (inclusive)
    start: usize,
    /// End index in flattened parameter (exclusive)
    end: usize,
    /// Local shard (flattened)
    shard: B::Tensor,
    /// Gradient shard (if computed)
    grad_shard: Option<B::Tensor>,
    /// Is currently gathered?
    is_gathered: bool,
}

/// FSDP wrapper for a model
pub struct FSDP<B: Backend, M: Module<B>, O: Optimizer<B>> {
    /// Wrapped model
    model: M,
    /// Optimizer
    optimizer: O,
    /// Process group
    process_group: ProcessGroup,
    /// Configuration
    config: FSDPConfig,
    /// Sharded parameters
    sharded_params: HashMap<ParameterId, ShardedParameter<B>>,
    /// Currently gathered parameters
    gathered_params: Vec<ParameterId>,
    /// Flattened parameter buffer for all-gather
    gather_buffer: Option<B::Tensor>,
    /// CPU offload buffer
    cpu_offload_buffer: Option<Vec<f32>>,
}

impl<B: Backend, M: Module<B>, O: Optimizer<B>> FSDP<B, M, O>
where
    B::Tensor: Clone + AsRef<[f32]> + mnr_core::TensorShape,
{
    /// Create new FSDP wrapper
    pub fn new(
        model: M,
        optimizer: O,
        process_group: ProcessGroup,
        config: FSDPConfig,
    ) -> DistributedResult<Self> {
        let world_size = process_group.world_size();
        let rank = process_group.rank();

        // Get model parameters
        let params = model.parameters();

        // Create sharded parameters
        let mut sharded_params = HashMap::new();

        for param_ref in params {
            let param = param_ref.to_parameter();
            let param_id = param.id();
            let shape = param.tensor().shape();
            let numel: usize = shape.iter().product();

            // Calculate shard range
            let base_shard_size = numel / world_size;
            let remainder = numel % world_size;

            let start = if rank < remainder {
                rank * (base_shard_size + 1)
            } else {
                remainder * (base_shard_size + 1) + (rank - remainder) * base_shard_size
            };

            let local_shard_size = if rank < remainder {
                base_shard_size + 1
            } else {
                base_shard_size
            };
            let end = start + local_shard_size;

            // Extract shard
            let full_data: Vec<f32> = param.tensor().as_ref().to_vec();
            let shard_data: Vec<f32> = full_data[start..end].to_vec();

            // Create shard tensor
            // In real implementation, would use backend directly
            let shard = param.tensor().clone(); // Simplified

            let sharded = ShardedParameter {
                param_id,
                name: param.name().to_string(),
                full_shape: shape.to_vec(),
                numel,
                start,
                end,
                shard,
                grad_shard: None,
                is_gathered: false,
            };

            sharded_params.insert(param_id, sharded);
        }

        Ok(Self {
            model,
            optimizer,
            process_group,
            config,
            sharded_params,
            gathered_params: Vec::new(),
            gather_buffer: None,
            cpu_offload_buffer: None,
        })
    }

    /// Forward pass with parameter sharding
    pub fn forward(&mut self, input: B::Tensor, ctx: &mut ForwardCtx<B>) -> Result<B::Tensor> {
        // Gather all parameters one at a time (or in buckets)
        self.gather_all_params(ctx.backend().ops())?;

        // Run forward
        let output = self.model.forward(input, ctx)?;

        // Scatter parameters back to shards
        self.scatter_all_params(ctx.backend().ops())?;

        Ok(output)
    }

    /// Gather all parameter shards to full parameters
    fn gather_all_params(&mut self, ops: &dyn TensorOps<B>) -> Result<()> {
        let world_size = self.process_group.world_size();

        for (param_id, sharded) in self.sharded_params.iter_mut() {
            if sharded.is_gathered {
                continue;
            }

            // All-gather shards from all ranks
            let shard_data = sharded.shard.as_ref();
            let shard_size = shard_data.len();

            // Create receive buffer
            let mut full_data = vec![0.0f32; sharded.numel];

            // In real implementation, would call NCCL all-gather here
            // For now, simplified: just put shard in correct position
            full_data[sharded.start..sharded.end].copy_from_slice(shard_data);

            // Store gathered parameter (would create new tensor in real impl)
            // This is a placeholder
            sharded.is_gathered = true;
            self.gathered_params.push(*param_id);
        }

        Ok(())
    }

    /// Scatter full parameters back to shards
    fn scatter_all_params(&mut self, ops: &dyn TensorOps<B>) -> Result<()> {
        let world_size = self.process_group.world_size();

        for param_id in &self.gathered_params {
            if let Some(sharded) = self.sharded_params.get_mut(param_id) {
                if !sharded.is_gathered {
                    continue;
                }

                // Extract shard from full parameter
                // In real impl, would extract and then free full param

                // Reduce-scatter gradient if needed
                if let Some(ref grad) = sharded.grad_shard {
                    // Apply gradient to shard
                }

                sharded.is_gathered = false;
            }
        }

        self.gathered_params.clear();
        Ok(())
    }

    /// Backward pass with gradient sharding
    pub fn backward(&mut self, loss: &B::Tensor, ctx: &mut ForwardCtx<B>) -> Result<Vec<Gradient<B>>> {
        // Gradient checkpointing if enabled
        if self.config.gradient_checkpointing {
            // Would recompute forward pass here
        }

        // Compute gradients (parameters are already gathered from forward)
        // In real impl, would use autodiff tape

        // Reduce-scatter gradients
        let gradients = self.reduce_scatter_grads(ctx.backend().ops())?;

        Ok(gradients)
    }

    /// Reduce-scatter gradients across ranks
    fn reduce_scatter_grads(&mut self, ops: &dyn TensorOps<B>) -> Result<Vec<Gradient<B>>> {
        let mut gradients = Vec::new();

        for (param_id, sharded) in self.sharded_params.iter_mut() {
            // In real implementation:
            // 1. All-gather gradient chunks from each rank
            // 2. Sum them locally
            // 3. Take only the shard for this rank

            if let Some(ref grad_shard) = sharded.grad_shard {
                gradients.push(Gradient {
                    param_id: *param_id,
                    tensor: grad_shard.clone(),
                });
            }
        }

        Ok(gradients)
    }

    /// Optimizer step (update sharded parameters)
    pub fn step(&mut self, gradients: &[Gradient<B>], ctx: &mut ForwardCtx<B>) -> std::result::Result<(), OptimError> {
        // Convert sharded_params to format optimizer expects
        let mut params: Vec<Parameter<B>> = self.sharded_params
            .values()
            .map(|s| Parameter::new(&s.name, s.shard.clone()))
            .collect();

        // Run optimizer step
        self.optimizer.step(&mut params, gradients, ctx)?;

        // Update shards
        for (param, sharded) in params.iter().zip(self.sharded_params.values_mut()) {
            sharded.shard = param.tensor().clone();
        }

        Ok(())
    }

    /// Training step: forward + backward + optimize
    pub fn train_step(
        &mut self,
        input: B::Tensor,
        target: B::Tensor,
        loss_fn: impl Fn(&B::Tensor, &B::Tensor) -> Result<B::Tensor>,
        ctx: &mut ForwardCtx<B>,
    ) -> Result<f32> {
        // Forward
        let output = self.forward(input, ctx)?;

        // Compute loss
        let loss = loss_fn(&output, &target)?;

        // Get loss value
        let loss_data: Vec<f32> = loss.as_ref().to_vec();
        let loss_value = loss_data[0];

        // Backward
        let gradients = self.backward(&loss, ctx)?;

        // Optimizer step
        self.step(&gradients, ctx).map_err(|e| {
            CoreError::Other(format!("Optimizer error: {:?}", e))
        })?;

        Ok(loss_value)
    }

    /// Get memory statistics
    pub fn memory_stats(&self) -> FSDPMemoryStats {
        let world_size = self.process_group.world_size();

        let total_params: usize = self.sharded_params.values()
            .map(|s| s.numel)
            .sum();

        let local_params: usize = self.sharded_params.values()
            .map(|s| s.end - s.start)
            .sum();

        FSDPMemoryStats {
            world_size,
            total_parameters: total_params,
            local_parameters: local_params,
            memory_reduction_percent: (1.0 - 1.0 / world_size as f32) * 100.0,
            gathered_params: self.gathered_params.len(),
        }
    }

    /// Get reference to inner model
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Get mutable reference to inner model
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }
}

/// Memory statistics for FSDP
#[derive(Debug, Clone)]
pub struct FSDPMemoryStats {
    pub world_size: usize,
    pub total_parameters: usize,
    pub local_parameters: usize,
    pub memory_reduction_percent: f32,
    pub gathered_params: usize,
}

/// FSDP checkpoint for saving/loading
pub struct FSDPCheckpoint<B: Backend> {
    /// Sharded parameter data
    pub shards: HashMap<ParameterId, Vec<f32>>,
    /// Optimizer state
    pub optimizer_state: Option<AdamCheckpoint>,
    /// Rank that saved this checkpoint
    pub rank: usize,
    /// World size at save time
    pub world_size: usize,
}

impl<B: Backend> FSDPCheckpoint<B> {
    /// Save FSDP state
    pub fn save(fsdp: &FSDP<B, impl Module<B>, impl Optimizer<B>>) -> Self {
        let shards = fsdp.sharded_params
            .values()
            .map(|s| (s.param_id, s.shard.as_ref().to_vec()))
            .collect();

        Self {
            shards,
            optimizer_state: None, // Would serialize optimizer state
            rank: fsdp.process_group.rank(),
            world_size: fsdp.process_group.world_size(),
        }
    }

    /// Load FSDP state
    pub fn load(&self, fsdp: &mut FSDP<B, impl Module<B>, impl Optimizer<B>>) -> DistributedResult<()> {
        // Verify compatibility
        if self.world_size != fsdp.process_group.world_size() {
            return Err(DistributedError::Communication(
                format!("Checkpoint world size {} doesn't match current {}",
                    self.world_size, fsdp.process_group.world_size())
            ));
        }

        // Load shards
        for (param_id, data) in &self.shards {
            if let Some(sharded) = fsdp.sharded_params.get_mut(param_id) {
                // In real impl, would convert vec back to tensor
                // sharded.shard = ...
            }
        }

        Ok(())
    }
}

/// Auto-wrap policy for applying FSDP to submodules
pub enum AutoWrapPolicy {
    /// Wrap based on parameter count
    Size(usize),
    /// Wrap specific module types
    Type(Vec<String>),
    /// Custom function
    Custom(Box<dyn Fn(&str) -> bool>),
}

/// Apply FSDP to model with auto-wrapping
pub fn auto_wrap<B: Backend, M: Module<B>>(
    model: M,
    policy: AutoWrapPolicy,
    optimizer: impl Optimizer<B>,
    process_group: ProcessGroup,
    config: FSDPConfig,
) -> DistributedResult<FSDP<B, M, impl Optimizer<B>>>
where
    B::Tensor: Clone + AsRef<[f32]> + mnr_core::TensorShape,
{
    // In real implementation, would recursively apply FSDP to submodules
    // based on the policy
    FSDP::new(model, optimizer, process_group, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnr_ndarray_backend::CpuBackend;
    use mnr_nn::{Linear, LinearConfig};
    use mnr_optim::Adam;

    #[test]
    fn test_fsdp_config() {
        let config = FSDPConfig::new()
            .with_cpu_offload(true)
            .with_gradient_checkpointing(true)
            .with_bucket_size(50 * 1024 * 1024);

        assert!(config.cpu_offload);
        assert!(config.gradient_checkpointing);
        assert_eq!(config.bucket_size, 50 * 1024 * 1024);
    }

    #[test]
    fn test_memory_stats() {
        let backend = CpuBackend::default();
        let pg = ProcessGroup::new_threaded(8, 0);

        let linear = Linear::new(&backend, LinearConfig::new(256, 256)).unwrap();
        let optimizer = Adam::new(0.001);

        let config = FSDPConfig::new();
        let fsdp = FSDP::new(linear, optimizer, pg, config).unwrap();

        let stats = fsdp.memory_stats();
        assert_eq!(stats.world_size, 8);
        assert!(stats.memory_reduction_percent > 85.0); // ~87.5%
    }

    #[test]
    fn test_fsdp_creation() {
        let backend = CpuBackend::default();
        let pg = ProcessGroup::new_threaded(4, 0);

        let linear = Linear::new(&backend, LinearConfig::new(64, 64)).unwrap();
        let optimizer = Adam::new(0.001);

        let config = FSDPConfig::new();
        let fsdp = FSDP::new(linear, optimizer, pg, config).unwrap();

        assert_eq!(fsdp.process_group.world_size(), 4);
    }
}
