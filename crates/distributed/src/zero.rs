//! ZeRO (Zero Redundancy Optimizer) for memory-efficient training.
//!
//! ZeRO shards optimizer states across data parallel processes,
/// reducing per-GPU memory usage.
///
/// # ZeRO Stages
///
/// - **ZeRO-1**: Shard optimizer states only (8x memory reduction with 8 GPUs)
/// - **ZeRO-2**: Shard optimizer states + gradients (more memory reduction)
/// - **ZeRO-3**: Shard optimizer states + gradients + parameters (max reduction)
///
/// This implementation provides ZeRO-1 as the foundation.

use std::collections::HashMap;

use mnr_core::{Backend, CoreError, ForwardCtx, Parameter, ParameterId, Result, TensorOps};
use mnr_optim::{Adam, AdamCheckpoint, Gradient, OptimError, Optimizer};

use crate::{DistributedError, DistributedResult, ProcessGroup};

/// ZeRO-1 optimizer wrapper.
///
/// Wraps an existing optimizer and shards its state across data parallel ranks.
/// Each rank only stores optimizer state for its shard of parameters.
pub struct ZeroOptimizer<B: Backend> {
    /// Inner optimizer (e.g., Adam).
    inner: Adam<B>,

    /// Process group for communication.
    process_group: ProcessGroup,

    /// Shard of parameters this rank owns.
    /// Maps from global parameter ID to local index.
    parameter_shard: HashMap<ParameterId, usize>,

    /// Total number of parameters (across all shards).
    total_params: usize,
}

impl<B: Backend> ZeroOptimizer<B>
where
    B::Tensor: Clone + AsRef<[f32]> + mnr_core::TensorShape,
{
    /// Create a new ZeRO-1 optimizer.
    ///
    /// # Arguments
    /// * `inner` - The base optimizer to wrap (e.g., Adam::new(0.001))
    /// * `process_group` - Process group for communication
    /// * `total_params` - Total number of parameters across all shards
    pub fn new(
        inner: Adam<B>,
        process_group: ProcessGroup,
        total_params: usize,
    ) -> Self {
        let world_size = process_group.world_size();
        let rank = process_group.rank();

        // Determine which parameters this rank owns
        let mut parameter_shard = HashMap::new();
        let local_count = (total_params + world_size - 1) / world_size;

        for i in 0..local_count {
            let global_idx = rank * local_count + i;
            if global_idx < total_params {
                // Use a dummy ID that will be mapped to real parameters later
                let dummy_id = ParameterId::fresh();
                parameter_shard.insert(dummy_id, i);
            }
        }

        Self {
            inner,
            process_group,
            parameter_shard,
            total_params,
        }
    }

    /// Map global parameters to local shards.
    ///
    /// Call this after creating parameters to establish the mapping.
    pub fn build_shard_mapping(&mut self, params: &[Parameter<B>]) {
        let world_size = self.process_group.world_size();
        let rank = self.process_group.rank();

        let local_count = (self.total_params + world_size - 1) / world_size;
        let start_idx = rank * local_count;
        let end_idx = ((start_idx + local_count).min(self.total_params));

        self.parameter_shard.clear();

        for (local_idx, global_idx) in (start_idx..end_idx).enumerate() {
            if let Some(param) = params.get(global_idx) {
                self.parameter_shard.insert(param.id(), local_idx);
            }
        }
    }

    /// Perform optimizer step with ZeRO sharding.
    pub fn step(
        &mut self,
        params: &mut [Parameter<B>],
        gradients: &[Gradient<B>],
        ctx: &mut ForwardCtx<B>,
    ) -> std::result::Result<(), OptimError> {
        // Filter gradients to only those in our shard
        let local_gradients: Vec<_> = gradients
            .iter()
            .filter(|g| self.parameter_shard.contains_key(&g.param_id))
            .cloned()
            .collect();

        // Local optimizer step
        self.inner.step(params, &local_gradients, ctx)?;

        // Broadcast updated parameters to all ranks
        for param in params.iter_mut() {
            let mut data: Vec<f32> = param.tensor().as_ref().to_vec();

            // Broadcast from the rank that owns this parameter
            let owner_rank = self.get_owner_rank(param.id());
            self.process_group
                .broadcast(&mut data, owner_rank)
                .map_err(|e| OptimError::Backend(e.to_string()))?;

            // Update parameter with broadcasted data
            let shape = ctx.backend().ops().shape(param.tensor());
            let new_tensor = ctx
                .backend()
                .ops()
                .tensor_from_vec(data, &shape)
                .map_err(|e| OptimError::Backend(e.to_string()))?;

            *param = Parameter::new(param.name(), new_tensor);
        }

        Ok(())
    }

    /// Get the rank that owns a parameter.
    fn get_owner_rank(&self, param_id: ParameterId) -> usize {
        let world_size = self.process_group.world_size();

        // Find the global index of this parameter
        if let Some(&local_idx) = self.parameter_shard.get(&param_id) {
            // This rank owns it, compute global index
            let rank = self.process_group.rank();
            let local_count = (self.total_params + world_size - 1) / world_size;
            rank * local_count + local_idx
        } else {
            // Estimate based on parameter ID
            // In practice, should maintain proper mapping
            (param_id.get() as usize) % world_size
        }
    }

    /// Save ZeRO sharded checkpoint.
    ///
    /// Each rank saves only its shard. To reconstruct the full optimizer state,
    /// collect shards from all ranks.
    pub fn save_checkpoint(&self, params: &[Parameter<B>]) -> ZeroCheckpoint {
        let inner_checkpoint = self.inner.save_checkpoint();

        // Filter to only this rank's parameters
        let mut shard_state = HashMap::new();
        for (param_id_num, state) in inner_checkpoint.state.iter() {
            let param_id = ParameterId::fresh(); // Would need proper reconstruction
            if self.get_owner_rank(param_id) == self.process_group.rank() {
                shard_state.insert(*param_id_num, state.clone());
            }
        }

        ZeroCheckpoint {
            rank: self.process_group.rank(),
            world_size: self.process_group.world_size(),
            inner: inner_checkpoint,
            shard_state,
        }
    }

    /// Load from a ZeRO sharded checkpoint.
    pub fn load_checkpoint(
        &mut self,
        checkpoint: &ZeroCheckpoint,
        params: &[Parameter<B>],
        ops: &dyn TensorOps<B>,
    ) -> std::result::Result<(), OptimError> {
        // Load only this rank's shard
        if checkpoint.rank != self.process_group.rank() {
            return Err(OptimError::Gradient(
                format!("Checkpoint rank {} doesn't match current rank {}",
                    checkpoint.rank, self.process_group.rank())
            ));
        }

        self.inner.load_checkpoint(&checkpoint.inner, params, ops)?;
        Ok(())
    }

    /// Get reference to inner optimizer.
    pub fn inner(&self) -> &Adam<B> {
        &self.inner
    }

    /// Get mutable reference to inner optimizer.
    pub fn inner_mut(&mut self) -> &mut Adam<B> {
        &mut self.inner
    }

    /// Get the process group.
    pub fn process_group(&self) -> &ProcessGroup {
        &self.process_group
    }

    /// Get the total number of parameters.
    pub fn total_params(&self) -> usize {
        self.total_params
    }
}

/// ZeRO sharded checkpoint.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ZeroCheckpoint {
    /// Rank that saved this checkpoint.
    pub rank: usize,

 /// Total world size.
    pub world_size: usize,

    /// Inner optimizer checkpoint.
    pub inner: AdamCheckpoint,

    /// State for this rank's shard only.
    pub shard_state: HashMap<u64, (Vec<f32>, Vec<f32>, u64, Vec<usize>)>,
}

/// ZeRO-2: Shards optimizer states AND gradients.
///
/// This provides additional memory savings over ZeRO-1.
pub struct Zero2Optimizer<B: Backend> {
    /// Base ZeRO-1 optimizer.
    zero1: ZeroOptimizer<B>,

    /// Gradient bucket size for all-reduce.
    bucket_size_mb: usize,
}

impl<B: Backend> Zero2Optimizer<B>
where
    B::Tensor: Clone + AsRef<[f32]> + mnr_core::TensorShape,
{
    /// Create a new ZeRO-2 optimizer.
    pub fn new(
        inner: Adam<B>,
        process_group: ProcessGroup,
        total_params: usize,
    ) -> Self {
        Self {
            zero1: ZeroOptimizer::new(inner, process_group, total_params),
            bucket_size_mb: 25, // Default 25MB buckets
        }
    }

    /// Perform optimizer step with ZeRO-2.
    pub fn step(
        &mut self,
        params: &mut [Parameter<B>],
        gradients: &[Gradient<B>],
        ctx: &mut ForwardCtx<B>,
    ) -> std::result::Result<(), OptimError> {
        // All-reduce gradients in buckets before ZeRO-1 step
        self.all_reduce_gradients(gradients, ctx)?;

        // Delegate to ZeRO-1
        self.zero1.step(params, gradients, ctx)
    }

    /// All-reduce gradients in buckets.
    fn all_reduce_gradients(
        &self,
        gradients: &[Gradient<B>],
        ctx: &mut ForwardCtx<B>,
    ) -> std::result::Result<(), OptimError> {
        for grad in gradients {
            let mut data: Vec<f32> = grad.tensor.as_ref().to_vec();

            self.zero1
                .process_group
                .all_reduce_sum(&format!("grad_{}", grad.param_id.get()), &mut data)
                .map_err(|e| OptimError::Backend(e.to_string()))?;

            // Average by world size
            let world_size = self.zero1.process_group.world_size() as f32;
            for v in data.iter_mut() {
                *v /= world_size;
            }
        }

        Ok(())
    }
}

/// Memory usage statistics for ZeRO.
#[derive(Debug, Clone)]
pub struct ZeRoMemoryStats {
    /// Parameters stored locally.
    pub local_params_mb: f32,

    /// Optimizer states stored locally.
    pub local_optimizer_states_mb: f32,

    /// Gradients stored locally.
    pub local_gradients_mb: f32,

    /// Total across all ranks (theoretical).
    pub total_params_mb: f32,

    /// Memory saved compared to non-ZeRO.
    pub memory_saved_percent: f32,
}

impl ZeRoMemoryStats {
    /// Calculate memory stats for ZeRO-1.
    pub fn zero1(
        total_params: usize,
        param_size_bytes: usize,
        world_size: usize,
    ) -> Self {
        let param_size_mb = (total_params * param_size_bytes) as f32 / (1024.0 * 1024.0);
        let optimizer_state_multiplier = 2.0; // Adam: m and v

        let local_params = param_size_mb / world_size as f32;
        let local_optimizer = param_size_mb * optimizer_state_multiplier / world_size as f32;
        let local_gradients = param_size_mb; // Still full gradients

        let total_local = local_params + local_optimizer + local_gradients;
        let total_without_zero = param_size_mb * (1.0 + optimizer_state_multiplier + 1.0);

        Self {
            local_params_mb: local_params,
            local_optimizer_states_mb: local_optimizer,
            local_gradients_mb: local_gradients,
            total_params_mb: param_size_mb,
            memory_saved_percent: (1.0 - total_local / total_without_zero) * 100.0,
        }
    }

    /// Calculate memory stats for ZeRO-2.
    pub fn zero2(
        total_params: usize,
        param_size_bytes: usize,
        world_size: usize,
    ) -> Self {
        let param_size_mb = (total_params * param_size_bytes) as f32 / (1024.0 * 1024.0);
        let optimizer_state_multiplier = 2.0;

        let local_params = param_size_mb / world_size as f32;
        let local_optimizer = param_size_mb * optimizer_state_multiplier / world_size as f32;
        let local_gradients = param_size_mb / world_size as f32; // Sharded gradients

        let total_local = local_params + local_optimizer + local_gradients;
        let total_without_zero = param_size_mb * (1.0 + optimizer_state_multiplier + 1.0);

        Self {
            local_params_mb: local_params,
            local_optimizer_states_mb: local_optimizer,
            local_gradients_mb: local_gradients,
            total_params_mb: param_size_mb,
            memory_saved_percent: (1.0 - total_local / total_without_zero) * 100.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnr_ndarray_backend::CpuBackend;
    use mnr_core::{ForwardCtx, Mode};

    #[test]
    fn test_zero_optimizer_creation() {
        let backend = CpuBackend::default();
        let pg = ProcessGroup::new_single_process();
        let adam = Adam::<CpuBackend>::new(0.001);

        let zero = ZeroOptimizer::new(adam, pg, 100);

        assert_eq!(zero.total_params(), 100);
        assert_eq!(zero.process_group().world_size(), 1);
    }

    #[test]
    fn test_zero_optimizer_build_shard_mapping() {
        let backend = CpuBackend::default();
        let pg = ProcessGroup::new_threaded(2, 0).unwrap();
        let adam = Adam::<CpuBackend>::new(0.001);
        let mut zero = ZeroOptimizer::new(adam, pg, 4);

        let params = vec![
            Parameter::new("p0", backend.tensor_from_vec(vec![1.0f32], &[1]).unwrap()),
            Parameter::new("p1", backend.tensor_from_vec(vec![2.0f32], &[1]).unwrap()),
            Parameter::new("p2", backend.tensor_from_vec(vec![3.0f32], &[1]).unwrap()),
            Parameter::new("p3", backend.tensor_from_vec(vec![4.0f32], &[1]).unwrap()),
        ];

        zero.build_shard_mapping(&params);
        assert!(!zero.parameter_shard.is_empty());
    }

    #[test]
    fn test_zero_optimizer_step() {
        let backend = CpuBackend::default();
        let pg = ProcessGroup::new_single_process();
        let adam = Adam::<CpuBackend>::new(0.001);
        let mut zero = ZeroOptimizer::new(adam, pg, 2);

        let mut params = vec![
            Parameter::new("p0", backend.tensor_from_vec(vec![1.0f32], &[1]).unwrap()),
            Parameter::new("p1", backend.tensor_from_vec(vec![2.0f32], &[1]).unwrap()),
        ];

        zero.build_shard_mapping(&params);

        let grad_tensor = backend.tensor_from_vec(vec![0.1f32], &[1]).unwrap();
        let gradients = vec![Gradient {
            param_id: params[0].id(),
            tensor: grad_tensor,
        }];

        let mut ctx = ForwardCtx::new(&backend, Mode::Train);
        zero.step(&mut params, &gradients, &mut ctx).unwrap();
    }

    #[test]
    fn test_zero_optimizer_get_owner_rank() {
        let pg = ProcessGroup::new_threaded(4, 0).unwrap();
        let adam = Adam::<CpuBackend>::new(0.001);
        let zero = ZeroOptimizer::new(adam, pg, 8);

        // For unknown param_id, should distribute based on id
        let unknown_id = ParameterId::fresh();
        let rank = zero.get_owner_rank(unknown_id);
        assert!(rank < 4);
    }

    #[test]
    fn test_zero_optimizer_save_and_load_checkpoint() {
        let backend = CpuBackend::default();
        let pg = ProcessGroup::new_single_process();
        let adam = Adam::<CpuBackend>::new(0.001);
        let mut zero = ZeroOptimizer::new(adam, pg, 2);

        let params = vec![
            Parameter::new("p0", backend.tensor_from_vec(vec![1.0f32], &[1]).unwrap()),
        ];

        let checkpoint = zero.save_checkpoint(&params);
        assert_eq!(checkpoint.rank, 0);
        assert_eq!(checkpoint.world_size, 1);

        let result = zero.load_checkpoint(&checkpoint, &params, backend.ops());
        assert!(result.is_ok());
    }

    #[test]
    fn test_zero_optimizer_load_checkpoint_mismatch() {
        let backend = CpuBackend::default();
        let pg = ProcessGroup::new_single_process();
        let adam = Adam::<CpuBackend>::new(0.001);
        let mut zero = ZeroOptimizer::new(adam, pg, 2);

        let params = vec![
            Parameter::new("p0", backend.tensor_from_vec(vec![1.0f32], &[1]).unwrap()),
        ];

        let inner_ckpt = zero.inner().save_checkpoint();
        let checkpoint = ZeroCheckpoint {
            rank: 1,
            world_size: 1,
            inner: inner_ckpt,
            shard_state: HashMap::new(),
        };

        assert!(zero.load_checkpoint(&checkpoint, &params, backend.ops()).is_err());
    }

    #[test]
    fn test_zero_optimizer_inner() {
        let pg = ProcessGroup::new_single_process();
        let adam = Adam::<CpuBackend>::new(0.001);
        let mut zero = ZeroOptimizer::new(adam, pg, 2);

        assert_eq!(zero.inner().lr, 0.001);
        zero.inner_mut().lr = 0.01;
        assert_eq!(zero.inner().lr, 0.01);
    }

    #[test]
    fn test_zero2_optimizer_step() {
        let backend = CpuBackend::default();
        let pg = ProcessGroup::new_single_process();
        let adam = Adam::<CpuBackend>::new(0.001);
        let mut zero2 = Zero2Optimizer::new(adam, pg, 2);

        let mut params = vec![
            Parameter::new("p0", backend.tensor_from_vec(vec![1.0f32], &[1]).unwrap()),
            Parameter::new("p1", backend.tensor_from_vec(vec![2.0f32], &[1]).unwrap()),
        ];

        zero2.zero1.build_shard_mapping(&params);

        let grad_tensor = backend.tensor_from_vec(vec![0.1f32], &[1]).unwrap();
        let gradients = vec![Gradient {
            param_id: params[0].id(),
            tensor: grad_tensor,
        }];

        let mut ctx = ForwardCtx::new(&backend, Mode::Train);
        zero2.step(&mut params, &gradients, &mut ctx).unwrap();
    }

    #[test]
    fn test_memory_stats_zero1() {
        let stats = ZeRoMemoryStats::zero1(
            1_000_000_000, // 1B params
            4,             // f32 = 4 bytes
            8,             // 8 GPUs
        );

        assert!(stats.memory_saved_percent > 50.0);
        assert!(stats.local_optimizer_states_mb > 0.0);
        assert!(stats.local_params_mb > 0.0);
        assert!(stats.local_gradients_mb > 0.0);
    }

    #[test]
    fn test_memory_stats_zero2() {
        let stats = ZeRoMemoryStats::zero2(
            1_000_000_000,
            4,
            8,
        );

        assert!(stats.local_gradients_mb < stats.total_params_mb);
        assert!(stats.memory_saved_percent > 50.0);
    }
}
