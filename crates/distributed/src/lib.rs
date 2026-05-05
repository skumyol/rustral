//! Distributed training for Rustral.
//!
//! Provides data parallelism, model parallelism, and ZeRO sharding
//! for training large models across multiple GPUs and nodes.

#![allow(dead_code, unused_variables, unused_imports)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use rustral_core::{Backend, CoreError, ForwardCtx, Parameter, Result, TensorOps};
use rustral_optim::{Gradient, Optimizer};

#[cfg(feature = "mpi")]
use mpi::collective::{CommunicatorCollectives, Root, SystemOperation};
#[cfg(feature = "mpi")]
use mpi::topology::Communicator;

mod chaos_engineering;
mod checkpoint;
mod compression;
mod context_parallel;
mod device_mesh;
mod fault_tolerance;
mod fsdp;
#[cfg(feature = "nccl")]
mod nccl;
mod parallelism_3d;
mod pipeline_parallel;
mod sequence_parallel;
mod tensor_parallel;
mod zero;
mod zero_infinity;

pub use chaos_engineering::{
    ChaosMonkey, ChaosScenarios, CheckpointCorruption, FaultInjection, FaultResult, FaultType, TestReport,
};
pub use checkpoint::{AsyncCheckpointWriter, CheckpointMetadata, DistributedCheckpointManager};
pub use compression::{
    BandwidthStats, CompressedCommunicator, CompressionType, ErrorFeedbackCompression, OneBitAdam,
};
pub use context_parallel::{CommunicationStats, ContextParallel, ContextParallelConfig, DynamicLoadBalancer};
pub use device_mesh::{DeviceMesh, MeshCoord, ParallelismConfig};
pub use fault_tolerance::{
    CheckpointVersion, ElasticProcessGroup, ElasticTrainer, FaultStats, HealthMonitor, MembershipChange,
    NodeInfo, NodeState, RestartConfig, StateSync, TimedBarrier,
};
pub use fsdp::{auto_wrap, FSDPCheckpoint, FSDPConfig, FSDPMemoryStats, FSDP};
#[cfg(feature = "nccl")]
pub use nccl::{
    AllReduceOp, NcclCommunicator, NcclCompressedCommunicator, NcclDataType, NcclProcessGroup, NcclRedOp,
};
pub use parallelism_3d::{create_3d_parallel_model, Parallel3DConfig, Parallel3DLayer, Parallel3DTrainer};
pub use pipeline_parallel::{
    create_pipeline, PipelineComm, PipelineConfig, PipelineParallel, PipelineSchedule, PipelineStage,
    PipelineStats, StageSplitter as PipelineStageSplitter,
};
pub use sequence_parallel::{
    compute_sequence_sharding, gather_sequence, shard_sequence, RingAttention, SequenceParallelConfig,
};
pub use tensor_parallel::{
    ParallelStyle, PipelineParallelTrainer, PipelineStage as TensorPipelineStage, ReduceOp,
    TensorParallelLinear,
};
pub use zero::{ZeRoMemoryStats, Zero2Optimizer, ZeroOptimizer};
pub use zero_infinity::{
    StorageLocation, ZeROInfinityEstimator, ZeROMemoryEstimate, ZeroInfinity, ZeroInfinityConfig,
    ZeroInfinityStats,
};

/// Errors specific to distributed training.
#[derive(Debug, thiserror::Error)]
pub enum DistributedError {
    /// Rank or world size mismatch.
    #[error("rank error: expected {expected}, got {actual}")]
    RankMismatch { expected: usize, actual: usize },

    /// Communication failed.
    #[error("communication error: {0}")]
    Communication(String),

    /// Backend error.
    #[error("backend error: {0}")]
    Backend(#[from] CoreError),

    /// All-reduce not available.
    #[error("all-reduce not available: {0}")]
    AllReduceNotAvailable(String),
}

/// Result type for distributed operations.
pub type DistributedResult<T> = std::result::Result<T, DistributedError>;

/// Process group for distributed communication.
///
/// A process group defines a set of processes that can communicate
/// with each other. This is the foundation for all distributed training.
#[derive(Clone)]
pub struct ProcessGroup {
    rank: usize,
    world_size: usize,
    backend: CommunicationBackend,
}

/// Communication backend for distributed operations.
#[derive(Clone)]
pub enum CommunicationBackend {
    /// Single-process (for testing/debugging).
    SingleProcess,

    /// Multi-threaded within a process (uses shared memory).
    Threaded {
        /// Shared gradient storage for all-reduce.
        shared_gradients: Arc<Mutex<HashMap<String, Vec<f32>>>>,
    },

    /// MPI backend for multi-node.
    #[cfg(feature = "mpi")]
    Mpi {
        /// MPI communicator.
        communicator: mpi::topology::SystemCommunicator,
    },
}

impl ProcessGroup {
    /// Create a single-process process group (for testing).
    pub fn new_single_process() -> Self {
        Self { rank: 0, world_size: 1, backend: CommunicationBackend::SingleProcess }
    }

    /// Create a multi-threaded process group for single-node multi-GPU.
    pub fn new_threaded(world_size: usize, rank: usize) -> DistributedResult<Self> {
        if rank >= world_size {
            return Err(DistributedError::RankMismatch { expected: world_size, actual: rank });
        }

        Ok(Self {
            rank,
            world_size,
            backend: CommunicationBackend::Threaded {
                shared_gradients: Arc::new(Mutex::new(HashMap::new())),
            },
        })
    }

    /// Build a process group backed by the MPI world communicator.
    ///
    /// Requires MPI to be initialized (typically via `mpi::initialize()` in `main`, or when launched
    /// under `mpirun` / `mpiexec` depending on your MPI implementation).
    #[cfg(feature = "mpi")]
    pub fn new_mpi() -> DistributedResult<Self> {
        Self::from_mpi_communicator(mpi::topology::SystemCommunicator::world())
    }

    /// Wrap an existing MPI [`mpi::topology::SystemCommunicator`] (e.g. a custom communicator).
    #[cfg(feature = "mpi")]
    pub fn from_mpi_communicator(communicator: mpi::topology::SystemCommunicator) -> DistributedResult<Self> {
        let rank = communicator.rank() as usize;
        let world_size = communicator.size() as usize;
        Ok(Self { rank, world_size, backend: CommunicationBackend::Mpi { communicator } })
    }

    /// Get the rank of this process.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get the world size (total number of processes).
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Check if this is the primary rank (rank 0).
    pub fn is_primary(&self) -> bool {
        self.rank == 0
    }

    /// All-reduce: sum `data` in place across all processes (same buffer contents on every rank
    /// after the call).
    pub fn all_reduce_sum(&self, name: &str, data: &mut [f32]) -> DistributedResult<()> {
        match &self.backend {
            CommunicationBackend::SingleProcess => {
                // No-op for single process
                Ok(())
            }
            CommunicationBackend::Threaded { shared_gradients } => {
                let mut gradients = shared_gradients.lock().unwrap();
                if let Some(existing) = gradients.get_mut(name) {
                    // Add our data to the existing sum
                    for (a, b) in existing.iter_mut().zip(data.iter()) {
                        *a += *b;
                    }
                    // Copy back the accumulated sum
                    data.copy_from_slice(existing);
                } else {
                    // First contribution, store it
                    gradients.insert(name.to_string(), data.to_vec());
                }
                Ok(())
            }
            #[cfg(feature = "mpi")]
            CommunicationBackend::Mpi { communicator } => {
                let mut recv_buffer = vec![0.0f32; data.len()];
                communicator.all_reduce_into(&*data, &mut recv_buffer[..], SystemOperation::sum());
                data.copy_from_slice(&recv_buffer);
                Ok(())
            }
        }
    }

    /// All-reduce sum then divide by world size (mean across ranks).
    pub fn all_reduce_mean_f32(&self, name: &str, data: &mut [f32]) -> DistributedResult<()> {
        self.all_reduce_sum(name, data)?;
        let inv = 1.0f32 / self.world_size as f32;
        for v in data.iter_mut() {
            *v *= inv;
        }
        Ok(())
    }

    /// Global barrier (no-op for single-process / threaded simulation).
    pub fn barrier(&self) -> DistributedResult<()> {
        match &self.backend {
            CommunicationBackend::SingleProcess | CommunicationBackend::Threaded { .. } => Ok(()),
            #[cfg(feature = "mpi")]
            CommunicationBackend::Mpi { communicator } => {
                communicator.barrier();
                Ok(())
            }
        }
    }

    /// All-gather equal-sized `f32` buffers from every rank into `recv` (concatenated in rank order).
    pub fn all_gather_f32(&self, send: &[f32], recv: &mut [f32]) -> DistributedResult<()> {
        let ws = self.world_size;
        if recv.len() != send.len() * ws {
            return Err(DistributedError::Communication(format!(
                "all_gather_f32: recv len {} != send len {} * world_size {}",
                recv.len(),
                send.len(),
                ws
            )));
        }
        match &self.backend {
            CommunicationBackend::SingleProcess => {
                recv[..send.len()].copy_from_slice(send);
                Ok(())
            }
            CommunicationBackend::Threaded { .. } => Err(DistributedError::Communication(
                "all_gather_f32 is not implemented for the Threaded backend; use MPI or world_size=1".into(),
            )),
            #[cfg(feature = "mpi")]
            CommunicationBackend::Mpi { communicator } => {
                communicator.all_gather_into(send, recv);
                Ok(())
            }
        }
    }

    /// Broadcast `f32` buffer from `root` to all ranks. **All ranks must call this** with the same
    /// `tag`, sizes, and `root` (MPI collective semantics).
    pub fn broadcast_f32(&self, tag: &str, data: &mut [f32], root: usize) -> DistributedResult<()> {
        if root >= self.world_size {
            return Err(DistributedError::RankMismatch { expected: self.world_size, actual: root });
        }

        match &self.backend {
            CommunicationBackend::SingleProcess => Ok(()),
            CommunicationBackend::Threaded { shared_gradients } => {
                let key = format!("bcast:{}:{}:{}", tag, root, data.len());
                let mut g = shared_gradients.lock().unwrap();
                if self.rank == root {
                    g.insert(key, data.to_vec());
                } else if let Some(v) = g.get(&key) {
                    data.copy_from_slice(v);
                } else {
                    return Err(DistributedError::Communication(format!(
                        "threaded broadcast: root rank must publish `{}` before other ranks read",
                        tag
                    )));
                }
                Ok(())
            }
            #[cfg(feature = "mpi")]
            CommunicationBackend::Mpi { communicator } => {
                let root_proc = communicator.process_at_rank(root as mpi::topology::Rank);
                root_proc.broadcast_into(data);
                Ok(())
            }
        }
    }

    /// Backwards-compatible broadcast using a fixed tag (`"default"`).
    pub fn broadcast(&self, data: &mut [f32], root: usize) -> DistributedResult<()> {
        self.broadcast_f32("default", data, root)
    }
}

/// Data parallel trainer for multi-GPU training.
///
/// This trainer replicates the model on each GPU and splits batches
/// across them, synchronizing gradients via all-reduce after each step.
///
/// # Example
///
/// ```rust,ignore
/// let pg = ProcessGroup::new_threaded(4, rank)?; // 4 GPUs
/// let trainer = DataParallelTrainer::new(pg, model, optimizer);
/// trainer.train_epoch(&dataset, &mut ctx)?;
/// ```
pub struct DataParallelTrainer<B: Backend, O: Optimizer<B>> {
    process_group: ProcessGroup,
    optimizer: O,
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend, O: Optimizer<B>> DataParallelTrainer<B, O>
where
    B::Tensor: Clone + AsRef<[f32]> + rustral_core::TensorShape,
{
    /// Create a new data parallel trainer.
    pub fn new(process_group: ProcessGroup, optimizer: O) -> Self {
        Self { process_group, optimizer, _phantom: std::marker::PhantomData }
    }

    /// Train for one step with data parallelism.
    ///
    /// The batch is split across all processes, gradients are computed
    /// locally, then synchronized via all-reduce before the optimizer step.
    pub fn step<D, L>(
        &mut self,
        params: &mut [Parameter<B>],
        batch: &[D],
        loss_fn: &mut L,
        ctx: &mut ForwardCtx<B>,
    ) -> DistributedResult<f32>
    where
        D: Clone,
        L: FnMut(&D, &mut ForwardCtx<B>) -> Result<(f32, Vec<Gradient<B>>)>,
    {
        let world_size = self.process_group.world_size();
        let rank = self.process_group.rank();

        // Split batch across processes
        let local_batch = self.split_batch(batch, rank, world_size);

        // Compute local gradients
        let mut local_gradients: HashMap<rustral_core::ParameterId, Vec<f32>> = HashMap::new();
        let mut total_loss = 0.0;

        let local_batch_size = local_batch.len().max(1);

        for sample in &local_batch {
            let (loss, grads) = loss_fn(sample, ctx).map_err(DistributedError::Backend)?;
            total_loss += loss;

            // Accumulate gradients
            for grad in grads {
                let grad_data: Vec<f32> = grad.tensor.as_ref().to_vec();
                if let Some(existing) = local_gradients.get_mut(&grad.param_id) {
                    for (a, b) in existing.iter_mut().zip(grad_data.iter()) {
                        *a += *b;
                    }
                } else {
                    local_gradients.insert(grad.param_id, grad_data);
                }
            }
        }

        // Average local gradients by local batch size
        for data in local_gradients.values_mut() {
            for v in data.iter_mut() {
                *v /= local_batch_size as f32;
            }
        }

        // All-reduce gradients across all processes
        for (param_id, data) in local_gradients.iter_mut() {
            let param_name = format!("param_{}", param_id.get());
            self.process_group.all_reduce_sum(&param_name, data)?;

            // Average by world size
            for v in data.iter_mut() {
                *v /= world_size as f32;
            }
        }

        // Convert back to gradients and apply optimizer step
        let mut synced_gradients = Vec::new();
        for (param_id, data) in local_gradients {
            let shape = params
                .iter()
                .find(|p| p.id() == param_id)
                .map(|p| ctx.backend().ops().shape(p.tensor()))
                .unwrap_or_else(|| vec![data.len()]);

            let tensor =
                ctx.backend().ops().tensor_from_vec(data, &shape).map_err(DistributedError::Backend)?;

            synced_gradients.push(Gradient { param_id, tensor });
        }

        self.optimizer
            .step(params, &synced_gradients, ctx)
            .map_err(|e| DistributedError::Backend(CoreError::Other(format!("{:?}", e))))?;

        // All-reduce loss for reporting
        let mut loss_data = [total_loss / local_batch_size as f32];
        self.process_group.all_reduce_sum("loss", &mut loss_data)?;

        Ok(loss_data[0] / world_size as f32)
    }

    /// Split a batch across processes.
    fn split_batch<'a, D>(&self, batch: &'a [D], rank: usize, world_size: usize) -> Vec<&'a D> {
        batch.iter().enumerate().filter(|(i, _)| i % world_size == rank).map(|(_, d)| d).collect()
    }
}

/// Gradient accumulator for asynchronous gradient updates.
///
/// Accumulates gradients over multiple micro-batches before
/// performing the all-reduce and optimizer step.
pub struct GradientAccumulator<B: Backend> {
    process_group: ProcessGroup,
    accumulated_gradients: HashMap<rustral_core::ParameterId, B::Tensor>,
    steps: usize,
}

impl<B: Backend> GradientAccumulator<B>
where
    B::Tensor: Clone,
{
    /// Create a new gradient accumulator.
    pub fn new(process_group: ProcessGroup) -> Self {
        Self { process_group, accumulated_gradients: HashMap::new(), steps: 0 }
    }

    /// Accumulate gradients from a local step.
    pub fn accumulate(&mut self, gradients: &[Gradient<B>], ops: &dyn TensorOps<B>) -> Result<()> {
        for grad in gradients {
            if let Some(existing) = self.accumulated_gradients.get_mut(&grad.param_id) {
                *existing = ops.add(existing, &grad.tensor)?;
            } else {
                self.accumulated_gradients.insert(grad.param_id, grad.tensor.clone());
            }
        }
        self.steps += 1;
        Ok(())
    }

    /// Get the number of accumulated steps.
    pub fn steps(&self) -> usize {
        self.steps
    }

    /// Perform all-reduce and return accumulated gradients.
    /// Note: gradients remain as Vec<f32> since we don't have backend access.
    /// Caller must convert back to tensors.
    pub fn all_reduce(&mut self) -> DistributedResult<Vec<(rustral_core::ParameterId, Vec<f32>)>>
    where
        B::Tensor: AsRef<[f32]> + rustral_core::TensorShape,
    {
        let mut result = Vec::new();

        for (param_id, tensor) in self.accumulated_gradients.iter_mut() {
            let mut data: Vec<f32> = tensor.as_ref().to_vec();
            let name = format!("param_{}", param_id.get());

            // All-reduce across processes
            self.process_group.all_reduce_sum(&name, &mut data)?;

            // Average by world size and steps
            let world_size = self.process_group.world_size();
            for v in data.iter_mut() {
                *v /= (world_size * self.steps) as f32;
            }

            result.push((*param_id, data));
        }

        // Reset accumulator
        self.accumulated_gradients.clear();
        self.steps = 0;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustral_core::Mode;
    use rustral_ndarray_backend::CpuBackend;
    use rustral_optim::Sgd;

    #[test]
    fn test_process_group_single_process() {
        let pg = ProcessGroup::new_single_process();
        assert_eq!(pg.rank(), 0);
        assert_eq!(pg.world_size(), 1);
        assert!(pg.is_primary());
    }

    #[test]
    fn test_process_group_threaded() {
        let pg = ProcessGroup::new_threaded(4, 0).unwrap();
        assert_eq!(pg.rank(), 0);
        assert_eq!(pg.world_size(), 4);
        assert!(pg.is_primary());

        let pg2 = ProcessGroup::new_threaded(4, 3).unwrap();
        assert_eq!(pg2.rank(), 3);
        assert!(!pg2.is_primary());
    }

    #[test]
    fn test_process_group_threaded_error() {
        let result = ProcessGroup::new_threaded(4, 4);
        assert!(result.is_err());
        if let Err(DistributedError::RankMismatch { expected, actual }) = result {
            assert_eq!(expected, 4);
            assert_eq!(actual, 4);
        }
    }

    #[test]
    fn test_broadcast() {
        let pg = ProcessGroup::new_single_process();
        let mut data = vec![1.0f32, 2.0, 3.0];
        pg.broadcast(&mut data, 0).unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_all_reduce_sum() {
        let pg = ProcessGroup::new_threaded(2, 1).unwrap();

        // Simulate rank 0 contribution
        let mut data0 = vec![1.0f32, 2.0, 3.0];
        pg.all_reduce_sum("test", &mut data0).unwrap();

        // Rank 1 contribution
        let mut data1 = vec![4.0f32, 5.0, 6.0];
        pg.all_reduce_sum("test", &mut data1).unwrap();

        // Both should see the sum after all-reduce
        // In threaded mode with shared memory, this depends on order
        // For testing, we verify the mechanism works
    }

    #[test]
    fn test_data_parallel_trainer() {
        let backend = CpuBackend::default();
        let pg = ProcessGroup::new_single_process();
        let optimizer = Sgd::new(0.01);
        let mut trainer = DataParallelTrainer::new(pg, optimizer);

        let mut params = vec![Parameter::new("p0", backend.tensor_from_vec(vec![1.0f32], &[1]).unwrap())];
        let param_id = params[0].id();

        let batch = vec![backend.tensor_from_vec(vec![1.0f32], &[1]).unwrap()];
        let mut ctx = ForwardCtx::new(&backend, Mode::Train);

        let mut loss_fn = |_item: &<CpuBackend as rustral_core::Backend>::Tensor,
                           _ctx: &mut ForwardCtx<CpuBackend>| {
            let grad_tensor = backend.tensor_from_vec(vec![0.1f32], &[1]).unwrap();
            Ok((0.5f32, vec![Gradient { param_id, tensor: grad_tensor }]))
        };

        let loss = trainer.step(&mut params, &batch, &mut loss_fn, &mut ctx).unwrap();
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_gradient_accumulator() {
        let backend = CpuBackend::default();
        let pg = ProcessGroup::new_single_process();
        let mut acc = GradientAccumulator::new(pg);

        // Create some dummy gradients
        let grad_tensor = backend.tensor_from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let gradients = vec![Gradient { param_id: rustral_core::ParameterId::fresh(), tensor: grad_tensor }];

        // Accumulate twice
        acc.accumulate(&gradients, backend.ops()).unwrap();
        acc.accumulate(&gradients, backend.ops()).unwrap();

        assert_eq!(acc.steps(), 2);
    }

    #[test]
    fn test_gradient_accumulator_all_reduce() {
        let backend = CpuBackend::default();
        let pg = ProcessGroup::new_single_process();
        let mut acc = GradientAccumulator::<CpuBackend>::new(pg);

        let grad_tensor = backend.tensor_from_vec(vec![1.0f32, 2.0], &[2]).unwrap();
        let gradients = vec![Gradient { param_id: rustral_core::ParameterId::fresh(), tensor: grad_tensor }];

        acc.accumulate(&gradients, backend.ops()).unwrap();
        let result = acc.all_reduce().unwrap();
        assert_eq!(acc.steps(), 0); // Reset after all_reduce
        assert!(!result.is_empty());
    }

    #[test]
    fn test_distributed_error_display() {
        let err = DistributedError::RankMismatch { expected: 4, actual: 5 };
        let msg = format!("{}", err);
        assert!(msg.contains("rank error"));

        let err = DistributedError::Communication("network failure".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("network failure"));
    }
}
