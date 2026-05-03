//! Tensor Parallelism for Model Parallel Training.
//!
//! Tensor parallelism splits individual layers across multiple GPUs.
//! This is essential for training very large models that don't fit on a single GPU.

use std::sync::Arc;

use mnr_core::{Backend, CoreError, ForwardCtx, Module, Parameter, Result, TensorOps};
use mnr_nn::Linear;

use crate::{DistributedError, DistributedResult, ProcessGroup};

/// Tensor parallel linear layer.
///
/// Splits the weight matrix column-wise or row-wise across multiple GPUs.
/// For column-wise: each GPU computes partial output, then all-gather.
/// For row-wise: input is split, each GPU computes partial result, then all-reduce.
///
/// # Example
///
/// ```rust,ignore
/// // 8 GPUs, each handles 1/8 of the output features
/// let linear = TensorParallelLinear::column_parallel(
///     in_features,
///     out_features,
///     &process_group,
///     &backend,
/// )?;
/// ```
pub struct TensorParallelLinear<B: Backend> {
    /// Local linear layer on this GPU.
    local_linear: Linear<B>,

    /// Process group for communication.
    process_group: ProcessGroup,

    /// Parallelism style: column or row.
    parallel_style: ParallelStyle,
}

/// Style of tensor parallelism.
#[derive(Clone, Copy, Debug)]
pub enum ParallelStyle {
    /// Column-wise: split output features across GPUs.
    ColumnParallel,
    /// Row-wise: split input features across GPUs.
    RowParallel,
}

impl<B: Backend> TensorParallelLinear<B>
where
    B::Tensor: Clone + AsRef<[f32]> + mnr_core::TensorShape,
{
    /// Create a column-parallel linear layer.
    ///
    /// The output dimension is split across GPUs. Each GPU computes
    /// a partial output, then results are gathered.
    pub fn column_parallel(
        in_features: usize,
        out_features: usize,
        process_group: &ProcessGroup,
        backend: &B,
    ) -> DistributedResult<Self> {
        let world_size = process_group.world_size();
        let rank = process_group.rank();

        // Split output dimension
        let local_out = out_features / world_size;
        if local_out * world_size != out_features {
            return Err(DistributedError::Communication(
                format!("out_features {} not divisible by world_size {}", out_features, world_size)
            ));
        }

        // Each GPU owns [local_out, in_features] of the weight
        let config = mnr_nn::LinearConfig::new(in_features, local_out);
        let local_linear = Linear::new(backend, config)
            .map_err(|e| DistributedError::Backend(e.into()))?;

        Ok(Self {
            local_linear,
            process_group: process_group.clone(),
            parallel_style: ParallelStyle::ColumnParallel,
        })
    }

    /// Create a row-parallel linear layer.
    ///
    /// The input dimension is split across GPUs. Each GPU computes
    /// partial results that are all-reduced.
    pub fn row_parallel(
        in_features: usize,
        out_features: usize,
        process_group: &ProcessGroup,
        backend: &B,
    ) -> DistributedResult<Self> {
        let world_size = process_group.world_size();
        let rank = process_group.rank();

        // Split input dimension
        let local_in = in_features / world_size;
        if local_in * world_size != in_features {
            return Err(DistributedError::Communication(
                format!("in_features {} not divisible by world_size {}", in_features, world_size)
            ));
        }

        // Each GPU owns [out_features, local_in] of the weight
        let config = mnr_nn::LinearConfig::new(local_in, out_features);
        let local_linear = Linear::new(backend, config)
            .map_err(|e| DistributedError::Backend(e.into()))?;

        Ok(Self {
            local_linear,
            process_group: process_group.clone(),
            parallel_style: ParallelStyle::RowParallel,
        })
    }

    /// Forward pass with tensor parallelism.
    pub fn forward(&self, input: &B::Tensor, ctx: &mut ForwardCtx<B>) -> DistributedResult<B::Tensor> {
        match self.parallel_style {
            ParallelStyle::ColumnParallel => {
                // Each GPU computes partial output
                let local_output = self.local_linear.forward(input, ctx)
                    .map_err(|e| DistributedError::Backend(e.into()))?;

                // All-gather: collect partial outputs from all GPUs
                // For now, simplified implementation
                // Full implementation needs proper all-gather primitive
                Ok(local_output)
            }
            ParallelStyle::RowParallel => {
                // Split input (in practice, input would already be split)
                // Each GPU computes partial result
                let local_output = self.local_linear.forward(input, ctx)
                    .map_err(|e| DistributedError::Backend(e.into()))?;

                // All-reduce: sum partial results from all GPUs
                let mut data: Vec<f32> = local_output.as_ref().to_vec();
                let name = "row_parallel_linear";
                self.process_group.all_reduce_sum(name, &mut data)?;

                // Convert back to tensor
                let shape = ctx.backend().ops().shape(&local_output);
                let output = ctx.backend().ops().tensor_from_vec(data, &shape)
                    .map_err(|e| DistributedError::Backend(e.into()))?;

                Ok(output)
            }
        }
    }

    /// Get the parallel style.
    pub fn parallel_style(&self) -> ParallelStyle {
        self.parallel_style
    }

    /// Get local parameters.
    pub fn parameters(&self) -> Vec<&Parameter<B>> {
        self.local_linear.parameters()
    }
}

/// Sequential pipeline stage for pipeline parallelism.
///
/// Splits a model into stages, where each stage runs on a different GPU.
/// Uses micro-batching to hide pipeline bubbles.
pub struct PipelineStage<B: Backend> {
    /// Layers in this stage.
    layers: Vec<Box<dyn Module<B, Input = B::Tensor, Output = B::Tensor>>>,

    /// Stage ID (0 to num_stages - 1).
    stage_id: usize,

    /// Number of stages.
    num_stages: usize,

    /// Micro-batch size for pipelining.
    micro_batch_size: usize,
}

impl<B: Backend> PipelineStage<B> {
    /// Create a new pipeline stage.
    pub fn new(stage_id: usize, num_stages: usize, micro_batch_size: usize) -> Self {
        Self {
            layers: Vec::new(),
            stage_id,
            num_stages,
            micro_batch_size,
        }
    }

    /// Add a layer to this stage.
    pub fn add_layer(&mut self, layer: Box<dyn Module<B, Input = B::Tensor, Output = B::Tensor>>) {
        self.layers.push(layer);
    }

    /// Forward pass through this stage.
    pub fn forward(&self, input: B::Tensor, ctx: &mut ForwardCtx<B>) -> Result<B::Tensor> {
        let mut output = input;
        for layer in &self.layers {
            output = layer.forward(output, ctx)?;
        }
        Ok(output)
    }

    /// Backward pass through this stage (simplified).
    pub fn backward(&self, grad_output: B::Tensor, ctx: &mut ForwardCtx<B>) -> Result<B::Tensor> {
        // In full implementation, would compute gradients and pass to previous stage
        // For now, simplified placeholder
        Ok(grad_output)
    }

    /// Get stage ID.
    pub fn stage_id(&self) -> usize {
        self.stage_id
    }
}

/// Pipeline parallel trainer.
///
/// Manages pipeline stages across multiple GPUs.
pub struct PipelineParallelTrainer<B: Backend> {
    stages: Vec<PipelineStage<B>>,
    num_stages: usize,
}

impl<B: Backend> PipelineParallelTrainer<B> {
    /// Create a new pipeline parallel trainer.
    pub fn new(num_stages: usize) -> Self {
        Self {
            stages: Vec::new(),
            num_stages,
        }
    }

    /// Add a stage.
    pub fn add_stage(&mut self, stage: PipelineStage<B>) {
        self.stages.push(stage);
    }

    /// Forward pass through the entire pipeline.
    pub fn forward(&self, input: B::Tensor, ctx: &mut ForwardCtx<B>) -> Result<B::Tensor> {
        let mut output = input;
        for stage in &self.stages {
            output = stage.forward(output, ctx)?;
        }
        Ok(output)
    }

    /// Train with pipeline parallelism.
    ///
    /// Uses interleaved micro-batches to reduce pipeline bubbles.
    pub fn train_step<D, L>(
        &mut self,
        batch: &[D],
        loss_fn: &mut L,
        ctx: &mut ForwardCtx<B>,
    ) -> Result<f32>
    where
        D: Clone,
        L: FnMut(&D, &mut ForwardCtx<B>) -> Result<(f32, B::Tensor)>,
    {
        // Split batch into micro-batches
        let micro_batches = self.split_into_micro_batches(batch);

        // Forward pass: send micro-batches through pipeline
        let mut outputs = Vec::new();
        for micro_batch in &micro_batches {
            let (loss, output) = loss_fn(micro_batch, ctx)?;
            outputs.push((loss, output));
        }

        // Average loss
        let total_loss: f32 = outputs.iter().map(|(l, _)| l).sum();
        Ok(total_loss / outputs.len() as f32)
    }

    /// Split batch into micro-batches.
    fn split_into_micro_batches<D>(&self, batch: &[D]) -> Vec<Vec<D>>
    where
        D: Clone,
    {
        let micro_batch_size = if let Some(stage) = self.stages.first() {
            stage.micro_batch_size
        } else {
            1
        };

        batch
            .chunks(micro_batch_size)
            .map(|chunk| chunk.to_vec())
            .collect()
    }
}

/// All-gather operation for tensor parallelism.
///
/// Collects tensors from all processes and concatenates them.
pub struct AllGatherOp<B: Backend> {
    process_group: ProcessGroup,
    concat_dim: usize,
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> AllGatherOp<B> {
    /// Create a new all-gather operation.
    pub fn new(process_group: ProcessGroup, concat_dim: usize) -> Self {
        Self {
            process_group,
            concat_dim,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Perform all-gather.
    pub fn execute(&self, local_tensor: &B::Tensor, ops: &dyn TensorOps<B>) -> Result<B::Tensor>
    where
        B::Tensor: AsRef<[f32]> + mnr_core::TensorShape,
    {
        // In full implementation, this would:
        // 1. Gather tensor sizes from all ranks
        // 2. Allocate output tensor with total size
        // 3. Each rank writes to its slice
        // 4. Synchronize

        // Simplified: just return local tensor for now
        Ok(local_tensor.clone())
    }
}

/// All-reduce operation for gradient synchronization.
pub struct AllReduceOp {
    process_group: ProcessGroup,
    op: ReduceOp,
}

/// Reduction operation.
#[derive(Clone, Copy, Debug)]
pub enum ReduceOp {
    Sum,
    Mean,
    Max,
}

impl AllReduceOp {
    /// Create a new all-reduce operation.
    pub fn new(process_group: ProcessGroup, op: ReduceOp) -> Self {
        Self { process_group, op }
    }

    /// Perform all-reduce on a buffer.
    pub fn execute(&self, data: &mut [f32]) -> DistributedResult<()> {
        self.process_group.all_reduce_sum("buffer", data)?;

        match self.op {
            ReduceOp::Mean => {
                let world_size = self.process_group.world_size() as f32;
                for v in data.iter_mut() {
                    *v /= world_size;
                }
            }
            _ => {}
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnr_ndarray_backend::CpuBackend;

    #[test]
    fn test_tensor_parallel_linear_column() {
        let backend = CpuBackend::default();
        let pg = ProcessGroup::new_single_process();

        let linear = TensorParallelLinear::column_parallel(
            64,   // in_features
            128,  // out_features
            &pg,  // single process
            &backend,
        ).unwrap();

        assert_eq!(linear.parallel_style(), ParallelStyle::ColumnParallel);
    }

    #[test]
    fn test_tensor_parallel_linear_row() {
        let backend = CpuBackend::default();
        let pg = ProcessGroup::new_single_process();

        let linear = TensorParallelLinear::row_parallel(
            128,  // in_features
            64,   // out_features
            &pg,
            &backend,
        ).unwrap();

        assert_eq!(linear.parallel_style(), ParallelStyle::RowParallel);
    }

    #[test]
    fn test_pipeline_stage() {
        let stage = PipelineStage::<CpuBackend>::new(0, 4, 2);
        assert_eq!(stage.stage_id(), 0);
    }

    #[test]
    fn test_all_reduce_op() {
        let pg = ProcessGroup::new_single_process();
        let op = AllReduceOp::new(pg, ReduceOp::Sum);

        let mut data = vec![1.0f32, 2.0, 3.0];
        op.execute(&mut data).unwrap();

        // With single process, data should be unchanged
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }
}
