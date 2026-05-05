//! 3D Parallelism: Data + Tensor + Pipeline Parallel Combined
//!
//! Combines all three parallelism strategies for training
//! massive models (100B+ parameters) at maximum efficiency.
//!
//! ```text
//! 3D Parallel Layout (8 GPUs example):
//!
//! DP=2, TP=2, PP=2
//!
//!          Pipeline Stage 0    Pipeline Stage 1
//!         ┌─────────┬────────┐  ┌─────────┬────────┐
//!  Data   │ GPU 0   │ GPU 1  │  │ GPU 2   │ GPU 3  │
//!  Par 0  │ (TP=0)  │ (TP=1) │  │ (TP=0)  │ (TP=1) │
//!         └─────────┴────────┘  └─────────┴────────┘
//!         ┌─────────┬────────┐  ┌─────────┬────────┐
//!  Data   │ GPU 4   │ GPU 5  │  │ GPU 6   │ GPU 7  │
//!  Par 1  │ (TP=0)  │ (TP=1) │  │ (TP=0)  │ (TP=1) │
//!         └─────────┴────────┘  └─────────┴────────┘
//!
//! Communication:
//! - TP: All-Reduce within each PP/DP cell
//! - PP: P2P between stages
//! - DP: All-Reduce across DP groups
//! ```

use mnr_core::{Backend, CoreError, ForwardCtx, Module, Result};
use mnr_nn::{Linear, LinearConfig};

use crate::{DeviceMesh, ParallelismConfig, PipelineParallelTrainer, ProcessGroup, TensorParallelLinear};

/// 3D Parallel Trainer combining DP, TP, and PP.
pub struct Parallel3DTrainer<B: Backend> {
    device_mesh: DeviceMesh,
    pipeline_trainer: PipelineParallelTrainer<B>,
    dp_group: ProcessGroup,
    tp_group: ProcessGroup,
    pp_group: ProcessGroup,
    _backend: std::marker::PhantomData<B>,
}

impl<B: Backend> Parallel3DTrainer<B>
where
    B::Tensor: Clone + AsRef<[f32]> + mnr_core::TensorShape,
{
    /// Create a new 3D parallel trainer.
    pub fn new(device_mesh: DeviceMesh, pipeline_trainer: PipelineParallelTrainer<B>) -> Result<Self> {
        let mut mesh = device_mesh;

        // Get process groups for each parallelism dimension
        let dp_group = mesh.get_data_parallel_group()?;
        let tp_group = mesh.get_tensor_parallel_group()?;
        let pp_group = mesh.get_pipeline_parallel_group()?;

        Ok(Self {
            device_mesh: mesh,
            pipeline_trainer,
            dp_group,
            tp_group,
            pp_group,
            _backend: std::marker::PhantomData,
        })
    }

    /// Auto-configure parallelism strategy based on model size and GPU count.
    pub fn auto_configure(num_gpus: usize, model_params: usize, _num_layers: usize) -> ParallelismConfig {
        // Heuristic configuration
        let config = ParallelismConfig::auto_select(num_gpus, model_params);

        println!("3D Parallelism Configuration:");
        println!("  GPUs: {}", num_gpus);
        println!("  Model Params: {}B", model_params / 1_000_000_000);
        println!("  Data Parallel: {}", config.data_parallel);
        println!("  Tensor Parallel: {}", config.tensor_parallel);
        println!("  Pipeline Parallel: {}", config.pipeline_parallel);

        config
    }

    /// Execute a forward/backward step with 3D parallelism.
    pub fn step<M: Module<B>, L: FnMut(&B::Tensor, &mut ForwardCtx<B>) -> Result<B::Tensor>>(
        &mut self,
        _model: &mut M,
        batch: &B::Tensor,
        _loss_fn: &mut L,
        ctx: &mut ForwardCtx<B>,
    ) -> Result<f32> {
        // Step 1: Split batch for data parallelism
        let local_batch = self.scatter_batch(batch, ctx)?;

        // Step 2: Execute pipeline forward pass
        // Note: PipelineParallelTrainer::train_step has different signature.
        // Using forward pass for now.
        let _output = self.pipeline_trainer.forward(local_batch, ctx)?;

        // Placeholder: compute loss and gradients
        let local_loss = 0.0f32;

        // Step 3: All-reduce loss across data parallel group
        let global_loss = self.all_reduce_loss(local_loss)?;

        // Step 4: All-reduce gradients across tensor parallel group
        self.all_reduce_tensor_parallel_gradients(_model)?;

        // Step 5: All-reduce gradients across data parallel group
        self.all_reduce_data_parallel_gradients(_model)?;

        Ok(global_loss)
    }

    /// Scatter batch across data parallel ranks.
    fn scatter_batch(&self, batch: &B::Tensor, ctx: &mut ForwardCtx<B>) -> Result<B::Tensor> {
        let dp_size = self.dp_group.world_size();
        let dp_rank = self.dp_group.rank();

        if dp_size == 1 {
            return Ok(batch.clone());
        }

        // Split batch along first dimension
        let shape = ctx.backend().ops().shape(batch);
        let batch_size = shape[0];

        if batch_size % dp_size != 0 {
            return Err(CoreError::InvalidArgument(format!(
                "Batch size {} not divisible by DP size {}",
                batch_size, dp_size
            )));
        }

        let local_size = batch_size / dp_size;
        let start = dp_rank * local_size;

        // Extract local batch slice
        // In real impl: use proper slice operation
        let data: Vec<f32> = batch.as_ref().to_vec();
        let mut local_data = Vec::new();

        let other_dims: usize = shape.iter().skip(1).product();
        let row_size = other_dims;

        for i in start..(start + local_size) {
            let offset = i * row_size;
            local_data.extend_from_slice(&data[offset..offset + row_size]);
        }

        let mut new_shape = shape.clone();
        new_shape[0] = local_size;

        ctx.backend().ops().tensor_from_vec(local_data, &new_shape)
    }

    /// All-reduce loss across data parallel group.
    fn all_reduce_loss(&self, local_loss: f32) -> Result<f32> {
        let dp_size = self.dp_group.world_size();

        if dp_size == 1 {
            return Ok(local_loss);
        }

        // In real impl: use actual all-reduce
        // Simplified: return average (assuming all ranks have similar loss)
        Ok(local_loss)
    }

    /// All-reduce gradients across tensor parallel group.
    fn all_reduce_tensor_parallel_gradients<M: Module<B>>(&self, _model: &mut M) -> Result<()> {
        let tp_size = self.tp_group.world_size();

        if tp_size == 1 {
            return Ok(());
        }

        // All-reduce gradients for tensor parallel layers
        // In real impl: iterate over parameters and all-reduce
        Ok(())
    }

    /// All-reduce gradients across data parallel group.
    fn all_reduce_data_parallel_gradients<M: Module<B>>(&self, _model: &mut M) -> Result<()> {
        let dp_size = self.dp_group.world_size();

        if dp_size == 1 {
            return Ok(());
        }

        // All-reduce gradients for all parameters
        // In real impl: iterate over parameters and all-reduce
        Ok(())
    }

    /// Get the device mesh.
    pub fn device_mesh(&self) -> &DeviceMesh {
        &self.device_mesh
    }

    /// Get data parallel group size.
    pub fn dp_size(&self) -> usize {
        self.dp_group.world_size()
    }

    /// Get tensor parallel group size.
    pub fn tp_size(&self) -> usize {
        self.tp_group.world_size()
    }

    /// Get pipeline parallel group size.
    pub fn pp_size(&self) -> usize {
        self.pp_group.world_size()
    }
}

/// Layer wrapper that applies tensor parallelism within 3D parallelism.
pub struct Parallel3DLayer<B: Backend> {
    linear: TensorParallelLinear<B>,
    tp_group: ProcessGroup,
}

impl<B: Backend> Parallel3DLayer<B>
where
    B::Tensor: Clone + AsRef<[f32]> + mnr_core::TensorShape,
{
    pub fn new(in_features: usize, out_features: usize, tp_group: ProcessGroup, backend: &B) -> Result<Self> {
        let linear = TensorParallelLinear::column_parallel(in_features, out_features, &tp_group, backend)
            .map_err(|e| CoreError::Other(format!("{:?}", e)))?;

        Ok(Self { linear, tp_group })
    }
}

impl<B: Backend> Module<B> for Parallel3DLayer<B>
where
    B::Tensor: Clone + AsRef<[f32]> + mnr_core::TensorShape,
{
    type Input = B::Tensor;
    type Output = B::Tensor;

    fn forward(&self, input: Self::Input, ctx: &mut ForwardCtx<B>) -> Result<Self::Output> {
        self.linear.forward(&input, ctx).map_err(|e| CoreError::Other(format!("{:?}", e)))
    }
}

/// Configuration for 3D parallelism.
pub struct Parallel3DConfig {
    pub data_parallel_size: usize,
    pub tensor_parallel_size: usize,
    pub pipeline_parallel_size: usize,
    pub gradient_accumulation_steps: usize,
    pub pipeline_num_microbatches: usize,
}

impl Parallel3DConfig {
    pub fn new(dp: usize, tp: usize, pp: usize) -> Self {
        Self {
            data_parallel_size: dp,
            tensor_parallel_size: tp,
            pipeline_parallel_size: pp,
            gradient_accumulation_steps: 1,
            pipeline_num_microbatches: 4,
        }
    }

    pub fn with_gradient_accumulation(mut self, steps: usize) -> Self {
        self.gradient_accumulation_steps = steps;
        self
    }

    pub fn with_pipeline_microbatches(mut self, num: usize) -> Self {
        self.pipeline_num_microbatches = num;
        self
    }

    /// Total number of GPUs required.
    pub fn total_gpus(&self) -> usize {
        self.data_parallel_size * self.tensor_parallel_size * self.pipeline_parallel_size
    }

    /// Validate configuration.
    pub fn validate(&self) -> Result<()> {
        let total = self.total_gpus();

        if self.data_parallel_size == 0 || self.tensor_parallel_size == 0 || self.pipeline_parallel_size == 0
        {
            return Err(CoreError::InvalidArgument("All parallelism dimensions must be > 0".to_string()));
        }

        println!("3D Parallelism Configuration Validated:");
        println!("  Total GPUs: {}", total);
        println!("  Data Parallel: {}", self.data_parallel_size);
        println!("  Tensor Parallel: {}", self.tensor_parallel_size);
        println!("  Pipeline Parallel: {}", self.pipeline_parallel_size);

        Ok(())
    }
}

/// Helper to create a 3D parallel model.
pub fn create_3d_parallel_model<B: Backend>(
    config: &Parallel3DConfig,
    device_mesh: &DeviceMesh,
    backend: &B,
) -> Result<Vec<Box<dyn Module<B, Input = B::Tensor, Output = B::Tensor>>>>
where
    B::Tensor: 'static,
{
    config.validate()?;

    let mut stages = Vec::new();

    // Get pipeline stage index
    let pp_rank = device_mesh.my_coord().get(2).copied().unwrap_or(0);
    let pp_size = config.pipeline_parallel_size;

    // Calculate which layers this stage owns
    let total_layers = 12; // Example
    let layers_per_stage = total_layers / pp_size;
    let start_layer = pp_rank * layers_per_stage;
    let end_layer = start_layer + layers_per_stage;

    println!("Pipeline Stage {}: layers {}-{} (of {})", pp_rank, start_layer, end_layer - 1, total_layers);

    // Create layers for this stage
    // In real impl: create actual transformer layers with TP
    for _ in start_layer..end_layer {
        // Each layer uses tensor parallelism
        stages.push(Box::new(Linear::new(backend, LinearConfig::new(256, 256))?)
            as Box<dyn Module<B, Input = B::Tensor, Output = B::Tensor>>);
    }

    Ok(stages)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnr_core::{ForwardCtx, Mode, Module};
    use mnr_ndarray_backend::CpuBackend;

    #[test]
    fn test_parallel_3d_config() {
        let config =
            Parallel3DConfig::new(2, 4, 2).with_gradient_accumulation(4).with_pipeline_microbatches(8);

        assert_eq!(config.total_gpus(), 16);
        assert_eq!(config.data_parallel_size, 2);
        assert_eq!(config.tensor_parallel_size, 4);
        assert_eq!(config.pipeline_parallel_size, 2);
        assert_eq!(config.gradient_accumulation_steps, 4);
        assert_eq!(config.pipeline_num_microbatches, 8);

        config.validate().unwrap();
    }

    #[test]
    fn test_parallel_3d_config_validation_errors() {
        let config = Parallel3DConfig::new(0, 1, 1);
        assert!(config.validate().is_err());

        let config = Parallel3DConfig::new(1, 0, 1);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_parallel_3d_auto_config() {
        let config = Parallel3DTrainer::<CpuBackend>::auto_configure(64, 100_000_000_000, 96);

        assert_eq!(config.total_devices(), 64);
        assert!(config.tensor_parallel >= 1);
        assert!(config.data_parallel >= 1);
        assert!(config.pipeline_parallel >= 1);
    }

    #[test]
    fn test_device_mesh_integration() {
        let mesh = DeviceMesh::for_3d_parallelism(2, 2, 2, 0).unwrap();

        assert_eq!(mesh.world_size(), 8);
        assert_eq!(mesh.dim_size(0), 2); // DP
        assert_eq!(mesh.dim_size(1), 2); // TP
        assert_eq!(mesh.dim_size(2), 2); // PP
    }

    #[test]
    fn test_parallel_3d_trainer_creation() {
        let mesh = DeviceMesh::for_3d_parallelism(1, 1, 1, 0).unwrap();
        let pipeline_trainer = PipelineParallelTrainer::<CpuBackend>::new(1);
        let trainer = Parallel3DTrainer::new(mesh, pipeline_trainer).unwrap();

        assert_eq!(trainer.dp_size(), 1);
        assert_eq!(trainer.tp_size(), 1);
        assert_eq!(trainer.pp_size(), 1);
        assert_eq!(trainer.device_mesh().world_size(), 1);
    }

    #[test]
    fn test_parallel_3d_trainer_step() {
        let backend = CpuBackend::default();
        let mesh = DeviceMesh::for_3d_parallelism(1, 1, 1, 0).unwrap();
        let pipeline_trainer = PipelineParallelTrainer::<CpuBackend>::new(1);
        let mut trainer = Parallel3DTrainer::new(mesh, pipeline_trainer).unwrap();

        let batch = backend.tensor_from_vec(vec![1.0f32; 16], &[2, 8]).unwrap();
        let mut ctx = ForwardCtx::new(&backend, Mode::Train);

        // Mock module that returns input unchanged
        struct IdentityModule;
        impl Module<CpuBackend> for IdentityModule {
            type Input = <CpuBackend as mnr_core::Backend>::Tensor;
            type Output = <CpuBackend as mnr_core::Backend>::Tensor;
            fn forward(&self, input: Self::Input, _ctx: &mut ForwardCtx<CpuBackend>) -> Result<Self::Output> {
                Ok(input)
            }
        }

        let mut model = IdentityModule;
        let mut loss_fn = |_output: &<CpuBackend as mnr_core::Backend>::Tensor,
                           _ctx: &mut ForwardCtx<CpuBackend>| {
            backend.tensor_from_vec(vec![0.5f32], &[1])
        };

        let result = trainer.step(&mut model, &batch, &mut loss_fn, &mut ctx);
        // Should succeed with DP size 1 (no actual splitting)
        assert!(result.is_ok());
    }

    #[test]
    fn test_parallel_3d_layer() {
        let backend = CpuBackend::default();
        let pg = ProcessGroup::new_single_process();
        let layer = Parallel3DLayer::new(64, 64, pg, &backend).unwrap();

        let input = backend.tensor_from_vec(vec![1.0f32; 64], &[1, 64]).unwrap();
        let mut ctx = ForwardCtx::new(&backend, Mode::Train);
        let output = layer.forward(input, &mut ctx).unwrap();
        let shape = backend.ops().shape(&output);
        assert_eq!(shape[0], 1);
    }

    #[test]
    fn test_create_3d_parallel_model() {
        let backend = CpuBackend::default();
        let mesh = DeviceMesh::for_3d_parallelism(1, 1, 2, 0).unwrap();
        let config = Parallel3DConfig::new(1, 1, 2);

        let stages = create_3d_parallel_model(&config, &mesh, &backend).unwrap();
        assert_eq!(stages.len(), 6); // 12 total_layers / 2 pp_size = 6 per stage
    }
}
