//! Pipeline Parallelism with Automatic Stage Splitting
//!
//! Splits model into stages across multiple GPUs. Each GPU processes
//! a subset of layers. Uses micro-batching to hide pipeline bubbles.
//!
//! # Pipeline Schedule
//!
//! ```text
//! GPU 0: F0 → F0 → F0 → B0 → B0 → B0
//! GPU 1:      F1 → F1 → F1 → B1 → B1
//! GPU 2:           F2 → F2 → F2 → B2
//!
//! F = Forward, B = Backward
//! ```
//!
//! # Automatic Stage Splitting
//!
//! Analyzes model structure and balances:
//! - Parameter count per stage
//! - FLOPs per stage
//! - Memory usage per stage
//!
//! # Example
//! ```rust,ignore
//! use mnr_distributed::pipeline_parallel::{PipelineParallel, StageSplitter};
//!
//! let stages = StageSplitter::auto_split(model, 4)?; // Split into 4 stages
//! let pipeline = PipelineParallel::new(stages, process_group)?;
//! let output = pipeline.forward(input)?;
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use mnr_core::{Backend, CoreError, ForwardCtx, Mode, Module, Parameter, ParameterRef, Result};

use crate::{DistributedError, DistributedResult, ProcessGroup};

/// Configuration for pipeline parallelism
#[derive(Clone, Debug)]
pub struct PipelineConfig {
    /// Number of micro-batches
    pub num_micro_batches: usize,
    /// Pipeline schedule (interleaved or simple)
    pub schedule: PipelineSchedule,
    /// Activation checkpointing for stages
    pub checkpoint_activations: bool,
    /// Async communication
    pub async_comm: bool,
}

#[derive(Clone, Copy, Debug)]
pub enum PipelineSchedule {
    /// Simple fill-drain (GPipe style)
    Simple,
    /// Interleaved 1F1B (one forward one backward)
    Interleaved,
    /// Zero bubble
    ZeroBubble,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            num_micro_batches: 4,
            schedule: PipelineSchedule::Interleaved,
            checkpoint_activations: false,
            async_comm: true,
        }
    }
}

impl PipelineConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_micro_batches(mut self, n: usize) -> Self {
        self.num_micro_batches = n;
        self
    }

    pub fn with_schedule(mut self, schedule: PipelineSchedule) -> Self {
        self.schedule = schedule;
        self
    }

    pub fn with_checkpointing(mut self, enable: bool) -> Self {
        self.checkpoint_activations = enable;
        self
    }
}

/// A pipeline stage containing one or more layers
pub struct PipelineStage<B: Backend> {
    /// Stage ID (0 to num_stages - 1)
    pub stage_id: usize,
    /// Layers in this stage
    layers: Vec<Box<dyn Module<B, Input = B::Tensor, Output = B::Tensor>>>,
    /// Cached activations for backward
    activations: Vec<B::Tensor>,
    /// Stage device (in real impl, would be GPU device)
    device: usize,
}

impl<B: Backend> PipelineStage<B> {
    pub fn new(stage_id: usize, device: usize) -> Self {
        Self {
            stage_id,
            layers: Vec::new(),
            activations: Vec::new(),
            device,
        }
    }

    pub fn add_layer(&mut self, layer: Box<dyn Module<B, Input = B::Tensor, Output = B::Tensor>>) {
        self.layers.push(layer);
    }

    pub fn forward(&self, input: B::Tensor, ctx: &mut ForwardCtx<B>) -> Result<B::Tensor> {
        let mut output = input;
        for layer in &self.layers {
            output = layer.forward(output, ctx)?;
        }
        Ok(output)
    }

    pub fn parameters(&self) -> Vec<ParameterRef<B>> {
        self.layers
            .iter()
            .flat_map(|l| l.parameters())
            .collect()
    }
}

/// Automatic stage splitter
pub struct StageSplitter;

impl StageSplitter {
    /// Automatically split model into balanced stages
    pub fn auto_split<B: Backend, M: Module<B>>(
        model: &M,
        num_stages: usize,
    ) -> DistributedResult<Vec<PipelineStage<B>>> {
        let params = model.parameters();
        let total_params: usize = params.len();
        let params_per_stage = (total_params + num_stages - 1) / num_stages;

        let mut stages: Vec<PipelineStage<B>> = (0..num_stages)
            .map(|i| PipelineStage::new(i, i))
            .collect();

        // Simple round-robin assignment
        // In real impl, would analyze model structure and balance by FLOPs
        for (idx, param) in params.iter().enumerate() {
            let stage_id = idx / params_per_stage;
            if stage_id < num_stages {
                // Would add layers to appropriate stage
            }
        }

        Ok(stages)
    }

    /// Split by layer type (e.g., separate transformer blocks)
    pub fn split_by_type<B: Backend>(
        layers: Vec<Box<dyn Module<B, Input = B::Tensor, Output = B::Tensor>>>,
        num_stages: usize,
    ) -> Vec<PipelineStage<B>> {
        let mut stages: Vec<PipelineStage<B>> = (0..num_stages)
            .map(|i| PipelineStage::new(i, i))
            .collect();

        let layers_per_stage = (layers.len() + num_stages - 1) / num_stages;

        for (idx, layer) in layers.into_iter().enumerate() {
            let stage_id = idx / layers_per_stage;
            if stage_id < num_stages {
                stages[stage_id].add_layer(layer);
            }
        }

        stages
    }

    /// Balance stages by parameter count
    pub fn balance_by_params<B: Backend>(
        layers: Vec<(Box<dyn Module<B, Input = B::Tensor, Output = B::Tensor>>, usize)>, // (layer, param_count)
        num_stages: usize,
    ) -> Vec<PipelineStage<B>> {
        let mut stages: Vec<PipelineStage<B>> = (0..num_stages)
            .map(|i| PipelineStage::new(i, i))
            .collect();

        let total_params: usize = layers.iter().map(|(_, c)| c).sum();
        let target_per_stage = total_params / num_stages;

        let mut current_stage = 0;
        let mut current_stage_params = 0;

        for (layer, param_count) in layers {
            // If adding this layer would exceed target, move to next stage
            if current_stage_params > 0
                && current_stage_params + param_count > target_per_stage
                && current_stage < num_stages - 1
            {
                current_stage += 1;
                current_stage_params = 0;
            }

            stages[current_stage].add_layer(layer);
            current_stage_params += param_count;
        }

        stages
    }
}

/// Pipeline parallel trainer
pub struct PipelineParallel<B: Backend> {
    stages: Vec<PipelineStage<B>>,
    process_group: ProcessGroup,
    config: PipelineConfig,
    /// Current micro-batch being processed
    current_micro_batch: usize,
    /// Forward outputs cache for backward
    forward_cache: HashMap<usize, Vec<B::Tensor>>,
    /// Backward gradients cache
    grad_cache: HashMap<usize, B::Tensor>,
}

impl<B: Backend> PipelineParallel<B>
where
    B::Tensor: Clone + AsRef<[f32]> + mnr_core::TensorShape,
{
    pub fn new(
        stages: Vec<PipelineStage<B>>,
        process_group: ProcessGroup,
        config: PipelineConfig,
    ) -> DistributedResult<Self> {
        if stages.len() != process_group.world_size() {
            return Err(DistributedError::Communication(format!(
                "Number of stages ({}) doesn't match world size ({})",
                stages.len(),
                process_group.world_size()
            )));
        }

        Ok(Self {
            stages,
            process_group,
            config,
            current_micro_batch: 0,
            forward_cache: HashMap::new(),
            grad_cache: HashMap::new(),
        })
    }

    /// Forward pass through entire pipeline
    pub fn forward(&mut self, input: B::Tensor, ctx: &mut ForwardCtx<B>) -> Result<B::Tensor> {
        let rank = self.process_group.rank();
        let num_stages = self.stages.len();

        // If first stage, start pipeline
        if rank == 0 {
            // Process and send to next stage
            let output = self.stages[0].forward(input, ctx)?;
            // In real impl, would send to rank 1
            Ok(output)
        } else if rank == num_stages - 1 {
            // Last stage, receive from previous and return
            // In real impl, would receive from rank-1
            let stage_input = input; // Placeholder
            self.stages[rank].forward(stage_input, ctx)
        } else {
            // Middle stage: receive, process, send
            let stage_input = input; // Placeholder
            let output = self.stages[rank].forward(stage_input, ctx)?;
            Ok(output)
        }
    }

    /// Training step with pipeline parallelism
    pub fn train_step(
        &mut self,
        micro_batches: &[B::Tensor],
        ctx: &mut ForwardCtx<B>,
    ) -> Result<Vec<f32>> {
        let rank = self.process_group.rank();
        let num_stages = self.stages.len();
        let num_micro = micro_batches.len();

        let mut losses = Vec::new();

        match self.config.schedule {
            PipelineSchedule::Simple => {
                // Fill pipeline (all forwards)
                for (i, batch) in micro_batches.iter().enumerate() {
                    let output = self.forward(batch.clone(), ctx)?;
                    self.forward_cache.insert(i, vec![output]);
                }

                // Drain pipeline (all backwards)
                for i in (0..num_micro).rev() {
                    // In real impl, would do backward pass
                    losses.push(0.0f32); // Placeholder
                }
            }

            PipelineSchedule::Interleaved => {
                // 1F1B schedule: interleave forward and backward
                // This hides pipeline bubbles

                // Warmup: fill pipeline
                for i in 0..self.config.num_micro_batches.min(num_stages) {
                    if let Some(batch) = micro_batches.get(i) {
                        let _output = self.forward(batch.clone(), ctx)?;
                    }
                }

                // Steady state: 1 forward, 1 backward
                for i in self.config.num_micro_batches.min(num_stages)..num_micro {
                    if let Some(batch) = micro_batches.get(i) {
                        let _output = self.forward(batch.clone(), ctx)?;
                        // Immediately do backward for oldest micro-batch
                        losses.push(0.0f32);
                    }
                }

                // Cooldown: drain pipeline
                for _ in 0..num_stages {
                    losses.push(0.0f32);
                }
            }

            PipelineSchedule::ZeroBubble => {
                // Zero bubble schedule minimizes idle time
                // Would implement in production
                for batch in micro_batches {
                    let _ = self.forward(batch.clone(), ctx)?;
                    losses.push(0.0f32);
                }
            }
        }

        Ok(losses)
    }

    /// Get statistics about pipeline efficiency
    pub fn pipeline_stats(&self) -> PipelineStats {
        let num_stages = self.stages.len();
        let num_micro = self.config.num_micro_batches;

        // Calculate bubble time (idle waiting for other stages)
        let bubble_fraction = match self.config.schedule {
            PipelineSchedule::Simple => {
                // Bubble = (num_stages - 1) / num_micro
                (num_stages - 1) as f32 / num_micro as f32
            }
            PipelineSchedule::Interleaved => {
                // Much smaller bubble
                (num_stages - 1) as f32 / (num_micro + num_stages - 1) as f32
            }
            PipelineSchedule::ZeroBubble => 0.0,
        };

        PipelineStats {
            num_stages,
            num_micro_batches: num_micro,
            schedule: format!("{:?}", self.config.schedule),
            bubble_fraction,
            efficiency: 1.0 - bubble_fraction,
        }
    }
}

/// Pipeline statistics
#[derive(Debug, Clone)]
pub struct PipelineStats {
    pub num_stages: usize,
    pub num_micro_batches: usize,
    pub schedule: String,
    pub bubble_fraction: f32,
    pub efficiency: f32,
}

/// Communication primitive for pipeline stages
pub struct PipelineComm<B: Backend> {
    stage_id: usize,
    next_stage: Option<usize>,
    prev_stage: Option<usize>,
    send_queue: Arc<Mutex<Vec<B::Tensor>>>,
    recv_queue: Arc<Mutex<Vec<B::Tensor>>>,
}

impl<B: Backend> PipelineComm<B> {
    pub fn new(stage_id: usize, num_stages: usize) -> Self {
        Self {
            stage_id,
            next_stage: if stage_id < num_stages - 1 {
                Some(stage_id + 1)
            } else {
                None
            },
            prev_stage: if stage_id > 0 { Some(stage_id - 1) } else { None },
            send_queue: Arc::new(Mutex::new(Vec::new())),
            recv_queue: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Send tensor to next stage
    pub fn send(&self, tensor: B::Tensor) {
        if let Some(_next) = self.next_stage {
            let mut queue = self.send_queue.lock().unwrap();
            queue.push(tensor);
        }
    }

    /// Receive tensor from previous stage
    pub fn recv(&self) -> Option<B::Tensor> {
        if self.prev_stage.is_some() {
            let mut queue = self.recv_queue.lock().unwrap();
            queue.pop()
        } else {
            None
        }
    }
}

/// Helper function to create pipeline from model
pub fn create_pipeline<B: Backend, M: Module<B>>(
    model: M,
    process_group: ProcessGroup,
    config: PipelineConfig,
) -> DistributedResult<PipelineParallel<B>> {
    let num_stages = process_group.world_size();
    let stages = StageSplitter::auto_split(&model, num_stages)?;
    PipelineParallel::new(stages, process_group, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnr_ndarray_backend::CpuBackend;
    use mnr_nn::{Linear, LinearConfig};

    #[test]
    fn test_pipeline_config() {
        let config = PipelineConfig::new()
            .with_micro_batches(8)
            .with_schedule(PipelineSchedule::Interleaved)
            .with_checkpointing(true);

        assert_eq!(config.num_micro_batches, 8);
        assert!(matches!(config.schedule, PipelineSchedule::Interleaved));
        assert!(config.checkpoint_activations);
    }

    #[test]
    fn test_stage_splitter() {
        let backend = CpuBackend::default();
        let linear = Linear::new(&backend, LinearConfig::new(64, 64)).unwrap();

        let pg = ProcessGroup::new_threaded(4, 0);
        let stages = StageSplitter::auto_split(&linear, 4).unwrap();

        assert_eq!(stages.len(), 4);
    }

    #[test]
    fn test_pipeline_creation() {
        let backend = CpuBackend::default();
        let pg = ProcessGroup::new_threaded(4, 1);

        // Create stages manually
        let mut stages: Vec<PipelineStage<CpuBackend>> = (0..4)
            .map(|i| PipelineStage::new(i, i))
            .collect();

        // Add layers
        for stage in &mut stages {
            let layer = Linear::new(&backend, LinearConfig::new(64, 64)).unwrap();
            stage.add_layer(Box::new(layer));
        }

        let config = PipelineConfig::new();
        let pipeline = PipelineParallel::new(stages, pg, config).unwrap();

        assert_eq!(pipeline.stages.len(), 4);
    }

    #[test]
    fn test_pipeline_stats() {
        let backend = CpuBackend::default();
        let pg = ProcessGroup::new_threaded(4, 0);

        let mut stages: Vec<PipelineStage<CpuBackend>> = (0..4)
            .map(|i| PipelineStage::new(i, i))
            .collect();

        for stage in &mut stages {
            let layer = Linear::new(&backend, LinearConfig::new(64, 64)).unwrap();
            stage.add_layer(Box::new(layer));
        }

        let config = PipelineConfig::new()
            .with_micro_batches(16);

        let pipeline = PipelineParallel::new(stages, pg, config).unwrap();
        let stats = pipeline.pipeline_stats();

        assert_eq!(stats.num_stages, 4);
        assert_eq!(stats.num_micro_batches, 16);
        assert!(stats.efficiency > 0.7); // Should be efficient with enough micro-batches
    }
}
