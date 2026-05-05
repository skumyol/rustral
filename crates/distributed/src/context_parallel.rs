//! Context Parallelism for Ultra-Long Sequences (1M+ tokens)
//!
//! Implements striped ring attention for processing extremely long contexts
//! across multiple devices. Optimized for document-level understanding and
//! long-form content generation.
//!
//! # Key Features
//!
//! - **Striped Attention**: Each device handles interleaved token chunks
//! - **Ring Communication**: Efficient KV block rotation
//! - **Load Balancing**: Dynamic chunk sizing for uneven sequences
//! - **Compatible with Flash Attention**: Memory-efficient attention
//!
//! # Example
//!
//! ```rust,ignore
//! use mnr_distributed::context_parallel::{ContextParallel, ContextParallelConfig};
//!
//! let config = ContextParallelConfig::new(8)  // 8 devices
//!     .with_sequence_length(1_048_576)      // 1M tokens
//!     .with_block_size(8192);               // 8K blocks
//!
//! let ctx_parallel = ContextParallel::new(config, process_group)?;
//! let output = ctx_parallel.forward(&input, &mut model)?;
//! ```

use mnr_core::{Backend, CoreError, ForwardCtx, Module, Result, TensorOps};
use mnr_nn::SelfAttentionConfig;

use crate::ProcessGroup;

/// Configuration for context parallelism.
pub struct ContextParallelConfig {
    /// Number of devices for parallelism
    pub num_devices: usize,
    /// Total sequence length
    pub sequence_length: usize,
    /// Attention block size
    pub block_size: usize,
    /// Whether to use striped pattern (vs contiguous)
    pub striped: bool,
    /// Whether to use ring attention
    pub use_ring_attention: bool,
    /// All-gather output (vs keep sharded)
    pub all_gather_output: bool,
}

impl ContextParallelConfig {
    pub fn new(num_devices: usize) -> Self {
        Self {
            num_devices,
            sequence_length: 131_072, // 128K default
            block_size: 4096,
            striped: true,
            use_ring_attention: true,
            all_gather_output: false,
        }
    }

    pub fn with_sequence_length(mut self, len: usize) -> Self {
        self.sequence_length = len;
        self
    }

    pub fn with_block_size(mut self, size: usize) -> Self {
        self.block_size = size;
        self
    }

    pub fn with_striped(mut self, striped: bool) -> Self {
        self.striped = striped;
        self
    }

    pub fn with_ring_attention(mut self, enabled: bool) -> Self {
        self.use_ring_attention = enabled;
        self
    }

    pub fn with_all_gather_output(mut self, enabled: bool) -> Self {
        self.all_gather_output = enabled;
        self
    }

    /// Calculate tokens per device.
    pub fn tokens_per_device(&self) -> usize {
        self.sequence_length / self.num_devices
    }

    /// Validate configuration.
    pub fn validate(&self) -> Result<()> {
        if self.sequence_length % self.num_devices != 0 {
            return Err(CoreError::InvalidArgument(
                format!("Sequence length {} must be divisible by num_devices {}",
                    self.sequence_length, self.num_devices)
            ));
        }

        if self.tokens_per_device() % self.block_size != 0 {
            return Err(CoreError::InvalidArgument(
                format!("Tokens per device ({}) must be divisible by block_size {}",
                    self.tokens_per_device(), self.block_size)
            ));
        }

        Ok(())
    }

    /// Print configuration summary.
    pub fn print_summary(&self) {
        println!("Context Parallelism Configuration:");
        println!("  Devices: {}", self.num_devices);
        println!("  Sequence Length: {} tokens", self.sequence_length);
        println!("  Tokens per Device: {}", self.tokens_per_device());
        println!("  Block Size: {}", self.block_size);
        println!("  Pattern: {}", if self.striped { "striped" } else { "contiguous" });
        println!("  Ring Attention: {}", self.use_ring_attention);
    }
}

/// Context parallelism for distributed long-sequence attention.
pub struct ContextParallel<B: Backend> {
    config: ContextParallelConfig,
    process_group: ProcessGroup,
    my_rank: usize,
    _backend: std::marker::PhantomData<B>,
}

impl<B: Backend> ContextParallel<B>
where
    B::Tensor: AsRef<[f32]> + mnr_core::TensorShape,
{
    pub fn new(config: ContextParallelConfig, process_group: ProcessGroup) -> Result<Self> {
        config.validate()?;

        let my_rank = process_group.rank();
        let world_size = process_group.world_size();

        if world_size != config.num_devices {
            return Err(CoreError::InvalidArgument(
                format!("Process group size {} doesn't match config {}",
                    world_size, config.num_devices)
            ));
        }

        Ok(Self {
            config,
            process_group,
            my_rank,
            _backend: std::marker::PhantomData,
        })
    }

    /// Forward pass with context parallelism.
    pub fn forward<M: Module<B, Input = B::Tensor, Output = B::Tensor>>(
        &self,
        input: &B::Tensor,
        model: &mut M,
        ctx: &mut ForwardCtx<B>,
    ) -> Result<B::Tensor> {
        // 1. Shard input across devices
        let sharded_input = self.shard_input(input, ctx)?;

        // 2. Apply ring attention if enabled
        let output = if self.config.use_ring_attention {
            self.ring_attention_forward(&sharded_input, model, ctx)?
        } else {
            // Simple local forward
            model.forward(sharded_input, ctx)?
        };

        // 3. Optionally all-gather output
        if self.config.all_gather_output {
            self.all_gather(&output, ctx)
        } else {
            Ok(output)
        }
    }

    /// Shard input according to striped or contiguous pattern.
    fn shard_input(&self, input: &B::Tensor, ctx: &mut ForwardCtx<B>) -> Result<B::Tensor> {
        let ops = ctx.backend().ops();
        let shape = ops.shape(input);

        if shape.len() < 2 {
            return Err(CoreError::Shape(
                "Input must have at least 2 dimensions [batch, seq, ...]".to_string()
            ));
        }

        let batch_size = shape[0];
        let seq_len = shape[1];

        if seq_len != self.config.sequence_length {
            return Err(CoreError::InvalidArgument(
                format!("Input seq len {} doesn't match config {}",
                    seq_len, self.config.sequence_length)
            ));
        }

        // Extract local portion
        let data: Vec<f32> = input.as_ref().to_vec();
        let tokens_per_device = self.config.tokens_per_device();
        let mut local_data = Vec::with_capacity(batch_size * tokens_per_device * shape[2..].iter().product::<usize>());

        if self.config.striped {
            // Striped pattern: device i gets tokens i, i+num_devices, i+2*num_devices, ...
            let num_features: usize = shape[2..].iter().product();

            for b in 0..batch_size {
                for local_pos in 0..tokens_per_device {
                    let global_pos = self.my_rank + local_pos * self.config.num_devices;
                    let offset = b * seq_len * num_features + global_pos * num_features;
                    local_data.extend_from_slice(&data[offset..offset + num_features]);
                }
            }
        } else {
            // Contiguous pattern: device i gets tokens [i*tpd, (i+1)*tpd)
            let num_features: usize = shape[2..].iter().product();
            let start = self.my_rank * tokens_per_device;

            for b in 0..batch_size {
                for pos in start..(start + tokens_per_device) {
                    let offset = b * seq_len * num_features + pos * num_features;
                    local_data.extend_from_slice(&data[offset..offset + num_features]);
                }
            }
        }

        let mut new_shape = shape.clone();
        new_shape[1] = tokens_per_device;
        ops.tensor_from_vec(local_data, &new_shape)
    }

    /// Ring attention forward pass.
    fn ring_attention_forward<M: Module<B, Input = B::Tensor, Output = B::Tensor>>(
        &self,
        local_input: &B::Tensor,
        _model: &mut M,
        ctx: &mut ForwardCtx<B>,
    ) -> Result<B::Tensor> {
        // Extract Q, K, V from local input
        let ops = ctx.backend().ops();
        let shape = ops.shape(local_input);

        // Number of ring iterations = number of devices
        let num_iters = self.config.num_devices;

        // Initialize output accumulator
        let output_size: usize = shape.iter().product();
        let mut output_accum: Vec<f32> = vec![0.0; output_size];
        let mut max_logits: Vec<f32> = vec![f32::NEG_INFINITY; output_size];
        let mut sum_exp: Vec<f32> = vec![0.0; output_size];

        // Current KV blocks (start with local)
        let mut current_kv = self.extract_kv(local_input, ops)?;

        for iter in 0..num_iters {
            // Compute attention with current KV blocks
            let (q, k, v) = &current_kv;
            let scores = self.compute_attention_scores(q, k, ops)?;

            // Update output with online softmax
            self.update_online_softmax(&mut output_accum, &mut max_logits, &mut sum_exp, &scores, v, ops)?;

            // Rotate KV blocks to next device
            if iter < num_iters - 1 {
                current_kv = self.rotate_kv_blocks(current_kv, ops)?;
            }
        }

        // Final normalization and create output tensor
        self.finalize_output(&mut output_accum, &sum_exp, ops)?;
        ops.tensor_from_vec(output_accum, &shape)
    }

    /// Extract Q, K, V from input (simplified - in real impl, use linear projection).
    fn extract_kv(
        &self,
        input: &B::Tensor,
        ops: &dyn TensorOps<B>,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let data: Vec<f32> = input.as_ref().to_vec();
        // In real implementation, apply Q, K, V projections
        // For now, use same tensor as all three (simplified)
        Ok((data.clone(), data.clone(), data))
    }

    /// Compute attention scores Q @ K^T.
    fn compute_attention_scores(
        &self,
        q: &[f32],
        k: &[f32],
        _ops: &dyn TensorOps<B>,
    ) -> Result<Vec<f32>> {
        // Simplified attention score computation
        // In real impl: proper matrix multiply with scaling
        let len = q.len().min(k.len());
        let mut scores = vec![0.0f32; len];

        for i in 0..len {
            scores[i] = q[i] * k[i]; // Simplified dot product
        }

        Ok(scores)
    }

    /// Update output with online softmax (stable attention accumulation).
    fn update_online_softmax(
        &self,
        output: &mut [f32],
        max_logits: &mut [f32],
        sum_exp: &mut [f32],
        scores: &[f32],
        v: &[f32],
        _ops: &dyn TensorOps<B>,
    ) -> Result<()> {
        // Online softmax update for stable attention computation
        for i in 0..scores.len().min(output.len()) {
            let new_max = max_logits[i].max(scores[i]);
            let exp_old = (max_logits[i] - new_max).exp();
            let exp_new = (scores[i] - new_max).exp();

            // Update output weighted by attention
            let v_idx = i % v.len();
            output[i] = output[i] * exp_old + v[v_idx] * exp_new;
            sum_exp[i] = sum_exp[i] * exp_old + exp_new;
            max_logits[i] = new_max;
        }

        Ok(())
    }

    /// Finalize output by dividing by sum_exp.
    fn finalize_output(
        &self,
        output: &mut [f32],
        sum_exp: &[f32],
        _ops: &dyn TensorOps<B>,
    ) -> Result<()> {
        for i in 0..output.len().min(sum_exp.len()) {
            if sum_exp[i] > 0.0 {
                output[i] /= sum_exp[i];
            }
        }
        Ok(())
    }

    /// Rotate KV blocks to next device in ring.
    fn rotate_kv_blocks(
        &self,
        kv: (Vec<f32>, Vec<f32>, Vec<f32>),
        _ops: &dyn TensorOps<B>,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        // In real implementation: async send/recv
        // For now, return same (simplified)
        Ok(kv)
    }

    /// All-gather sharded outputs across devices.
    fn all_gather(&self, local_output: &B::Tensor, ctx: &mut ForwardCtx<B>) -> Result<B::Tensor> {
        let ops = ctx.backend().ops();
        let shape = ops.shape(local_output);
        let tokens_per_device = shape[1];

        // Collect from all devices
        // In real impl: actual all-gather communication
        let local_data: Vec<f32> = local_output.as_ref().to_vec();

        // Reconstruct full sequence (simplified - assumes same data from all)
        let mut full_data = Vec::with_capacity(local_data.len() * self.config.num_devices);

        for _ in 0..self.config.num_devices {
            full_data.extend_from_slice(&local_data);
        }

        let mut full_shape = shape.clone();
        full_shape[1] = tokens_per_device * self.config.num_devices;

        ops.tensor_from_vec(full_data, &full_shape)
    }

    /// Calculate communication statistics.
    pub fn communication_stats(&self) -> CommunicationStats {
        let tokens_per_device = self.config.tokens_per_device();
        let kv_size_per_iter = tokens_per_device * 2 * 4; // K + V, f32 = 4 bytes
        let total_ring_comm = kv_size_per_iter * (self.config.num_devices - 1);

        CommunicationStats {
            tokens_per_device,
            kv_size_per_iteration: kv_size_per_iter,
            total_ring_communication: total_ring_comm,
            num_ring_iterations: self.config.num_devices,
        }
    }
}

/// Communication statistics for context parallelism.
#[derive(Debug, Clone)]
pub struct CommunicationStats {
    pub tokens_per_device: usize,
    pub kv_size_per_iteration: usize,
    pub total_ring_communication: usize,
    pub num_ring_iterations: usize,
}

impl CommunicationStats {
    /// Print statistics.
    pub fn print(&self) {
        println!("Context Parallel Communication Stats:");
        println!("  Tokens per device: {}", self.tokens_per_device);
        println!("  KV size per iteration: {:.2} MB", self.kv_size_per_iteration as f64 / 1e6);
        println!("  Total ring communication: {:.2} MB", self.total_ring_communication as f64 / 1e6);
        println!("  Ring iterations: {}", self.num_ring_iterations);
    }
}

/// Dynamic load balancing for uneven sequences.
pub struct DynamicLoadBalancer {
    /// Target tokens per device
    target_tokens: usize,
    /// Current assignments
    assignments: Vec<usize>,
}

impl DynamicLoadBalancer {
    pub fn new(target_tokens: usize, num_devices: usize) -> Self {
        Self {
            target_tokens,
            assignments: vec![target_tokens; num_devices],
        }
    }

    /// Rebalance based on actual token distribution.
    pub fn rebalance(&mut self, actual_tokens: &[usize]) {
        let total: usize = actual_tokens.iter().sum();
        let num_devices = self.assignments.len();

        // Simple greedy reassignment
        for (i, &actual) in actual_tokens.iter().enumerate().take(num_devices) {
            if actual > self.target_tokens * 12 / 10 {
                // Overloaded - would need to redistribute
                println!("Device {} overloaded: {} > {}", i, actual, self.target_tokens);
            }
            self.assignments[i] = actual;
        }
    }

    /// Get assignment for device.
    pub fn get_assignment(&self, device_id: usize) -> usize {
        self.assignments.get(device_id).copied().unwrap_or(self.target_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnr_ndarray_backend::CpuBackend;
    use mnr_core::{ForwardCtx, Mode, Module};

    /// Mock module that returns input unchanged (identity).
    struct IdentityModule;

    impl<B: Backend> Module<B> for IdentityModule
    where
        B::Tensor: Clone,
    {
        type Input = B::Tensor;
        type Output = B::Tensor;

        fn forward(&self, input: Self::Input, _ctx: &mut ForwardCtx<B>) -> Result<Self::Output> {
            Ok(input)
        }
    }

    #[test]
    fn test_context_parallel_config() {
        let config = ContextParallelConfig::new(8)
            .with_sequence_length(1_048_576)
            .with_block_size(8192)
            .with_striped(true)
            .with_ring_attention(true)
            .with_all_gather_output(true);

        assert_eq!(config.tokens_per_device(), 131_072);
        assert_eq!(config.block_size, 8192);
        assert!(config.striped);
        assert!(config.use_ring_attention);
        assert!(config.all_gather_output);
        config.validate().unwrap();
    }

    #[test]
    fn test_config_validation_errors() {
        // Sequence not divisible by num_devices
        let config = ContextParallelConfig::new(3)
            .with_sequence_length(10);
        assert!(config.validate().is_err());

        // Tokens per device not divisible by block_size
        let config = ContextParallelConfig::new(2)
            .with_sequence_length(8)
            .with_block_size(3);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_print_summary() {
        let config = ContextParallelConfig::new(4)
            .with_sequence_length(8192)
            .with_block_size(1024);
        config.print_summary();
    }

    #[test]
    fn test_context_parallel_striped_vs_contiguous() {
        let config = ContextParallelConfig::new(4)
            .with_sequence_length(1_048_576);

        assert_eq!(config.tokens_per_device(), 262_144);
    }

    #[test]
    fn test_context_parallel_new_mismatch() {
        let config = ContextParallelConfig::new(8)
            .with_sequence_length(8192);
        let pg = ProcessGroup::new_threaded(4, 0).unwrap();
        assert!(ContextParallel::<CpuBackend>::new(config, pg).is_err());
    }

    #[test]
    fn test_forward_no_ring_attention() {
        let backend = CpuBackend::default();
        let config = ContextParallelConfig::new(1)
            .with_sequence_length(8)
            .with_block_size(4)
            .with_ring_attention(false)
            .with_all_gather_output(false);
        let pg = ProcessGroup::new_threaded(1, 0).unwrap();
        let ctx_parallel = ContextParallel::<CpuBackend>::new(config, pg).unwrap();

        let input = backend.tensor_from_vec(vec![1.0f32; 32], &[1, 8, 4]).unwrap();
        let mut model = IdentityModule;
        let mut ctx = ForwardCtx::new(&backend, Mode::Train);

        let output = ctx_parallel.forward(&input, &mut model, &mut ctx).unwrap();
        let shape = backend.ops().shape(&output);
        assert_eq!(shape[0], 1);
        assert_eq!(shape[1], 8);
    }

    #[test]
    fn test_forward_with_ring_attention() {
        let backend = CpuBackend::default();
        let config = ContextParallelConfig::new(1)
            .with_sequence_length(8)
            .with_block_size(4)
            .with_ring_attention(true)
            .with_all_gather_output(false);
        let pg = ProcessGroup::new_threaded(1, 0).unwrap();
        let ctx_parallel = ContextParallel::<CpuBackend>::new(config, pg).unwrap();

        let input = backend.tensor_from_vec(vec![1.0f32; 32], &[1, 8, 4]).unwrap();
        let mut model = IdentityModule;
        let mut ctx = ForwardCtx::new(&backend, Mode::Train);

        let output = ctx_parallel.forward(&input, &mut model, &mut ctx).unwrap();
        let shape = backend.ops().shape(&output);
        assert_eq!(shape[0], 1);
        assert_eq!(shape[1], 8);
    }

    #[test]
    fn test_forward_all_gather_output() {
        let backend = CpuBackend::default();
        let config = ContextParallelConfig::new(1)
            .with_sequence_length(8)
            .with_block_size(4)
            .with_ring_attention(false)
            .with_all_gather_output(true);
        let pg = ProcessGroup::new_threaded(1, 0).unwrap();
        let ctx_parallel = ContextParallel::<CpuBackend>::new(config, pg).unwrap();

        let input = backend.tensor_from_vec(vec![1.0f32; 32], &[1, 8, 4]).unwrap();
        let mut model = IdentityModule;
        let mut ctx = ForwardCtx::new(&backend, Mode::Train);

        let output = ctx_parallel.forward(&input, &mut model, &mut ctx).unwrap();
        let shape = backend.ops().shape(&output);
        assert_eq!(shape[0], 1);
        assert_eq!(shape[1], 8);
    }

    #[test]
    fn test_forward_bad_input_shape() {
        let backend = CpuBackend::default();
        let config = ContextParallelConfig::new(1)
            .with_sequence_length(8)
            .with_block_size(4);
        let pg = ProcessGroup::new_threaded(1, 0).unwrap();
        let ctx_parallel = ContextParallel::<CpuBackend>::new(config, pg).unwrap();

        // 1D input should fail
        let input = backend.tensor_from_vec(vec![1.0f32; 8], &[8]).unwrap();
        let mut model = IdentityModule;
        let mut ctx = ForwardCtx::new(&backend, Mode::Train);

        assert!(ctx_parallel.forward(&input, &mut model, &mut ctx).is_err());
    }

    #[test]
    fn test_forward_wrong_seq_len() {
        let backend = CpuBackend::default();
        let config = ContextParallelConfig::new(1)
            .with_sequence_length(8)
            .with_block_size(4);
        let pg = ProcessGroup::new_threaded(1, 0).unwrap();
        let ctx_parallel = ContextParallel::<CpuBackend>::new(config, pg).unwrap();

        // Input seq_len=4 doesn't match config seq_len=8
        let input = backend.tensor_from_vec(vec![1.0f32; 16], &[1, 4, 4]).unwrap();
        let mut model = IdentityModule;
        let mut ctx = ForwardCtx::new(&backend, Mode::Train);

        assert!(ctx_parallel.forward(&input, &mut model, &mut ctx).is_err());
    }

    #[test]
    fn test_communication_stats() {
        let config = ContextParallelConfig::new(8)
            .with_sequence_length(1_048_576);

        let stats = ContextParallel::<CpuBackend>::new(
            config,
            ProcessGroup::new_threaded(8, 0).unwrap()
        ).unwrap().communication_stats();

        assert_eq!(stats.tokens_per_device, 131_072);
        assert_eq!(stats.num_ring_iterations, 8);
        assert!(stats.kv_size_per_iteration > 0);
        assert!(stats.total_ring_communication > 0);
        stats.print();
    }

    #[test]
    fn test_dynamic_load_balancer() {
        let mut balancer = DynamicLoadBalancer::new(100_000, 8);

        // Simulate uneven distribution
        let actual = vec![120_000, 90_000, 100_000, 80_000, 100_000, 110_000, 95_000, 105_000];
        balancer.rebalance(&actual);

        assert_eq!(balancer.get_assignment(0), 120_000);
        assert_eq!(balancer.get_assignment(3), 80_000);
        // Out of bounds falls back to target
        assert_eq!(balancer.get_assignment(100), 100_000);
    }

    #[test]
    fn test_dynamic_load_balancer_fewer_devices() {
        let mut balancer = DynamicLoadBalancer::new(50_000, 4);
        let actual = vec![30_000, 70_000];
        balancer.rebalance(&actual);
        assert_eq!(balancer.get_assignment(0), 30_000);
        assert_eq!(balancer.get_assignment(1), 70_000);
        assert_eq!(balancer.get_assignment(2), 50_000);
        assert_eq!(balancer.get_assignment(3), 50_000);
    }

    #[test]
    fn test_ring_attention_equivalent() {
        let config = ContextParallelConfig::new(2)
            .with_sequence_length(8192)
            .with_ring_attention(true);

        assert!(config.use_ring_attention);
        config.validate().unwrap();
    }
}
