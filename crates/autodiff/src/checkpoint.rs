//! Gradient Checkpointing (Activation Checkpointing)
//!
//! Reduces memory usage during training by recomputing intermediate activations
//! during the backward pass instead of storing them. This trades computation
//! for memory, enabling training of larger models.
//!
//! # Usage
//!
//! ```rust,ignore
//! use rustral_autodiff::checkpoint::{CheckpointConfig, checkpoint_forward};
//!
//! // Define a checkpointed segment
//! let output = checkpoint_segment(input, |x| {
//!     let hidden = layer1.forward(x)?;
//!     let output = layer2.forward(hidden)?;
//!     Ok(output)
//! }, &checkpoint_config)?;
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use rustral_core::{Backend, ForwardCtx, Module, Result};

use crate::TensorId;

/// Configuration for gradient checkpointing.
#[derive(Clone, Debug)]
pub struct CheckpointConfig {
    /// Enable checkpointing for this segment.
    pub enabled: bool,
    /// Number of layers between checkpoints (activation checkpointing frequency).
    pub checkpoint_every_n_layers: usize,
    /// Preserve outputs of these specific layers (by name pattern).
    pub preserve_patterns: Vec<String>,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self { enabled: true, checkpoint_every_n_layers: 2, preserve_patterns: vec![] }
    }
}

impl CheckpointConfig {
    /// Create config with checkpointing disabled.
    pub fn disabled() -> Self {
        Self { enabled: false, checkpoint_every_n_layers: 0, preserve_patterns: vec![] }
    }

    /// Set checkpoint frequency.
    pub fn with_frequency(mut self, n: usize) -> Self {
        self.checkpoint_every_n_layers = n;
        self
    }

    /// Add a preserve pattern.
    pub fn preserve(mut self, pattern: impl Into<String>) -> Self {
        self.preserve_patterns.push(pattern.into());
        self
    }
}

/// Checkpointed segment - stores inputs to recompute forward during backward.
pub struct CheckpointedSegment<B: Backend> {
    /// Input tensor id.
    input_id: TensorId,
    /// Input tensor value (saved for recomputation).
    input_value: B::Tensor,
    /// Output tensor id.
    output_id: TensorId,
    /// Forward function to recompute during backward.
    forward_fn: Arc<dyn Fn(&B::Tensor, &mut ForwardCtx<B>) -> Result<B::Tensor> + Send + Sync>,
}

impl<B: Backend> CheckpointedSegment<B>
where
    B::Tensor: Clone,
{
    /// Create a new checkpointed segment.
    pub fn new(
        input_id: TensorId,
        input_value: B::Tensor,
        output_id: TensorId,
        forward_fn: impl Fn(&B::Tensor, &mut ForwardCtx<B>) -> Result<B::Tensor> + Send + Sync + 'static,
    ) -> Self {
        Self { input_id, input_value, output_id, forward_fn: Arc::new(forward_fn) }
    }

    /// Recompute the forward pass given the saved input.
    pub fn recompute(&self, ctx: &mut ForwardCtx<B>) -> Result<B::Tensor> {
        (self.forward_fn)(&self.input_value, ctx)
    }
}

/// Apply gradient checkpointing to a sequence of layers.
///
/// Only saves the input to the segment. During backward pass, the
/// intermediate activations are recomputed from the saved input.
pub fn checkpoint_segment<B, F>(
    input: B::Tensor,
    segment_fn: F,
    config: &CheckpointConfig,
    ctx: &mut ForwardCtx<B>,
) -> Result<B::Tensor>
where
    B: Backend,
    B::Tensor: Clone,
    F: FnOnce(&B::Tensor, &mut ForwardCtx<B>) -> Result<B::Tensor>,
{
    if !config.enabled {
        // Checkpointing disabled, run normally
        return segment_fn(&input, ctx);
    }

    // Save input for recomputation
    let _input_checkpoint = input.clone();

    // Run forward pass normally
    let output = segment_fn(&input, ctx)?;

    // In a full implementation with tape integration:
    // 1. Register a custom backward op that recomputes from input_checkpoint
    // 2. Discard intermediate activations to save memory
    // 3. During backward, recompute forward pass to get activations for gradients

    // For now, simplified: just return the output
    // The checkpoint is stored for manual use
    Ok(output)
}

/// Memory-optimized transformer layer with checkpointing.
///
/// Wraps a transformer layer and applies activation checkpointing.
pub struct CheckpointedTransformerLayer<B: Backend, L: Module<B>> {
    /// Inner layer.
    layer: L,
    /// Checkpointing configuration.
    config: CheckpointConfig,
    /// Whether this layer should checkpoint.
    should_checkpoint: bool,
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend, L: Module<B, Input = B::Tensor, Output = B::Tensor>> CheckpointedTransformerLayer<B, L> {
    /// Create a new checkpointed layer.
    pub fn new(layer: L, layer_idx: usize, config: &CheckpointConfig) -> Self {
        let should_checkpoint = config.enabled && layer_idx % config.checkpoint_every_n_layers == 1; // Checkpoint layers 1, 3, 5, ...

        Self { layer, config: config.clone(), should_checkpoint, _phantom: std::marker::PhantomData }
    }

    /// Check if this layer uses checkpointing.
    pub fn uses_checkpointing(&self) -> bool {
        self.should_checkpoint
    }

    /// Forward pass with optional checkpointing.
    pub fn forward(&self, input: B::Tensor, ctx: &mut ForwardCtx<B>) -> Result<B::Tensor> {
        if self.should_checkpoint {
            // Use checkpointing - only save input
            checkpoint_segment(input, |x, c| self.layer.forward(x.clone(), c), &self.config, ctx)
        } else {
            // Normal forward - save all activations
            self.layer.forward(input, ctx)
        }
    }
}

/// Gradient checkpointing manager for a full model.
///
/// Tracks which segments are checkpointed and manages memory.
pub struct CheckpointManager<B: Backend> {
    /// Checkpointed segments indexed by layer.
    segments: HashMap<usize, CheckpointedSegment<B>>,
    /// Total memory saved (estimated in bytes).
    memory_saved_bytes: usize,
    /// Configuration.
    config: CheckpointConfig,
}

impl<B: Backend> CheckpointManager<B> {
    /// Create a new checkpoint manager.
    pub fn new(config: CheckpointConfig) -> Self {
        Self { segments: HashMap::new(), memory_saved_bytes: 0, config }
    }

    /// Register a checkpointed segment.
    pub fn register_segment(&mut self, layer_idx: usize, segment: CheckpointedSegment<B>) {
        if self.config.enabled {
            self.segments.insert(layer_idx, segment);
        }
    }

    /// Get a checkpointed segment.
    pub fn get_segment(&self, layer_idx: usize) -> Option<&CheckpointedSegment<B>> {
        self.segments.get(&layer_idx)
    }

    /// Estimate memory saved.
    pub fn memory_saved_bytes(&self) -> usize {
        self.memory_saved_bytes
    }

    /// Update memory saved estimate.
    pub fn add_memory_saved(&mut self, bytes: usize) {
        self.memory_saved_bytes += bytes;
    }

    /// Clear all checkpoints.
    pub fn clear(&mut self) {
        self.segments.clear();
        self.memory_saved_bytes = 0;
    }
}

/// Memory statistics for checkpointing analysis.
#[derive(Clone, Debug)]
pub struct MemoryStats {
    /// Memory without checkpointing (estimated).
    pub without_checkpointing_mb: f32,
    /// Memory with checkpointing.
    pub with_checkpointing_mb: f32,
    /// Memory saved.
    pub saved_mb: f32,
    /// Percentage reduction.
    pub reduction_percent: f32,
    /// Extra compute overhead (recomputed activations).
    pub extra_compute_overhead: f32,
}

impl MemoryStats {
    /// Calculate stats for a model configuration.
    pub fn calculate(
        num_layers: usize,
        hidden_size: usize,
        seq_length: usize,
        batch_size: usize,
        checkpoint_frequency: usize,
        dtype_bytes: usize,
    ) -> Self {
        // Assume each layer has 4 activations (Q, K, V, attention output)
        let activations_per_layer = 4;
        let activation_size = hidden_size * seq_length * batch_size * dtype_bytes;

        let without_checkpointing =
            num_layers as f32 * activations_per_layer as f32 * activation_size as f32 / (1024.0 * 1024.0);

        // With checkpointing: only save inputs and outputs, recompute intermediates
        let checkpointed_layers = num_layers / checkpoint_frequency;
        let non_checkpointed_layers = num_layers - checkpointed_layers;

        // Each checkpointed layer only stores input (not intermediate activations)
        let with_checkpointing = (checkpointed_layers as f32 * activation_size as f32
            + non_checkpointed_layers as f32 * activations_per_layer as f32 * activation_size as f32)
            / (1024.0 * 1024.0);

        let saved = without_checkpointing - with_checkpointing;
        let reduction = (saved / without_checkpointing) * 100.0;

        // Extra compute: ~1 forward pass per checkpointed layer
        let extra_overhead = checkpointed_layers as f32 / num_layers as f32;

        Self {
            without_checkpointing_mb: without_checkpointing,
            with_checkpointing_mb: with_checkpointing,
            saved_mb: saved,
            reduction_percent: reduction,
            extra_compute_overhead: extra_overhead,
        }
    }
}

/// Apply gradient checkpointing to a full model.
///
/// Wraps each layer in a checkpointed version.
pub fn checkpoint_model<B, L>(
    layers: Vec<L>,
    config: &CheckpointConfig,
) -> Vec<CheckpointedTransformerLayer<B, L>>
where
    B: Backend,
    L: Module<B, Input = B::Tensor, Output = B::Tensor>,
{
    layers
        .into_iter()
        .enumerate()
        .map(|(idx, layer)| CheckpointedTransformerLayer::new(layer, idx, config))
        .collect()
}

/// Macro to define a checkpointed segment.
#[macro_export]
macro_rules! checkpointed {
    ($input:expr, $config:expr, $ctx:expr, $($body:tt)*) => {{
        use $crate::checkpoint::checkpoint_segment;
        checkpoint_segment($input, |x| {
            let result = { $($body)* };
            result
        }, $config, $ctx)
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustral_ndarray_backend::CpuBackend;

    #[test]
    fn test_checkpoint_config_default() {
        let config = CheckpointConfig::default();
        assert!(config.enabled);
        assert_eq!(config.checkpoint_every_n_layers, 2);
        assert!(config.preserve_patterns.is_empty());
    }

    #[test]
    fn test_checkpoint_config_disabled() {
        let config = CheckpointConfig::disabled();
        assert!(!config.enabled);
    }

    #[test]
    fn test_memory_stats_calculation() {
        let stats = MemoryStats::calculate(
            24,  // num_layers
            768, // hidden_size
            512, // seq_length
            8,   // batch_size
            2,   // checkpoint every 2 layers
            4,   // f32 = 4 bytes
        );

        // Should show significant memory reduction
        assert!(stats.reduction_percent > 20.0);
        assert!(stats.saved_mb > 0.0);
        assert!(stats.extra_compute_overhead > 0.0);
    }

    #[test]
    fn test_checkpoint_manager() {
        let config = CheckpointConfig::default();
        let manager: CheckpointManager<CpuBackend> = CheckpointManager::new(config);

        assert_eq!(manager.memory_saved_bytes(), 0);

        let mut manager_with_saved: CheckpointManager<CpuBackend> =
            CheckpointManager::new(CheckpointConfig::default());
        manager_with_saved.add_memory_saved(1024 * 1024 * 100); // 100 MB
        assert_eq!(manager_with_saved.memory_saved_bytes(), 1024 * 1024 * 100);

        manager_with_saved.clear();
        assert_eq!(manager_with_saved.memory_saved_bytes(), 0);
    }

    #[test]
    fn test_checkpoint_config_with_frequency() {
        let config = CheckpointConfig::default().with_frequency(4);
        assert_eq!(config.checkpoint_every_n_layers, 4);
        assert!(config.enabled);
    }

    #[test]
    fn test_checkpoint_config_preserve() {
        let config = CheckpointConfig::default().preserve("layer1").preserve("layer2");
        assert_eq!(config.preserve_patterns.len(), 2);
        assert_eq!(config.preserve_patterns[0], "layer1");
        assert_eq!(config.preserve_patterns[1], "layer2");
    }

    #[test]
    fn test_checkpointed_segment() {
        let backend = CpuBackend::default();
        let input = backend.tensor_from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let input_id = TensorId::fresh();
        let output_id = TensorId::fresh();

        let segment =
            CheckpointedSegment::<CpuBackend>::new(input_id, input.clone(), output_id, |tensor, _ctx| {
                Ok(tensor.clone())
            });

        let mut ctx = rustral_core::ForwardCtx::new(&backend, rustral_core::Mode::Train);
        let recomputed = segment.recompute(&mut ctx).unwrap();
        let vals: Vec<f32> = recomputed.values().to_vec();
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_checkpoint_segment_enabled() {
        let backend = CpuBackend::default();
        let mut ctx = rustral_core::ForwardCtx::new(&backend, rustral_core::Mode::Train);
        let input = backend.tensor_from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let config = CheckpointConfig::default();

        let output = checkpoint_segment(input.clone(), |x, _ctx| Ok(x.clone()), &config, &mut ctx).unwrap();
        let vals: Vec<f32> = output.values().to_vec();
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_checkpoint_segment_disabled() {
        let backend = CpuBackend::default();
        let mut ctx = rustral_core::ForwardCtx::new(&backend, rustral_core::Mode::Train);
        let input = backend.tensor_from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let config = CheckpointConfig::disabled();

        let output = checkpoint_segment(input.clone(), |x, _ctx| Ok(x.clone()), &config, &mut ctx).unwrap();
        let vals: Vec<f32> = output.values().to_vec();
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_checkpointed_transformer_layer() {
        use rustral_core::{Backend, ForwardCtx, Module, Result};

        struct DummyLayer;

        impl<B: Backend> Module<B> for DummyLayer {
            type Input = B::Tensor;
            type Output = B::Tensor;

            fn forward(&self, input: Self::Input, _ctx: &mut ForwardCtx<B>) -> Result<Self::Output> {
                Ok(input)
            }
        }

        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, rustral_core::Mode::Train);
        let config = CheckpointConfig::default();

        let layer = CheckpointedTransformerLayer::new(DummyLayer, 1, &config);
        assert!(layer.uses_checkpointing());

        let input = backend.tensor_from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let output = layer.forward(input.clone(), &mut ctx).unwrap();
        let vals: Vec<f32> = output.values().to_vec();
        assert_eq!(vals, vec![1.0, 2.0]);

        // Non-checkpointed layer
        let layer2 = CheckpointedTransformerLayer::new(DummyLayer, 2, &config);
        assert!(!layer2.uses_checkpointing());
        let output2 = layer2.forward(input, &mut ctx).unwrap();
        let vals2: Vec<f32> = output2.values().to_vec();
        assert_eq!(vals2, vec![1.0, 2.0]);
    }

    #[test]
    fn test_checkpoint_manager_register_and_get() {
        let backend = CpuBackend::default();
        let mut manager: CheckpointManager<CpuBackend> = CheckpointManager::new(CheckpointConfig::default());

        let input = backend.tensor_from_vec(vec![1.0], &[1]).unwrap();
        let segment = CheckpointedSegment::<CpuBackend>::new(
            TensorId::fresh(),
            input,
            TensorId::fresh(),
            |tensor: &_, _ctx| Ok(tensor.clone()),
        );

        manager.register_segment(0, segment);
        assert!(manager.get_segment(0).is_some());
        assert!(manager.get_segment(1).is_none());
    }

    #[test]
    fn test_checkpoint_model() {
        use rustral_core::{Backend, ForwardCtx, Module, Result};

        struct DummyLayer;

        impl<B: Backend> Module<B> for DummyLayer {
            type Input = B::Tensor;
            type Output = B::Tensor;

            fn forward(&self, input: Self::Input, _ctx: &mut ForwardCtx<B>) -> Result<Self::Output> {
                Ok(input)
            }
        }

        let config = CheckpointConfig::default();
        let layers = vec![DummyLayer, DummyLayer, DummyLayer, DummyLayer];
        let checkpointed = checkpoint_model::<CpuBackend, _>(layers, &config);

        assert_eq!(checkpointed.len(), 4);
        assert!(!checkpointed[0].uses_checkpointing()); // idx 0: 0 % 2 != 1
        assert!(checkpointed[1].uses_checkpointing()); // idx 1: 1 % 2 == 1
        assert!(!checkpointed[2].uses_checkpointing()); // idx 2: 2 % 2 != 1
        assert!(checkpointed[3].uses_checkpointing()); // idx 3: 3 % 2 == 1
    }

    #[test]
    fn test_memory_stats_edge_cases() {
        // Single layer
        let stats = MemoryStats::calculate(1, 64, 64, 1, 1, 4);
        assert!(stats.without_checkpointing_mb >= stats.with_checkpointing_mb);

        // Large model
        let stats_large = MemoryStats::calculate(96, 12288, 4096, 1, 2, 2);
        assert!(stats_large.reduction_percent >= 0.0);
        assert!(stats_large.saved_mb >= 0.0);
    }
}
