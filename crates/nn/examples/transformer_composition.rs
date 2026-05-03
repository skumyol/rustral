//! Composing Transformer Architectures
//!
//! Demonstrates how to combine different transformer components
//! for advanced use cases like:
//! - Multi-task learning
//! - Adapter layers
//! - Pre-trained model fine-tuning
//!
//! # Architecture Patterns
//!
//! ```text
//! Multi-Task:
//!   Shared Encoder → Task A Head
//!                → Task B Head
//!                → Task C Head
//!
//! Adapter Tuning:
//!   Frozen Encoder → Adapter Layer → Fine-tuned Head
//!
//! Cascaded Models://!   Encoder1 → Decoder1 → Encoder2 → Decoder2
//! ```

use mnr_core::{ForwardCtx, Mode, Module, Trainable, ParameterRef};
use mnr_nn::{
    Linear, LinearConfig,
    TransformerEncoder, TransformerEncoderConfig,
    TransformerDecoder, TransformerDecoderConfig,
    LayerNorm, LayerNormConfig,
};

/// Multi-task transformer with shared encoder and task-specific heads.
struct MultiTaskTransformer<B: mnr_core::Backend> {
    encoder: TransformerEncoder<B>,
    task_heads: Vec<Linear<B>>,
}

impl<B: mnr_core::Backend> MultiTaskTransformer<B>
where
    B::Tensor: Clone + AsRef<[f32]> + mnr_core::TensorShape,
{
    fn new(
        backend: &B,
        encoder_config: TransformerEncoderConfig,
        vocab_size: usize,
        task_sizes: Vec<usize>, // Output size for each task
        seed: u64,
    ) -> mnr_core::Result<Self> {
        let encoder = TransformerEncoder::new(backend, encoder_config, vocab_size, seed)?;

        let mut task_heads = Vec::new();
        for (i, task_size) in task_sizes.iter().enumerate() {
            let head = Linear::new(
                backend,
                LinearConfig::new(encoder.config().d_model, *task_size).with_bias(true),
            )?;
            task_heads.push(head);
        }

        Ok(Self {
            encoder,
            task_heads,
        })
    }

    /// Forward pass returning outputs for all tasks.
    fn forward_all(&self, input: B::Tensor, ctx: &mut ForwardCtx<B>) -> mnr_core::Result<Vec<B::Tensor>> {
        // Shared encoding
        let encoded = self.encoder.forward(input, ctx)?;

        // Get CLS token representation
        let cls = self.encoder.cls_token(&encoded, ctx.backend().ops())?;

        // Task-specific outputs
        let mut outputs = Vec::new();
        for head in &self.task_heads {
            let output = head.forward(cls.clone(), ctx)?;
            outputs.push(output);
        }

        Ok(outputs)
    }
}

impl<B: mnr_core::Backend> Trainable<B> for MultiTaskTransformer<B> {
    fn parameters(&self) -> Vec<ParameterRef<B>> {
        let mut params = self.encoder.parameters();
        for head in &self.task_heads {
            params.extend(head.parameters());
        }
        params
    }
}

/// Adapter layer for parameter-efficient fine-tuning.
struct Adapter<B: mnr_core::Backend> {
    /// Down-projection
    down: Linear<B>,
    /// Up-projection
    up: Linear<B>,
    /// Layer norm
    norm: LayerNorm<B>,
    /// Bottleneck dimension
    bottleneck_dim: usize,
}

impl<B: mnr_core::Backend> Adapter<B>
where
    B::Tensor: Clone + AsRef<[f32]> + mnr_core::TensorShape,
{
    fn new(backend: &B, d_model: usize, bottleneck_dim: usize, seed: u64) -> mnr_core::Result<Self> {
        let down = Linear::new(
            backend,
            LinearConfig::new(d_model, bottleneck_dim).with_bias(false),
        )?;

        let up = Linear::new(
            backend,
            LinearConfig::new(bottleneck_dim, d_model).with_bias(false),
        )?;

        let norm = LayerNorm::new(
            backend,
            LayerNormConfig::new(vec![d_model]).with_eps(1e-5),
            seed,
        )?;

        Ok(Self {
            down,
            up,
            norm,
            bottleneck_dim,
        })
    }

    fn forward(&self, input: B::Tensor, ctx: &mut ForwardCtx<B>) -> mnr_core::Result<B::Tensor> {
        // Residual connection with bottleneck
        let normed = self.norm.forward(input.clone(), ctx)?;
        let bottleneck = self.down.forward(normed, ctx)?;
        // Activation (simplified as ReLU)
        let activated = ctx.backend().ops().relu(&bottleneck)?;
        let up_projected = self.up.forward(activated, ctx)?;

        // Residual connection
        ctx.backend().ops().add(&input, &up_projected)
    }
}

impl<B: mnr_core::Backend> Trainable<B> for Adapter<B> {
    fn parameters(&self) -> Vec<ParameterRef<B>> {
        let mut params = self.down.parameters();
        params.extend(self.up.parameters());
        params.extend(self.norm.parameters());
        params
    }
}

fn main() {
    use mnr_ndarray_backend::CpuBackend;

    println!("Composing Transformer Architectures");
    println!("==================================\n");

    let backend = CpuBackend::default();

    // Example 1: Multi-Task Learning
    println!("Example 1: Multi-Task Transformer");
    println!("----------------------------------");

    let encoder_config = TransformerEncoderConfig::new(256, 8, 4, 1024);

    // 3 tasks: sentiment (3 classes), NER (17 tags), QA (start/end positions)
    let task_sizes = vec![3, 17, 512];

    let multi_task = MultiTaskTransformer::new(
        &backend,
        encoder_config,
        10000, // vocab size
        task_sizes.clone(),
        42,
    ).unwrap();

    // Input
    let input = backend
        .tensor_from_vec(vec![101u32, 200, 300, 400, 102], &[1, 5])
        .unwrap();

    let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
    let task_outputs = multi_task.forward_all(input, &mut ctx).unwrap();

    println!("Multi-task outputs:");
    for (i, (output, task_size)) in task_outputs.iter().zip(&task_sizes).enumerate() {
        let shape = backend.ops().shape(output);
        let task_name = match i {
            0 => "Sentiment",
            1 => "NER",
            2 => "QA",
            _ => "Unknown",
        };
        println!("  Task {} ({}): shape={:?}, classes={}", i + 1, task_name, shape, task_size);
    }

    // Count parameters
    let total_params: usize = multi_task
        .parameters()
        .iter()
        .map(|p| p.as_tensor(backend.ops()).as_ref().len())
        .sum();
    println!("\nTotal parameters: ~{}M", total_params / 1_000_000);

    // Example 2: Parameter-Efficient Fine-Tuning with Adapters
    println!("\nExample 2: Adapter Layers");
    println!("--------------------------");

    let d_model = 768;
    let bottleneck_dim = 64;

    let adapter = Adapter::new(&backend, d_model, bottleneck_dim, 42).unwrap();

    let adapter_params: usize = adapter
        .parameters()
        .iter()
        .map(|p| p.as_tensor(backend.ops()).as_ref().len())
        .sum();

    // Full fine-tuning would need d_model * d_model params per layer
    // Adapter only needs: d_model * bottleneck + bottleneck * d_model
    let full_params = d_model * d_model;
    let adapter_only = d_model * bottleneck_dim + bottleneck_dim * d_model;

    println!("Adapter configuration:");
    println!("  d_model: {}", d_model);
    println!("  bottleneck_dim: {}", bottleneck_dim);
    println!("  Adapter parameters: ~{}", adapter_params);
    println!("  Full layer parameters: ~{}", full_params);
    println!("  Parameter reduction: {:.1}%", (1.0 - adapter_only as f32 / full_params as f32) * 100.0);

    println!("\nAdapter forward pass:");
    let features = backend
        .tensor_from_vec(vec![0.1f32; d_model], &[1, d_model])
        .unwrap();

    let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
    let adapted = adapter.forward(features, &mut ctx).unwrap();
    println!("  Input shape: [1, {}]", d_model);
    println!("  Output shape: {:?}", backend.ops().shape(&adapted));

    // Example 3: Architecture Comparison
    println!("\nExample 3: Architecture Design Patterns");
    println!("--------------------------------------");

    println!("  Pattern 1: Single-Task");
    println!("    Encoder → Classification Head");
    println!("    Use case: Standard fine-tuning");
    println!("    Params: All parameters trainable");
    println!();

    println!("  Pattern 2: Multi-Task");
    println!("    Shared Encoder → [Task A, Task B, Task C]");
    println!("    Use case: Related tasks with shared representations");
    println!("    Benefit: Transfer learning between tasks");
    println!();

    println!("  Pattern 3: Adapter Tuning");
    println!("    Frozen Encoder → Adapter → Task Head");
    println!("    Use case: Many downstream tasks, limited compute");
    println!("    Benefit: <1% additional parameters per task");
    println!();

    println!("  Pattern 4: Cascaded Generation");
    println!("    Draft Model → Full Model");
    println!("    Use case: Fast speculative decoding");
    println!("    Benefit: 2-3x speedup");

    // Example 4: Memory comparison
    println!("\nExample 4: Memory Footprint");
    println!("----------------------------");

    let d_model = 768;
    let vocab_size = 32000;
    let seq_len = 512;

    // Embedding memory
    let embedding_mem = vocab_size * d_model * 4; // 4 bytes per f32
    println!("  Embeddings: {:.1} MB", embedding_mem as f32 / 1_048_576.0);

    // Per-layer memory
    let per_layer_params = d_model * d_model * 4; // attn weights
    let per_layer_mem = per_layer_params * 4 / 1_048_576; // MB
    println!("  Per-layer (attn): ~{} MB", per_layer_mem);

    // KV cache memory
    let kv_cache_per_seq = 2 * seq_len * d_model * 4; // 2 for K and V
    println!("  KV cache (1 seq): {:.1} MB", kv_cache_per_seq as f32 / 1_048_576.0);

    println!("\nDone! Architecture composition patterns demonstrated.");
}
