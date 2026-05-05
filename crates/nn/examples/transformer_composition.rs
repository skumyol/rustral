//! Composing Transformer Architectures
//!
//! Demonstrates architecture patterns for:
//! - Multi-task learning
//! - Adapter layers
//! - Pre-trained model fine-tuning
//!
//! Run with: `cargo run -p mnr-nn --example transformer_composition`

use mnr_core::{Backend, ForwardCtx, Mode, Module};
use mnr_nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};

fn main() {
    use mnr_ndarray_backend::CpuBackend;

    println!("Composing Transformer Architectures");
    println!("====================================\n");

    let backend = CpuBackend::default();
    let ops = backend.ops();

    let d_model = 64;
    let vocab_size = 1000;

    // Example 1: Multi-Task Learning
    println!("Example 1: Multi-Task Learning");
    println!("--------------------------------");

    // Shared encoder
    let token_embedding = Embedding::new(&backend, EmbeddingConfig::new(vocab_size, d_model), 42).unwrap();
    let shared_proj = Linear::new(&backend, LinearConfig::new(d_model, d_model)).unwrap();

    // Task-specific heads
    let sentiment_head = Linear::new(&backend, LinearConfig::new(d_model, 3).with_bias(true)).unwrap();
    let ner_head = Linear::new(&backend, LinearConfig::new(d_model, 17).with_bias(true)).unwrap();

    // Sample input
    let tokens: Vec<usize> = vec![101, 50, 80, 102, 0];
    let mut ctx = ForwardCtx::new(&backend, Mode::Inference);

    // Shared encoding
    let embedded = token_embedding.forward(tokens, &mut ctx).unwrap();
    let encoded = shared_proj.forward(embedded.clone(), &mut ctx).unwrap();

    // First token for classification tasks
    let cls = ops.slice(&encoded, 0, 1).unwrap();

    // Task outputs
    let sentiment_logits = sentiment_head.forward(cls.clone(), &mut ctx).unwrap();
    let ner_logits = ner_head.forward(encoded, &mut ctx).unwrap();

    println!("Sentiment logits shape: {:?}", ops.shape(&sentiment_logits));
    println!("  Classes: [negative, neutral, positive]");
    println!("NER logits shape: {:?}", ops.shape(&ner_logits));
    println!("  Tags: 17 different entity types\n");

    // Example 2: Adapter Layers
    println!("Example 2: Adapter Layers");
    println!("--------------------------");

    let bottleneck_dim = 16;

    // Adapter: d_model -> bottleneck -> d_model
    let adapter_down = Linear::new(&backend, LinearConfig::new(d_model, bottleneck_dim)).unwrap();
    let adapter_up = Linear::new(&backend, LinearConfig::new(bottleneck_dim, d_model)).unwrap();
    let adapter_norm = LayerNorm::new(&backend, LayerNormConfig::new(vec![d_model]), 42).unwrap();

    // Full model would be: Frozen Encoder -> Adapter -> Task Head
    let features = embedded.clone();
    let normed = adapter_norm.forward(features.clone(), &mut ctx).unwrap();
    let bottleneck = adapter_down.forward(normed, &mut ctx).unwrap();
    let activated = ops.relu(&bottleneck).unwrap();
    let up_projected = adapter_up.forward(activated, &mut ctx).unwrap();
    let adapted = ops.add(&features, &up_projected).unwrap();

    println!("Adapter configuration:");
    println!("  d_model: {}", d_model);
    println!("  bottleneck_dim: {}", bottleneck_dim);

    let full_params = d_model * d_model;
    let adapter_params = d_model * bottleneck_dim + bottleneck_dim * d_model;
    println!("  Full projection params: {}", full_params);
    println!("  Adapter params: {}", adapter_params);
    println!("  Reduction: {:.1}%\n", (1.0 - adapter_params as f32 / full_params as f32) * 100.0);

    let adapted_shape = ops.shape(&adapted);
    println!("Adapted features shape: {:?}", adapted_shape);

    // Example 3: Architecture Patterns
    println!("\nExample 3: Architecture Design Patterns");
    println!("----------------------------------------");

    println!("Pattern 1: Single-Task");
    println!("  Encoder → Classification Head");
    println!("  All parameters trainable\n");

    println!("Pattern 2: Multi-Task");
    println!("  Shared Encoder → [Task A, Task B, Task C]");
    println!("  Transfer learning between tasks\n");

    println!("Pattern 3: Adapter Tuning");
    println!("  Frozen Encoder → Adapter → Task Head");
    println!("  <1%% additional parameters per task\n");

    println!("Pattern 4: Cascaded Generation");
    println!("  Draft Model → Full Model");
    println!("  2-3x speedup with speculative decoding\n");

    // Example 4: Memory Footprint
    println!("Example 4: Memory Footprint");
    println!("---------------------------");

    let seq_len = 512;

    let embedding_mem = vocab_size * d_model * 4;
    let per_layer_mem = d_model * d_model * 4;
    let kv_cache_mem = 2 * seq_len * d_model * 4;

    println!("  Embeddings: {:.1} MB", embedding_mem as f32 / 1_048_576.0);
    println!("  Per-layer: ~{} elements ({:.1} MB)", per_layer_mem / 4, per_layer_mem as f32 / 1_048_576.0);
    println!("  KV cache: {:.1} MB", kv_cache_mem as f32 / 1_048_576.0);

    println!("\nDone! Architecture composition patterns demonstrated.");
}
