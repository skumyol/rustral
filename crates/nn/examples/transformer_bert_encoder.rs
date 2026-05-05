//! BERT-Style Transformer Encoder Example
//!
//! Demonstrates transformer encoder architecture concepts for classification tasks,
//! similar to how BERT encodes text for downstream tasks.
//!
//! Architecture:
//! - Token Embeddings
//! - Linear Projections (simulating attention/feedforward)
//! - Classification Head
//!
//! # Example Output
//!
//! ```text
//! BERT-Style Encoder Example
//! =========================
//! Configuration: d_model=64, vocab_size=1000
//! Input: Token IDs [101, 50, 80, 120, 102, 0]
//! Output: Encoded representation [6, 64]
//! CLS token → Linear(64, 3) → Classification logits
//! ```

use rustral_core::{Backend, ForwardCtx, Mode, Module};
use rustral_nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};

fn main() {
    use rustral_ndarray_backend::CpuBackend;

    println!("BERT-Style Transformer Encoder Example");
    println!("======================================\n");

    let backend = CpuBackend::default();
    let ops = backend.ops();

    // Configuration
    let d_model = 64;
    let seq_len = 6;
    let vocab_size = 1000;

    println!("Configuration:");
    println!("  d_model: {}", d_model);
    println!("  seq_len: {}", seq_len);
    println!("  vocab_size: {}", vocab_size);
    println!();

    // Create components
    let token_embedding = Embedding::new(&backend, EmbeddingConfig::new(vocab_size, d_model), 42).unwrap();

    // Simulate transformer layers with linear projections
    let projection = Linear::new(&backend, LinearConfig::new(d_model, d_model)).unwrap();
    let feedforward = Linear::new(&backend, LinearConfig::new(d_model, d_model * 4).with_bias(true)).unwrap();
    let output_proj = Linear::new(&backend, LinearConfig::new(d_model * 4, d_model).with_bias(true)).unwrap();
    let norm1 = LayerNorm::new(&backend, LayerNormConfig::new(vec![d_model]), 42).unwrap();
    let norm2 = LayerNorm::new(&backend, LayerNormConfig::new(vec![d_model]), 43).unwrap();

    // Example 1: Token embedding
    println!("Example 1: Token Embedding");
    println!("----------------------------");

    let token_ids: Vec<usize> = vec![101, 50, 80, 120, 102, 0];
    println!("Input tokens: {:?}", token_ids);

    let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
    let embedded = token_embedding.forward(token_ids, &mut ctx).unwrap();

    let embed_shape = ops.shape(&embedded);
    println!("Embedded shape: {:?}", embed_shape);
    println!("  Each token → {}-dimensional vector\n", d_model);

    // Example 2: Transformer-style processing (simplified)
    println!("Example 2: Transformer Processing");
    println!("--------------------------------");

    // Simulate: Norm → Linear → Residual → Norm → FF → Residual
    let normed1 = norm1.forward(embedded.clone(), &mut ctx).unwrap();
    let projected = projection.forward(normed1, &mut ctx).unwrap();
    let residual1 = ops.add(&embedded, &projected).unwrap();

    let normed2 = norm2.forward(residual1.clone(), &mut ctx).unwrap();
    let ff_out = feedforward.forward(normed2, &mut ctx).unwrap();
    // Apply activation (simplified as relu)
    let activated = ops.relu(&ff_out).unwrap();
    let ff_proj = output_proj.forward(activated, &mut ctx).unwrap();
    let encoded = ops.add(&residual1, &ff_proj).unwrap();

    let shape = ops.shape(&encoded);
    println!("Encoded shape: {:?}", shape);
    println!("  [seq_len={}, d_model={}]\n", shape[0], shape[1]);

    // Example 3: Classification
    println!("Example 3: Classification");
    println!("---------------------------");

    // Extract first token (CLS) - take first row
    let cls = ops.slice(&encoded, 0, 1).unwrap();
    let cls_shape = ops.shape(&cls);
    println!("CLS token shape: {:?}", cls_shape);

    let num_classes = 3;
    let classifier = Linear::new(&backend, LinearConfig::new(d_model, num_classes).with_bias(true)).unwrap();

    let logits = classifier.forward(cls, &mut ctx).unwrap();
    let logits_shape = ops.shape(&logits);
    println!("Classification logits shape: {:?}", logits_shape);
    println!("  Classes: [negative, neutral, positive]\n");

    // Example 4: Model scaling
    println!("Example 4: BERT Model Sizes");
    println!("---------------------------");

    let sizes = vec![
        ("Tiny", 128, 2, 2, 512),
        ("Small", 256, 4, 4, 1024),
        ("Base", 768, 12, 12, 3072),
        ("Large", 1024, 16, 24, 4096),
    ];

    for (name, d_m, heads, layers, ff) in sizes {
        println!("  {:8}: d_model={:4}, heads={:2}, layers={:2}, ff_dim={}", name, d_m, heads, layers, ff);
    }

    println!("\nKey concepts:");
    println!("  - Token embeddings convert token IDs to dense vectors");
    println!("  - LayerNorm stabilizes training");
    println!("  - Residual connections help gradient flow");
    println!("  - Feedforward layers increase model capacity");
    println!("  - [CLS] token represents the entire sequence for classification");

    println!("\nDone! BERT-style transformer demonstrated.");
}
