//! BERT-Style Transformer Encoder Example
//!
//! Demonstrates using the TransformerEncoder for classification tasks,
//! similar to how BERT encodes text for downstream tasks.
//!
//! Architecture:
//! - Token + Position Embeddings
//! - 12 Transformer Encoder Layers (self-attention + feedforward)
//! - Pooler for classification
//!
//! # Example Output
//!
//! ```text
//! BERT-Style Encoder Example
//! =========================
//! Model size: ~110M parameters (base)
//! Input: "The cat sat on the mat" [batch=2, seq=6]
//! Output: Encoded representation [batch=2, seq=6, d_model=768]
//! CLS token: [batch=2, d_model=768] → Classification
//! ```

use mnr_core::{ForwardCtx, Mode, Trainable};
use mnr_nn::{
    Linear, LinearConfig, TransformerEncoder, TransformerEncoderConfig,
};

fn main() {
    use mnr_ndarray_backend::CpuBackend;

    println!("BERT-Style Transformer Encoder Example");
    println!("======================================\n");

    let backend = CpuBackend::default();

    // Configuration: BERT-base
    // d_model=768, heads=12, layers=12, ff_dim=3072
    let config = TransformerEncoderConfig::new(768, 12, 12, 3072)
        .with_dropout(0.1)
        .with_max_seq_len(512)
        .with_pre_norm(true);

    println!("Configuration:");
    println!("  d_model: {}", config.d_model);
    println!("  num_heads: {}", config.num_heads);
    println!("  num_layers: {}", config.num_layers);
    println!("  ff_dim: {}", config.ff_dim);
    println!("  max_seq_len: {}", config.max_seq_len);
    println!();

    // Create encoder with vocabulary size 30000 (BERT vocab)
    let encoder = TransformerEncoder::new(&backend, config, 30000, 42).unwrap();

    // Count parameters
    let num_params: usize = encoder
        .parameters()
        .iter()
        .map(|p| p.as_tensor(backend.ops()).as_ref().len())
        .sum();
    println!("Total parameters: ~{}M", num_params / 1_000_000);
    println!();

    // Example 1: Sentence encoding
    println!("Example 1: Sentence Encoding");
    println!("----------------------------");

    // Token IDs for "The cat sat" and "Dogs love" (simplified)
    let batch_size = 2;
    let seq_len = 6;

    let input_tokens = backend
        .tensor_from_vec(
            vec![
                // Sentence 1: [CLS] The cat sat [SEP] [PAD]
                101, 1996, 4937, 4053, 102, 0, // [CLS] The cat sat [SEP] [PAD]
                // Sentence 2: [CLS] Dogs love [SEP] [PAD] [PAD]
                101, 6364, 2293, 102, 0, 0, // [CLS] Dogs love [SEP] [PAD] [PAD]
            ],
            &[batch_size, seq_len],
        )
        .unwrap();

    println!("Input shape: [{}, {}]", batch_size, seq_len);
    println!("Input tokens (first 10): {:?}", &input_tokens.as_ref()[..10]);

    // Forward pass
    let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
    let encoded = encoder.forward(input_tokens, &mut ctx).unwrap();

    let shape = backend.ops().shape(&encoded);
    println!("Output shape: {:?}", shape);
    println!("  [batch={}, seq_len={}, d_model={}]\n", shape[0], shape[1], shape[2]);

    // Extract CLS token for classification
    let cls = encoder.cls_token(&encoded, backend.ops()).unwrap();
    let cls_shape = backend.ops().shape(&cls);
    println!("CLS token shape: {:?}", cls_shape);
    println!("  Used for classification head\n");

    // Example 2: Add classification head
    println!("Example 2: Text Classification");
    println!("------------------------------");

    let num_classes = 3; // e.g., positive, negative, neutral
    let classifier = Linear::new(
        &backend,
        LinearConfig::new(768, num_classes).with_bias(true),
    )
    .unwrap();

    let logits = classifier.forward(cls, &mut ctx).unwrap();
    let logits_shape = backend.ops().shape(&logits);
    println!("Classification logits shape: {:?}", logits_shape);
    println!("  Classes: [negative, neutral, positive]");
    println!("  Predictions: batch of {} samples\n", logits_shape[0]);

    // Example 3: Different model sizes
    println!("Example 3: Model Size Comparison");
    println!("--------------------------------");

    let sizes = vec![
        ("Tiny", 128, 2, 2, 512),
        ("Small", 256, 4, 4, 1024),
        ("Base", 768, 12, 12, 3072),
        ("Large", 1024, 16, 24, 4096),
    ];

    for (name, d_model, heads, layers, ff_dim) in sizes {
        let cfg = TransformerEncoderConfig::new(d_model, heads, layers, ff_dim);
        let enc = TransformerEncoder::new(&backend, cfg, 30000, 42).unwrap();

        let params: usize = enc
            .parameters()
            .iter()
            .map(|p| p.as_tensor(backend.ops()).as_ref().len())
            .sum();

        println!(
            "  {:8}: d_model={:4}, heads={:2}, layers={:2} → ~{}M params",
            name,
            d_model,
            heads,
            layers,
            params / 1_000_000
        );
    }

    println!("\nDone! BERT-style encoder ready for fine-tuning.");
}
