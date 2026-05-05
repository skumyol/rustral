//! BERT Fine-tuning Example
//!
//! Demonstrates transformer encoder architecture for NLP.
//!
//! Run with: `cargo run --example bert_fine_tuning`

use rustral_core::{Backend, ForwardCtx, Mode, Module};
use rustral_ndarray_backend::CpuBackend;
use rustral_nn::{Linear, LinearConfig, TransformerEncoder, TransformerEncoderConfig};

fn main() {
    println!("BERT Fine-tuning Example");
    println!("=======================\n");

    let backend = CpuBackend::default();
    let ops = backend.ops();

    // Create a transformer encoder
    let config = TransformerEncoderConfig::new(64, 2, 128, 4);
    let encoder = TransformerEncoder::new(&backend, config).unwrap();

    println!("Model configuration:");
    println!("  Embedding dim: 64");
    println!("  Num layers: 2");
    println!("  Hidden dim: 128");
    println!("  Num heads: 4\n");

    // Simulate token embeddings [batch=1, seq_len=4, dim=64]
    let tokens = backend.tensor_from_vec(vec![0.1f32; 256], &[1, 4, 64]).unwrap();
    println!("Input tokens shape: {:?}", ops.shape(&tokens));

    // Forward pass
    let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
    let output = encoder.forward(tokens, &mut ctx).unwrap();

    println!("Encoder output shape: {:?}", ops.shape(&output));

    // Classification head
    let classifier = Linear::new(&backend, LinearConfig::new(64, 2)).unwrap();
    let reshaped = ops.reshape(&output, &[4, 64]).unwrap();
    let logits = classifier.forward(reshaped, &mut ctx).unwrap();
    println!("Classification logits shape: {:?}", ops.shape(&logits));

    println!("\nKey concepts:");
    println!("  - Transformer encoder processes all tokens in parallel");
    println!("  - Self-attention captures contextual relationships");
    println!("  - Classification head maps to task-specific outputs");
}
