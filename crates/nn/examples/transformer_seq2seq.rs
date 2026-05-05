//! Sequence-to-Sequence Transformer Example (T5/BART-style)
//!
//! Demonstrates encoder-decoder architecture for sequence-to-sequence tasks
//! such as translation, summarization, and question answering.
//!
//! Architecture:
//! - Encoder: Processes source sequence
//! - Decoder: Generates target sequence
//!
//! Run with: `cargo run -p rustral-nn --example transformer_seq2seq`

use rustral_core::{Backend, ForwardCtx, Mode, Module};
use rustral_nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};

fn main() {
    use rustral_ndarray_backend::CpuBackend;

    println!("Sequence-to-Sequence Transformer Example");
    println!("=========================================\n");

    let backend = CpuBackend::default();
    let ops = backend.ops();

    // Configuration
    let d_model = 64;
    let vocab_size = 1000;
    let src_len = 8;
    let tgt_len = 10;

    println!("Configuration:");
    println!("  d_model: {}", d_model);
    println!("  vocab_size: {}", vocab_size);
    println!("  Source length: {}", src_len);
    println!("  Target length: {}", tgt_len);
    println!();

    // Create components
    let src_embedding = Embedding::new(&backend, EmbeddingConfig::new(vocab_size, d_model), 42).unwrap();
    let tgt_embedding = Embedding::new(&backend, EmbeddingConfig::new(vocab_size, d_model), 43).unwrap();
    let encoder_proj = Linear::new(&backend, LinearConfig::new(d_model, d_model)).unwrap();
    let decoder_proj = Linear::new(&backend, LinearConfig::new(d_model, d_model)).unwrap();
    let output_proj = Linear::new(&backend, LinearConfig::new(d_model, vocab_size).with_bias(true)).unwrap();
    let norm = LayerNorm::new(&backend, LayerNormConfig::new(vec![d_model]), 42).unwrap();

    // Example 1: Source encoding
    println!("Example 1: Source Encoding");
    println!("---------------------------");

    let src_tokens: Vec<usize> = vec![101, 50, 80, 120, 102, 0, 0, 0]; // "The cat sat on the mat"
    println!("Source tokens: {:?}", &src_tokens[..6]);

    let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
    let src_embedded = src_embedding.forward(src_tokens, &mut ctx).unwrap();
    let src_encoded = encoder_proj.forward(src_embedded, &mut ctx).unwrap();

    let src_shape = ops.shape(&src_encoded);
    println!("Source encoded shape: {:?}", src_shape);
    println!("  [src_len={}, d_model={}]\n", src_shape[0], src_shape[1]);

    // Example 2: Target generation
    println!("Example 2: Target Generation");
    println!("-----------------------------");

    let tgt_tokens: Vec<usize> = vec![2, 30, 60, 90, 3, 0, 0, 0, 0, 0]; // "Le chat etait sur le tapis"
    println!("Target tokens: {:?}", &tgt_tokens[..6]);

    let tgt_embedded = tgt_embedding.forward(tgt_tokens, &mut ctx).unwrap();
    let tgt_decoded = decoder_proj.forward(tgt_embedded, &mut ctx).unwrap();

    // In full implementation, cross-attention would attend from decoder to encoder
    // Here we just use the decoder output for demonstration
    let combined = tgt_decoded;
    let normed = norm.forward(combined, &mut ctx).unwrap();
    let logits = output_proj.forward(normed, &mut ctx).unwrap();

    let logits_shape = ops.shape(&logits);
    println!("Output logits shape: {:?}", logits_shape);
    println!("  [seq_len={}, vocab_size={}]\n", logits_shape[0], logits_shape[1]);

    // Example 3: Model comparison
    println!("Example 3: Model Comparison");
    println!("----------------------------");

    let models = vec![
        ("T5-Small", 512, 8, 6, 60_000_000usize),
        ("T5-Base", 768, 12, 12, 220_000_000),
        ("T5-Large", 1024, 16, 24, 770_000_000),
        ("BART-Base", 768, 12, 6, 140_000_000),
    ];

    for (name, d_m, heads, layers, _params) in models {
        println!("  {:12}: d_model={:4}, heads={:2}, layers={:2}", name, d_m, heads, layers);
    }

    println!("\nUse cases:");
    println!("  - Machine Translation: EN → FR, DE, ES");
    println!("  - Summarization: Long article → Short summary");
    println!("  - Question Answering: Context + Question → Answer");
    println!("  - Text-to-SQL: Natural language → SQL query");

    println!("\nDone! Seq2Seq transformer demonstrated.");
}
