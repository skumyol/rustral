//! GPT Training Example
//!
//! Demonstrates causal (autoregressive) transformer decoder.
//!
//! Run with: `cargo run --example gpt_training`

use mnr_core::{Backend, ForwardCtx, Mode, Module};
use mnr_ndarray_backend::CpuBackend;
use mnr_nn::{Linear, LinearConfig, Embedding, EmbeddingConfig};

fn main() {
    println!("GPT Training Example");
    println!("====================\n");

    let backend = CpuBackend::default();
    let ops = backend.ops();

    // Vocabulary and model config
    let vocab_size = 1000;
    let embedding_dim = 64;
    let seq_len = 8;

    println!("Model configuration:");
    println!("  Vocab size: {}", vocab_size);
    println!("  Embedding dim: {}", embedding_dim);
    println!("  Sequence length: {}\n", seq_len);

    // Create embedding layer
    let embedding = Embedding::new(&backend, EmbeddingConfig::new(vocab_size, embedding_dim)).unwrap();

    // Simulate token IDs
    let token_ids: Vec<usize> = (0..seq_len).collect();

    // Look up embeddings
    let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
    let embedded = embedding.forward(&token_ids, ops).unwrap();
    println!("Embedded tokens shape: {:?}", ops.shape(&embedded));

    // Language modeling head
    let lm_head = Linear::new(&backend, LinearConfig::new(embedding_dim, vocab_size)).unwrap();
    let reshaped = ops.reshape(&embedded, &[seq_len, embedding_dim]).unwrap();
    let logits = lm_head.forward(reshaped, &mut ctx).unwrap();

    println!("LM logits shape: {:?}", ops.shape(&logits));

    // Get next token prediction
    let last_token_logits = ops.reshape(&logits, &[seq_len, vocab_size]).unwrap();
    let last_logit_idx = ops.argmax(&last_token_logits).unwrap();
    println!("Predicted next token ID: {}\n", last_logit_idx);

    println!("Key concepts:");
    println!("  - GPT is a decoder-only transformer");
    println!("  - Causal masking prevents looking at future tokens");
    println!("  - Next-token prediction is the training objective");
    println!("  - Generation: predict next token, append, repeat");
}
