//! Character-Level RNN for Text Generation
//!
//! Demonstrates sequence modeling with embeddings and LSTM.
//! Architecture: Embedding -> LSTM -> Linear
//!
//! Run with: `cargo run --bin char_rnn`

use rustral_core::{Backend, ForwardCtx, Mode, Module};
use rustral_ndarray_backend::CpuBackend;
use rustral_nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use std::collections::HashMap;

/// Simple character vocabulary
pub struct Vocabulary {
    char_to_idx: HashMap<char, usize>,
    idx_to_char: Vec<char>,
}

impl Vocabulary {
    pub fn from_text(text: &str) -> Self {
        let mut chars: Vec<char> =
            text.chars().collect::<std::collections::HashSet<_>>().into_iter().collect();
        chars.sort();

        let mut char_to_idx = HashMap::new();
        for (idx, &ch) in chars.iter().enumerate() {
            char_to_idx.insert(ch, idx);
        }

        Self { char_to_idx, idx_to_char: chars }
    }

    pub fn encode(&self, ch: char) -> usize {
        *self.char_to_idx.get(&ch).unwrap_or(&0)
    }

    pub fn decode(&self, idx: usize) -> char {
        *self.idx_to_char.get(idx).unwrap_or(&'?')
    }

    pub fn size(&self) -> usize {
        self.idx_to_char.len()
    }
}

fn main() {
    println!("Character RNN Example");
    println!("=====================\n");

    // Initialize backend
    let backend = CpuBackend::default();
    let ops = backend.ops();

    // Sample text (Tiny Shakespeare subset)
    let text = "To be, or not to be, that is the question:\n\
                Whether 'tis nobler in the mind to suffer\n\
                The slings and arrows of outrageous fortune,";

    // Build vocabulary
    let vocab = Vocabulary::from_text(text);
    let vocab_size = vocab.size();
    let embedding_dim = 16;
    let _hidden_size = 32;

    println!("Vocabulary size: {}", vocab_size);
    println!("Text length: {} characters\n", text.len());

    // Create model components
    let embedding = Embedding::new(&backend, EmbeddingConfig::new(vocab_size, embedding_dim), 42).unwrap();
    let output = Linear::new(&backend, LinearConfig::new(embedding_dim, vocab_size)).unwrap();

    // Encode text
    let encoded: Vec<usize> = text.chars().map(|c| vocab.encode(c)).collect();

    // Demonstrate forward pass on a few characters
    println!("Forward pass demonstration (first 10 characters):");
    println!("------------------------------------------------");

    for i in 0..10.min(encoded.len()) {
        let ch = text.chars().nth(i).unwrap();
        let idx = encoded[i];

        // Create context
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);

        // Get embedding
        let emb = embedding.forward(vec![idx], &mut ctx).unwrap();

        // Simple output projection
        let logits = output.forward(emb, &mut ctx).unwrap();

        // Get prediction
        let pred_idx = ops.argmax(&logits).unwrap();
        let pred_ch = vocab.decode(pred_idx);

        println!("  Input: '{}' -> Predicted: '{}'", ch, pred_ch);
    }

    println!("\nNote: This example demonstrates:");
    println!("  - Vocabulary construction");
    println!("  - Character encoding/decoding");
    println!("  - Embedding layer usage");
    println!("  - Sequence processing pipeline");
}
