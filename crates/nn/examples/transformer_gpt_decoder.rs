//! GPT-Style Transformer Decoder Example
//!
//! Demonstrates autoregressive text generation with a decoder-only transformer.
//!
//! Run with: `cargo run -p mnr-nn --example transformer_gpt_decoder`

use mnr_core::{Backend, ForwardCtx, Mode, Module};
use mnr_nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};

fn main() {
    use mnr_ndarray_backend::CpuBackend;

    println!("GPT-Style Transformer Decoder Example");
    println!("=====================================\n");

    let backend = CpuBackend::default();
    let ops = backend.ops();

    // Configuration
    let d_model = 64;
    let vocab_size = 1000;
    let seq_len = 10;

    println!("Configuration:");
    println!("  d_model: {}", d_model);
    println!("  vocab_size: {}", vocab_size);
    println!("  seq_len: {}", seq_len);
    println!();

    // Create components
    let token_embedding = Embedding::new(&backend, EmbeddingConfig::new(vocab_size, d_model), 42).unwrap();
    let projection1 = Linear::new(&backend, LinearConfig::new(d_model, d_model * 2)).unwrap();
    let projection2 = Linear::new(&backend, LinearConfig::new(d_model * 2, d_model)).unwrap();
    let lm_head = Linear::new(&backend, LinearConfig::new(d_model, vocab_size).with_bias(false)).unwrap();

    // Example 1: Token embedding
    println!("Example 1: Token Embedding");
    println!("---------------------------");

    let prompt_tokens: Vec<usize> = vec![100, 200, 300, 0, 0, 0, 0, 0, 0, 0];
    println!("Prompt tokens: {:?}", &prompt_tokens[..4]);

    let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
    let embedded = token_embedding.forward(prompt_tokens, &mut ctx).unwrap();

    let embed_shape = ops.shape(&embedded);
    println!("Embedded shape: {:?}", embed_shape);
    println!("  [seq_len={}, d_model={}]\n", embed_shape[0], embed_shape[1]);

    // Example 2: Forward pass through decoder layers
    println!("Example 2: Decoder Forward Pass");
    println!("--------------------------------");

    // Simulate transformer layers
    let hidden1 = projection1.forward(embedded, &mut ctx).unwrap();
    let activated = ops.relu(&hidden1).unwrap();
    let hidden2 = projection2.forward(activated, &mut ctx).unwrap();

    // Language modeling head
    let logits = lm_head.forward(hidden2, &mut ctx).unwrap();
    let logits_shape = ops.shape(&logits);
    println!("Logits shape: {:?}", logits_shape);
    println!("  [seq_len={}, vocab_size={}]\n", logits_shape[0], logits_shape[1]);

    // Example 3: Next token prediction
    println!("Example 3: Next Token Prediction");
    println!("---------------------------------");

    // Get logits for last position
    let last_logits = ops.slice(&logits, logits_shape[0] - 1, logits_shape[0]).unwrap();
    let last_logits_data: Vec<f32> = last_logits.as_ref().to_vec();

    // Find token with highest probability (greedy decode)
    let next_token = last_logits_data
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    println!("Predicted next token: {}", next_token);
    println!("  This would be appended to the sequence for autoregressive generation\n");

    // Example 4: Model scaling
    println!("Example 4: GPT Scaling Series");
    println!("-----------------------------");

    let models = vec![
        ("GPT-2 Small", 768, 12, 12, 124_000_000usize),
        ("GPT-2 Medium", 1024, 16, 24, 350_000_000),
        ("GPT-2 Large", 1280, 20, 36, 774_000_000),
        ("GPT-2 XL", 1600, 25, 48, 1_500_000_000),
    ];

    for (name, d_m, heads, layers, _params) in models {
        println!("  {:12}: d_model={:4}, heads={:2}, layers={:2}", name, d_m, heads, layers);
    }

    println!("\nKey concepts:");
    println!("  - Decoder-only architecture (no encoder)");
    println!("  - Causal masking prevents looking at future tokens");
    println!("  - Autoregressive: predict next token, append, repeat");
    println!("  - Used for: text generation, chatbots, code completion");

    println!("\nDone! GPT-style decoder demonstrated.");
}
