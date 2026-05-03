//! GPT-Style Transformer Decoder Example
//!
//! Demonstrates using the TransformerDecoder for autoregressive text generation,
//! similar to how GPT generates text token by token.
//!
//! Architecture:
//! - Token + Position Embeddings
//! - 12 Decoder Layers with causal (masked) self-attention
//! - Language Modeling Head
//!
//! # Example Output
//!
//! ```text
//! GPT-Style Decoder Example
//! =========================
//! Model size: ~117M parameters (GPT-small)
//! Input: "Once upon a time" → Output: "..."
//! Generation: Autoregressive token-by-token
//! ```

use mnr_core::{ForwardCtx, Mode, Trainable};
use mnr_nn::{TransformerDecoder, TransformerDecoderConfig};

fn main() {
    use mnr_ndarray_backend::CpuBackend;

    println!("GPT-Style Transformer Decoder Example");
    println!("=====================================\n");

    let backend = CpuBackend::default();

    // Configuration: GPT-small
    let config = TransformerDecoderConfig::new(768, 12, 12, 3072)
        .with_max_seq_len(1024);

    println!("Configuration:");
    println!("  d_model: {}", config.d_model);
    println!("  num_heads: {}", config.num_heads);
    println!("  num_layers: {}", config.num_layers);
    println!("  ff_dim: {}", config.ff_dim);
    println!("  max_seq_len: {}", config.max_seq_len);
    println!();

    // Create decoder with vocabulary size 50000
    let vocab_size = 50000;
    let decoder = TransformerDecoder::new(&backend, config, vocab_size, 42).unwrap();

    // Count parameters
    let num_params: usize = decoder
        .parameters()
        .iter()
        .map(|p| p.as_tensor(backend.ops()).as_ref().len())
        .sum();
    println!("Total parameters: ~{}M", num_params / 1_000_000);
    println!();

    // Example 1: Forward pass (training)
    println!("Example 1: Training Forward Pass");
    println!("--------------------------------");

    let batch_size = 2;
    let seq_len = 10;

    // Input tokens
    let input_tokens = backend
        .tensor_from_vec(
            vec![
                100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
                50, 150, 250, 350, 450, 550, 650, 750, 850, 950,
            ],
            &[batch_size, seq_len],
        )
        .unwrap();

    println!("Input tokens shape: [{}, {}]", batch_size, seq_len);

    let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
    let logits = decoder.forward(input_tokens, &mut ctx).unwrap();

    let shape = backend.ops().shape(&logits);
    println!("Output logits shape: {:?}", shape);
    println!("  [batch={}, seq_len={}, vocab_size={}]\n", shape[0], shape[1], shape[2]);

    // Example 2: Autoregressive generation
    println!("Example 2: Text Generation");
    println!("---------------------------");

    // Start with prompt
    let prompt_tokens = vec![100u32, 200, 300]; // "Once upon a"
    let mut generated = prompt_tokens.clone();

    println!("Prompt: {:?}", prompt_tokens);
    println!("Generating 5 tokens autoregressively...\n");

    for i in 0..5 {
        let prefix = backend
            .tensor_from_vec(generated.clone(), &[1, generated.len()])
            .unwrap();

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);

        // Get logits for the sequence
        let logits = decoder.forward(prefix, &mut ctx).unwrap();

        // Get last position logits (simplified greedy decode)
        let logits_data: Vec<f32> = logits.as_ref().to_vec();
        let vocab_size = 50000;
        let last_pos = (generated.len() - 1) * vocab_size;
        let last_logits = &logits_data[last_pos..last_pos + 100.min(vocab_size)]; // Top 100

        // Find max (greedy)
        let next_token = last_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0);

        println!("  Step {}: Generated token {}", i + 1, next_token);
        generated.push(next_token);
    }

    println!("\nFinal sequence length: {}", generated.len());
    println!("Full sequence: {:?}\n", generated);

    // Example 3: Model scaling
    println!("Example 3: GPT Scaling Series");
    println!("-----------------------------");

    let models = vec![
        ("GPT-2 Small", 768, 12, 12, 3072, 124_000_000usize),
        ("GPT-2 Medium", 1024, 16, 24, 4096, 350_000_000),
        ("GPT-2 Large", 1280, 20, 36, 5120, 774_000_000),
        ("GPT-2 XL", 1600, 25, 48, 6400, 1_500_000_000),
    ];

    for (name, d_model, heads, layers, ff_dim, expected_params) in models {
        let cfg = TransformerDecoderConfig::new(d_model, heads, layers, ff_dim)
            .with_max_seq_len(1024);

        let dec = TransformerDecoder::new(&backend, cfg, 50000, 42).unwrap();
        let actual_params: usize = dec
            .parameters()
            .iter()
            .map(|p| p.as_tensor(backend.ops()).as_ref().len())
            .sum();

        println!(
            "  {:12}: ~{:>5}M params (expected: ~{:>5}M)",
            name,
            actual_params / 1_000_000,
            expected_params / 1_000_000
        );
    }

    // Example 4: Temperature sampling (conceptual)
    println!("\nExample 4: Sampling Strategies");
    println!("------------------------------");
    println!("  Greedy: Always pick highest probability token");
    println!("  Temperature: Adjust logits with temperature T");
    println!("    - T < 1.0: More deterministic");
    println!("    - T > 1.0: More random/creative");
    println!("  Top-K: Sample from K most likely tokens");
    println!("  Top-P (nucleus): Sample from tokens with cumulative prob > P");

    println!("\nDone! GPT-style decoder ready for generation tasks.");
}
