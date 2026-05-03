//! Sequence-to-Sequence Transformer Example (T5/BART-style)
//!
//! Demonstrates using TransformerEncoderDecoder for translation and
//! other sequence-to-sequence tasks.
//!
//! Architecture:
//! - Encoder: Processes source sequence
//! - Decoder: Generates target sequence with cross-attention to encoder
//! - Shared embeddings between encoder/decoder
//!
//! # Example Output
//!
//! ```text
//! Encoder-Decoder Example
//! =======================
//! Task: English → French Translation
//! Source: "Hello, how are you?"
//! Target: "Bonjour, comment allez-vous?"
//! ```

use mnr_core::{ForwardCtx, Mode, Trainable};
use mnr_nn::{
    EncoderDecoderConfig, TransformerEncoderDecoder,
    TransformerEncoderConfig, TransformerDecoderConfig,
};

fn main() {
    use mnr_ndarray_backend::CpuBackend;

    println!("Sequence-to-Sequence Transformer (T5/BART-style)");
    println!("=================================================\n");

    let backend = CpuBackend::default();

    // Configuration
    println!("Configuration:");
    println!("  Architecture: Encoder-Decoder");
    println!("  Shared Embeddings: true");
    println!();

    // T5-small configuration
    let config = EncoderDecoderConfig::symmetric(512, 8, 6, 2048)
        .with_shared_embeddings(true);

    println!("Encoder/Decoder Configuration:");
    println!("  d_model: {}", config.encoder.d_model);
    println!("  num_heads: {}", config.encoder.num_heads);
    println!("  num_layers: {}", config.encoder.num_layers);
    println!("  ff_dim: {}", config.encoder.ff_dim);
    println!();

    // Create model
    let src_vocab = 32000; // Source vocabulary size
    let tgt_vocab = 32000; // Target vocabulary size
    let model = TransformerEncoderDecoder::new(&backend, config.clone(), src_vocab, tgt_vocab, 42).unwrap();

    // Count parameters
    let num_params: usize = model
        .parameters()
        .iter()
        .map(|p| p.as_tensor(backend.ops()).as_ref().len())
        .sum();
    println!("Total parameters: ~{}M", num_params / 1_000_000);
    println!();

    // Example 1: Training forward pass
    println!("Example 1: Training Forward Pass");
    println!("-------------------------------");

    // Source: "The cat sat on the mat" (English)
    let src_tokens = backend
        .tensor_from_vec(
            vec![101u32, 1996, 4937, 4053, 2006, 1996, 5353, 102], // BERT-style tokens
            &[1, 8],
        )
        .unwrap();

    // Target: "Le chat etait sur le tapis" (French, shifted right)
    let tgt_tokens = backend
        .tensor_from_vec(
            vec![2u32, 120, 453, 892, 234, 56, 120, 789, 3], // <s> Le chat ... </s>
            &[1, 9],
        )
        .unwrap();

    println!("Source tokens: {:?}", &src_tokens.as_ref()[..6.min(8)]);
    println!("Target tokens: {:?}", &tgt_tokens.as_ref()[..6.min(9)]);

    let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
    let logits = model.forward(src_tokens.clone(), tgt_tokens, &mut ctx).unwrap();

    let shape = backend.ops().shape(&logits);
    println!("Output logits shape: {:?}", shape);
    println!("  [batch={}, tgt_len={}, vocab_size={}]\n", shape[0], shape[1], shape[2]);

    // Example 2: Inference (greedy decoding)
    println!("Example 2: Autoregressive Translation");
    println!("------------------------------------");

    let bos_token = 2u32; // Beginning of sequence
    let eos_token = 3u32; // End of sequence
    let max_len = 20;

    println!("Source: [English sentence tokens]");
    println!("Translating (greedy decoding)...\n");

    let generated = model.generate(
        src_tokens.clone(),
        max_len,
        bos_token,
        eos_token,
        &mut ctx,
    ).unwrap();

    println!("Generated sequence: {:?}", generated);
    println!("  Length: {} tokens\n", generated.len());

    // Example 3: Different seq2seq configurations
    println!("Example 3: Seq2Seq Configurations");
    println!("----------------------------------");

    let configs = vec![
        ("T5-Small", 512, 8, 6, 60_000_000usize),
        ("T5-Base", 768, 12, 12, 220_000_000),
        ("T5-Large", 1024, 16, 24, 770_000_000),
        ("BART-Base", 768, 12, 6, 140_000_000), // Note: 6 enc + 6 dec layers
    ];

    for (name, d_model, heads, layers, expected_params) in configs {
        let cfg = EncoderDecoderConfig::symmetric(d_model, heads, layers, d_model * 4)
            .with_shared_embeddings(true);

        let m = TransformerEncoderDecoder::new(&backend, cfg, 32000, 32000, 42).unwrap();
        let params: usize = m
            .parameters()
            .iter()
            .map(|p| p.as_tensor(backend.ops()).as_ref().len())
            .sum();

        println!(
            "  {:12}: d_model={:4}, heads={:2}, layers={:2} → ~{:>4}M params (expected: ~{}M)",
            name,
            d_model,
            heads,
            layers,
            params / 1_000_000,
            expected_params / 1_000_000
        );
    }

    // Example 4: Task comparison
    println!("\nExample 4: Seq2Seq Task Examples");
    println!("---------------------------------");
    println!("  Machine Translation:");
    println!("    EN: 'Hello, world!' → FR: 'Bonjour le monde !'");
    println!();
    println!("  Summarization:");
    println!("    Input: [Long article text...]");
    println!("    Output: 'Key point summary'");
    println!();
    println!("  Question Answering:");
    println!("    Context: 'Paris is the capital of France.'");
    println!("    Question: 'What is the capital of France?'");
    println!("    Answer: 'Paris'");
    println!();
    println!("  Text-to-SQL:");
    println!("    NL: 'Show me all users over 18'");
    println!("    SQL: 'SELECT * FROM users WHERE age > 18'");

    // Example 5: Architecture comparison
    println!("\nExample 5: Encoder-Decoder vs Decoder-Only");
    println!("------------------------------------------");
    println!("  Encoder-Decoder (T5/BART):");
    println!("    - Pros: Explicit separation of input/output processing");
    println!("    - Pros: Bidirectional encoder sees full context");
    println!("    - Cons: Slower (must process both encoder and decoder)");
    println!("    - Best for: Translation, summarization, structured tasks");
    println!();
    println!("  Decoder-Only (GPT):");
    println!("    - Pros: Simpler architecture");
    println!("    - Pros: Better for open-ended generation");
    println!("    - Cons: Can't see future input tokens");
    println!("    - Best for: Chat, completion, creative writing");

    println!("\nDone! Seq2Seq transformer ready for translation/generation tasks.");
}
