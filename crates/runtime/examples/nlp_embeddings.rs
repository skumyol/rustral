//! NLP Embeddings Task Example
//!
//! Demonstrates generating embeddings for a document using a Transformer encoder.

use std::time::Instant;
use rustral_core::{ForwardCtx, Mode, Backend};
use rustral_ndarray_backend::CpuBackend;
use rustral_nn::{TransformerEncoder, TransformerEncoderConfig};
use rustral_symbolic::{Document, Sentence, Span, Token};

fn main() -> anyhow::Result<()> {
    let backend = CpuBackend::default();

    // Configure a small transformer for the example
    let d_model = 128;
    let config = TransformerEncoderConfig::new(d_model, 4, 2, 512);
    let encoder = TransformerEncoder::new(&backend, config, 1000, 42)?;

    let mut ctx = ForwardCtx::new(&backend, Mode::Inference);

    let doc = Document {
        text: "The model learns quickly.".to_string(),
        sentences: vec![
            Sentence {
                tokens: vec![
                    Token { text: "The".to_string(), id: 0, span: Span::new(0, 3), pos: None },
                    Token { text: "model".to_string(), id: 1, span: Span::new(4, 9), pos: None },
                    Token { text: "learns".to_string(), id: 2, span: Span::new(10, 16), pos: None },
                    Token { text: "quickly".to_string(), id: 3, span: Span::new(17, 24), pos: None },
                    Token { text: ".".to_string(), id: 4, span: Span::new(24, 25), pos: None },
                ],
                dependency_graph: None,
            }
        ],
        entities: vec![],
    };

    println!("Generating embeddings for: '{}'", doc.text);

    // In a real scenario, we'd use a tokenizer. Here we mock input IDs.
    let input_ids = vec![1usize, 12, 45, 7, 2];

    let start = Instant::now();
    let embeddings = encoder.forward(input_ids, &mut ctx)?;
    let duration = start.elapsed();

    let ops = backend.ops();
    let shape = ops.shape(&embeddings);

    println!("Embedding generation took {:?}", duration);
    println!("Output tensor shape: {:?}", shape);

    Ok(())
}
