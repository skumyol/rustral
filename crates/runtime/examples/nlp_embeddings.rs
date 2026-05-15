//! Sentence Embeddings Task Example
//!
//! Demonstrates generating and comparing sentence embeddings
//! using Rustral's high-performance transformer kernels.

use rustral_core::{ForwardCtx, Mode, Module};
use rustral_ndarray_backend::CpuBackend;
use rustral_nn::transformer::{TransformerEncoder, TransformerEncoderConfig};

fn main() -> anyhow::Result<()> {
    let backend = CpuBackend::default();
    let d_model = 128;
    let config = TransformerEncoderConfig::new(d_model, 4, 2, 512);
    let encoder = TransformerEncoder::new(&backend, config, 1000, 42)?;

    println!("--- Rustral Sentence Embeddings Task Example ---");

    // Two similar sentences
    let sent1 = vec![1usize, 2, 3, 4];
    let sent2 = vec![1usize, 2, 3, 5];

    let mut ctx = ForwardCtx::new(&backend, Mode::Inference);

    // Generate embeddings
    let emb1 = encoder.forward(sent1, &mut ctx)?;
    let emb2 = encoder.forward(sent2, &mut ctx)?;

    // Pooling (mean pool across sequence)
    let ops = backend.ops();
    let pool1 = ops.mean_dim(&emb1, 1, false)?;
    let pool2 = ops.mean_dim(&emb2, 1, false)?;

    println!("Generated embedding shapes: {:?}", ops.shape(&pool1));

    // Cosine similarity would go here in a full impl
    println!("Embedding generation completed successfully.");

    Ok(())
}
