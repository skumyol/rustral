//! MNIST Handwritten Digit Classification Example
//!
//! Demonstrates a simple neural network for image classification.
//! Architecture: Linear -> ReLU -> Linear (simplified for demonstration)
//!
//! Run with: `cargo run --bin mnist`

use mnr_core::{Backend, ForwardCtx, Mode, Module};
use mnr_ndarray_backend::CpuBackend;
use mnr_nn::{Linear, LinearConfig};

/// Simple MNIST sample
#[derive(Clone)]
pub struct MnistSample {
    pub image: Vec<f32>,  // 784 elements (28x28)
    pub label: usize,     // 0-9
}

fn main() {
    println!("MNIST Digit Classification Example");
    println!("=================================\n");

    // Initialize backend
    let backend = CpuBackend::default();
    let ops = backend.ops();

    // Create a simple neural network
    // Input: 784 pixels -> Hidden: 128 -> Output: 10 classes
    let hidden = Linear::new(&backend, LinearConfig::new(784, 128)).unwrap();
    let output = Linear::new(&backend, LinearConfig::new(128, 10)).unwrap();

    println!("Model architecture:");
    println!("  Input: 784 pixels (28x28 flattened)");
    println!("  Linear(784 -> 128)");
    println!("  ReLU activation");
    println!("  Linear(128 -> 10)");
    println!("  Output: 10 class logits\n");

    // Create synthetic data for demonstration
    println!("Generating synthetic MNIST-like samples...");
    let num_samples = 4;
    let samples: Vec<MnistSample> = (0..num_samples)
        .map(|i| MnistSample {
            image: vec![0.5f32; 784], // Flat gray image
            label: i % 10,
        })
        .collect();

    println!("Testing forward pass on {} samples:\n", num_samples);

    // Test forward pass on each sample
    for (idx, sample) in samples.iter().enumerate() {
        // Create context
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);

        // Create input tensor [1, 784]
        let input = backend.tensor_from_vec(
            sample.image.clone(),
            &[1, 784]
        ).unwrap();

        // Hidden layer + ReLU
        let hidden_out = hidden.forward(input, &mut ctx).unwrap();
        let activated = ops.relu(&hidden_out).unwrap();

        // Output layer
        let logits = output.forward(activated, &mut ctx).unwrap();

        // Get prediction
        let pred_class = ops.argmax(&logits).unwrap();
        let true_class = sample.label;

        println!("  Sample {}: True = {}, Predicted = {}", idx + 1, true_class, pred_class);
    }

    println!("\nKey concepts:");
    println!("  - Flattening 2D images to 1D vectors");
    println!("  - Hidden layer learns features");
    println!("  - ReLU introduces non-linearity");
    println!("  - Output layer produces class scores");
    println!("  - Argmax selects predicted class");
}
