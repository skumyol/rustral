//! ResNet Image Classification Example
//!
//! Demonstrates residual connections in deep neural networks.
//!
//! Run with: `cargo run --example resnet_image_classification`

use rustral_core::{Backend, ForwardCtx, Mode, Module};
use rustral_ndarray_backend::CpuBackend;
use rustral_nn::{Linear, LinearConfig, BatchNorm, BatchNormConfig};

fn main() {
    println!("ResNet Image Classification Example");
    println!("==================================\n");

    let backend = CpuBackend::default();
    let ops = backend.ops();

    // Demonstrate residual connection concept
    println!("Residual Connection Demonstration:");
    println!("----------------------------------");

    // Create a simple residual block
    let linear = Linear::new(&backend, LinearConfig::new(10, 10)).unwrap();

    // Input
    let input = backend.tensor_from_vec(vec![1.0f32; 10], &[1, 10]).unwrap();
    println!("Input shape: {:?}", ops.shape(&input));

    // Forward through layer
    let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
    let output = linear.forward(input.clone(), &mut ctx).unwrap();

    // Residual: output = input + layer(input)
    let residual = ops.add(&input, &output).unwrap();
    println!("Residual output shape: {:?}", ops.shape(&residual));

    println!("\nKey concept:");
    println!("  Residual: output = input + F(input)");
    println!("  This allows gradients to flow directly through");
    println!("  the network, enabling training of very deep models.");
}
