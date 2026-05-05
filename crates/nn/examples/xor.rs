//! XOR Classification Example
//!
//! Demonstrates a simple neural network forward pass.
//! Note: This example focuses on the inference pipeline.
//! Training with full autodiff is shown in train_demo.rs.

use rustral_core::{Backend, ForwardCtx, Mode, Module};
use rustral_ndarray_backend::CpuBackend;
use rustral_nn::{Linear, LinearConfig};

/// XOR dataset: inputs and expected outputs
fn xor_dataset() -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = vec![
        vec![0.0],  // 0 XOR 0 = 0
        vec![1.0],  // 0 XOR 1 = 1
        vec![1.0],  // 1 XOR 0 = 1
        vec![0.0],  // 1 XOR 1 = 0
    ];
    (inputs, targets)
}

fn main() {
    println!("XOR Classification Example");
    println!("==========================\n");

    // Initialize backend
    let backend = CpuBackend::default();
    let ops = backend.ops();

    // Create a simple network: 2 inputs -> 2 hidden (ReLU) -> 1 output (Sigmoid)
    let hidden = Linear::new(&backend, LinearConfig::new(2, 2))
        .expect("Failed to create hidden layer");
    let output = Linear::new(&backend, LinearConfig::new(2, 1))
        .expect("Failed to create output layer");

    println!("Model architecture: 2 -> 2 -> 1");
    println!("Hidden layer: 2 -> 2, bias: true");
    println!("Output layer: 2 -> 1, bias: true\n");

    // Get dataset
    let (inputs, targets) = xor_dataset();

    // Demonstrate forward pass on all samples
    println!("Forward pass demonstration:");
    println!("---------------------------");

    for (i, (input_vec, target_vec)) in inputs.iter().zip(targets.iter()).enumerate() {
        // Create forward context
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);

        // Create input tensor
        let input_tensor = backend.tensor_from_vec(input_vec.clone(), &[1, 2]).unwrap();

        // Forward pass: hidden layer with ReLU
        let hidden_out = hidden.forward(input_tensor.clone(), &mut ctx).unwrap();
        let hidden_activated = ops.relu(&hidden_out).unwrap();

        // Forward pass: output layer with Sigmoid
        let output_out = output.forward(hidden_activated.clone(), &mut ctx).unwrap();
        let prediction = ops.sigmoid(&output_out).unwrap();

        let pred_val = ops.tensor_element(&prediction, 0).unwrap();
        let target_val = target_vec[0];

        println!(
            "Sample {}: Input [{:.0}, {:.0}] -> Output {:.4} (Target: {:.0})",
            i + 1,
            input_vec[0], input_vec[1],
            pred_val, target_val
        );
    }

    println!("\nNote: This example demonstrates forward pass only.");
    println!("For training, see train_demo.rs which shows gradient computation.");
    println!("\nKey concepts demonstrated:");
    println!("  - Backend initialization");
    println!("  - Linear layer creation");
    println!("  - Forward context (Mode::Inference)");
    println!("  - Activation functions (ReLU, Sigmoid)");
    println!("  - Tensor operations");
}
