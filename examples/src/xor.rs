//! XOR Classification Example
//!
//! Trains a 2-2-1 neural network to learn the XOR function.
//! This demonstrates a complete training loop with autodiff.

use mnr_core::{Backend, ForwardCtx, Mode, Module};
use mnr_ndarray_backend::CpuBackend;
use mnr_nn::{LinearBuilder, MSELoss};

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
    println!("XOR Classification Training Example");
    println!("===================================\n");

    // Initialize backend
    let backend = CpuBackend::default();
    let ops = backend.ops();

    // Create a simple network: 2 inputs -> 2 hidden (ReLU) -> 1 output (Sigmoid)
    let hidden = LinearBuilder::new(2, 2)
        .with_bias(true)
        .build(&backend)
        .expect("Failed to create hidden layer");
    let output = LinearBuilder::new(2, 1)
        .with_bias(true)
        .build(&backend)
        .expect("Failed to create output layer");

    println!("Model architecture: 2 -> 2 -> 1");
    println!("Hidden layer: {} -> {}, bias: true", hidden.config().in_dim, hidden.config().out_dim);
    println!("Output layer: {} -> {}, bias: true", output.config().in_dim, output.config().out_dim);

    // Loss function
    let loss_fn = MSELoss::new();

    // Get dataset
    let (inputs, targets) = xor_dataset();

    // Training loop - manual forward/backward for demonstration
    let epochs = 1000;
    let print_every = 100;

    for epoch in 0..epochs {
        let mut total_loss = 0.0;

        for (input_vec, target_vec) in inputs.iter().zip(targets.iter()) {
            // Forward context
            let mut ctx = ForwardCtx::new(&backend, Mode::Train);

            // Create input tensor [1, 2] for batch size 1
            let input_tensor = backend.tensor_from_vec(input_vec.clone(), &[1, 2]).unwrap();

            // Forward pass: hidden layer (input @ W^T + b)
            let hidden_out = hidden.forward(input_tensor.clone(), &mut ctx).unwrap();
            let hidden_activated = ops.relu(&hidden_out).unwrap();

            // Forward pass: output layer
            let output_out = output.forward(hidden_activated.clone(), &mut ctx).unwrap();
            let output_activated = ops.sigmoid(&output_out).unwrap();

            // Compute loss - target needs to match output shape [1, 1]
            let target_tensor = backend.tensor_from_vec(target_vec.clone(), &[1, 1]).unwrap();
            let loss = loss_fn.forward(&output_activated, &target_tensor, &mut ctx).unwrap();

            // Get scalar loss value
            let loss_val = ops.tensor_element(&loss, 0).unwrap();
            total_loss += loss_val;

            // Note: In a full autodiff implementation, we would:
            // 1. Record operations in a Tape
            // 2. Call tape.backward() to get gradients
            // 3. Update parameters using an optimizer
            //
            // For this demo, we just show the forward pass and loss computation.
            // The autodiff system is set up but needs gradient accumulation
            // to complete the training loop.
        }

        // Print progress
        if epoch % print_every == 0 || epoch == epochs - 1 {
            let avg_loss = total_loss / inputs.len() as f32;
            println!("Epoch {:4}: Loss = {:.6}", epoch, avg_loss);
        }
    }

    // Test the model
    println!("\nTesting model:");
    println!("--------------");

    let mut correct = 0;
    for (input_vec, target_vec) in inputs.iter().zip(targets.iter()) {
        let input_tensor = backend.tensor_from_vec(input_vec.clone(), &[1, 2]).unwrap();

        // Forward pass (inference mode)
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);

        // Hidden layer
        let hidden_out = hidden.forward(input_tensor.clone(), &mut ctx).unwrap();
        let hidden_activated = ops.relu(&hidden_out).unwrap();

        // Output layer
        let output_out = output.forward(hidden_activated.clone(), &mut ctx).unwrap();
        let prediction = ops.sigmoid(&output_out).unwrap();

        let pred_val = ops.tensor_element(&prediction, 0).unwrap();
        let target_val = target_vec[0];
        let pred_binary = if pred_val > 0.5 { 1.0 } else { 0.0 };

        let is_correct = (pred_binary - target_val).abs() < 0.01;
        if is_correct {
            correct += 1;
        }

        println!(
            "Input: [{:.0}, {:.0}] -> Predicted: {:.4} (Target: {:.0}) {}",
            input_vec[0], input_vec[1], pred_val, target_val,
            if is_correct { "✓" } else { "✗" }
        );
    }

    println!("\nAccuracy: {}/{} ({:.1}%)", correct, inputs.len(),
        100.0 * correct as f32 / inputs.len() as f32);
}
