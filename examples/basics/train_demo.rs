//! Training Demo
//!
//! Demonstrates a complete training loop:
//! 1. Create a simple linear model
//! 2. Forward pass with tape recording
//! 3. Compute loss
//! 4. Backward pass for gradients
//! 5. Manual SGD parameter update
//!
//! Run with: `cargo run --bin train_demo`

use mnr_autodiff::{Tape, GradExt, GradExtFromStore};
use mnr_core::{Backend, ForwardCtx, Mode, Module};
use mnr_ndarray_backend::CpuBackend;

fn main() {
    println!("Training Demo");
    println!("=============\n");

    // Initialize backend
    let backend = CpuBackend::default();
    let ops = backend.ops();

    // Create simple linear model: y = w * x + b
    // Initialize parameters
    let mut w = backend.normal_parameter("weight", &[1], 42, 0.1).unwrap();
    let mut b = backend.normal_parameter("bias", &[1], 43, 0.1).unwrap();

    println!("Initial parameters:");
    println!("  w = {:.4}", w.tensor().values()[0]);
    println!("  b = {:.4}", b.tensor().values()[0]);

    // Training data: y = 2x + 1
    let training_data = vec![
        (vec![1.0], vec![3.0]),   // y = 2*1 + 1 = 3
        (vec![2.0], vec![5.0]),   // y = 2*2 + 1 = 5
        (vec![3.0], vec![7.0]),   // y = 2*3 + 1 = 7
        (vec![4.0], vec![9.0]),   // y = 2*4 + 1 = 9
    ];

    // Training loop
    let num_epochs = 500;
    let learning_rate = 0.01;
    println!("\nTraining for {} epochs with lr={}...", num_epochs, learning_rate);

    for epoch in 0..num_epochs {
        let mut epoch_loss = 0.0;

        for (x_val, y_target) in &training_data {
            // Create forward context and tape
            let mut ctx = ForwardCtx::new(&backend, Mode::Train);
            let mut tape = Tape::new();

            // Watch parameters
            let w_id = tape.watch_parameter(&w);
            let b_id = tape.watch_parameter(&b);

            // Create input tensor
            let x_tensor = backend.tensor_from_vec(x_val.clone(), &[1]).unwrap();
            let x_id = tape.watch(x_tensor);

            // Forward: y_pred = w * x + b
            let wx = tape.mul(w_id, x_id, &mut ctx).unwrap();
            let y_pred = tape.add(wx, b_id, &mut ctx).unwrap();

            // Compute MSE loss
            let target_tensor = backend.tensor_from_vec(y_target.clone(), &[1]).unwrap();
            let target_id = tape.watch(target_tensor);

            let pred_val = tape.value(y_pred).unwrap().values()[0];
            let target_val = tape.value(target_id).unwrap().values()[0];
            let loss_val = (pred_val - target_val).powi(2);
            epoch_loss += loss_val;

            // Backward pass
            let param_map = tape.param_map().clone();
            let make_ones = |_data: Vec<f32>, shape: &[usize]| -> Result<_, _> {
                backend.tensor_from_vec(vec![1.0f32], shape)
            };

            let grads = tape.backward(y_pred, make_ones, ops).unwrap();

            // Get gradients
            let w_grad = w.gradient_from_store(&grads, &param_map)
                .expect("Missing w gradient").values()[0];
            let b_grad = b.gradient_from_store(&grads, &param_map)
                .expect("Missing b gradient").values()[0];

            // SGD update
            let w_val = w.tensor().values()[0];
            let b_val = b.tensor().values()[0];
            let new_w = backend.tensor_from_vec(vec![w_val - learning_rate * w_grad], &[1]).unwrap();
            let new_b = backend.tensor_from_vec(vec![b_val - learning_rate * b_grad], &[1]).unwrap();
            w = mnr_core::Parameter::new("weight", new_w);
            b = mnr_core::Parameter::new("bias", new_b);
        }

        if epoch % 20 == 0 {
            println!("Epoch {}: loss = {:.6}", epoch, epoch_loss / training_data.len() as f32);
        }
    }

    println!("\nFinal parameters:");
    println!("  w = {:.4} (target: 2.0)", w.tensor().values()[0]);
    println!("  b = {:.4} (target: 1.0)", b.tensor().values()[0]);

    // Test
    println!("\nTesting:");
    for (x_val, y_expected) in &training_data {
        let x = x_val[0];
        let y_pred = w.tensor().values()[0] * x + b.tensor().values()[0];
        println!("  x = {}: predicted = {:.4}, expected = {:.4}", x, y_pred, y_expected[0]);
    }

    println!("\nTraining complete!");
}
