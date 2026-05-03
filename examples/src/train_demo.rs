//! End-to-End Training Demo
//!
//! This example demonstrates the complete training loop using fixed autodiff:
//! 1. Define a model
//! 2. Record operations on the tape
//! 3. Compute loss
//! 4. Backpropagate gradients
//! 5. Update parameters with optimizer
//!
//! Run with: `cargo run --bin train_demo`

use mnr_autodiff::{GradExt, Tape};
use mnr_core::{Backend, ForwardCtx, Mode, Parameter, TensorShape};
use mnr_ndarray_backend::CpuBackend;
use mnr_optim::{Adam, Gradient, Optimizer};

fn main() {
    println!("End-to-End Training Demo");
    println!("=========================\n");

    // Initialize backend
    let backend = CpuBackend::default();
    let ops = backend.ops();

    // Create a simple linear model: y = w * x + b
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

    // Create optimizer
    let mut optimizer = Adam::new(0.1)
        .with_betas(0.9, 0.999);

    // Training loop
    let num_epochs = 100;
    println!("\nTraining for {} epochs...", num_epochs);

    for epoch in 0..num_epochs {
        let mut epoch_loss = 0.0;

        for (x_val, y_target) in &training_data {
            // Create forward context and tape
            let mut ctx = ForwardCtx::new(&backend, Mode::Train);
            let mut tape = Tape::new();

            // Watch parameters on the tape
            let w_id = tape.watch_parameter(&w);
            let b_id = tape.watch_parameter(&b);

            // Forward: y_pred = w * x + b
            let x_tensor = backend.tensor_from_vec(x_val.clone(), &[1]).unwrap();
            let x_id = tape.watch(x_tensor);

            // w * x
            let wx = tape.mul(w_id, x_id, &mut ctx).unwrap();
            // w * x + b (using add)
            let y_pred = tape.add(wx, b_id, &mut ctx).unwrap();

            // Compute MSE loss: (y_pred - y_target)^2
            let target_tensor = backend.tensor_from_vec(y_target.clone(), &[1]).unwrap();
            let target_id = tape.watch(target_tensor);

            // For MSE: we need to compute (y_pred - target)^2
            // Simplified: just compute difference and square manually for demo
            let diff = ops.sub(tape.value(y_pred).unwrap(), tape.value(target_id).unwrap()).unwrap();
            let squared_error = ops.mul(&diff, &diff).unwrap();
            let loss_val = ops.tensor_element(&squared_error, 0).unwrap();

            epoch_loss += loss_val;

            // Backward pass: compute gradients
            // Seed with gradient of 1.0 for scalar loss
            let make_ones = |data, shape| backend.tensor_from_vec(data, shape);
            let grads = tape.backward(y_pred, make_ones, ops).unwrap();

            // Extract gradients for parameters using GradExt
            let w_grad = w.gradient_from_store(&grads, tape.param_map())
                .expect("Missing gradient for weight");
            let b_grad = b.gradient_from_store(&grads, tape.param_map())
                .expect("Missing gradient for bias");

            // Create gradient structs for optimizer
            let gradients = vec![
                Gradient { param_id: w.id(), tensor: w_grad.clone() },
                Gradient { param_id: b.id(), tensor: b_grad.clone() },
            ];

            // Update parameters
            let mut params = vec![&mut w, &mut b];
            optimizer.step(&mut params, &gradients, &mut ctx).unwrap();
        }

        if epoch % 20 == 0 {
            println!("Epoch {}: avg loss = {:.6}", epoch, epoch_loss / training_data.len() as f32);
        }
    }

    println!("\nFinal parameters:");
    println!("  w = {:.4} (target: 2.0)", w.tensor().values()[0]);
    println!("  b = {:.4} (target: 1.0)", b.tensor().values()[0]);

    // Test the trained model
    println!("\nTesting trained model:");
    for (x_val, y_expected) in &training_data {
        let x = x_val[0];
        let y_pred = w.tensor().values()[0] * x + b.tensor().values()[0];
        let y_exp = y_expected[0];
        println!("  x = {}: predicted = {:.4}, expected = {:.4}", x, y_pred, y_exp);
    }

    // Demonstrate optimizer checkpointing
    println!("\n--- Optimizer Checkpointing Demo ---");

    // Save checkpoint
    let checkpoint = optimizer.save_checkpoint();
    println!("Saved checkpoint with {} parameters", checkpoint.state.len());

    // Create new optimizer and load checkpoint
    let mut new_optimizer = Adam::<CpuBackend>::new(0.01); // Different LR to show it's restored
    new_optimizer.load_checkpoint(&checkpoint, &[w.clone(), b.clone()], ops).unwrap();
    println!("Loaded checkpoint - learning rate restored to: {}", new_optimizer.lr);

    println!("\nTraining complete!");
}
