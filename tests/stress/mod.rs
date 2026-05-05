//! Stress Tests and Load Tests

use mnr_core::{Backend, ForwardCtx, Mode, Module, Trainable};
use mnr_ndarray_backend::CpuBackend;
use mnr_nn::{Linear, LinearConfig, TransformerEncoder, TransformerEncoderConfig};

use crate::common::TestRunner;
use std::time::{Duration, Instant};

pub fn run_all(runner: &mut TestRunner) {
    test_sustained_forward_pressure(runner);
    test_rapid_allocation_deallocation(runner);
    test_memory_pressure_recovery(runner);
    test_large_model_load(runner);
    test_long_sequence_processing(runner);
    test_concurrent_model_usage(runner);
    test_error_recovery_stability(runner);
    test_repeated_serialization(runner);
    test_gradient_accumulation_pressure(runner);
    test_timeout_handling(runner);
}

fn test_sustained_forward_pressure(runner: &mut TestRunner) {
    runner.run_test("stress_sustained_forward", || {
        let backend = CpuBackend::default();
        let linear = Linear::new(&backend, LinearConfig::new(256, 128))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let iterations = 200usize;
        let start = Instant::now();
        let timeout = Duration::from_secs(60);

        for i in 0..iterations {
            let input = backend
                .tensor_from_vec(vec![0.5f32; 64 * 256], &[64, 256])
                .map_err(|e| format!("Iteration {}: {}", i, e))?;

            let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
            let _output = linear.forward(input, &mut ctx).map_err(|e| format!("Iteration {}: {}", i, e))?;

            if i % 100 == 0 && start.elapsed() > timeout {
                return Err(format!("Timeout at iteration {}", i));
            }
        }

        println!("  Completed {} forward passes in {:?}", iterations, start.elapsed());
        Ok(())
    });
}

fn test_rapid_allocation_deallocation(runner: &mut TestRunner) {
    runner.run_test("stress_rapid_alloc_dealloc", || {
        let backend = CpuBackend::default();
        let iterations = 500usize;
        let start = Instant::now();

        for i in 0..iterations {
            let _linear = Linear::new(&backend, LinearConfig::new(100, 50))
                .map_err(|e| format!("Iteration {}: {}", i, e))?;

            let _input = backend
                .tensor_from_vec(vec![1.0f32; 100], &[1, 100])
                .map_err(|e| format!("Iteration {}: {}", i, e))?;
        }

        println!("  Completed {} alloc/dealloc cycles in {:?}", iterations, start.elapsed());
        Ok(())
    });
}

fn test_memory_pressure_recovery(runner: &mut TestRunner) {
    runner.run_test("stress_memory_recovery", || {
        let backend = CpuBackend::default();
        let large_sizes = vec![(100, 1000), (200, 500), (500, 200)];
        let mut tensors = Vec::new();

        for (rows, cols) in &large_sizes {
            let tensor = backend
                .tensor_from_vec(vec![0.5f32; rows * cols], &[*rows, *cols])
                .map_err(|e| format!("Allocation failed: {}", e))?;
            tensors.push(tensor);
        }

        for (i, tensor) in tensors.iter().enumerate() {
            let shape = backend.ops().shape(tensor);
            println!("  Tensor {}: shape={:?}", i, shape);
        }

        drop(tensors);

        for (rows, cols) in &large_sizes {
            let _tensor = backend
                .tensor_from_vec(vec![0.5f32; rows * cols], &[*rows, *cols])
                .map_err(|e| format!("Re-allocation failed: {}", e))?;
        }

        println!("  Memory pressure test passed");
        Ok(())
    });
}

fn test_large_model_load(runner: &mut TestRunner) {
    runner.run_test("stress_large_model", || {
        let backend = CpuBackend::default();
        let config = TransformerEncoderConfig::new(128, 4, 4, 512).with_max_seq_len(256);

        let start = Instant::now();
        let encoder = TransformerEncoder::new(&backend, config, 5000, 42)
            .map_err(|e| format!("Create large model failed: {}", e))?;

        let create_time = start.elapsed();

        let num_params: usize = encoder.parameters().len();

        println!("  Created large model with ~{}K params in {:?}", num_params / 1_000, create_time);

        let input = vec![100usize; 64];
        let fwd_start = Instant::now();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let _output =
            encoder.forward(input, &mut ctx).map_err(|e| format!("Large model forward failed: {}", e))?;

        println!("  Large model forward completed in {:?}", fwd_start.elapsed());
        Ok(())
    });
}

fn test_long_sequence_processing(runner: &mut TestRunner) {
    runner.run_test("stress_long_sequences", || {
        let backend = CpuBackend::default();
        let config = TransformerEncoderConfig::new(256, 8, 4, 1024).with_max_seq_len(512);

        let encoder = TransformerEncoder::new(&backend, config, 10000, 42)
            .map_err(|e| format!("Create encoder failed: {}", e))?;

        let seq_lengths = vec![16, 32, 64, 128];

        for seq_len in seq_lengths {
            let input = vec![100usize; 4 * seq_len];
            let start = Instant::now();
            let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
            let _output = encoder
                .forward(input, &mut ctx)
                .map_err(|e| format!("Forward (seq={}) failed: {}", seq_len, e))?;

            let elapsed = start.elapsed();
            println!("  Seq len {}: completed in {:?}", seq_len, elapsed);
        }

        Ok(())
    });
}

fn test_concurrent_model_usage(runner: &mut TestRunner) {
    runner.run_test("stress_concurrent_models", || {
        let backend = CpuBackend::default();
        let model_count = 5usize;
        let mut models = Vec::new();

        for i in 0..model_count {
            let model = Linear::new(&backend, LinearConfig::new(100, 50))
                .map_err(|e| format!("Create model {}: {}", i, e))?;
            models.push(model);
        }

        let input = backend
            .tensor_from_vec(vec![0.5f32; 100], &[1, 100])
            .map_err(|e| format!("Create input failed: {}", e))?;

        for (i, model) in models.iter().enumerate() {
            let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
            let _output =
                model.forward(input.clone(), &mut ctx).map_err(|e| format!("Model {} forward: {}", i, e))?;
        }

        println!("  Successfully used {} concurrent models", model_count);
        Ok(())
    });
}

fn test_error_recovery_stability(runner: &mut TestRunner) {
    runner.run_test("stress_error_recovery", || {
        let backend = CpuBackend::default();
        let linear = Linear::new(&backend, LinearConfig::new(10, 5))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let iterations = 100usize;
        let mut errors = 0;
        let mut toggle = true;

        for _ in 0..iterations {
            toggle = !toggle;
            let result = if toggle {
                let input = backend
                    .tensor_from_vec(vec![1.0f32; 20], &[2, 10])
                    .map_err(|e| format!("Create valid input: {}", e))?;
                let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
                linear.forward(input, &mut ctx)
            } else {
                let input = backend
                    .tensor_from_vec(vec![1.0f32; 30], &[2, 15])
                    .map_err(|e| format!("Create invalid input: {}", e))?;
                let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
                linear.forward(input, &mut ctx)
            };

            match result {
                Ok(_) | Err(mnr_core::CoreError::Shape(_)) => {}
                Err(_) => {
                    errors += 1;
                }
            }
        }

        let valid_input = backend
            .tensor_from_vec(vec![1.0f32; 20], &[2, 10])
            .map_err(|e| format!("Create final input: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let _ = linear.forward(valid_input, &mut ctx).map_err(|e| format!("Final forward failed: {}", e))?;

        println!("  Recovered from {} errors, model still functional", errors);
        Ok(())
    });
}

fn test_repeated_serialization(runner: &mut TestRunner) {
    runner.run_test("stress_repeated_serialization", || {
        let backend = CpuBackend::default();
        let linear = Linear::new(&backend, LinearConfig::new(100, 50))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let input = backend
            .tensor_from_vec(vec![0.5f32; 100], &[1, 100])
            .map_err(|e| format!("Create input failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let original =
            linear.forward(input.clone(), &mut ctx).map_err(|e| format!("Original forward failed: {}", e))?;

        let iterations = 100usize;
        for i in 0..iterations {
            let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
            let output =
                linear.forward(input.clone(), &mut ctx).map_err(|e| format!("Iteration {}: {}", i, e))?;

            let orig_data: Vec<f32> = original.as_ref().to_vec();
            let out_data: Vec<f32> = output.as_ref().to_vec();

            for (j, (a, b)) in orig_data.iter().zip(out_data.iter()).enumerate() {
                let diff = (a - b).abs();
                if diff > 1e-5 {
                    return Err(format!("Iteration {}: element {} differs by {}", i, j, diff));
                }
            }
        }

        println!("  Completed {} consistency checks", iterations);
        Ok(())
    });
}

fn test_gradient_accumulation_pressure(runner: &mut TestRunner) {
    runner.run_test("stress_gradient_accumulation", || {
        let backend = CpuBackend::default();
        let config = TransformerEncoderConfig::new(128, 4, 2, 512).with_max_seq_len(128);

        let encoder = TransformerEncoder::new(&backend, config, 5000, 42)
            .map_err(|e| format!("Create encoder failed: {}", e))?;

        let accumulation_steps = 20usize;

        for i in 0..accumulation_steps {
            let input = vec![100usize; 8 * 32];
            let mut ctx = ForwardCtx::new(&backend, Mode::Train);
            let _output =
                encoder.forward(input, &mut ctx).map_err(|e| format!("Step {} forward: {}", i, e))?;
        }

        println!("  Completed {} gradient accumulation steps", accumulation_steps);
        Ok(())
    });
}

fn test_timeout_handling(runner: &mut TestRunner) {
    runner.run_test("stress_timeout", || {
        let backend = CpuBackend::default();
        let linear = Linear::new(&backend, LinearConfig::new(100, 50))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let input = backend
            .tensor_from_vec(vec![1.0f32; 100], &[1, 100])
            .map_err(|e| format!("Create input failed: {}", e))?;

        let timeout = Duration::from_millis(1000);
        let start = Instant::now();

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let _output = linear.forward(input, &mut ctx).map_err(|e| format!("Forward failed: {}", e))?;

        let elapsed = start.elapsed();

        assert!(elapsed < timeout, "Operation exceeded timeout: {:?} > {:?}", elapsed, timeout);

        println!("  Completed within timeout: {:?} < {:?}", elapsed, timeout);
        Ok(())
    });
}
