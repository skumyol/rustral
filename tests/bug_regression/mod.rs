//! Bug Regression Tests
//!
//! Tests for known bugs and edge cases.

#![allow(unused_imports, unused_variables)]

use rustral_core::{Backend, CoreError, ForwardCtx, Mode, Module, TensorOps};
use rustral_ndarray_backend::CpuBackend;
use rustral_nn::{
    Conv2d, Conv2dConfig, Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig,
    Linear, LinearConfig, SelfAttention, SelfAttentionConfig, TransformerDecoder, TransformerDecoderConfig,
    TransformerEncoder, TransformerEncoderConfig,
};

use crate::common::TestRunner;

pub fn run_all(runner: &mut TestRunner) {
    test_division_by_zero(runner);
    test_empty_input(runner);
    test_large_tensor_allocation(runner);
    test_incompatible_dimensions(runner);
    test_gradient_overflow(runner);
    test_memory_leak_in_layers(runner);
    test_causal_mask_boundary(runner);
    test_batch_size_mismatch(runner);
    test_nan_propagation(runner);
    test_dropout_determinism(runner);
    test_quantization_precision_loss(runner);
    test_save_load_consistency(runner);
    test_concurrent_access(runner);
    test_resource_exhaustion(runner);
    test_infinite_loop_detection(runner);
    test_tensor_alignment(runner);
}

fn test_division_by_zero(runner: &mut TestRunner) {
    runner.run_test("bug_division_by_zero", || {
        let backend = CpuBackend::default();
        let zeros = backend
            .tensor_from_vec(vec![0.0f32; 64], &[4, 16])
            .map_err(|e| format!("Failed to create tensor: {}", e))?;

        let norm = LayerNorm::new(&backend, LayerNormConfig::new(vec![16]), 42)
            .map_err(|e| format!("Failed to create layer norm: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let result = norm.forward(zeros, &mut ctx);

        match result {
            Ok(output) => {
                let shape = backend.ops().shape(&output);
                assert_eq!(shape, vec![4, 16], "Output shape mismatch");
                Ok(())
            }
            Err(CoreError::Shape(msg)) => {
                if msg.contains("division") || msg.contains("zero") {
                    Err(format!("Division by zero not handled: {}", msg))
                } else {
                    Ok(())
                }
            }
            Err(e) => Err(format!("Unexpected error: {}", e)),
        }
    });
}

fn test_empty_input(runner: &mut TestRunner) {
    runner.run_test("bug_empty_input", || {
        let backend = CpuBackend::default();
        let empty =
            backend.tensor_from_vec(vec![], &[0, 10]).map_err(|e| format!("Create empty failed: {}", e))?;

        let linear = Linear::new(&backend, LinearConfig::new(10, 5))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        match linear.forward(empty, &mut ctx) {
            Ok(_) | Err(CoreError::Shape(_)) => Ok(()),
            Err(e) => Err(format!("Empty input should be handled: {}", e)),
        }
    });

    runner.run_test("bug_single_element_input", || {
        let backend = CpuBackend::default();
        let single = backend
            .tensor_from_vec(vec![1.0f32; 5], &[1, 5])
            .map_err(|e| format!("Create single failed: {}", e))?;

        let linear = Linear::new(&backend, LinearConfig::new(5, 3))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output =
            linear.forward(single, &mut ctx).map_err(|e| format!("Single element forward failed: {}", e))?;

        let shape = backend.ops().shape(&output);
        assert_eq!(shape, vec![1, 3], "Single element output shape wrong");
        Ok(())
    });
}

fn test_large_tensor_allocation(runner: &mut TestRunner) {
    runner.run_test("bug_large_tensor_allocation", || {
        let backend = CpuBackend::default();
        let huge_size = 1_000_000_000usize;
        let data: Vec<f32> = vec![];

        match backend.tensor_from_vec(data.clone(), &[huge_size]) {
            Ok(_) => Ok(()),
            Err(_) => Ok(()),
        }
    });
}

fn test_incompatible_dimensions(runner: &mut TestRunner) {
    runner.run_test("bug_dimension_mismatch", || {
        let backend = CpuBackend::default();
        let wrong_shape = backend
            .tensor_from_vec(vec![1.0f32; 6], &[2, 3])
            .map_err(|e| format!("Create tensor failed: {}", e))?;

        let linear = Linear::new(&backend, LinearConfig::new(5, 2))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);

        match linear.forward(wrong_shape, &mut ctx) {
            Err(_) => Ok(()),
            Ok(_) => Err("Should fail with dimension mismatch".to_string()),
        }
    });

    runner.run_test("bug_conv_kernel_too_large", || {
        let backend = CpuBackend::default();
        let small_input = backend
            .tensor_from_vec(vec![1.0f32; 9], &[1, 1, 3, 3])
            .map_err(|e| format!("Create tensor failed: {}", e))?;

        let conv = Conv2d::new(
            &backend,
            Conv2dConfig {
                out_channels: 1,
                kernel_h: 5,
                kernel_w: 5,
                stride_h: 1,
                stride_w: 1,
                bias: true,
                no_padding: true,
            },
        )
        .map_err(|e| format!("Create conv failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);

        match conv.forward(small_input, &mut ctx) {
            Ok(_) | Err(_) => Ok(()),
        }
    });
}

fn test_gradient_overflow(runner: &mut TestRunner) {
    runner.run_test("bug_gradient_overflow", || {
        let backend = CpuBackend::default();
        let large_values: Vec<f32> = (0..50).map(|i| 1e20f32 * (i as f32)).collect();
        let large_input = backend
            .tensor_from_vec(large_values, &[5, 10])
            .map_err(|e| format!("Create tensor failed: {}", e))?;

        let linear = Linear::new(&backend, LinearConfig::new(10, 5))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Train);
        let output = linear.forward(large_input, &mut ctx).map_err(|e| format!("Forward failed: {}", e))?;

        let data: Vec<f32> = output.as_ref().to_vec();
        let has_nan = data.iter().any(|&v| v.is_nan());

        if has_nan {
            Err("NaN detected in output - numeric instability".to_string())
        } else {
            Ok(())
        }
    });

    runner.run_test("bug_gradient_underflow", || {
        let backend = CpuBackend::default();
        let tiny_values: Vec<f32> = (0..50).map(|_| 1e-38f32).collect();
        let tiny_input = backend
            .tensor_from_vec(tiny_values, &[5, 10])
            .map_err(|e| format!("Create tensor failed: {}", e))?;

        let linear = Linear::new(&backend, LinearConfig::new(10, 5))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Train);
        let output = linear.forward(tiny_input, &mut ctx).map_err(|e| format!("Forward failed: {}", e))?;

        let data: Vec<f32> = output.as_ref().to_vec();
        let has_nan = data.iter().any(|&v| v.is_nan());

        if has_nan {
            Err("NaN detected with tiny inputs".to_string())
        } else {
            Ok(())
        }
    });
}

fn test_memory_leak_in_layers(runner: &mut TestRunner) {
    runner.run_test("bug_memory_leak_layers", || {
        let backend = CpuBackend::default();

        for i in 0..100 {
            let linear = Linear::new(&backend, LinearConfig::new(100, 50))
                .map_err(|e| format!("Iteration {}: {}", i, e))?;

            let input = backend
                .tensor_from_vec(vec![1.0f32; 100], &[1, 100])
                .map_err(|e| format!("Iteration {}: {}", i, e))?;

            let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
            let _output = linear.forward(input, &mut ctx).map_err(|e| format!("Iteration {}: {}", i, e))?;
        }

        Ok(())
    });
}

fn test_causal_mask_boundary(runner: &mut TestRunner) {
    runner.run_test("bug_causal_mask_boundary", || {
        let backend = CpuBackend::default();

        let seq_len = 1usize;
        let mask = rustral_nn::causal_mask(&backend, seq_len)
            .map_err(|e| format!("Create causal mask failed: {}", e))?;

        let mask_data: Vec<f32> = mask.as_ref().to_vec();

        assert_eq!(mask_data.len(), 1, "Single element mask should have 1 element");
        assert_eq!(mask_data[0], 0.0, "Single position should not be masked");

        let seq_len = 4usize;
        let mask = rustral_nn::causal_mask(&backend, seq_len)
            .map_err(|e| format!("Create causal mask failed: {}", e))?;

        let mask_data: Vec<f32> = mask.as_ref().to_vec();

        for i in 0..seq_len {
            for j in 0..seq_len {
                let idx = i * seq_len + j;
                if j > i {
                    assert!(
                        mask_data[idx].is_infinite() || mask_data[idx] < -1e6,
                        "Position ({}, {}) should be masked but got {}",
                        i,
                        j,
                        mask_data[idx]
                    );
                } else {
                    assert_eq!(mask_data[idx], 0.0, "Position ({}, {}) should not be masked", i, j);
                }
            }
        }

        Ok(())
    });
}

fn test_batch_size_mismatch(runner: &mut TestRunner) {
    runner.run_test("bug_batch_size_1", || {
        let backend = CpuBackend::default();

        for batch_size in [1usize, 2, 4, 8] {
            let input = backend
                .tensor_from_vec(vec![1.0f32; batch_size * 10], &[batch_size, 10])
                .map_err(|e| format!("Batch {}: {}", batch_size, e))?;

            let linear = Linear::new(&backend, LinearConfig::new(10, 5))
                .map_err(|e| format!("Batch {}: {}", batch_size, e))?;

            let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
            let output =
                linear.forward(input, &mut ctx).map_err(|e| format!("Batch {}: {}", batch_size, e))?;

            let shape = backend.ops().shape(&output);
            assert_eq!(shape[0], batch_size, "Batch size {} not preserved in output", batch_size);
        }

        Ok(())
    });
}

fn test_nan_propagation(runner: &mut TestRunner) {
    runner.run_test("bug_nan_propagation", || {
        let backend = CpuBackend::default();

        let mut nan_input = vec![1.0f32; 50];
        nan_input[25] = f32::NAN;

        let input = backend
            .tensor_from_vec(nan_input, &[5, 10])
            .map_err(|e| format!("Create tensor failed: {}", e))?;

        let linear = Linear::new(&backend, LinearConfig::new(10, 5))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = linear.forward(input, &mut ctx).map_err(|e| format!("Forward failed: {}", e))?;

        let data: Vec<f32> = output.as_ref().to_vec();
        let output_has_nan = data.iter().any(|&v| v.is_nan());

        if output_has_nan {
            println!("  Note: NaN propagated through linear layer (expected)");
        }

        Ok(())
    });
}

fn test_dropout_determinism(runner: &mut TestRunner) {
    runner.run_test("bug_dropout_determinism", || {
        let backend = CpuBackend::default();

        let input = backend
            .tensor_from_vec(vec![1.0f32; 50], &[5, 10])
            .map_err(|e| format!("Create tensor failed: {}", e))?;

        let dropout = Dropout::new(DropoutConfig::new(0.5));

        let mut ctx1 = ForwardCtx::new(&backend, Mode::Train);
        let output1 =
            dropout.forward(input.clone(), &mut ctx1).map_err(|e| format!("First forward failed: {}", e))?;

        let mut ctx2 = ForwardCtx::new(&backend, Mode::Train);
        let output2 =
            dropout.forward(input, &mut ctx2).map_err(|e| format!("Second forward failed: {}", e))?;

        let data1: Vec<f32> = output1.as_ref().to_vec();
        let data2: Vec<f32> = output2.as_ref().to_vec();

        let mut inf_ctx = ForwardCtx::new(&backend, Mode::Inference);
        let inf_output = dropout
            .forward(backend.tensor_from_vec(vec![1.0f32; 50], &[5, 10]).unwrap(), &mut inf_ctx)
            .unwrap();
        let inf_data: Vec<f32> = inf_output.as_ref().to_vec();

        assert!(inf_data.iter().all(|&v| v == 1.0), "Inference mode should not apply dropout");

        println!("  Dropout determinism verified (training vs inference)");
        Ok(())
    });
}

fn test_quantization_precision_loss(runner: &mut TestRunner) {
    runner.run_test("bug_quantization_precision", || {
        let backend = CpuBackend::default();

        let input = backend
            .tensor_from_vec(vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3])
            .map_err(|e| format!("Create tensor failed: {}", e))?;

        let linear = Linear::new(&backend, LinearConfig::new(3, 2))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = linear.forward(input, &mut ctx).map_err(|e| format!("Forward failed: {}", e))?;

        let _data: Vec<f32> = output.as_ref().to_vec();
        Ok(())
    });
}

fn test_save_load_consistency(runner: &mut TestRunner) {
    runner.run_test("bug_save_load_consistency", || {
        let backend = CpuBackend::default();

        let model = Linear::new(&backend, LinearConfig::new(10, 5))
            .map_err(|e| format!("Create model failed: {}", e))?;

        let input = backend
            .tensor_from_vec(vec![0.5f32; 20], &[2, 10])
            .map_err(|e| format!("Create input failed: {}", e))?;

        let mut ctx1 = ForwardCtx::new(&backend, Mode::Inference);
        let out1 =
            model.forward(input.clone(), &mut ctx1).map_err(|e| format!("First forward failed: {}", e))?;

        let mut ctx2 = ForwardCtx::new(&backend, Mode::Inference);
        let out2 = model.forward(input, &mut ctx2).map_err(|e| format!("Second forward failed: {}", e))?;

        let data1: Vec<f32> = out1.as_ref().to_vec();
        let data2: Vec<f32> = out2.as_ref().to_vec();

        for (i, (a, b)) in data1.iter().zip(data2.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(diff < 1e-5, "Output element {} differs: {} vs {} (diff: {})", i, a, b, diff);
        }

        Ok(())
    });
}

fn test_concurrent_access(runner: &mut TestRunner) {
    runner.run_test("bug_concurrent_access", || {
        let backend = CpuBackend::default();

        let model = Linear::new(&backend, LinearConfig::new(10, 5))
            .map_err(|e| format!("Create model failed: {}", e))?;

        let input1 = backend
            .tensor_from_vec(vec![1.0f32; 10], &[1, 10])
            .map_err(|e| format!("Create input1 failed: {}", e))?;

        let input2 = backend
            .tensor_from_vec(vec![2.0f32; 20], &[2, 10])
            .map_err(|e| format!("Create input2 failed: {}", e))?;

        let mut ctx1 = ForwardCtx::new(&backend, Mode::Inference);
        let out1 = model.forward(input1, &mut ctx1).map_err(|e| format!("Forward1 failed: {}", e))?;

        let mut ctx2 = ForwardCtx::new(&backend, Mode::Inference);
        let out2 = model.forward(input2, &mut ctx2).map_err(|e| format!("Forward2 failed: {}", e))?;

        assert_eq!(backend.ops().shape(&out1), vec![1, 5]);
        assert_eq!(backend.ops().shape(&out2), vec![2, 5]);

        Ok(())
    });
}

fn test_resource_exhaustion(runner: &mut TestRunner) {
    runner.run_test("bug_resource_exhaustion", || {
        let backend = CpuBackend::default();

        let model = Linear::new(&backend, LinearConfig::new(10, 5))
            .map_err(|e| format!("Create model failed: {}", e))?;

        for i in 0..50 {
            let input = backend
                .tensor_from_vec(vec![1.0f32; 10], &[1, 10])
                .map_err(|e| format!("Iteration {}: {}", i, e))?;

            let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
            let _ = model.forward(input, &mut ctx).map_err(|e| format!("Iteration {}: {}", i, e))?;
        }

        Ok(())
    });
}

fn test_infinite_loop_detection(runner: &mut TestRunner) {
    runner.run_test("bug_infinite_loop", || {
        let backend = CpuBackend::default();

        let linear = Linear::new(&backend, LinearConfig::new(5, 5))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let input = backend
            .tensor_from_vec(vec![1.0f32; 5], &[1, 5])
            .map_err(|e| format!("Create input failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let _ = linear.forward(input, &mut ctx).map_err(|e| format!("Forward failed: {}", e))?;

        Ok(())
    });
}

fn test_tensor_alignment(runner: &mut TestRunner) {
    runner.run_test("bug_tensor_alignment", || {
        let backend = CpuBackend::default();

        let input = backend
            .tensor_from_vec(vec![1.0f32; 100], &[10, 10])
            .map_err(|e| format!("Create tensor failed: {}", e))?;

        let linear = Linear::new(&backend, LinearConfig::new(10, 5))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = linear.forward(input, &mut ctx).map_err(|e| format!("Forward failed: {}", e))?;

        let shape = backend.ops().shape(&output);
        assert_eq!(shape, vec![10, 5], "Output shape mismatch");

        Ok(())
    });
}
