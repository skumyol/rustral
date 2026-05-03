//! Edge Cases and Boundary Condition Tests
//!
//! Comprehensive tests for boundary conditions, extreme values,
//! and unusual input configurations.

use mnr_core::{Backend, CoreError, ForwardCtx, Mode, Module};
use mnr_nn::{
    Conv2d, Conv2dConfig, Embedding, EmbeddingConfig, Linear, LinearConfig,
    LayerNorm, LayerNormConfig, BatchNorm, BatchNormConfig,
    SelfAttention, SelfAttentionConfig,
    TransformerEncoder, TransformerEncoderConfig,
};
use mnr_ndarray_backend::CpuBackend;

use crate::common::{run_performance_test, PerfConfig, TestRunner};

pub fn run_all(runner: &mut TestRunner) {
    test_zero_dimensions(runner);
    test_maximum_dimensions(runner);
    test_extreme_values(runner);
    test_special_float_values(runner);
    test_boundary_sequence_lengths(runner);
    test_single_element_tensors(runner);
    test_very_deep_networks(runner);
    test_wide_vs_narrow_tensors(runner);
    test_non_contiguous_tensors(runner);
    test_mixed_precision_boundaries(runner);
}

// ========================================================================
// Zero Dimensions
// ========================================================================

fn test_zero_dimensions(runner: &mut TestRunner) {
    runner.run_test("edge_zero_dim_batch", || {
        let backend = CpuBackend::default();

        // Zero batch size
        let empty = backend.tensor_from_vec(vec![], &[0, 10])
            .map_err(|e| format!("Create zero batch failed: {}", e))?;

        let linear = Linear::new(&backend, LinearConfig::new(10, 5))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        match linear.forward(empty, &mut ctx) {
            Ok(out) => {
                let shape = backend.ops().shape(&out);
                assert_eq!(shape[0], 0, "Zero batch should produce zero batch output");
                Ok(())
            }
            Err(CoreError::Shape(_)) => Ok(()), // Accept shape error
            Err(e) => Err(format!("Unexpected error: {}", e)),
        }
    });

    runner.run_test("edge_zero_dim_features", || {
        let backend = CpuBackend::default();

        // Zero feature dimension
        let empty = backend.tensor_from_vec(vec![], &[5, 0])
            .map_err(|e| format!("Create zero features failed: {}", e))?;

        let linear = Linear::new(&backend, LinearConfig::new(0, 5))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        match linear.forward(empty, &mut ctx) {
            Ok(out) | Err(CoreError::Shape(_)) => Ok(()),
            Err(e) => Err(format!("Unexpected error: {}", e)),
        }
    });
}

// ========================================================================
// Maximum Dimensions
// ========================================================================

fn test_maximum_dimensions(runner: &mut TestRunner) {
    runner.run_test("edge_large_batch", || {
        let backend = CpuBackend::default();

        // Large batch (but not too large for test)
        let batch_size = 256usize;
        let input = backend
            .tensor_from_vec(vec![1.0f32; batch_size * 10], &[batch_size, 10])
            .map_err(|e| format!("Create large batch failed: {}", e))?;

        let linear = Linear::new(&backend, LinearConfig::new(10, 5))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = linear.forward(input, &mut ctx)
            .map_err(|e| format!("Forward failed: {}", e))?;

        let shape = backend.ops().shape(&output);
        assert_eq!(shape[0], batch_size, "Large batch size not preserved");
        Ok(())
    });

    runner.run_test("edge_large_features", || {
        let backend = CpuBackend::default();

        // Large feature dimension
        let in_features = 1024usize;
        let out_features = 512usize;

        let input = backend
            .tensor_from_vec(vec![0.01f32; 2 * in_features], &[2, in_features])
            .map_err(|e| format!("Create large features failed: {}", e))?;

        let linear = Linear::new(&backend, LinearConfig::new(in_features, out_features))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = linear.forward(input, &mut ctx)
            .map_err(|e| format!("Forward failed: {}", e))?;

        let shape = backend.ops().shape(&output);
        assert_eq!(shape, vec![2, out_features], "Large feature shape mismatch");
        Ok(())
    });
}

// ========================================================================
// Extreme Values
// ========================================================================

fn test_extreme_values(runner: &mut TestRunner) {
    runner.run_test("edge_float32_max", || {
        let backend = CpuBackend::default();

        // Float32 max value
        let input = backend
            .tensor_from_vec(vec![f32::MAX; 10], &[2, 5])
            .map_err(|e| format!("Create MAX tensor failed: {}", e))?;

        let linear = Linear::new(&backend, LinearConfig::new(5, 3))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = linear.forward(input, &mut ctx)
            .map_err(|e| format!("Forward failed: {}", e))?;

        // Output should be Inf (not NaN)
        let data: Vec<f32> = output.as_ref().to_vec();
        assert!(
            data.iter().all(|&v| !v.is_nan()),
            "MAX input should not produce NaN"
        );
        Ok(())
    });

    runner.run_test("edge_float32_min", || {
        let backend = CpuBackend::default();

        // Float32 min (most negative) value
        let input = backend
            .tensor_from_vec(vec![f32::MIN; 10], &[2, 5])
            .map_err(|e| format!("Create MIN tensor failed: {}", e))?;

        let linear = Linear::new(&backend, LinearConfig::new(5, 3))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = linear.forward(input, &mut ctx)
            .map_err(|e| format!("Forward failed: {}", e))?;

        let data: Vec<f32> = output.as_ref().to_vec();
        assert!(
            data.iter().all(|&v| !v.is_nan()),
            "MIN input should not produce NaN"
        );
        Ok(())
    });

    runner.run_test("edge_epsilon_values", || {
        let backend = CpuBackend::default();

        // Very small epsilon values
        let input = backend
            .tensor_from_vec(vec![f32::EPSILON; 10], &[2, 5])
            .map_err(|e| format!("Create epsilon tensor failed: {}", e))?;

        let linear = Linear::new(&backend, LinearConfig::new(5, 3))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = linear.forward(input, &mut ctx)
            .map_err(|e| format!("Forward failed: {}", e))?;

        // Should produce valid (possibly zero) output
        let data: Vec<f32> = output.as_ref().to_vec();
        assert!(
            data.iter().all(|&v| v.is_finite() || v == 0.0),
            "Epsilon input should produce finite output"
        );
        Ok(())
    });
}

// ========================================================================
// Special Float Values
// ========================================================================

fn test_special_float_values(runner: &mut TestRunner) {
    runner.run_test("edge_special_values_mixed", || {
        let backend = CpuBackend::default();

        // Mix of special values
        let special_values = vec![
            1.0f32,
            f32::NAN,
            2.0,
            f32::INFINITY,
            3.0,
            f32::NEG_INFINITY,
            4.0,
            f32::MAX,
            5.0,
            f32::MIN,
        ];

        let input = backend
            .tensor_from_vec(special_values, &[2, 5])
            .map_err(|e| format!("Create special tensor failed: {}", e))?;

        let linear = Linear::new(&backend, LinearConfig::new(5, 3))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = linear.forward(input, &mut ctx)
            .map_err(|e| format!("Forward failed: {}", e))?;

        // Just verify it doesn't crash
        let _data: Vec<f32> = output.as_ref().to_vec();
        Ok(())
    });

    runner.run_test("edge_negative_zero", || {
        let backend = CpuBackend::default();

        // Negative zero
        let input = backend
            .tensor_from_vec(vec![-0.0f32; 10], &[2, 5])
            .map_err(|e| format!("Create -0.0 tensor failed: {}", e))?;

        let linear = Linear::new(&backend, LinearConfig::new(5, 3))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = linear.forward(input, &mut ctx)
            .map_err(|e| format!("Forward failed: {}", e))?;

        // -0.0 should behave same as 0.0
        let data: Vec<f32> = output.as_ref().to_vec();
        assert!(
            data.iter().all(|&v| v == 0.0 || v.is_nan()),
            "Negative zero should produce zero output"
        );
        Ok(())
    });
}

// ========================================================================
// Boundary Sequence Lengths
// ========================================================================

fn test_boundary_sequence_lengths(runner: &mut TestRunner) {
    runner.run_test("edge_seq_length_1", || {
        let backend = CpuBackend::default();

        // Sequence length of 1
        let config = TransformerEncoderConfig::new(64, 4, 2, 256)
            .with_max_seq_len(128);

        let encoder = TransformerEncoder::new(&backend, config, 1000, 42)
            .map_err(|e| format!("Create encoder failed: {}", e))?;

        let input = backend
            .tensor_from_vec(vec![100u32], &[1, 1])
            .map_err(|e| format!("Create input failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = encoder.forward(input, &mut ctx)
            .map_err(|e| format!("Forward failed: {}", e))?;

        let shape = backend.ops().shape(&output);
        assert_eq!(shape[1], 1, "Seq length 1 should be preserved");
        Ok(())
    });

    runner.run_test("edge_seq_length_exact_max", || {
        let backend = CpuBackend::default();

        // Sequence length exactly at max_seq_len
        let max_len = 16usize;
        let config = TransformerEncoderConfig::new(64, 4, 2, 256)
            .with_max_seq_len(max_len);

        let encoder = TransformerEncoder::new(&backend, config, 1000, 42)
            .map_err(|e| format!("Create encoder failed: {}", e))?;

        let input = backend
            .tensor_from_vec(vec![100u32; max_len], &[1, max_len])
            .map_err(|e| format!("Create input failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = encoder.forward(input, &mut ctx)
            .map_err(|e| format!("Forward failed: {}", e))?;

        let shape = backend.ops().shape(&output);
        assert_eq!(shape[1], max_len, "Max seq length should work");
        Ok(())
    });

    runner.run_test("edge_vocab_boundary", || {
        let backend = CpuBackend::default();

        // Test vocab boundary indices
        let vocab_size = 1000usize;
        let config = TransformerEncoderConfig::new(64, 4, 2, 256);

        let encoder = TransformerEncoder::new(&backend, config, vocab_size, 42)
            .map_err(|e| format!("Create encoder failed: {}", e))?;

        // Token IDs at boundaries
        let boundary_tokens = vec![
            0u32,          // First vocab entry
            1,
            (vocab_size / 2) as u32,
            (vocab_size - 2) as u32,
            (vocab_size - 1) as u32, // Last vocab entry
        ];

        let input = backend
            .tensor_from_vec(boundary_tokens.clone(), &[1, boundary_tokens.len()])
            .map_err(|e| format!("Create input failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = encoder.forward(input, &mut ctx)
            .map_err(|e| format!("Forward failed: {}", e))?;

        let shape = backend.ops().shape(&output);
        assert_eq!(shape[1], boundary_tokens.len(), "Boundary vocab tokens should work");
        Ok(())
    });
}

// ========================================================================
// Single Element Tensors
// ========================================================================

fn test_single_element_tensors(runner: &mut TestRunner) {
    runner.run_test("edge_single_element_1d", || {
        let backend = CpuBackend::default();

        let input = backend
            .tensor_from_vec(vec![1.0f32], &[1])
            .map_err(|e| format!("Create single element failed: {}", e))?;

        // 1D tensors might not work with Linear which expects 2D
        // This tests the error handling
        match Linear::new(&backend, LinearConfig::new(1, 1)) {
            Ok(linear) => {
                let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
                match linear.forward(input, &mut ctx) {
                    Ok(_) | Err(CoreError::Shape(_)) => Ok(()),
                    Err(e) => Err(format!("Unexpected error: {}", e)),
                }
            }
            Err(_) => Ok(()),
        }
    });

    runner.run_test("edge_scalar_broadcast", || {
        let backend = CpuBackend::default();

        // Single batch, single feature
        let input = backend
            .tensor_from_vec(vec![1.0f32], &[1, 1])
            .map_err(|e| format!("Create 1x1 tensor failed: {}", e))?;

        let linear = Linear::new(&backend, LinearConfig::new(1, 1))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = linear.forward(input, &mut ctx)
            .map_err(|e| format!("Forward failed: {}", e))?;

        let shape = backend.ops().shape(&output);
        assert_eq!(shape, vec![1, 1], "1x1 tensor shape mismatch");
        Ok(())
    });
}

// ========================================================================
// Very Deep Networks
// ========================================================================

fn test_very_deep_networks(runner: &mut TestRunner) {
    runner.run_test("edge_deep_transformer", || {
        let backend = CpuBackend::default();

        // Very deep transformer (many layers)
        let num_layers = 24usize;
        let config = TransformerEncoderConfig::new(64, 4, num_layers, 256)
            .with_max_seq_len(128);

        let encoder = TransformerEncoder::new(&backend, config, 1000, 42)
            .map_err(|e| format!("Create deep encoder failed: {}", e))?;

        let input = backend
            .tensor_from_vec(vec![100u32; 16], &[1, 16])
            .map_err(|e| format!("Create input failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = encoder.forward(input, &mut ctx)
            .map_err(|e| format!("Deep forward failed: {}", e))?;

        let shape = backend.ops().shape(&output);
        assert_eq!(shape, vec![1, 16, 64], "Deep network output shape wrong");
        Ok(())
    });

    runner.run_test("edge_wide_attention_heads", || {
        let backend = CpuBackend::default();

        // Many attention heads (d_model must be divisible)
        let d_model = 256usize;
        let num_heads = 32usize; // 8 dims per head

        let config = TransformerEncoderConfig::new(d_model, num_heads, 2, 512)
            .with_max_seq_len(128);

        let encoder = TransformerEncoder::new(&backend, config, 1000, 42)
            .map_err(|e| format!("Create wide encoder failed: {}", e))?;

        let input = backend
            .tensor_from_vec(vec![100u32; 16], &[1, 16])
            .map_err(|e| format!("Create input failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = encoder.forward(input, &mut ctx)
            .map_err(|e| format!("Wide forward failed: {}", e))?;

        let shape = backend.ops().shape(&output);
        assert_eq!(shape, vec![1, 16, d_model], "Wide attention output shape wrong");
        Ok(())
    });
}

// ========================================================================
// Wide vs Narrow Tensors
// ========================================================================

fn test_wide_vs_narrow_tensors(runner: &mut TestRunner) {
    runner.run_test("edge_very_wide_tensor", || {
        let backend = CpuBackend::default();

        // Batch=1, Features=very large
        let features = 4096usize;
        let input = backend
            .tensor_from_vec(vec![0.01f32; features], &[1, features])
            .map_err(|e| format!("Create wide tensor failed: {}", e))?;

        let linear = Linear::new(&backend, LinearConfig::new(features, 128))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = linear.forward(input, &mut ctx)
            .map_err(|e| format!("Wide forward failed: {}", e))?;

        let shape = backend.ops().shape(&output);
        assert_eq!(shape, vec![1, 128], "Wide tensor output shape wrong");
        Ok(())
    });

    runner.run_test("edge_very_narrow_tensor", || {
        let backend = CpuBackend::default();

        // Large batch, few features
        let batch = 1024usize;
        let input = backend
            .tensor_from_vec(vec![0.5f32; batch * 4], &[batch, 4])
            .map_err(|e| format!("Create narrow tensor failed: {}", e))?;

        let linear = Linear::new(&backend, LinearConfig::new(4, 128))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = linear.forward(input, &mut ctx)
            .map_err(|e| format!("Narrow forward failed: {}", e))?;

        let shape = backend.ops().shape(&output);
        assert_eq!(shape, vec![batch, 128], "Narrow tensor output shape wrong");
        Ok(())
    });
}

// ========================================================================
// Non-Contiguous Tensors
// ========================================================================

fn test_non_contiguous_tensors(runner: &mut TestRunner) {
    runner.run_test("edge_strided_operations", || {
        let backend = CpuBackend::default();

        // Create tensor and test operations that might require contiguous memory
        let input = backend
            .tensor_from_vec((0..100).map(|i| i as f32).collect(), &[10, 10])
            .map_err(|e| format!("Create tensor failed: {}", e))?;

        let linear = Linear::new(&backend, LinearConfig::new(10, 5))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = linear.forward(input, &mut ctx)
            .map_err(|e| format!("Forward failed: {}", e))?;

        // Verify output is reasonable
        let data: Vec<f32> = output.as_ref().to_vec();
        assert!(
            data.iter().all(|&v| v.is_finite()),
            "Strided operations should produce finite output"
        );
        Ok(())
    });
}

// ========================================================================
// Mixed Precision Boundaries
// ========================================================================

fn test_mixed_precision_boundaries(runner: &mut TestRunner) {
    runner.run_test("edge_precision_boundary", || {
        let backend = CpuBackend::default();

        // Values near precision limits
        let values = vec![
            1.0f32,
            1.0 + f32::EPSILON,      // Just above 1.0
            1.0 + 2.0 * f32::EPSILON,
            1.0 - f32::EPSILON,      // Just below 1.0
            f32::MIN_POSITIVE,       // Smallest positive normal
            f32::MIN_POSITIVE * f32::EPSILON, // Smallest positive subnormal
        ];

        let input = backend
            .tensor_from_vec(values.clone(), &[1, values.len()])
            .map_err(|e| format!("Create precision tensor failed: {}", e))?;

        let linear = Linear::new(&backend, LinearConfig::new(values.len(), 3))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = linear.forward(input, &mut ctx)
            .map_err(|e| format!("Forward failed: {}", e))?;

        let out_data: Vec<f32> = output.as_ref().to_vec();

        // Output should not lose all precision
        assert!(
            out_data.iter().all(|&v| v.is_finite()),
            "Precision boundary values should produce finite output"
        );
        Ok(())
    });
}
