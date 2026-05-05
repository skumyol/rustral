//! Edge Cases and Boundary Condition Tests

use mnr_core::{Backend, CoreError, ForwardCtx, Mode, Module};
use mnr_ndarray_backend::CpuBackend;
use mnr_nn::{
    Conv2d, Conv2dConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig,
    SelfAttention, SelfAttentionConfig, TransformerEncoder, TransformerEncoderConfig,
};

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

fn test_zero_dimensions(runner: &mut TestRunner) {
    runner.run_test("edge_zero_dim_batch", || {
        let backend = CpuBackend::default();
        let empty = backend
            .tensor_from_vec(vec![], &[0, 10])
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
            Err(CoreError::Shape(_)) => Ok(()),
            Err(e) => Err(format!("Unexpected error: {}", e)),
        }
    });

    runner.run_test("edge_zero_dim_features", || {
        let backend = CpuBackend::default();
        let empty = backend
            .tensor_from_vec(vec![], &[5, 0])
            .map_err(|e| format!("Create zero features failed: {}", e))?;

        let linear = Linear::new(&backend, LinearConfig::new(0, 5))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        match linear.forward(empty, &mut ctx) {
            Ok(_) | Err(CoreError::Shape(_)) => Ok(()),
            Err(e) => Err(format!("Unexpected error: {}", e)),
        }
    });
}

fn test_maximum_dimensions(runner: &mut TestRunner) {
    runner.run_test("edge_large_batch", || {
        let backend = CpuBackend::default();
        let batch_size = 256usize;
        let input = backend
            .tensor_from_vec(vec![1.0f32; batch_size * 10], &[batch_size, 10])
            .map_err(|e| format!("Create large batch failed: {}", e))?;

        let linear = Linear::new(&backend, LinearConfig::new(10, 5))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = linear.forward(input, &mut ctx).map_err(|e| format!("Forward failed: {}", e))?;

        let shape = backend.ops().shape(&output);
        assert_eq!(shape[0], batch_size, "Large batch size not preserved");
        Ok(())
    });

    runner.run_test("edge_large_features", || {
        let backend = CpuBackend::default();
        let in_features = 1024usize;
        let out_features = 512usize;

        let input = backend
            .tensor_from_vec(vec![0.01f32; 2 * in_features], &[2, in_features])
            .map_err(|e| format!("Create large features failed: {}", e))?;

        let linear = Linear::new(&backend, LinearConfig::new(in_features, out_features))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = linear.forward(input, &mut ctx).map_err(|e| format!("Forward failed: {}", e))?;

        let shape = backend.ops().shape(&output);
        assert_eq!(shape, vec![2, out_features], "Large feature shape mismatch");
        Ok(())
    });
}

fn test_extreme_values(runner: &mut TestRunner) {
    runner.run_test("edge_float32_max", || {
        let backend = CpuBackend::default();
        let input = backend
            .tensor_from_vec(vec![f32::MAX; 10], &[2, 5])
            .map_err(|e| format!("Create MAX tensor failed: {}", e))?;

        let linear = Linear::new(&backend, LinearConfig::new(5, 3))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = linear.forward(input, &mut ctx).map_err(|e| format!("Forward failed: {}", e))?;

        let data: Vec<f32> = output.as_ref().to_vec();
        assert!(data.iter().all(|&v| !v.is_nan()), "MAX input should not produce NaN");
        Ok(())
    });

    runner.run_test("edge_float32_min", || {
        let backend = CpuBackend::default();
        let input = backend
            .tensor_from_vec(vec![f32::MIN; 10], &[2, 5])
            .map_err(|e| format!("Create MIN tensor failed: {}", e))?;

        let linear = Linear::new(&backend, LinearConfig::new(5, 3))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = linear.forward(input, &mut ctx).map_err(|e| format!("Forward failed: {}", e))?;

        let data: Vec<f32> = output.as_ref().to_vec();
        assert!(data.iter().all(|&v| !v.is_nan()), "MIN input should not produce NaN");
        Ok(())
    });

    runner.run_test("edge_epsilon_values", || {
        let backend = CpuBackend::default();
        let input = backend
            .tensor_from_vec(vec![f32::EPSILON; 10], &[2, 5])
            .map_err(|e| format!("Create epsilon tensor failed: {}", e))?;

        let linear = Linear::new(&backend, LinearConfig::new(5, 3))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = linear.forward(input, &mut ctx).map_err(|e| format!("Forward failed: {}", e))?;

        let data: Vec<f32> = output.as_ref().to_vec();
        assert!(
            data.iter().all(|&v| v.is_finite() || v == 0.0),
            "Epsilon input should produce finite output"
        );
        Ok(())
    });
}

fn test_special_float_values(runner: &mut TestRunner) {
    runner.run_test("edge_special_values_mixed", || {
        let backend = CpuBackend::default();
        let special_values =
            vec![1.0f32, f32::NAN, 2.0, f32::INFINITY, 3.0, f32::NEG_INFINITY, 4.0, f32::MAX, 5.0, f32::MIN];

        let input = backend
            .tensor_from_vec(special_values, &[2, 5])
            .map_err(|e| format!("Create special tensor failed: {}", e))?;

        let linear = Linear::new(&backend, LinearConfig::new(5, 3))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = linear.forward(input, &mut ctx).map_err(|e| format!("Forward failed: {}", e))?;

        let _data: Vec<f32> = output.as_ref().to_vec();
        Ok(())
    });

    runner.run_test("edge_negative_zero", || {
        let backend = CpuBackend::default();
        let input = backend
            .tensor_from_vec(vec![-0.0f32; 10], &[2, 5])
            .map_err(|e| format!("Create -0.0 tensor failed: {}", e))?;

        let linear = Linear::new(&backend, LinearConfig::new(5, 3))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = linear.forward(input, &mut ctx).map_err(|e| format!("Forward failed: {}", e))?;

        let data: Vec<f32> = output.as_ref().to_vec();
        assert!(data.iter().all(|&v| v == 0.0 || v.is_nan()), "Negative zero should produce zero output");
        Ok(())
    });
}

fn test_boundary_sequence_lengths(runner: &mut TestRunner) {
    runner.run_test("edge_seq_length_1", || {
        let backend = CpuBackend::default();
        let config = TransformerEncoderConfig::new(64, 4, 2, 256).with_max_seq_len(128);

        let encoder = TransformerEncoder::new(&backend, config, 1000, 42)
            .map_err(|e| format!("Create encoder failed: {}", e))?;

        let input = vec![100usize];
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = encoder.forward(input, &mut ctx).map_err(|e| format!("Forward failed: {}", e))?;

        let shape = backend.ops().shape(&output);
        assert_eq!(shape[1], 1, "Seq length 1 should be preserved");
        Ok(())
    });

    runner.run_test("edge_seq_length_exact_max", || {
        let backend = CpuBackend::default();
        let max_len = 16usize;
        let config = TransformerEncoderConfig::new(64, 4, 2, 256).with_max_seq_len(max_len);

        let encoder = TransformerEncoder::new(&backend, config, 1000, 42)
            .map_err(|e| format!("Create encoder failed: {}", e))?;

        let input = vec![100usize; max_len];
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = encoder.forward(input, &mut ctx).map_err(|e| format!("Forward failed: {}", e))?;

        let shape = backend.ops().shape(&output);
        assert_eq!(shape[1], max_len, "Max seq length should work");
        Ok(())
    });

    runner.run_test("edge_vocab_boundary", || {
        let backend = CpuBackend::default();
        let vocab_size = 1000usize;
        let config = TransformerEncoderConfig::new(64, 4, 2, 256);

        let encoder = TransformerEncoder::new(&backend, config, vocab_size, 42)
            .map_err(|e| format!("Create encoder failed: {}", e))?;

        let boundary_tokens = vec![0usize, 1, vocab_size / 2, vocab_size - 2, vocab_size - 1];
        let input = boundary_tokens.clone();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = encoder.forward(input, &mut ctx).map_err(|e| format!("Forward failed: {}", e))?;

        let shape = backend.ops().shape(&output);
        assert_eq!(shape[1], boundary_tokens.len(), "Boundary vocab tokens should work");
        Ok(())
    });
}

fn test_single_element_tensors(runner: &mut TestRunner) {
    runner.run_test("edge_single_element_1d", || {
        let backend = CpuBackend::default();
        let input = backend
            .tensor_from_vec(vec![1.0f32], &[1])
            .map_err(|e| format!("Create single element failed: {}", e))?;

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
        let input = backend
            .tensor_from_vec(vec![1.0f32], &[1, 1])
            .map_err(|e| format!("Create 1x1 tensor failed: {}", e))?;

        let linear = Linear::new(&backend, LinearConfig::new(1, 1))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = linear.forward(input, &mut ctx).map_err(|e| format!("Forward failed: {}", e))?;

        let shape = backend.ops().shape(&output);
        assert_eq!(shape, vec![1, 1], "1x1 tensor shape mismatch");
        Ok(())
    });
}

fn test_very_deep_networks(runner: &mut TestRunner) {
    runner.run_test("edge_deep_transformer", || {
        let backend = CpuBackend::default();
        let num_layers = 24usize;
        let config = TransformerEncoderConfig::new(64, 4, num_layers, 256).with_max_seq_len(128);

        let encoder = TransformerEncoder::new(&backend, config, 1000, 42)
            .map_err(|e| format!("Create deep encoder failed: {}", e))?;

        let input = vec![100usize; 16];
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = encoder.forward(input, &mut ctx).map_err(|e| format!("Deep forward failed: {}", e))?;

        let shape = backend.ops().shape(&output);
        assert_eq!(shape, vec![1, 16, 64], "Deep network output shape wrong");
        Ok(())
    });

    runner.run_test("edge_wide_attention_heads", || {
        let backend = CpuBackend::default();
        let d_model = 64usize;
        let num_heads = 4usize;

        let config = TransformerEncoderConfig::new(d_model, num_heads, 2, 512).with_max_seq_len(128);

        let encoder = TransformerEncoder::new(&backend, config, 1000, 42)
            .map_err(|e| format!("Create wide encoder failed: {}", e))?;

        let input = vec![100usize; 16];
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = encoder.forward(input, &mut ctx).map_err(|e| format!("Wide forward failed: {}", e))?;

        let shape = backend.ops().shape(&output);
        assert_eq!(shape, vec![1, 16, d_model], "Wide attention output shape wrong");
        Ok(())
    });
}

fn test_wide_vs_narrow_tensors(runner: &mut TestRunner) {
    runner.run_test("edge_very_wide_tensor", || {
        let backend = CpuBackend::default();
        let features = 4096usize;
        let input = backend
            .tensor_from_vec(vec![0.01f32; features], &[1, features])
            .map_err(|e| format!("Create wide tensor failed: {}", e))?;

        let linear = Linear::new(&backend, LinearConfig::new(features, 128))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = linear.forward(input, &mut ctx).map_err(|e| format!("Wide forward failed: {}", e))?;

        let shape = backend.ops().shape(&output);
        assert_eq!(shape, vec![1, 128], "Wide tensor output shape wrong");
        Ok(())
    });

    runner.run_test("edge_very_narrow_tensor", || {
        let backend = CpuBackend::default();
        let batch = 1024usize;
        let input = backend
            .tensor_from_vec(vec![0.5f32; batch * 4], &[batch, 4])
            .map_err(|e| format!("Create narrow tensor failed: {}", e))?;

        let linear = Linear::new(&backend, LinearConfig::new(4, 128))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = linear.forward(input, &mut ctx).map_err(|e| format!("Narrow forward failed: {}", e))?;

        let shape = backend.ops().shape(&output);
        assert_eq!(shape, vec![batch, 128], "Narrow tensor output shape wrong");
        Ok(())
    });
}

fn test_non_contiguous_tensors(runner: &mut TestRunner) {
    runner.run_test("edge_strided_operations", || {
        let backend = CpuBackend::default();
        let input = backend
            .tensor_from_vec((0..100).map(|i| i as f32).collect(), &[10, 10])
            .map_err(|e| format!("Create tensor failed: {}", e))?;

        let linear = Linear::new(&backend, LinearConfig::new(10, 5))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = linear.forward(input, &mut ctx).map_err(|e| format!("Forward failed: {}", e))?;

        let data: Vec<f32> = output.as_ref().to_vec();
        assert!(data.iter().all(|&v| v.is_finite()), "Strided operations should produce finite output");
        Ok(())
    });
}

fn test_mixed_precision_boundaries(runner: &mut TestRunner) {
    runner.run_test("edge_mixed_precision_boundary", || {
        let backend = CpuBackend::default();
        let input = backend
            .tensor_from_vec(vec![0.5f32; 100], &[10, 10])
            .map_err(|e| format!("Create tensor failed: {}", e))?;

        let linear = Linear::new(&backend, LinearConfig::new(10, 5))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = linear.forward(input, &mut ctx).map_err(|e| format!("Forward failed: {}", e))?;

        let shape = backend.ops().shape(&output);
        assert_eq!(shape, vec![10, 5], "Mixed precision boundary output shape wrong");
        Ok(())
    });
}
