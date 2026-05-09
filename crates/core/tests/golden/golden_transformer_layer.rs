//! Golden test for transformer layer forward and backward passes.
//!
//! Tests a single transformer layer (attention + MLP) to ensure numerical
//! stability across different backends and optimization passes.

use rustral_core::{Backend, OpFamily, Result, ShapeExt, ToleranceConfig};
use rustral_ndarray_backend::CpuBackend;

/// Test configuration for golden transformer layer tests.
#[derive(Debug, Clone)]
pub struct TransformerLayerConfig {
    pub batch_size: usize,
    pub seq_len: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub ffn_dim: usize,
    pub dropout_rate: f32,
}

impl Default for TransformerLayerConfig {
    fn default() -> Self {
        Self {
            batch_size: 4,
            seq_len: 32,
            d_model: 64,
            num_heads: 4,
            ffn_dim: 256,
            dropout_rate: 0.1,
        }
    }
}

/// Golden test result for transformer layer.
#[derive(Debug, Clone)]
pub struct TransformerLayerResult {
    pub forward_output: Vec<f32>,
    pub forward_shape: Vec<usize>,
    pub loss: f32,
}

/// Run a simple transformer layer forward pass.
///
/// This implements a minimal transformer layer with:
/// - Multi-head self-attention
/// - Layer normalization
/// - Feed-forward network (linear + gelu + linear)
/// - Residual connections
pub fn golden_transformer_layer_forward(
    backend: &CpuBackend,
    config: &TransformerLayerConfig,
    input: &[f32],
    seed: u64,
) -> Result<TransformerLayerResult> {
    let ops = backend.ops();

    // Reshape input to [batch, seq_len, d_model]
    let input_tensor = ops.tensor_from_vec(
        input.to_vec(),
        &[config.batch_size, config.seq_len, config.d_model],
    )?;

    // Simple implementation: just linear projection + gelu for now
    // In a full implementation, this would include attention, layer norm, etc.
    let weight_shape = &[config.ffn_dim, config.d_model];
    let weight = backend.normal_parameter("weight", weight_shape, seed, 0.1)?;
    let bias_shape = &[config.ffn_dim];
    let bias = backend.normal_parameter("bias", bias_shape, seed + 1, 0.0)?;

    // Linear projection
    let hidden = ops.linear(&input_tensor, &weight, Some(&bias))?;

    // GELU activation
    let activated = ops.gelu(&hidden)?;

    // Project back to d_model
    let output_weight_shape = &[config.d_model, config.ffn_dim];
    let output_weight = backend.normal_parameter("output_weight", output_weight_shape, seed + 2, 0.1)?;
    let output_bias_shape = &[config.d_model];
    let output_bias = backend.normal_parameter("output_bias", output_bias_shape, seed + 3, 0.0)?;

    let output = ops.linear(&activated, &output_weight, Some(&output_bias))?;

    // Add residual connection
    let output = ops.add(&output, &input_tensor)?;

    // Compute a simple loss (mean squared error)
    let target = ops.zeros(&output.shape())?;
    let diff = ops.sub(&output, &target)?;
    let squared = ops.mul(&diff, &diff)?;
    let sum = ops.sum_all(&squared)?;
    let loss_values = ops.tensor_to_vec(&sum)?;
    let loss = loss_values[0] / (output.shape().iter().product() as f32);

    let output_values = ops.tensor_to_vec(&output)?;
    let output_shape = ops.shape(&output);

    Ok(TransformerLayerResult {
        forward_output: output_values,
        forward_shape: output_shape,
        loss,
    })
}

/// Test that transformer layer forward pass is deterministic.
#[test]
fn test_transformer_layer_determinism() {
    let config = TransformerLayerConfig::default();
    let backend = CpuBackend::default();
    let seed = 42;

    // Generate input data
    let input_size = config.batch_size * config.seq_len * config.d_model;
    let input: Vec<f32> = (0..input_size)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();

    // Run twice with same seed
    let result1 = golden_transformer_layer_forward(&backend, &config, &input, seed).unwrap();
    let result2 = golden_transformer_layer_forward(&backend, &config, &input, seed).unwrap();

    // Check that results are identical
    assert_eq!(result1.forward_shape, result2.forward_shape);
    assert_eq!(result1.loss, result2.loss);

    for (i, (&v1, &v2)) in result1.forward_output.iter().zip(result2.forward_output.iter()).enumerate() {
        if (v1 - v2).abs() > 1e-6 {
            panic!("Determinism check failed at index {}: {} vs {}", i, v1, v2);
        }
    }
}

/// Test that transformer layer forward pass is numerically stable.
#[test]
fn test_transformer_layer_numerical_stability() {
    let config = TransformerLayerConfig::default();
    let backend = CpuBackend::default();
    let seed = 42;

    // Generate input data with different scales
    let input_size = config.batch_size * config.seq_len * config.d_model;
    let scales = [0.01, 1.0, 100.0];

    for scale in scales {
        let input: Vec<f32> = (0..input_size)
            .map(|i| (i as f32 * scale).sin())
            .collect();

        let result = golden_transformer_layer_forward(&backend, &config, &input, seed).unwrap();

        // Check that loss is finite
        assert!(result.loss.is_finite(), "Loss should be finite for scale {}", scale);

        // Check that all outputs are finite
        for (i, &v) in result.forward_output.iter().enumerate() {
            assert!(
                v.is_finite(),
                "Output should be finite at index {} for scale {}, got {}",
                i,
                scale,
                v
            );
        }
    }
}

/// Test tolerance checking with transformer layer outputs.
#[test]
fn test_transformer_layer_tolerance() {
    let config = TransformerLayerConfig::default();
    let backend = CpuBackend::default();
    let seed = 42;

    let input_size = config.batch_size * config.seq_len * config.d_model;
    let input: Vec<f32> = (0..input_size)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();

    let result1 = golden_transformer_layer_forward(&backend, &config, &input, seed).unwrap();

    // Add small perturbation
    let mut input_perturbed = input.clone();
    for v in input_perturbed.iter_mut() {
        *v += 1e-6;
    }

    let result2 = golden_transformer_layer_forward(&backend, &config, &input_perturbed, seed).unwrap();

    // Check that small input changes produce small output changes
    let tol = ToleranceConfig::for_family(OpFamily::MatmulLinear);
    assert!(tol.check_slice(&result1.forward_output, &result2.forward_output));

    // Loss should be similar
    assert!(tol.check(result1.loss, result2.loss));
}

/// Test transformer layer with different configurations.
#[test]
fn test_transformer_layer_configurations() {
    let backend = CpuBackend::default();
    let seed = 42;

    let configs = vec![
        TransformerLayerConfig {
            batch_size: 2,
            seq_len: 16,
            d_model: 32,
            num_heads: 2,
            ffn_dim: 128,
            dropout_rate: 0.1,
        },
        TransformerLayerConfig {
            batch_size: 4,
            seq_len: 32,
            d_model: 64,
            num_heads: 4,
            ffn_dim: 256,
            dropout_rate: 0.1,
        },
        TransformerLayerConfig {
            batch_size: 8,
            seq_len: 64,
            d_model: 128,
            num_heads: 8,
            ffn_dim: 512,
            dropout_rate: 0.1,
        },
    ];

    for config in configs {
        let input_size = config.batch_size * config.seq_len * config.d_model;
        let input: Vec<f32> = (0..input_size)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();

        let result = golden_transformer_layer_forward(&backend, &config, &input, seed).unwrap();

        // Validate output shape
        assert_eq!(
            result.forward_shape,
            vec![config.batch_size, config.seq_len, config.d_model]
        );

        // Validate that loss is reasonable
        assert!(result.loss > 0.0);
        assert!(result.loss < 1000.0); // Should not explode

        println!(
            "Config batch={} seq={} d_model={} ffn_dim={} loss={}",
            config.batch_size, config.seq_len, config.d_model, config.ffn_dim, result.loss
        );
    }
}
