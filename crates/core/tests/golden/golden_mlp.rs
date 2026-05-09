//! Golden test for MLP (Multi-Layer Perceptron) block.
//!
//! Tests the feed-forward network component of transformer blocks
//! with linear layers, GELU activation, and dropout.

use rustral_core::{Backend, OpFamily, Result, ShapeExt, ToleranceConfig};
use rustral_ndarray_backend::CpuBackend;

/// Test configuration for golden MLP tests.
#[derive(Debug, Clone)]
pub struct MlpConfig {
    pub batch_size: usize,
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    pub dropout_rate: f32,
    pub activation: Activation,
}

#[derive(Debug, Clone, Copy)]
pub enum Activation {
    ReLU,
    GELU,
    Sigmoid,
    Tanh,
}

impl Default for MlpConfig {
    fn default() -> Self {
        Self {
            batch_size: 4,
            input_dim: 64,
            hidden_dim: 256,
            output_dim: 64,
            dropout_rate: 0.1,
            activation: Activation::GELU,
        }
    }
}

/// Golden test result for MLP block.
#[derive(Debug, Clone)]
pub struct MlpResult {
    pub output: Vec<f32>,
    pub output_shape: Vec<usize>,
    pub loss: f32,
}

/// Run a simplified MLP forward pass.
///
/// This implements a standard MLP:
/// - Linear layer 1: input -> hidden
/// - Activation function
/// - Dropout (training mode only)
/// - Linear layer 2: hidden -> output
pub fn golden_mlp_forward(
    backend: &CpuBackend,
    config: &MlpConfig,
    input: &[f32],
    seed: u64,
    training: bool,
) -> Result<MlpResult> {
    let ops = backend.ops();

    // Reshape input to [batch, input_dim]
    let input_tensor = ops.tensor_from_vec(input.to_vec(), &[config.batch_size, config.input_dim])?;

    // First linear layer: input -> hidden
    let w1_shape = &[config.hidden_dim, config.input_dim];
    let w1 = backend.normal_parameter("w1", w1_shape, seed, 0.1)?;
    let b1_shape = &[config.hidden_dim];
    let b1 = backend.normal_parameter("b1", b1_shape, seed + 1, 0.0)?;

    let hidden = ops.linear(&input_tensor, &w1, Some(&b1))?;

    // Activation function
    let activated = match config.activation {
        Activation::ReLU => ops.relu(&hidden)?,
        Activation::GELU => ops.gelu(&hidden)?,
        Activation::Sigmoid => ops.sigmoid(&hidden)?,
        Activation::Tanh => ops.tanh(&hidden)?,
    };

    // Dropout (only during training)
    let dropped = if training {
        ops.dropout(&activated, config.dropout_rate, true)?
    } else {
        activated
    };

    // Second linear layer: hidden -> output
    let w2_shape = &[config.output_dim, config.hidden_dim];
    let w2 = backend.normal_parameter("w2", w2_shape, seed + 2, 0.1)?;
    let b2_shape = &[config.output_dim];
    let b2 = backend.normal_parameter("b2", b2_shape, seed + 3, 0.0)?;

    let output = ops.linear(&dropped, &w2, Some(&b2))?;

    // Add residual connection if dimensions match
    let output = if config.input_dim == config.output_dim {
        ops.add(&output, &input_tensor)?
    } else {
        output
    };

    // Compute loss (mean squared error with zero target)
    let target = ops.zeros(&output.shape())?;
    let diff = ops.sub(&output, &target)?;
    let squared = ops.mul(&diff, &diff)?;
    let sum = ops.sum_all(&squared)?;
    let loss_values = ops.tensor_to_vec(&sum)?;
    let loss = loss_values[0] / (output.shape().iter().product() as f32);

    let output_values = ops.tensor_to_vec(&output)?;
    let output_shape = ops.shape(&output);

    Ok(MlpResult {
        output: output_values,
        output_shape,
        loss,
    })
}

/// Test that MLP forward pass is deterministic.
#[test]
fn test_mlp_determinism() {
    let config = MlpConfig::default();
    let backend = CpuBackend::default();
    let seed = 42;

    let input_size = config.batch_size * config.input_dim;
    let input: Vec<f32> = (0..input_size)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();

    let result1 = golden_mlp_forward(&backend, &config, &input, seed, false).unwrap();
    let result2 = golden_mlp_forward(&backend, &config, &input, seed, false).unwrap();

    // Check that results are identical
    assert_eq!(result1.output_shape, result2.output_shape);
    assert_eq!(result1.loss, result2.loss);

    for (i, (&v1, &v2)) in result1.output.iter().zip(result2.output.iter()).enumerate() {
        if (v1 - v2).abs() > 1e-6 {
            panic!("Determinism check failed at index {}: {} vs {}", i, v1, v2);
        }
    }
}

/// Test that MLP forward pass is numerically stable.
#[test]
fn test_mlp_numerical_stability() {
    let config = MlpConfig::default();
    let backend = CpuBackend::default();
    let seed = 42;

    // Test with different input scales
    let input_size = config.batch_size * config.input_dim;
    let scales = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0];

    for scale in scales {
        let input: Vec<f32> = (0..input_size)
            .map(|i| (i as f32 * scale).sin())
            .collect();

        let result = golden_mlp_forward(&backend, &config, &input, seed, false).unwrap();

        // Check that loss is finite
        assert!(
            result.loss.is_finite(),
            "Loss should be finite for scale {}, got {}",
            scale,
            result.loss
        );

        // Check that all outputs are finite
        for (i, &v) in result.output.iter().enumerate() {
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

/// Test different activation functions.
#[test]
fn test_mlp_activations() {
    let config = MlpConfig::default();
    let backend = CpuBackend::default();
    let seed = 42;

    let input_size = config.batch_size * config.input_dim;
    let input: Vec<f32> = (0..input_size)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();

    let activations = [
        Activation::ReLU,
        Activation::GELU,
        Activation::Sigmoid,
        Activation::Tanh,
    ];

    for activation in activations {
        let config_with_act = MlpConfig {
            activation,
            ..config.clone()
        };

        let result = golden_mlp_forward(&backend, &config_with_act, &input, seed, false).unwrap();

        // Validate output shape
        assert_eq!(result.output_shape, vec![config.batch_size, config.output_dim]);

        // Check that loss is reasonable
        assert!(result.loss > 0.0);
        assert!(result.loss < 1000.0);

        // Check that all outputs are finite
        for (i, &v) in result.output.iter().enumerate() {
            assert!(
                v.is_finite(),
                "Output should be finite at index {} for {:?}, got {}",
                i,
                activation,
                v
            );
        }

        println!(
            "Activation {:?}: loss={}, output_range=[{}, {}]",
            activation,
            result.loss,
            result.output.iter().cloned().fold(f32::INFINITY, f32::min),
            result.output.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        );
    }
}

/// Test training vs inference mode.
#[test]
fn test_mlp_training_vs_inference() {
    let config = MlpConfig::default();
    let backend = CpuBackend::default();
    let seed = 42;

    let input_size = config.batch_size * config.input_dim;
    let input: Vec<f32> = (0..input_size)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();

    // Run in inference mode (dropout disabled)
    let result_inference = golden_mlp_forward(&backend, &config, &input, seed, false).unwrap();

    // Run in training mode (dropout enabled)
    let result_training = golden_mlp_forward(&backend, &config, &input, seed, true).unwrap();

    // Outputs should differ due to dropout
    // But shapes should be the same
    assert_eq!(result_inference.output_shape, result_training.output_shape);

    // Loss should be different (dropout affects forward pass)
    assert_ne!(result_inference.loss, result_training.loss);

    println!(
        "Inference loss: {}, Training loss: {}",
        result_inference.loss, result_training.loss
    );
}

/// Test tolerance checking with MLP outputs.
#[test]
fn test_mlp_tolerance() {
    let config = MlpConfig::default();
    let backend = CpuBackend::default();
    let seed = 42;

    let input_size = config.batch_size * config.input_dim;
    let input: Vec<f32> = (0..input_size)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();

    let result1 = golden_mlp_forward(&backend, &config, &input, seed, false).unwrap();

    // Add small perturbation
    let mut input_perturbed = input.clone();
    for v in input_perturbed.iter_mut() {
        *v += 1e-6;
    }

    let result2 = golden_mlp_forward(&backend, &config, &input_perturbed, seed, false).unwrap();

    // Check that small input changes produce small output changes
    let tol = ToleranceConfig::for_family(OpFamily::MatmulLinear);
    assert!(tol.check_slice(&result1.output, &result2.output));

    // Loss should be similar
    assert!(tol.check(result1.loss, result2.loss));
}

/// Test MLP with different configurations.
#[test]
fn test_mlp_configurations() {
    let backend = CpuBackend::default();
    let seed = 42;

    let configs = vec![
        MlpConfig {
            batch_size: 2,
            input_dim: 32,
            hidden_dim: 128,
            output_dim: 32,
            dropout_rate: 0.1,
            activation: Activation::GELU,
        },
        MlpConfig {
            batch_size: 4,
            input_dim: 64,
            hidden_dim: 256,
            output_dim: 64,
            dropout_rate: 0.1,
            activation: Activation::GELU,
        },
        MlpConfig {
            batch_size: 8,
            input_dim: 128,
            hidden_dim: 512,
            output_dim: 128,
            dropout_rate: 0.1,
            activation: Activation::GELU,
        },
    ];

    for config in configs {
        let input_size = config.batch_size * config.input_dim;
        let input: Vec<f32> = (0..input_size)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();

        let result = golden_mlp_forward(&backend, &config, &input, seed, false).unwrap();

        // Validate output shape
        assert_eq!(
            result.output_shape,
            vec![config.batch_size, config.output_dim]
        );

        // Validate that loss is reasonable
        assert!(result.loss > 0.0);
        assert!(result.loss < 1000.0);

        println!(
            "Config batch={} in={} hidden={} out={} loss={}",
            config.batch_size, config.input_dim, config.hidden_dim, config.output_dim, result.loss
        );
    }
}

/// Test MLP residual connection.
#[test]
fn test_mlp_residual_connection() {
    let backend = CpuBackend::default();
    let seed = 42;

    // Config with matching input/output dimensions (residual connection)
    let config_residual = MlpConfig {
        batch_size: 4,
        input_dim: 64,
        hidden_dim: 256,
        output_dim: 64, // Same as input_dim
        dropout_rate: 0.0,
        activation: Activation::GELU,
    };

    // Config without residual connection
    let config_no_residual = MlpConfig {
        batch_size: 4,
        input_dim: 64,
        hidden_dim: 256,
        output_dim: 128, // Different from input_dim
        dropout_rate: 0.0,
        activation: Activation::GELU,
    };

    let input_size = config_residual.batch_size * config_residual.input_dim;
    let input: Vec<f32> = (0..input_size)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();

    let result_residual = golden_mlp_forward(&backend, &config_residual, &input, seed, false).unwrap();
    let result_no_residual = golden_mlp_forward(&backend, &config_no_residual, &input, seed, false).unwrap();

    // Both should succeed
    assert_eq!(
        result_residual.output_shape,
        vec![config_residual.batch_size, config_residual.output_dim]
    );
    assert_eq!(
        result_no_residual.output_shape,
        vec![config_no_residual.batch_size, config_no_residual.output_dim]
    );

    println!(
        "With residual: loss={}, Without residual: loss={}",
        result_residual.loss, result_no_residual.loss
    );
}
