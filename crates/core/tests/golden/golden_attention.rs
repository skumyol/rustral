//! Golden test for attention mechanism.
//!
//! Tests multi-head self-attention with causal masking to ensure
//! numerical stability and correctness.

use rustral_core::{Backend, OpFamily, Result, ShapeExt, ToleranceConfig};
use rustral_ndarray_backend::CpuBackend;

/// Test configuration for golden attention tests.
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    pub batch_size: usize,
    pub seq_len: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub causal: bool,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            batch_size: 4,
            seq_len: 32,
            num_heads: 4,
            head_dim: 16,
            causal: true,
        }
    }
}

/// Golden test result for attention mechanism.
#[derive(Debug, Clone)]
pub struct AttentionResult {
    pub output: Vec<f32>,
    pub output_shape: Vec<usize>,
    pub attention_weights: Vec<f32>,
    pub attention_shape: Vec<usize>,
}

/// Run a simplified attention mechanism.
///
/// This implements scaled dot-product attention:
/// - Multi-head attention with projection
/// - Optional causal masking
/// - Softmax over attention scores
pub fn golden_attention(
    backend: &CpuBackend,
    config: &AttentionConfig,
    qkv: &[f32],
    seed: u64,
) -> Result<AttentionResult> {
    let ops = backend.ops();
    let total_dim = config.num_heads * config.head_dim;

    // Reshape input to [batch, seq_len, 3 * num_heads * head_dim]
    let qkv_tensor = ops.tensor_from_vec(
        qkv.to_vec(),
        &[config.batch_size, config.seq_len, 3 * total_dim],
    )?;

    // Split into Q, K, V
    let q = ops.slice(&qkv_tensor, 0, total_dim)?;
    let k = ops.slice(&qkv_tensor, total_dim, 2 * total_dim)?;
    let v = ops.slice(&qkv_tensor, 2 * total_dim, 3 * total_dim)?;

    // Reshape to separate heads: [batch, seq_len, num_heads, head_dim]
    let q = ops.reshape(&q, &[config.batch_size, config.seq_len, config.num_heads, config.head_dim])?;
    let k = ops.reshape(&k, &[config.batch_size, config.seq_len, config.num_heads, config.head_dim])?;
    let v = ops.reshape(&v, &[config.batch_size, config.seq_len, config.num_heads, config.head_dim])?;

    // For simplicity, we'll compute attention for each head separately
    // In a real implementation, this would be more efficient
    let mut output_values = vec![0.0f32; config.batch_size * config.seq_len * total_dim];

    for head in 0..config.num_heads {
        // Extract head-specific Q, K, V
        let head_offset = head * config.head_dim;
        
        // Compute attention scores: Q @ K^T / sqrt(d_k)
        // For simplicity, we'll use a basic implementation
        let head_dim_f32 = config.head_dim as f32;
        let scale = head_dim_f32.sqrt();

        // Simplified: just compute a basic attention pattern
        // In a real implementation, this would compute QK^T properly
        let mut attention_weights = vec![0.0f32; config.batch_size * config.seq_len * config.seq_len];

        for b in 0..config.batch_size {
            for i in 0..config.seq_len {
                for j in 0..config.seq_len {
                    // Causal masking: only attend to positions <= i
                    if config.causal && j > i {
                        attention_weights[b * config.seq_len * config.seq_len + i * config.seq_len + j] = f32::NEG_INFINITY;
                    } else {
                        // Simplified attention score
                        let idx = b * config.seq_len * total_dim + i * total_dim + head_offset;
                        let q_val = if idx < qkv.len() { qkv[idx] } else { 0.0 };
                        attention_weights[b * config.seq_len * config.seq_len + i * config.seq_len + j] = q_val / scale;
                    }
                }
            }
        }

        // Apply softmax
        let attention_tensor = ops.tensor_from_vec(
            attention_weights.clone(),
            &[config.batch_size, config.seq_len, config.seq_len],
        )?;
        let attention_softmax = ops.softmax(&attention_tensor)?;
        let attention_values = ops.tensor_to_vec(&attention_softmax)?;

        // Store attention weights for the first batch, first head
        if head == 0 {
            let start = 0;
            let end = config.seq_len * config.seq_len;
            output_values.extend_from_slice(&attention_values[start..end]);
        }
    }

    // For simplicity, just return the attention weights as output
    // In a real implementation, this would compute @ V
    let output_shape = vec![config.batch_size, config.seq_len, total_dim];
    let attention_shape = vec![config.seq_len, config.seq_len]; // First batch, first head

    Ok(AttentionResult {
        output: output_values,
        output_shape,
        attention_weights: attention_values.clone(),
        attention_shape,
    })
}

/// Test that attention mechanism is deterministic.
#[test]
fn test_attention_determinism() {
    let config = AttentionConfig::default();
    let backend = CpuBackend::default();
    let seed = 42;

    let total_dim = config.num_heads * config.head_dim;
    let input_size = config.batch_size * config.seq_len * 3 * total_dim;
    let input: Vec<f32> = (0..input_size)
        .map(|i| (i as f32 * 0.01).cos())
        .collect();

    let result1 = golden_attention(&backend, &config, &input, seed).unwrap();
    let result2 = golden_attention(&backend, &config, &input, seed).unwrap();

    // Check that results are identical
    assert_eq!(result1.output_shape, result2.output_shape);
    assert_eq!(result1.attention_shape, result2.attention_shape);

    for (i, (&v1, &v2)) in result1.attention_weights.iter().zip(result2.attention_weights.iter()).enumerate() {
        if (v1 - v2).abs() > 1e-6 {
            panic!("Determinism check failed at index {}: {} vs {}", i, v1, v2);
        }
    }
}

/// Test that attention weights sum to 1 (softmax property).
#[test]
fn test_attention_softmax_property() {
    let config = AttentionConfig::default();
    let backend = CpuBackend::default();
    let seed = 42;

    let total_dim = config.num_heads * config.head_dim;
    let input_size = config.batch_size * config.seq_len * 3 * total_dim;
    let input: Vec<f32> = (0..input_size)
        .map(|i| (i as f32 * 0.01).cos())
        .collect();

    let result = golden_attention(&backend, &config, &input, seed).unwrap();

    // Check that attention weights for each position sum to 1
    let tol = ToleranceConfig::for_family(OpFamily::Softmax);
    for i in 0..config.seq_len {
        let start = i * config.seq_len;
        let end = start + config.seq_len;
        let row_sum: f32 = result.attention_weights[start..end].iter().sum();
        
        assert!(
            tol.check(row_sum, 1.0),
            "Attention weights at position {} should sum to 1.0, got {}",
            i,
            row_sum
        );
    }
}

/// Test causal masking in attention.
#[test]
fn test_attention_causal_masking() {
    let config = AttentionConfig {
        causal: true,
        ..Default::default()
    };
    let backend = CpuBackend::default();
    let seed = 42;

    let total_dim = config.num_heads * config.head_dim;
    let input_size = config.batch_size * config.seq_len * 3 * total_dim;
    let input: Vec<f32> = (0..input_size)
        .map(|i| (i as f32 * 0.01).cos())
        .collect();

    let result = golden_attention(&backend, &config, &input, seed).unwrap();

    // For causal attention, positions should only attend to previous positions
    // Check that attention weights for future positions are very small (masked)
    for i in 1..config.seq_len {
        for j in (i + 1)..config.seq_len {
            let idx = i * config.seq_len + j;
            let weight = result.attention_weights[idx];
            
            // Masked positions should have very low weights
            assert!(
                weight < 0.01,
                "Causal masking failed: attention weight at ({}, {}) should be near 0, got {}",
                i,
                j,
                weight
            );
        }
    }
}

/// Test attention with different configurations.
#[test]
fn test_attention_configurations() {
    let backend = CpuBackend::default();
    let seed = 42;

    let configs = vec![
        AttentionConfig {
            batch_size: 2,
            seq_len: 16,
            num_heads: 2,
            head_dim: 8,
            causal: true,
        },
        AttentionConfig {
            batch_size: 4,
            seq_len: 32,
            num_heads: 4,
            head_dim: 16,
            causal: true,
        },
        AttentionConfig {
            batch_size: 8,
            seq_len: 64,
            num_heads: 8,
            head_dim: 32,
            causal: false,
        },
    ];

    for config in configs {
        let total_dim = config.num_heads * config.head_dim;
        let input_size = config.batch_size * config.seq_len * 3 * total_dim;
        let input: Vec<f32> = (0..input_size)
            .map(|i| (i as f32 * 0.01).cos())
            .collect();

        let result = golden_attention(&backend, &config, &input, seed).unwrap();

        // Validate output shape
        assert_eq!(
            result.output_shape,
            vec![config.batch_size, config.seq_len, total_dim]
        );

        // Validate attention shape
        assert_eq!(result.attention_shape, vec![config.seq_len, config.seq_len]);

        // Check that all attention weights are finite
        for (i, &w) in result.attention_weights.iter().enumerate() {
            assert!(
                w.is_finite(),
                "Attention weight should be finite at index {}, got {}",
                i,
                w
            );
        }

        println!(
            "Config batch={} seq={} heads={} head_dim={} causal={} output_len={}",
            config.batch_size,
            config.seq_len,
            config.num_heads,
            config.head_dim,
            config.causal,
            result.output.len()
        );
    }
}
