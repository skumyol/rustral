//! Normalization layers for neural networks.

use mnr_core::{Backend, ForwardCtx, Module, Parameter, ParameterRef, Result, ShapeExt, Trainable};
use serde::{Deserialize, Serialize};

/// Configuration for layer normalization.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerNormConfig {
    /// Feature dimension to normalize over.
    pub normalized_shape: Vec<usize>,
    /// Epsilon for numerical stability.
    pub eps: f32,
}

impl LayerNormConfig {
    /// Create a new LayerNorm configuration.
    pub fn new(normalized_shape: impl Into<Vec<usize>>) -> Self {
        Self {
            normalized_shape: normalized_shape.into(),
            eps: 1e-5,
        }
    }

    /// Set epsilon value.
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }
}

/// Layer normalization layer.
///
/// Normalizes across the last D dimensions where D is the length of normalized_shape.
pub struct LayerNorm<B: Backend> {
    config: LayerNormConfig,
    /// Learnable scale parameter (gamma).
    weight: Parameter<B>,
    /// Learnable shift parameter (beta).
    bias: Parameter<B>,
}

impl<B: Backend> LayerNorm<B> {
    /// Create a LayerNorm layer from explicit parameters.
    pub fn from_parameters(config: LayerNormConfig, weight: Parameter<B>, bias: Parameter<B>) -> Self {
        Self { config, weight, bias }
    }

    /// Access the configuration.
    pub fn config(&self) -> &LayerNormConfig {
        &self.config
    }
}

impl<B: Backend> Module<B> for LayerNorm<B> {
    type Input = B::Tensor;
    type Output = B::Tensor;

    fn forward(&self, input: Self::Input, ctx: &mut ForwardCtx<B>) -> Result<Self::Output> {
        let ops = ctx.backend().ops();
        let input_shape = ops.shape(&input);

        // Layer normalization normalizes over the last D dimensions
        // where D is the length of normalized_shape
        let ndim = self.config.normalized_shape.len();
        if input_shape.len() < ndim {
            return Err(mnr_core::CoreError::InvalidShape {
                shape: input_shape,
                reason: format!("LayerNorm input rank must be >= normalized_shape rank ({})", ndim),
            });
        }

        // Calculate the number of elements to normalize over
        let norm_elem_count: usize = self.config.normalized_shape.iter().product();
        if norm_elem_count == 0 {
            return Ok(input);
        }

        // For each normalization group, compute mean and variance
        // and apply: (x - mean) / sqrt(var + eps) * gamma + beta
        let total_elems = input_shape.elem_count();
        let num_groups = total_elems / norm_elem_count;

        // Extract input values
        let input_values: Vec<f32> = (0..total_elems)
            .filter_map(|i| ops.tensor_element(&input, i).ok())
            .collect();

        if input_values.len() != total_elems {
            return Ok(input);
        }

        // Get gamma and beta parameters
        let gamma_values: Vec<f32> = (0..norm_elem_count)
            .filter_map(|i| ops.tensor_element(self.weight.tensor(), i).ok())
            .collect();
        let beta_values: Vec<f32> = (0..norm_elem_count)
            .filter_map(|i| ops.tensor_element(self.bias.tensor(), i).ok())
            .collect();

        let mut output_values = vec![0.0f32; total_elems];
        let eps = self.config.eps;

        for g in 0..num_groups {
            let group_start = g * norm_elem_count;

            // Compute mean
            let sum: f32 = input_values[group_start..group_start + norm_elem_count].iter().sum();
            let mean = sum / norm_elem_count as f32;

            // Compute variance
            let var_sum: f32 = input_values[group_start..group_start + norm_elem_count]
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum();
            let var = var_sum / norm_elem_count as f32;
            let std = (var + eps).sqrt();

            // Normalize and apply affine transform
            for i in 0..norm_elem_count {
                let idx = group_start + i;
                let normalized = (input_values[idx] - mean) / std;
                let gamma = gamma_values.get(i).copied().unwrap_or(1.0);
                let beta = beta_values.get(i).copied().unwrap_or(0.0);
                output_values[idx] = normalized * gamma + beta;
            }
        }

        ops.tensor_from_vec(output_values, &input_shape)
    }
}

impl<B: Backend> Trainable<B> for LayerNorm<B> {
    fn parameters(&self) -> Vec<ParameterRef> {
        vec![
            ParameterRef { id: self.weight.id() },
            ParameterRef { id: self.bias.id() },
        ]
    }
}

/// Configuration for batch normalization.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchNormConfig {
    /// Number of features/channels.
    pub num_features: usize,
    /// Epsilon for numerical stability.
    pub eps: f32,
    /// Momentum for running statistics.
    pub momentum: f32,
}

impl BatchNormConfig {
    /// Create a new BatchNorm configuration.
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            eps: 1e-5,
            momentum: 0.1,
        }
    }
}

/// Batch normalization layer.
pub struct BatchNorm<B: Backend> {
    config: BatchNormConfig,
    /// Learnable scale parameter (gamma).
    weight: Parameter<B>,
    /// Learnable shift parameter (beta).
    bias: Parameter<B>,
}

impl<B: Backend> BatchNorm<B> {
    /// Create a BatchNorm layer from explicit parameters.
    pub fn from_parameters(config: BatchNormConfig, weight: Parameter<B>, bias: Parameter<B>) -> Self {
        Self { config, weight, bias }
    }

    /// Access the configuration.
    pub fn config(&self) -> &BatchNormConfig {
        &self.config
    }
}

impl<B: Backend> Module<B> for BatchNorm<B> {
    type Input = B::Tensor;
    type Output = B::Tensor;

    fn forward(&self, input: Self::Input, ctx: &mut ForwardCtx<B>) -> Result<Self::Output> {
        let ops = ctx.backend().ops();
        let input_shape = ops.shape(&input);

        // BatchNorm normalizes per-channel across the batch and spatial dimensions
        // Expected input: [N, C, H, W] for 4D or [N, C] for 2D
        if input_shape.len() != 2 && input_shape.len() != 4 {
            return Err(mnr_core::CoreError::InvalidShape {
                shape: input_shape.clone(),
                reason: "BatchNorm expects 2D [N,C] or 4D [N,C,H,W] input".into(),
            });
        }

        let num_features = self.config.num_features;
        let (batch, channels, spatial) = if input_shape.len() == 2 {
            (input_shape[0], input_shape[1], 1)
        } else {
            (input_shape[0], input_shape[1], input_shape[2] * input_shape[3])
        };

        if channels != num_features {
            return Err(mnr_core::CoreError::ShapeMismatch {
                expected: vec![num_features],
                actual: vec![channels],
            });
        }

        let total_elems = input_shape.elem_count();
        let elems_per_channel = batch * spatial;

        // Extract input values
        let input_values: Vec<f32> = (0..total_elems)
            .filter_map(|i| ops.tensor_element(&input, i).ok())
            .collect();

        if input_values.len() != total_elems {
            return Ok(input);
        }

        // Get gamma and beta parameters
        let gamma_values: Vec<f32> = (0..num_features)
            .filter_map(|i| ops.tensor_element(self.weight.tensor(), i).ok())
            .collect();
        let beta_values: Vec<f32> = (0..num_features)
            .filter_map(|i| ops.tensor_element(self.bias.tensor(), i).ok())
            .collect();

        let mut output_values = vec![0.0f32; total_elems];
        let eps = self.config.eps;

        // Process each channel independently
        for c in 0..channels {
            // Collect all values for this channel across batch and spatial dims
            let mut channel_values = Vec::with_capacity(elems_per_channel);
            for n in 0..batch {
                for s in 0..spatial {
                    let idx = if input_shape.len() == 2 {
                        n * channels + c
                    } else {
                        let h = input_shape[2];
                        let w = input_shape[3];
                        n * channels * h * w + c * h * w + s
                    };
                    if idx < input_values.len() {
                        channel_values.push(input_values[idx]);
                    }
                }
            }

            // Compute mean and variance for this channel
            let sum: f32 = channel_values.iter().sum();
            let mean = sum / channel_values.len() as f32;

            let var_sum: f32 = channel_values.iter().map(|&x| (x - mean).powi(2)).sum();
            let var = var_sum / channel_values.len() as f32;
            let std = (var + eps).sqrt();

            // Normalize and apply affine transform
            let gamma = gamma_values.get(c).copied().unwrap_or(1.0);
            let beta = beta_values.get(c).copied().unwrap_or(0.0);

            for n in 0..batch {
                for s in 0..spatial {
                    let input_idx = if input_shape.len() == 2 {
                        n * channels + c
                    } else {
                        let h = input_shape[2];
                        let w = input_shape[3];
                        n * channels * h * w + c * h * w + s
                    };

                    if input_idx < input_values.len() {
                        let normalized = (input_values[input_idx] - mean) / std;
                        output_values[input_idx] = normalized * gamma + beta;
                    }
                }
            }
        }

        ops.tensor_from_vec(output_values, &input_shape)
    }
}

impl<B: Backend> Trainable<B> for BatchNorm<B> {
    fn parameters(&self) -> Vec<ParameterRef> {
        vec![
            ParameterRef { id: self.weight.id() },
            ParameterRef { id: self.bias.id() },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnr_core::Parameter;
    use mnr_ndarray_backend::CpuBackend;

    #[test]
    fn test_layer_norm_config() {
        let config = LayerNormConfig::new(vec![64, 64]).with_eps(1e-6);
        assert_eq!(config.normalized_shape, vec![64, 64]);
        assert_eq!(config.eps, 1e-6);
    }

    #[test]
    fn test_layer_norm_parameters() {
        let backend = CpuBackend::default();
        let weight: Parameter<CpuBackend> = Parameter::new("gamma", backend.tensor_from_vec(vec![1.0; 64], &[64]).unwrap());
        let bias: Parameter<CpuBackend> = Parameter::new("beta", backend.tensor_from_vec(vec![0.0; 64], &[64]).unwrap());

        let config = LayerNormConfig::new(vec![64]);
        let ln = LayerNorm::from_parameters(config, weight, bias);

        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_batch_norm_config() {
        let config = BatchNormConfig::new(128);
        assert_eq!(config.num_features, 128);
        assert_eq!(config.eps, 1e-5);
        assert_eq!(config.momentum, 0.1);
    }

    #[test]
    fn test_batch_norm_parameters() {
        let backend = CpuBackend::default();
        let weight: Parameter<CpuBackend> = Parameter::new("gamma", backend.tensor_from_vec(vec![1.0; 32], &[32]).unwrap());
        let bias: Parameter<CpuBackend> = Parameter::new("beta", backend.tensor_from_vec(vec![0.0; 32], &[32]).unwrap());

        let config = BatchNormConfig::new(32);
        let bn = BatchNorm::from_parameters(config, weight, bias);

        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_layer_norm_forward() {
        let backend = CpuBackend::default();
        let mut ctx = mnr_core::ForwardCtx::new(&backend, mnr_core::Mode::Inference);

        let weight: Parameter<CpuBackend> = Parameter::new("gamma", backend.tensor_from_vec(vec![1.0; 4], &[4]).unwrap());
        let bias: Parameter<CpuBackend> = Parameter::new("beta", backend.tensor_from_vec(vec![0.0; 4], &[4]).unwrap());

        let config = LayerNormConfig::new(vec![4]);
        let ln = LayerNorm::from_parameters(config, weight, bias);

        // Input: 2 groups of 4 values each
        let input = backend.tensor_from_vec(vec![1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0], &[2, 4]).unwrap();
        let output = ln.forward(input, &mut ctx).unwrap();

        let shape = backend.ops().shape(&output);
        assert_eq!(shape, &[2, 4]);

        // With gamma=1, beta=0, mean of first group is 2.5, second group is 5.0
        // Values should be roughly centered around 0 for each group
        let values: Vec<f32> = (0..8)
            .filter_map(|i| backend.ops().tensor_element(&output, i).ok())
            .collect();

        // First group: [-1.34, -0.45, 0.45, 1.34] approx
        assert!(values[0] < values[1]);
        assert!(values[1] < values[2]);
        assert!(values[2] < values[3]);

        // Second group should have same relative pattern
        assert!(values[4] < values[5]);
        assert!(values[5] < values[6]);
        assert!(values[6] < values[7]);
    }

    #[test]
    fn test_batch_norm_forward() {
        let backend = CpuBackend::default();
        let mut ctx = mnr_core::ForwardCtx::new(&backend, mnr_core::Mode::Inference);

        let weight: Parameter<CpuBackend> = Parameter::new("gamma", backend.tensor_from_vec(vec![1.0; 2], &[2]).unwrap());
        let bias: Parameter<CpuBackend> = Parameter::new("beta", backend.tensor_from_vec(vec![0.0; 2], &[2]).unwrap());

        let config = BatchNormConfig::new(2);
        let bn = BatchNorm::from_parameters(config, weight, bias);

        // Input: batch=2, channels=2
        let input = backend.tensor_from_vec(vec![1.0, 10.0, 2.0, 12.0], &[2, 2]).unwrap();
        let output = bn.forward(input, &mut ctx).unwrap();

        let shape = backend.ops().shape(&output);
        assert_eq!(shape, &[2, 2]);

        // Both output values per channel should be present
        let values: Vec<f32> = (0..4)
            .filter_map(|i| backend.ops().tensor_element(&output, i).ok())
            .collect();
        assert_eq!(values.len(), 4);
    }
}
