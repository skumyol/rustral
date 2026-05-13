//! Normalization layers for neural networks.

use rustral_core::{
    Backend, ForwardCtx, Module, NamedParameters, Parameter, ParameterRef, Result, Trainable,
};
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
        Self { normalized_shape: normalized_shape.into(), eps: 1e-5 }
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

    /// Create a LayerNorm layer with randomly initialized weights.
    pub fn new(backend: &B, config: LayerNormConfig, seed: u64) -> Result<Self> {
        let norm_elem_count: usize = config.normalized_shape.iter().product();
        let weight = backend.normal_parameter("weight", &[norm_elem_count], seed, 1.0)?;
        let bias = backend.normal_parameter("bias", &[norm_elem_count], seed.wrapping_add(1), 0.0)?;
        Ok(Self::from_parameters(config, weight, bias))
    }

    /// Access the configuration.
    pub fn config(&self) -> &LayerNormConfig {
        &self.config
    }

    /// Learnable scale (gamma), shape `[product(normalized_shape)]`.
    pub fn weight(&self) -> &Parameter<B> {
        &self.weight
    }

    /// Learnable shift (beta), shape `[product(normalized_shape)]`.
    pub fn bias(&self) -> &Parameter<B> {
        &self.bias
    }
}

impl<B: Backend> NamedParameters<B> for LayerNorm<B> {
    fn visit_parameters(&self, f: &mut dyn FnMut(&str, &Parameter<B>)) {
        f("weight", &self.weight);
        f("bias", &self.bias);
    }

    fn visit_parameters_mut(&mut self, f: &mut dyn FnMut(&str, &mut Parameter<B>)) {
        f("weight", &mut self.weight);
        f("bias", &mut self.bias);
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
            return Err(rustral_core::CoreError::InvalidShape {
                shape: input_shape,
                reason: format!("LayerNorm input rank must be >= normalized_shape rank ({})", ndim),
            });
        }

        // Standard layer_norm in trait handles normalization over the last dimension.
        // If we have multi-dim normalized_shape, we reshape to [prefix, product(normalized_shape)]
        let norm_elem_count: usize = self.config.normalized_shape.iter().product();
        if norm_elem_count == 0 {
            return Ok(input);
        }

        let prefix_len = input_shape.len() - ndim;
        let prefix_shape = &input_shape[..prefix_len];
        let prefix_prod: usize = prefix_shape.iter().product();

        // Reshape to [prefix_prod, norm_elem_count]
        let flattened = ops.reshape(&input, &[prefix_prod, norm_elem_count])?;

        let output = ops.layer_norm(&flattened, self.weight.tensor(), self.bias.tensor(), self.config.eps)?;

        // Reshape back to original shape
        ops.reshape(&output, &input_shape)
    }
}
impl<B: Backend> Trainable<B> for LayerNorm<B> {
    fn parameters(&self) -> Vec<ParameterRef> {
        vec![ParameterRef { id: self.weight.id() }, ParameterRef { id: self.bias.id() }]
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
        Self { num_features, eps: 1e-5, momentum: 0.1 }
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
            return Err(rustral_core::CoreError::InvalidShape {
                shape: input_shape.clone(),
                reason: "BatchNorm expects 2D [N,C] or 4D [N,C,H,W] input".into(),
            });
        }

        let num_features = self.config.num_features;
        let channels = input_shape[1];

        if channels != num_features {
            return Err(rustral_core::CoreError::ShapeMismatch {
                expected: vec![num_features],
                actual: vec![channels],
            });
        }

        if input_shape.len() == 2 {
            // [N, C] -> Normalize over dim 0
            let mean = ops.mean_dim(&input, 0, true)?; // [1, C]
            let var = ops.var_dim(&input, 0, false, true)?; // [1, C]

            let x_centered = ops.sub(&input, &ops.broadcast_to(&mean, &input_shape)?)?;
            let std = ops.sqrt(&ops.add_scalar(&var, self.config.eps)?)?;
            let x_hat = ops.div(&x_centered, &ops.broadcast_to(&std, &input_shape)?)?;

            // Apply learnable parameters: x_hat * weight + bias
            let weight_b = ops.broadcast_to(self.weight.tensor(), &input_shape)?;
            let bias_b = ops.broadcast_to(self.bias.tensor(), &input_shape)?;

            let y = ops.mul(&x_hat, &weight_b)?;
            ops.add(&y, &bias_b)
        } else {
            // [N, C, H, W]
            // We want to normalize over dimensions 0, 2, 3.
            // Correct multi-dim variance: Var(X) = E[X^2] - (E[X])^2

            // mean = E[X]
            let m0 = ops.mean_dim(&input, 0, true)?; // [1, C, H, W]
            let m02 = ops.mean_dim(&m0, 2, true)?; // [1, C, 1, W]
            let mean = ops.mean_dim(&m02, 3, true)?; // [1, C, 1, 1]

            // e_x2 = E[X^2]
            let x2 = ops.mul(&input, &input)?;
            let e0 = ops.mean_dim(&x2, 0, true)?;
            let e02 = ops.mean_dim(&e0, 2, true)?;
            let e_x2 = ops.mean_dim(&e02, 3, true)?; // [1, C, 1, 1]

            let mean2 = ops.mul(&mean, &mean)?;
            let var = ops.sub(&e_x2, &mean2)?; // [1, C, 1, 1]

            // Now normalize
            let mean_b = ops.broadcast_to(&mean, &input_shape)?;
            let x_centered = ops.sub(&input, &mean_b)?;
            let var_eps = ops.add_scalar(&var, self.config.eps)?;
            let std = ops.sqrt(&var_eps)?;
            let std_b = ops.broadcast_to(&std, &input_shape)?;
            let x_hat = ops.div(&x_centered, &std_b)?;

            // Apply weight and bias
            // weight is [C]. Reshape to [1, C, 1, 1] then broadcast.
            let weight_4d = ops.reshape(self.weight.tensor(), &[1, channels, 1, 1])?;
            let bias_4d = ops.reshape(self.bias.tensor(), &[1, channels, 1, 1])?;

            let weight_b = ops.broadcast_to(&weight_4d, &input_shape)?;
            let bias_b = ops.broadcast_to(&bias_4d, &input_shape)?;

            let y = ops.mul(&x_hat, &weight_b)?;
            ops.add(&y, &bias_b)
        }
    }
}
impl<B: Backend> Trainable<B> for BatchNorm<B> {
    fn parameters(&self) -> Vec<ParameterRef> {
        vec![ParameterRef { id: self.weight.id() }, ParameterRef { id: self.bias.id() }]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustral_core::Parameter;
    use rustral_ndarray_backend::CpuBackend;

    #[test]
    fn test_layer_norm_config() {
        let config = LayerNormConfig::new(vec![64, 64]).with_eps(1e-6);
        assert_eq!(config.normalized_shape, vec![64, 64]);
        assert_eq!(config.eps, 1e-6);
    }

    #[test]
    fn test_layer_norm_parameters() {
        let backend = CpuBackend::default();
        let weight: Parameter<CpuBackend> =
            Parameter::new("gamma", backend.tensor_from_vec(vec![1.0; 64], &[64]).unwrap());
        let bias: Parameter<CpuBackend> =
            Parameter::new("beta", backend.tensor_from_vec(vec![0.0; 64], &[64]).unwrap());

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
        let weight: Parameter<CpuBackend> =
            Parameter::new("gamma", backend.tensor_from_vec(vec![1.0; 32], &[32]).unwrap());
        let bias: Parameter<CpuBackend> =
            Parameter::new("beta", backend.tensor_from_vec(vec![0.0; 32], &[32]).unwrap());

        let config = BatchNormConfig::new(32);
        let bn = BatchNorm::from_parameters(config, weight, bias);

        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_layer_norm_forward() {
        let backend = CpuBackend::default();
        let mut ctx = rustral_core::ForwardCtx::new(&backend, rustral_core::Mode::Inference);

        let weight: Parameter<CpuBackend> =
            Parameter::new("gamma", backend.tensor_from_vec(vec![1.0; 4], &[4]).unwrap());
        let bias: Parameter<CpuBackend> =
            Parameter::new("beta", backend.tensor_from_vec(vec![0.0; 4], &[4]).unwrap());

        let config = LayerNormConfig::new(vec![4]);
        let ln = LayerNorm::from_parameters(config, weight, bias);

        // Input: 2 groups of 4 values each
        let input = backend.tensor_from_vec(vec![1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0], &[2, 4]).unwrap();
        let output = ln.forward(input, &mut ctx).unwrap();

        let shape = backend.ops().shape(&output);
        assert_eq!(shape, &[2, 4]);

        // With gamma=1, beta=0, mean of first group is 2.5, second group is 5.0
        // Values should be roughly centered around 0 for each group
        let values: Vec<f32> = (0..8).filter_map(|i| backend.ops().tensor_element(&output, i).ok()).collect();

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
        let mut ctx = rustral_core::ForwardCtx::new(&backend, rustral_core::Mode::Inference);

        let weight: Parameter<CpuBackend> =
            Parameter::new("gamma", backend.tensor_from_vec(vec![1.0; 2], &[2]).unwrap());
        let bias: Parameter<CpuBackend> =
            Parameter::new("beta", backend.tensor_from_vec(vec![0.0; 2], &[2]).unwrap());

        let config = BatchNormConfig::new(2);
        let bn = BatchNorm::from_parameters(config, weight, bias);

        // Input: batch=2, channels=2
        let input = backend.tensor_from_vec(vec![1.0, 10.0, 2.0, 12.0], &[2, 2]).unwrap();
        let output = bn.forward(input, &mut ctx).unwrap();

        let shape = backend.ops().shape(&output);
        assert_eq!(shape, &[2, 2]);

        // Both output values per channel should be present
        let values: Vec<f32> = (0..4).filter_map(|i| backend.ops().tensor_element(&output, i).ok()).collect();
        assert_eq!(values.len(), 4);
    }

    #[test]
    fn test_layer_norm_config_accessor() {
        let backend = CpuBackend::default();
        let weight: Parameter<CpuBackend> =
            Parameter::new("gamma", backend.tensor_from_vec(vec![1.0; 4], &[4]).unwrap());
        let bias: Parameter<CpuBackend> =
            Parameter::new("beta", backend.tensor_from_vec(vec![0.0; 4], &[4]).unwrap());
        let config = LayerNormConfig::new(vec![4]);
        let ln = LayerNorm::from_parameters(config, weight, bias);
        assert_eq!(ln.config().normalized_shape, vec![4]);
    }

    #[test]
    fn test_layer_norm_forward_invalid_rank() {
        let backend = CpuBackend::default();
        let mut ctx = rustral_core::ForwardCtx::new(&backend, rustral_core::Mode::Inference);
        let weight: Parameter<CpuBackend> =
            Parameter::new("gamma", backend.tensor_from_vec(vec![1.0; 4], &[4]).unwrap());
        let bias: Parameter<CpuBackend> =
            Parameter::new("beta", backend.tensor_from_vec(vec![0.0; 4], &[4]).unwrap());
        let config = LayerNormConfig::new(vec![4, 4]);
        let ln = LayerNorm::from_parameters(config, weight, bias);

        let input = backend.tensor_from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap(); // rank 1 < 2
        let result = ln.forward(input, &mut ctx);
        assert!(result.is_err());
    }

    #[test]
    fn test_layer_norm_forward_empty_norm_shape() {
        let backend = CpuBackend::default();
        let mut ctx = rustral_core::ForwardCtx::new(&backend, rustral_core::Mode::Inference);
        let weight: Parameter<CpuBackend> =
            Parameter::new("gamma", backend.tensor_from_vec(vec![1.0; 1], &[1]).unwrap());
        let bias: Parameter<CpuBackend> =
            Parameter::new("beta", backend.tensor_from_vec(vec![0.0; 1], &[1]).unwrap());
        let config = LayerNormConfig::new(vec![]);
        let ln = LayerNorm::from_parameters(config, weight, bias);

        let input = backend.tensor_from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let output = ln.forward(input, &mut ctx).unwrap();
        // norm_elem_count == 0 => return input unchanged
        let shape = backend.ops().shape(&output);
        assert_eq!(shape, &[4]);
    }

    #[test]
    fn test_batch_norm_config_accessor() {
        let backend = CpuBackend::default();
        let weight: Parameter<CpuBackend> =
            Parameter::new("gamma", backend.tensor_from_vec(vec![1.0; 2], &[2]).unwrap());
        let bias: Parameter<CpuBackend> =
            Parameter::new("beta", backend.tensor_from_vec(vec![0.0; 2], &[2]).unwrap());
        let config = BatchNormConfig::new(2);
        let bn = BatchNorm::from_parameters(config, weight, bias);
        assert_eq!(bn.config().num_features, 2);
    }

    #[test]
    fn test_batch_norm_forward_invalid_shape() {
        let backend = CpuBackend::default();
        let mut ctx = rustral_core::ForwardCtx::new(&backend, rustral_core::Mode::Inference);
        let weight: Parameter<CpuBackend> =
            Parameter::new("gamma", backend.tensor_from_vec(vec![1.0; 2], &[2]).unwrap());
        let bias: Parameter<CpuBackend> =
            Parameter::new("beta", backend.tensor_from_vec(vec![0.0; 2], &[2]).unwrap());
        let config = BatchNormConfig::new(2);
        let bn = BatchNorm::from_parameters(config, weight, bias);

        let input = backend.tensor_from_vec(vec![1.0f32; 10], &[2, 5]).unwrap(); // 2D but channels=5
        let result = bn.forward(input, &mut ctx);
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_norm_forward_4d() {
        let backend = CpuBackend::default();
        let mut ctx = rustral_core::ForwardCtx::new(&backend, rustral_core::Mode::Inference);
        let weight: Parameter<CpuBackend> =
            Parameter::new("gamma", backend.tensor_from_vec(vec![1.0; 2], &[2]).unwrap());
        let bias: Parameter<CpuBackend> =
            Parameter::new("beta", backend.tensor_from_vec(vec![0.0; 2], &[2]).unwrap());
        let config = BatchNormConfig::new(2);
        let bn = BatchNorm::from_parameters(config, weight, bias);

        // Input: [N=1, C=2, H=2, W=2]
        let input =
            backend.tensor_from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 2, 2, 2]).unwrap();
        let output = bn.forward(input, &mut ctx).unwrap();
        let shape = backend.ops().shape(&output);
        assert_eq!(shape, &[1, 2, 2, 2]);
    }

    #[test]
    fn test_batch_norm_channel_mismatch() {
        let backend = CpuBackend::default();
        let mut ctx = rustral_core::ForwardCtx::new(&backend, rustral_core::Mode::Inference);
        let weight: Parameter<CpuBackend> =
            Parameter::new("gamma", backend.tensor_from_vec(vec![1.0; 2], &[2]).unwrap());
        let bias: Parameter<CpuBackend> =
            Parameter::new("beta", backend.tensor_from_vec(vec![0.0; 2], &[2]).unwrap());
        let config = BatchNormConfig::new(2);
        let bn = BatchNorm::from_parameters(config, weight, bias);

        let input = backend.tensor_from_vec(vec![1.0f32; 6], &[2, 3]).unwrap(); // channels=3 != 2
        let result = bn.forward(input, &mut ctx);
        assert!(result.is_err());
    }
}
