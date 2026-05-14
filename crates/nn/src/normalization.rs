//! Normalization layers for neural networks.

use rustral_core::{
    Backend, ForwardCtx, Module, NamedParameters, Parameter, ParameterRef, Result, Trainable,
};
use serde::{Deserialize, Serialize};

/// Configuration for layer normalization.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerNormConfig {
    pub normalized_shape: Vec<usize>,
    pub eps: f32,
}

impl LayerNormConfig {
    pub fn new(normalized_shape: impl Into<Vec<usize>>) -> Self {
        Self { normalized_shape: normalized_shape.into(), eps: 1e-5 }
    }
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }
}

pub struct LayerNorm<B: Backend> {
    config: LayerNormConfig,
    weight: Parameter<B>,
    bias: Parameter<B>,
}

impl<B: Backend> LayerNorm<B> {
    pub fn from_parameters(config: LayerNormConfig, weight: Parameter<B>, bias: Parameter<B>) -> Self {
        Self { config, weight, bias }
    }
    pub fn new(backend: &B, config: LayerNormConfig, seed: u64) -> Result<Self> {
        let norm_elem_count: usize = config.normalized_shape.iter().product();
        let weight = backend.normal_parameter("weight", &[norm_elem_count], seed, 1.0)?;
        let bias = backend.normal_parameter("bias", &[norm_elem_count], seed.wrapping_add(1), 0.0)?;
        Ok(Self::from_parameters(config, weight, bias))
    }
    pub fn config(&self) -> &LayerNormConfig {
        &self.config
    }
    pub fn weight(&self) -> &Parameter<B> {
        &self.weight
    }
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
        let ndim = self.config.normalized_shape.len();
        if input_shape.len() < ndim {
            return Err(rustral_core::CoreError::InvalidShape {
                shape: input_shape,
                reason: format!("LayerNorm input rank must be >= normalized_shape rank ({})", ndim),
            });
        }
        let norm_elem_count: usize = self.config.normalized_shape.iter().product();
        if norm_elem_count == 0 {
            return Ok(input);
        }
        let prefix_len = input_shape.len() - ndim;
        let prefix_prod: usize = input_shape[..prefix_len].iter().product();
        let flattened = ops.reshape(&input, &[prefix_prod, norm_elem_count])?;
        let output = ops.layer_norm(&flattened, self.weight.tensor(), self.bias.tensor(), self.config.eps)?;
        ops.reshape(&output, &input_shape)
    }
}
impl<B: Backend> Trainable<B> for LayerNorm<B> {
    fn parameters(&self) -> Vec<ParameterRef> {
        vec![ParameterRef { id: self.weight.id() }, ParameterRef { id: self.bias.id() }]
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchNormConfig {
    pub num_features: usize,
    pub eps: f32,
    pub momentum: f32,
}
impl BatchNormConfig {
    pub fn new(num_features: usize) -> Self {
        Self { num_features, eps: 1e-5, momentum: 0.1 }
    }
}

pub struct BatchNorm<B: Backend> {
    config: BatchNormConfig,
    weight: Parameter<B>,
    bias: Parameter<B>,
}
impl<B: Backend> BatchNorm<B> {
    pub fn new(backend: &B, config: BatchNormConfig, seed: u64) -> Result<Self> {
        let weight = backend.normal_parameter("weight", &[config.num_features], seed, 1.0)?;
        let bias = backend.normal_parameter("bias", &[config.num_features], seed.wrapping_add(1), 0.0)?;
        Ok(Self::from_parameters(config, weight, bias))
    }
    pub fn from_parameters(config: BatchNormConfig, weight: Parameter<B>, bias: Parameter<B>) -> Self {
        Self { config, weight, bias }
    }
    pub fn config(&self) -> &BatchNormConfig {
        &self.config
    }
}

impl<B: Backend> Module<B> for BatchNorm<B> {
    type Input = B::Tensor;
    type Output = B::Tensor;
    fn forward(&self, input: Self::Input, ctx: &mut ForwardCtx<B>) -> Result<Self::Output> {
        ctx.backend().ops().batch_norm(&input, self.weight.tensor(), self.bias.tensor(), self.config.eps)
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
    fn test_batch_norm_forward() {
        let backend = CpuBackend::default();
        let mut ctx = rustral_core::ForwardCtx::new(&backend, rustral_core::Mode::Inference);
        let weight: Parameter<CpuBackend> =
            Parameter::new("gamma", backend.tensor_from_vec(vec![1.0; 2], &[2]).unwrap());
        let bias: Parameter<CpuBackend> =
            Parameter::new("beta", backend.tensor_from_vec(vec![0.0; 2], &[2]).unwrap());
        let config = BatchNormConfig::new(2);
        let bn = BatchNorm::from_parameters(config, weight, bias);
        let input = backend.tensor_from_vec(vec![1.0, 10.0, 2.0, 12.0], &[2, 2]).unwrap();
        let output = bn.forward(input, &mut ctx).unwrap();
        assert_eq!(backend.ops().shape(&output), &[2, 2]);
    }
}
