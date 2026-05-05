use mnr_core::{Backend, ForwardCtx, Module, Parameter, ParameterRef, Result, Saveable, Trainable};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Configuration for a dense affine projection.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LinearConfig {
    /// Input feature dimension.
    pub in_dim: usize,

    /// Output feature dimension.
    pub out_dim: usize,

    /// Whether the layer uses a bias vector.
    pub bias: bool,
}

/// Dense affine projection: `y = x * W^T + b`.
pub struct Linear<B: Backend> {
    config: LinearConfig,
    weight: Parameter<B>,
    bias: Option<Parameter<B>>,
}

impl<B: Backend> Linear<B> {
    /// Create a new Linear layer from config and backend.
    ///
    /// This is a convenience method equivalent to using LinearBuilder.
    pub fn new(backend: &B, config: LinearConfig) -> Result<Self> {
        LinearBuilder::new(config.in_dim, config.out_dim)
            .with_bias(config.bias)
            .build(backend)
    }

    /// Build a linear layer from explicit parameters.
    ///
    /// This constructor intentionally avoids lazy initialization. The caller is
    /// responsible for creating correctly shaped parameters.
    pub fn from_parameters(config: LinearConfig, weight: Parameter<B>, bias: Option<Parameter<B>>) -> Self {
        Self { config, weight, bias }
    }

    /// Build a linear layer from a [`LinearBuilder`].
    ///
    /// Use [`LinearBuilder::new`] instead of this method directly.
    pub fn from_builder(builder: LinearBuilder, backend: &B) -> Result<Self> {
        builder.build(backend)
    }

    /// Borrow the immutable layer configuration.
    pub fn config(&self) -> &LinearConfig {
        &self.config
    }

    /// Borrow the weight parameter.
    pub fn weight(&self) -> &Parameter<B> {
        &self.weight
    }

    /// Borrow the optional bias parameter.
    pub fn bias(&self) -> Option<&Parameter<B>> {
        self.bias.as_ref()
    }
}

/// Builder for [`Linear`] layers with backend-initialized parameters.
///
/// This eliminates the boilerplate of manually creating `Parameter` objects
/// and `LinearConfig`.
pub struct LinearBuilder {
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    seed: u64,
    scale: f32,
}

impl LinearBuilder {
    /// Start building a linear layer with the given input and output dimensions.
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        Self {
            in_dim,
            out_dim,
            bias: false,
            seed: 42,
            scale: 0.1,
        }
    }

    /// Enable or disable the bias vector (default: disabled).
    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    /// Set the random seed for parameter initialization (default: 42).
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set the initialization scale for parameter values (default: 0.1).
    pub fn scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// Build the [`Linear`] layer using the given backend.
    pub fn build<B: Backend>(self, backend: &B) -> Result<Linear<B>> {
        let weight = backend.normal_parameter(
            "weight",
            &[self.out_dim, self.in_dim],
            self.seed,
            self.scale,
        )?;
        let bias = if self.bias {
            Some(backend.normal_parameter(
                "bias",
                &[self.out_dim],
                self.seed.wrapping_add(1),
                self.scale,
            )?)
        } else {
            None
        };
        let config = LinearConfig {
            in_dim: self.in_dim,
            out_dim: self.out_dim,
            bias: self.bias,
        };
        Ok(Linear::from_parameters(config, weight, bias))
    }
}

impl<B: Backend> Module<B> for Linear<B> {
    type Input = B::Tensor;
    type Output = B::Tensor;

    /// Apply the affine projection for one forward context.
    fn forward(&self, input: Self::Input, ctx: &mut ForwardCtx<B>) -> Result<Self::Output> {
        ctx.backend().ops().linear(&input, &self.weight, self.bias.as_ref())
    }
}

impl<B: Backend> Trainable<B> for Linear<B> {
    /// Return weight and optional bias parameter references.
    fn parameters(&self) -> Vec<ParameterRef> {
        let mut out = vec![ParameterRef { id: self.weight.id() }];
        if let Some(b) = &self.bias {
            out.push(ParameterRef { id: b.id() });
        }
        out
    }
}

impl<B: Backend> Clone for Linear<B>
where
    B::Tensor: Clone,
{
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            weight: self.weight.clone(),
            bias: self.bias.clone(),
        }
    }
}

impl LinearConfig {
    /// Create a new Linear configuration.
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        Self {
            in_dim,
            out_dim,
            bias: false,
        }
    }

    /// Set bias.
    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }
}

impl<B: Backend> Saveable<B> for Linear<B> {
    fn state_dict(&self) -> Vec<(String, ParameterRef)> {
        let mut out = vec![("weight".to_string(), ParameterRef { id: self.weight.id() })];
        if let Some(b) = &self.bias {
            out.push(("bias".to_string(), ParameterRef { id: b.id() }));
        }
        out
    }

    fn load_state_dict(&mut self, dict: &HashMap<String, Vec<f32>>, backend: &B) -> Result<()> {
        // Load weight
        let weight_data = dict.get("weight")
            .ok_or_else(|| mnr_core::CoreError::InvalidArgument("Missing 'weight' in state_dict".into()))?;
        let weight_shape = &[self.config.out_dim, self.config.in_dim];
        let new_weight = backend.parameter_from_vec("weight", weight_data.clone(), weight_shape)?;
        self.weight = new_weight;

        // Load bias if present
        if self.config.bias {
            let bias_data = dict.get("bias")
                .ok_or_else(|| mnr_core::CoreError::InvalidArgument("Missing 'bias' in state_dict".into()))?;
            let bias_shape = &[self.config.out_dim];
            let new_bias = backend.parameter_from_vec("bias", bias_data.clone(), bias_shape)?;
            self.bias = Some(new_bias);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnr_core::{ForwardCtx, Mode};
    use mnr_ndarray_backend::CpuBackend;

    fn create_mock_linear(in_dim: usize, out_dim: usize, bias: bool) -> Linear<CpuBackend> {
        let backend = CpuBackend::default();
        LinearBuilder::new(in_dim, out_dim)
            .with_bias(bias)
            .seed(42)
            .build(&backend)
            .unwrap()
    }

    #[test]
    fn test_linear_forward_shape() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let linear = create_mock_linear(10, 5, false);

        let input = backend.tensor_from_vec(vec![1.0; 10], &[10]).unwrap();
        let output = linear.forward(input, &mut ctx).unwrap();

        // CPU backend treats 1D input as [1, 10] batch, so output is [1, 5]
        assert_eq!(output.shape(), &[1, 5]);
    }

    #[test]
    fn test_linear_with_bias() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let linear = create_mock_linear(4, 3, true);

        let input = backend.tensor_from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let output = linear.forward(input, &mut ctx).unwrap();

        assert_eq!(output.shape(), &[1, 3]);
    }

    #[test]
    fn test_linear_parameters() {
        let linear_no_bias = create_mock_linear(10, 5, false);
        assert_eq!(linear_no_bias.parameters().len(), 1);

        let linear_with_bias = create_mock_linear(10, 5, true);
        assert_eq!(linear_with_bias.parameters().len(), 2);
    }

    #[test]
    fn test_linear_accessors() {
        let linear = create_mock_linear(10, 5, true);
        assert_eq!(linear.config().in_dim, 10);
        assert_eq!(linear.config().out_dim, 5);
        assert!(linear.config().bias);
        assert_eq!(linear.weight().id(), linear.weight().id());
        assert!(linear.bias().is_some());

        let linear_no_bias = create_mock_linear(10, 5, false);
        assert!(linear_no_bias.bias().is_none());
    }

    #[test]
    fn test_linear_from_parameters() {
        let backend = CpuBackend::default();
        let weight = backend.normal_parameter("w", &[5, 10], 1, 0.1).unwrap();
        let bias = Some(backend.normal_parameter("b", &[5], 2, 0.1).unwrap());
        let config = LinearConfig { in_dim: 10, out_dim: 5, bias: true };
        let linear = Linear::from_parameters(config.clone(), weight.clone(), bias);
        assert_eq!(linear.config().in_dim, 10);
        assert_eq!(linear.weight().id(), weight.id());
    }

    #[test]
    fn test_linear_from_builder() {
        let backend = CpuBackend::default();
        let builder = LinearBuilder::new(10, 5).with_bias(true);
        let linear = Linear::from_builder(builder, &backend).unwrap();
        assert_eq!(linear.config().in_dim, 10);
        assert_eq!(linear.config().out_dim, 5);
        assert!(linear.config().bias);
    }

    #[test]
    fn test_linear_clone() {
        let linear = create_mock_linear(10, 5, true);
        let cloned = linear.clone();
        assert_eq!(cloned.config().in_dim, 10);
        assert_eq!(cloned.parameters().len(), 2);
    }

    #[test]
    fn test_linear_state_dict() {
        let linear = create_mock_linear(10, 5, true);
        let state = linear.state_dict();
        assert_eq!(state.len(), 2);
        assert_eq!(state[0].0, "weight");
        assert_eq!(state[1].0, "bias");
    }

    #[test]
    fn test_linear_load_state_dict() {
        let backend = CpuBackend::default();
        let mut linear = LinearBuilder::new(10, 5)
            .with_bias(true)
            .build(&backend)
            .unwrap();

        let weight_data = vec![0.1f32; 50];
        let bias_data = vec![0.2f32; 5];
        let mut dict = std::collections::HashMap::new();
        dict.insert("weight".to_string(), weight_data);
        dict.insert("bias".to_string(), bias_data);

        linear.load_state_dict(&dict, &backend).unwrap();
        assert_eq!(linear.weight().tensor().shape(), &[5, 10]);
    }

    #[test]
    fn test_linear_config_with_bias() {
        let config = LinearConfig::new(10, 5).with_bias(true);
        assert!(config.bias);
    }
}
