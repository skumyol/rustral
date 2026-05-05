//! Dropout layer for regularization.
//!
//! During training, randomly zeroes elements of the input tensor with
//! probability `p` using samples from a Bernoulli distribution. Each
//! channel is zeroed out independently on every forward call.
//!
//! During inference, the module computes an identity function.
//!
//! This follows the legacy DyNet wrapper design where dropout behavior
//! is controlled by a training mode guard.

use rustral_core::{Backend, ForwardCtx, Module, Result};
use serde::{Deserialize, Serialize};

/// Configuration for dropout regularization.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DropoutConfig {
    /// Probability of zeroing an element (0 <= p <= 1).
    pub p: f32,
}

impl DropoutConfig {
    /// Create a dropout config with the given probability.
    pub fn new(p: f32) -> Self {
        Self { p: p.clamp(0.0, 1.0) }
    }
}

impl Default for DropoutConfig {
    fn default() -> Self {
        Self { p: 0.1 }
    }
}

/// Dropout regularization layer.
///
/// # Example
///
/// ```
/// use rustral_nn::{Dropout, DropoutConfig};
/// use rustral_core::{ForwardCtx, Mode, Module};
/// use rustral_ndarray_backend::CpuBackend;
///
/// let backend = CpuBackend::default();
/// let mut ctx = ForwardCtx::new(&backend, Mode::Train);  // Dropout active
/// let dropout = Dropout::<CpuBackend>::new(DropoutConfig::new(0.5));
/// // let output = dropout.forward(input, &mut ctx)?;
/// ```
pub struct Dropout<B: Backend> {
    config: DropoutConfig,
    _backend: std::marker::PhantomData<B>,
}

impl<B: Backend> Dropout<B> {
    /// Create a new dropout layer with the given configuration.
    pub fn new(config: DropoutConfig) -> Self {
        Self { config, _backend: std::marker::PhantomData }
    }

    /// Borrow the dropout configuration.
    pub fn config(&self) -> &DropoutConfig {
        &self.config
    }
}

impl<B: Backend> Module<B> for Dropout<B> {
    type Input = B::Tensor;
    type Output = B::Tensor;

    /// Apply dropout during training, identity during inference.
    fn forward(&self, input: Self::Input, ctx: &mut ForwardCtx<B>) -> Result<Self::Output> {
        let training = ctx.is_training();
        ctx.backend().ops().dropout(&input, self.config.p, training)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustral_core::{ForwardCtx, Mode};
    use rustral_ndarray_backend::CpuBackend;

    #[test]
    fn test_dropout_inference_is_identity() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let dropout = Dropout::<CpuBackend>::new(DropoutConfig::new(0.5));

        let input = backend.tensor_from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let output = dropout.forward(input.clone(), &mut ctx).unwrap();

        // During inference, output should equal input
        assert_eq!(input.values(), output.values());
    }

    #[test]
    fn test_dropout_zero_is_identity() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Train);
        let dropout = Dropout::<CpuBackend>::new(DropoutConfig::new(0.0));

        let input = backend.tensor_from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let output = dropout.forward(input.clone(), &mut ctx).unwrap();

        // With p=0, output should equal input
        assert_eq!(input.values(), output.values());
    }

    #[test]
    fn test_dropout_config_default() {
        let config: DropoutConfig = Default::default();
        assert_eq!(config.p, 0.1);
    }

    #[test]
    fn test_dropout_config_clamp() {
        let config = DropoutConfig::new(1.5);
        assert_eq!(config.p, 1.0);
        let config2 = DropoutConfig::new(-0.5);
        assert_eq!(config2.p, 0.0);
    }

    #[test]
    fn test_dropout_config_accessor() {
        let dropout = Dropout::<CpuBackend>::new(DropoutConfig::new(0.3));
        assert_eq!(dropout.config().p, 0.3);
    }
}
