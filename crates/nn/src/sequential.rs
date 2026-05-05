use mnr_core::{Backend, ForwardCtx, Module, ParameterRef, Result, Trainable};

/// Composition of exactly two modules.
///
/// This keeps the first architecture simple while proving the module contract:
/// the output of the first module becomes the input of the second module.
pub struct Sequential2<A, Bm> {
    first: A,
    second: Bm,
}

impl<A, Bm> Sequential2<A, Bm> {
    /// Create a two-module sequence.
    pub fn new(first: A, second: Bm) -> Self {
        Self { first, second }
    }
}

/// Chain two modules into a [`Sequential2`].
///
/// This is a convenience function equivalent to `Sequential2::new(first, second)`.
pub fn chain<A, Bm>(first: A, second: Bm) -> Sequential2<A, Bm> {
    Sequential2::new(first, second)
}

impl<BE, A, Bm, I, H, O> Module<BE> for Sequential2<A, Bm>
where
    BE: Backend,
    A: Module<BE, Input = I, Output = H>,
    Bm: Module<BE, Input = H, Output = O>,
{
    type Input = I;
    type Output = O;

    /// Run the first module, then run the second module on the first output.
    fn forward(&self, input: Self::Input, ctx: &mut ForwardCtx<BE>) -> Result<Self::Output> {
        let hidden = self.first.forward(input, ctx)?;
        self.second.forward(hidden, ctx)
    }
}

impl<BE, A, Bm> Trainable<BE> for Sequential2<A, Bm>
where
    BE: Backend,
    A: Trainable<BE>,
    Bm: Trainable<BE>,
{
    fn parameters(&self) -> Vec<ParameterRef> {
        let mut params = self.first.parameters();
        params.extend(self.second.parameters());
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LinearBuilder;
    use mnr_core::{ForwardCtx, Mode};
    use mnr_ndarray_backend::CpuBackend;

    #[test]
    fn test_sequential2_composition() {
        let backend = CpuBackend::default();

        // Create layers with the builder
        let linear1 = LinearBuilder::new(10, 5).seed(42).build(&backend).unwrap();
        let linear2 = LinearBuilder::new(5, 3).seed(43).build(&backend).unwrap();

        // Compose them using chain()
        let seq = chain(linear1, linear2);

        // Test forward pass
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let input = backend.tensor_from_vec(vec![0.1; 10], &[10]).unwrap();
        let output = seq.forward(input, &mut ctx).unwrap();

        // CPU backend treats 1D as batch, so output is [1, 3]
        assert_eq!(output.shape(), &[1, 3]);
    }

    #[test]
    fn test_sequential2_parameters() {
        let backend = CpuBackend::default();

        let linear1 = LinearBuilder::new(10, 5).with_bias(true).seed(42).build(&backend).unwrap();
        let linear2 = LinearBuilder::new(5, 3).seed(43).build(&backend).unwrap();

        let seq = chain(linear1, linear2);

        // Should have 3 parameters: W1, b1, W2
        assert_eq!(seq.parameters().len(), 3);
    }
}
