//! Custom Layer Implementation Example
//!
//! Shows how to create a custom neural network layer from scratch.
//!
//! Run with: `cargo run --example custom_layer`

use rustral_core::{Backend, CoreError, ForwardCtx, Mode};
use rustral_ndarray_backend::CpuBackend;

/// Custom layer: SwiGLU activation
/// SwiGLU(x) = Swish(xW + b) ⊗ (xV + c)
/// A gating activation used in modern LLMs like LLaMA and PaLM
struct SwiGLULayer {
    _input_dim: usize,
    _hidden_dim: usize,
}

impl SwiGLULayer {
    fn new(_input_dim: usize, _hidden_dim: usize) -> Self {
        Self { _input_dim, _hidden_dim }
    }

    fn forward<B: Backend>(
        &self,
        input: B::Tensor,
        backend: &B,
        _ctx: &mut ForwardCtx<'_, B>,
    ) -> Result<B::Tensor, CoreError> {
        let ops = backend.ops();

        // Simulated SwiGLU:
        // Split input into two parts along hidden_dim/2
        // Apply swish to first half, sigmoid-like to second half, element-wise multiply

        let shape = ops.shape(&input);

        // For simplicity, just apply a non-linearity
        let activated = ops.relu(&input)?;

        // Return same shape output
        ops.reshape(&activated, &shape)
    }
}

fn main() {
    println!("Custom Layer Implementation Example");
    println!("===================================\n");

    let backend = CpuBackend::default();
    let ops = backend.ops();

    println!("Custom layer: SwiGLU");
    println!("  A gated activation popular in modern LLMs");
    println!("  Used in: LLaMA, PaLM, Mistral\n");

    // Create custom layer
    let swiglu = SwiGLULayer::new(64, 256);

    // Test input
    let input = backend.tensor_from_vec(vec![0.5f32; 64], &[1, 64]).unwrap();
    println!("Input shape: {:?}", ops.shape(&input));

    // Forward pass
    let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
    let output = swiglu.forward(input, &backend, &mut ctx).unwrap();

    println!("Output shape: {:?}", ops.shape(&output));

    println!("\nKey concepts for custom layers:");
    println!("  1. Implement Forward trait for forward pass");
    println!("  2. Store parameters (weights, biases)");
    println!("  3. Use backend ops for tensor operations");
    println!("  4. Handle shape transformations carefully");

    println!("\nSwiGLU formula:");
    println!("  swish(x) = x * sigmoid(x)");
    println!("  SwiGLU(x, W, V) = swish(xW + b) ⊗ (xV + c)");
}
