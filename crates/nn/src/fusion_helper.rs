//! Fusion helper utilities for neural network modules.
//!
//! This module provides unified fusion entry points for common neural network patterns,
//! eliminating duplicate fusion policy across modules and maintaining consistency.

use crate::linear::Linear;
use rustral_core::{Backend, ForwardCtx, FusionOptimizer, Result};

/// Unified fusion helper for linear + activation patterns.
///
/// This provides a single entry point for fused linear operations with various activations,
/// eliminating duplicate fusion policy across different module types.
pub struct FusionHelper;

impl FusionHelper {
    /// Try fused linear + ReLU, fall back to unfused sequence.
    ///
    /// This attempts to use the backend's fused operation if available, otherwise
    /// falls back to the unfused sequence (linear + relu). This provides a unified
    /// fusion policy for all linear + activation patterns.
    pub fn try_linear_relu<B: Backend>(
        input: &B::Tensor,
        linear: &Linear<B>,
        ctx: &mut ForwardCtx<B>,
    ) -> Result<B::Tensor> {
        let Some(bias) = linear.bias() else {
            let output = ctx.backend().ops().linear(input, linear.weight(), None)?;
            return ctx.backend().ops().relu(&output);
        };
        FusionOptimizer::apply_matmul_bias_relu(ctx.backend(), input, linear.weight(), bias)
    }

    /// Try fused linear + GELU, fall back to unfused sequence.
    ///
    /// This attempts to use the backend's fused operation if available, otherwise
    /// falls back to the unfused sequence (linear + gelu). This provides a unified
    /// fusion policy for all linear + activation patterns.
    pub fn try_linear_gelu<B: Backend>(
        input: &B::Tensor,
        linear: &Linear<B>,
        ctx: &mut ForwardCtx<B>,
    ) -> Result<B::Tensor> {
        let Some(bias) = linear.bias() else {
            let output = ctx.backend().ops().linear(input, linear.weight(), None)?;
            return ctx.backend().ops().gelu(&output);
        };
        FusionOptimizer::apply_matmul_bias_gelu(ctx.backend(), input, linear.weight(), bias)
    }

    /// Try fused linear + bias (no activation), fall back to unfused sequence.
    ///
    /// This attempts to use the backend's fused operation if available, otherwise
    /// falls back to the unfused sequence (linear + bias add).
    pub fn try_linear_bias<B: Backend>(
        input: &B::Tensor,
        linear: &Linear<B>,
        ctx: &mut ForwardCtx<B>,
    ) -> Result<B::Tensor> {
        let Some(bias) = linear.bias() else {
            return ctx.backend().ops().linear(input, linear.weight(), None);
        };
        FusionOptimizer::apply_matmul_bias(ctx.backend(), input, linear.weight(), bias)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fusion_helper_compiles() {
        // This test just verifies the fusion helper module compiles
        // Actual fusion behavior tests would need real backend implementations
        assert!(true);
    }

    #[test]
    fn test_fusion_helper_public_api() {
        // Verify that FusionHelper is publicly accessible
        // This is a simple compile-time check that the type exists
        let _ = FusionHelper;
    }
}
