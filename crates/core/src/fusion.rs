//! Operation fusion pattern matching and optimization.
//!
//! Provides unified fusion pipeline that pattern-matches on operation sequences
//! and calls fused backend entry points when available, with fallback to unfused
//! operations when fusion is not supported.

use crate::{Backend, Result};
use std::fmt;

/// Fusion pattern that can be matched and optimized.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FusionPattern {
    /// Matmul + bias + ReLU: y = relu(x @ w^T + b)
    MatmulBiasRelu,
    /// Matmul + bias + GELU: y = gelu(x @ w^T + b)
    MatmulBiasGelu,
    /// Matmul + bias: y = x @ w^T + b
    MatmulBias,
}

impl fmt::Display for FusionPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FusionPattern::MatmulBiasRelu => write!(f, "matmul_bias_relu"),
            FusionPattern::MatmulBiasGelu => write!(f, "matmul_bias_gelu"),
            FusionPattern::MatmulBias => write!(f, "matmul_bias"),
        }
    }
}

/// Operation type for pattern matching.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OpType {
    /// Matrix multiplication
    Matmul,
    /// Bias addition
    AddBias,
    /// ReLU activation
    ReLU,
    /// GELU activation
    GELU,
    /// Unknown operation
    Unknown,
}

/// Operation in a sequence for pattern matching.
#[derive(Debug, Clone)]
pub struct Op {
    /// Operation type
    pub op_type: OpType,
    /// Input shapes (for pattern matching context)
    pub input_shapes: Vec<Vec<usize>>,
    /// Output shape (for pattern matching context)
    pub output_shape: Vec<usize>,
}

impl Op {
    /// Create a new operation.
    pub fn new(op_type: OpType, input_shapes: Vec<Vec<usize>>, output_shape: Vec<usize>) -> Self {
        Self {
            op_type,
            input_shapes,
            output_shape,
        }
    }

    /// Get the operation type.
    pub fn op_type(&self) -> &OpType {
        &self.op_type
    }
}

/// Fusion pattern matcher that detects optimizable operation sequences.
pub struct PatternMatcher;

impl PatternMatcher {
    /// Match a sequence of operations against known fusion patterns.
    ///
    /// Returns the first matching pattern, or None if no pattern matches.
    pub fn match_pattern(ops: &[Op]) -> Option<FusionPattern> {
        if ops.len() < 2 {
            return None;
        }

        // Match matmul + bias + relu
        if Self::matches_matmul_bias_relu(ops) {
            return Some(FusionPattern::MatmulBiasRelu);
        }

        // Match matmul + bias + gelu
        if Self::matches_matmul_bias_gelu(ops) {
            return Some(FusionPattern::MatmulBiasGelu);
        }

        // Match matmul + bias
        if Self::matches_matmul_bias(ops) {
            return Some(FusionPattern::MatmulBias);
        }

        None
    }

    /// Check if operations match matmul + bias + relu pattern.
    fn matches_matmul_bias_relu(ops: &[Op]) -> bool {
        if ops.len() != 3 {
            return false;
        }

        ops[0].op_type == OpType::Matmul
            && ops[1].op_type == OpType::AddBias
            && ops[2].op_type == OpType::ReLU
    }

    /// Check if operations match matmul + bias + gelu pattern.
    fn matches_matmul_bias_gelu(ops: &[Op]) -> bool {
        if ops.len() != 3 {
            return false;
        }

        ops[0].op_type == OpType::Matmul
            && ops[1].op_type == OpType::AddBias
            && ops[2].op_type == OpType::GELU
    }

    /// Check if operations match matmul + bias pattern.
    fn matches_matmul_bias(ops: &[Op]) -> bool {
        if ops.len() != 2 {
            return false;
        }

        ops[0].op_type == OpType::Matmul && ops[1].op_type == OpType::AddBias
    }
}

/// Fusion optimizer that applies pattern matching and calls fused operations.
pub struct FusionOptimizer<B: Backend> {
    backend: B,
    enable_fusion: bool,
}

impl<B: Backend> FusionOptimizer<B> {
    /// Create a new fusion optimizer.
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            enable_fusion: true,
        }
    }

    /// Enable or disable fusion optimization.
    pub fn set_enable_fusion(&mut self, enable: bool) {
        self.enable_fusion = enable;
    }

    /// Check if fusion is enabled.
    pub fn is_fusion_enabled(&self) -> bool {
        self.enable_fusion
    }

    /// Get the backend.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Try to apply fusion for a matmul + bias + relu sequence.
    ///
    /// Returns the fused result if the backend supports it, otherwise None.
    pub fn try_fuse_matmul_bias_relu(
        &self,
        input: &B::Tensor,
        weight: &crate::Parameter<B>,
        bias: &crate::Parameter<B>,
    ) -> Result<Option<B::Tensor>> {
        if !self.enable_fusion {
            return Ok(None);
        }

        // Check if backend supports fusion
        if let Some(fusion_ops) = self.backend.fusion_ops() {
            match fusion_ops.fused_linear_bias_relu(input, weight, bias) {
                Ok(result) => Ok(Some(result)),
                Err(_) => Ok(None), // Fallback if fusion fails
            }
        } else {
            Ok(None)
        }
    }

    /// Try to apply fusion for a matmul + bias + gelu sequence.
    ///
    /// Returns the fused result if the backend supports it, otherwise None.
    pub fn try_fuse_matmul_bias_gelu(
        &self,
        input: &B::Tensor,
        weight: &crate::Parameter<B>,
        bias: &crate::Parameter<B>,
    ) -> Result<Option<B::Tensor>> {
        if !self.enable_fusion {
            return Ok(None);
        }

        // Check if backend supports fusion
        if let Some(fusion_ops) = self.backend.fusion_ops() {
            match fusion_ops.fused_linear_bias_gelu(input, weight, bias) {
                Ok(result) => Ok(Some(result)),
                Err(_) => Ok(None), // Fallback if fusion fails
            }
        } else {
            Ok(None)
        }
    }

    /// Try to apply fusion for a matmul + bias sequence.
    ///
    /// Returns the fused result if the backend supports it, otherwise None.
    pub fn try_fuse_matmul_bias(
        &self,
        input: &B::Tensor,
        weight: &crate::Parameter<B>,
        bias: &crate::Parameter<B>,
    ) -> Result<Option<B::Tensor>> {
        if !self.enable_fusion {
            return Ok(None);
        }

        // Check if backend supports fusion
        if let Some(fusion_ops) = self.backend.fusion_ops() {
            match fusion_ops.fused_linear_bias(input, weight, bias) {
                Ok(result) => Ok(Some(result)),
                Err(_) => Ok(None), // Fallback if fusion fails
            }
        } else {
            Ok(None)
        }
    }

    /// Execute matmul + bias + relu with automatic fusion fallback.
    ///
    /// Tries fused operation first, falls back to unfused sequence if fusion
    /// is not available or fails.
    pub fn matmul_bias_relu(
        &self,
        input: &B::Tensor,
        weight: &crate::Parameter<B>,
        bias: &crate::Parameter<B>,
    ) -> Result<B::Tensor> {
        // Try fused operation first
        if let Some(fused_result) = self.try_fuse_matmul_bias_relu(input, weight, bias)? {
            return Ok(fused_result);
        }

        // Fallback to unfused sequence
        let ops = self.backend.ops();
        let h = ops.matmul(input, weight.tensor())?;
        let h = ops.add(&h, bias.tensor())?;
        ops.relu(&h)
    }

    /// Execute matmul + bias + gelu with automatic fusion fallback.
    ///
    /// Tries fused operation first, falls back to unfused sequence if fusion
    /// is not available or fails.
    pub fn matmul_bias_gelu(
        &self,
        input: &B::Tensor,
        weight: &crate::Parameter<B>,
        bias: &crate::Parameter<B>,
    ) -> Result<B::Tensor> {
        // Try fused operation first
        if let Some(fused_result) = self.try_fuse_matmul_bias_gelu(input, weight, bias)? {
            return Ok(fused_result);
        }

        // Fallback to unfused sequence
        let ops = self.backend.ops();
        let h = ops.matmul(input, weight.tensor())?;
        let h = ops.add(&h, bias.tensor())?;
        ops.gelu(&h)
    }

    /// Execute matmul + bias with automatic fusion fallback.
    ///
    /// Tries fused operation first, falls back to unfused sequence if fusion
    /// is not available or fails.
    pub fn matmul_bias(
        &self,
        input: &B::Tensor,
        weight: &crate::Parameter<B>,
        bias: &crate::Parameter<B>,
    ) -> Result<B::Tensor> {
        // Try fused operation first
        if let Some(fused_result) = self.try_fuse_matmul_bias(input, weight, bias)? {
            return Ok(fused_result);
        }

        // Fallback to unfused sequence
        let ops = self.backend.ops();
        let h = ops.matmul(input, weight.tensor())?;
        ops.add(&h, bias.tensor())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fusion_pattern_display() {
        assert_eq!(FusionPattern::MatmulBiasRelu.to_string(), "matmul_bias_relu");
        assert_eq!(FusionPattern::MatmulBiasGelu.to_string(), "matmul_bias_gelu");
        assert_eq!(FusionPattern::MatmulBias.to_string(), "matmul_bias");
    }

    #[test]
    fn test_pattern_matcher_matmul_bias_relu() {
        let ops = vec![
            Op::new(OpType::Matmul, vec![vec![10, 20]], vec![10, 30]),
            Op::new(OpType::AddBias, vec![vec![10, 30]], vec![10, 30]),
            Op::new(OpType::ReLU, vec![vec![10, 30]], vec![10, 30]),
        ];

        let pattern = PatternMatcher::match_pattern(&ops);
        assert_eq!(pattern, Some(FusionPattern::MatmulBiasRelu));
    }

    #[test]
    fn test_pattern_matcher_matmul_bias_gelu() {
        let ops = vec![
            Op::new(OpType::Matmul, vec![vec![10, 20]], vec![10, 30]),
            Op::new(OpType::AddBias, vec![vec![10, 30]], vec![10, 30]),
            Op::new(OpType::GELU, vec![vec![10, 30]], vec![10, 30]),
        ];

        let pattern = PatternMatcher::match_pattern(&ops);
        assert_eq!(pattern, Some(FusionPattern::MatmulBiasGelu));
    }

    #[test]
    fn test_pattern_matcher_matmul_bias() {
        let ops = vec![
            Op::new(OpType::Matmul, vec![vec![10, 20]], vec![10, 30]),
            Op::new(OpType::AddBias, vec![vec![10, 30]], vec![10, 30]),
        ];

        let pattern = PatternMatcher::match_pattern(&ops);
        assert_eq!(pattern, Some(FusionPattern::MatmulBias));
    }

    #[test]
    fn test_pattern_matcher_no_match() {
        let ops = vec![
            Op::new(OpType::Matmul, vec![vec![10, 20]], vec![10, 30]),
            Op::new(OpType::ReLU, vec![vec![10, 30]], vec![10, 30]),
        ];

        let pattern = PatternMatcher::match_pattern(&ops);
        assert_eq!(pattern, None);
    }

    #[test]
    fn test_pattern_matcher_too_short() {
        let ops = vec![Op::new(OpType::Matmul, vec![vec![10, 20]], vec![10, 30])];

        let pattern = PatternMatcher::match_pattern(&ops);
        assert_eq!(pattern, None);
    }

    #[test]
    fn test_pattern_matcher_wrong_order() {
        let ops = vec![
            Op::new(OpType::ReLU, vec![vec![10, 20]], vec![10, 20]),
            Op::new(OpType::Matmul, vec![vec![10, 20]], vec![10, 30]),
            Op::new(OpType::AddBias, vec![vec![10, 30]], vec![10, 30]),
        ];

        let pattern = PatternMatcher::match_pattern(&ops);
        assert_eq!(pattern, None);
    }
}
