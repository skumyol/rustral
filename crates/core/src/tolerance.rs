//! Numerical stability and tolerance framework for cross-backend comparisons.
//!
//! This module provides tolerance configurations for different operation families
//! to enable safe cross-backend comparisons (e.g., CPU vs GPU, different backends).

use std::fmt;

/// Operation families for tolerance specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpFamily {
    /// Element-wise operations (add, mul, relu, etc.)
    Elementwise,
    /// Matrix multiplication and linear layers
    MatmulLinear,
    /// Softmax and log-softmax operations
    Softmax,
    /// Layer normalization and batch normalization
    LayerNorm,
    /// Attention operations
    Attention,
    /// Reduction operations (sum, mean, etc.)
    Reduction,
}

impl fmt::Display for OpFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpFamily::Elementwise => write!(f, "elementwise"),
            OpFamily::MatmulLinear => write!(f, "matmul_linear"),
            OpFamily::Softmax => write!(f, "softmax"),
            OpFamily::LayerNorm => write!(f, "layer_norm"),
            OpFamily::Attention => write!(f, "attention"),
            OpFamily::Reduction => write!(f, "reduction"),
        }
    }
}

/// Tolerance configuration for numerical comparisons.
#[derive(Debug, Clone, Copy)]
pub struct ToleranceConfig {
    /// Relative tolerance (as a fraction of the magnitude)
    pub relative_tol: f32,
    /// Absolute tolerance (minimum acceptable difference)
    pub absolute_tol: f32,
    /// ULP tolerance for exact bit-level comparisons (None disables)
    pub ulp_tolerance: Option<u32>,
}

impl ToleranceConfig {
    /// Create tolerance configuration for a given operation family.
    pub fn for_family(family: OpFamily) -> Self {
        match family {
            OpFamily::Elementwise => Self {
                relative_tol: 1e-5,
                absolute_tol: 1e-6,
                ulp_tolerance: None,
            },
            OpFamily::MatmulLinear => Self {
                relative_tol: 1e-4,
                absolute_tol: 1e-5,
                ulp_tolerance: None,
            },
            OpFamily::Softmax => Self {
                relative_tol: 1e-5,
                absolute_tol: 1e-6,
                ulp_tolerance: None,
            },
            OpFamily::LayerNorm => Self {
                relative_tol: 1e-4,
                absolute_tol: 1e-5,
                ulp_tolerance: None,
            },
            OpFamily::Attention => Self {
                relative_tol: 1e-4,
                absolute_tol: 1e-5,
                ulp_tolerance: None,
            },
            OpFamily::Reduction => Self {
                relative_tol: 1e-5,
                absolute_tol: 1e-6,
                ulp_tolerance: None,
            },
        }
    }

    /// Create custom tolerance configuration.
    pub fn custom(relative_tol: f32, absolute_tol: f32) -> Self {
        Self {
            relative_tol,
            absolute_tol,
            ulp_tolerance: None,
        }
    }

    /// Check if two values are within tolerance.
    pub fn check(&self, a: f32, b: f32) -> bool {
        let diff = (a - b).abs();
        let magnitude = a.abs().max(b.abs());

        // Check absolute tolerance
        if diff <= self.absolute_tol {
            return true;
        }

        // Check relative tolerance
        if magnitude > 0.0 && diff / magnitude <= self.relative_tol {
            return true;
        }

        // Check ULP tolerance if specified
        if let Some(ulp) = self.ulp_tolerance {
            let a_bits = a.to_bits();
            let b_bits = b.to_bits();
            let ulp_diff = a_bits.abs_diff(b_bits);
            if (ulp_diff as u32) <= ulp {
                return true;
            }
        }

        false
    }

    /// Check if two slices of values are within tolerance.
    pub fn check_slice(&self, a: &[f32], b: &[f32]) -> bool {
        if a.len() != b.len() {
            return false;
        }

        a.iter()
            .zip(b.iter())
            .all(|(&x, &y)| self.check(x, y))
    }
}

impl Default for ToleranceConfig {
    fn default() -> Self {
        Self::for_family(OpFamily::Elementwise)
    }
}

/// Assert that two values are within tolerance.
#[macro_export]
macro_rules! assert_close {
    ($a:expr, $b:expr, $tol:expr) => {
        let a_val = $a;
        let b_val = $b;
        let tol = $tol;
        
        if !tol.check(a_val, b_val) {
            panic!(
                "Values not within tolerance: {} vs {} (rel_tol={}, abs_tol={})",
                a_val, b_val, tol.relative_tol, tol.absolute_tol
            );
        }
    };
}

/// Assert that two tensor slices are within tolerance.
#[macro_export]
macro_rules! assert_slices_close {
    ($a:expr, $b:expr, $tol:expr) => {
        let a_slice = $a;
        let b_slice = $b;
        let tol = $tol;
        
        if !tol.check_slice(a_slice, b_slice) {
            // Find first differing element
            for (i, (&x, &y)) in a_slice.iter().zip(b_slice.iter()).enumerate() {
                if !tol.check(x, y) {
                    panic!(
                        "Slices not within tolerance at index {}: {} vs {} (rel_tol={}, abs_tol={})",
                        i, x, y, tol.relative_tol, tol.absolute_tol
                    );
                }
            }
            panic!("Slices have different lengths: {} vs {}", a_slice.len(), b_slice.len());
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tolerance_exact_match() {
        let tol = ToleranceConfig::for_family(OpFamily::Elementwise);
        assert!(tol.check(1.0, 1.0));
    }

    #[test]
    fn test_tolerance_within_absolute() {
        let tol = ToleranceConfig::custom(0.0, 1e-3);
        assert!(tol.check(1.0, 1.0005));
        assert!(!tol.check(1.0, 1.002));
    }

    #[test]
    fn test_tolerance_within_relative() {
        let tol = ToleranceConfig::custom(1e-4, 0.0);
        assert!(tol.check(1000.0, 1000.1));
        assert!(!tol.check(1000.0, 1000.2));
    }

    #[test]
    fn test_tolerance_check_slice() {
        let tol = ToleranceConfig::for_family(OpFamily::Elementwise);
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.000001, 2.000001, 3.000001];
        assert!(tol.check_slice(&a, &b));
    }

    #[test]
    fn test_tolerance_check_slice_different_lengths() {
        let tol = ToleranceConfig::for_family(OpFamily::Elementwise);
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!(!tol.check_slice(&a, &b));
    }

    #[test]
    fn test_op_family_display() {
        assert_eq!(format!("{}", OpFamily::Elementwise), "elementwise");
        assert_eq!(format!("{}", OpFamily::MatmulLinear), "matmul_linear");
    }
}
