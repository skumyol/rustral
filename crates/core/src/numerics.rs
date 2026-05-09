//! Numerics policy for optimization validation.
//!
//! Provides dtype-specific tolerances and validation utilities to ensure
//! that fused operations and other optimizations maintain numerical accuracy
//! within acceptable bounds.

use std::fmt;

/// Data type for numerics policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// 32-bit floating point
    F32,
    /// 16-bit floating point (half precision)
    F16,
    /// 16-bit brain floating point
    BF16,
    /// 64-bit floating point (double precision)
    F64,
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F32 => write!(f, "f32"),
            DType::F16 => write!(f, "f16"),
            DType::BF16 => write!(f, "bf16"),
            DType::F64 => write!(f, "f64"),
        }
    }
}

/// Numerics tolerance specification.
#[derive(Debug, Clone, Copy)]
pub struct Tolerance {
    /// Relative tolerance (rtol)
    pub rtol: f64,
    /// Absolute tolerance (atol)
    pub atol: f64,
    /// Maximum allowed ULP distance (for integer comparisons)
    pub max_ulps: Option<u32>,
}

impl Tolerance {
    /// Create a new tolerance specification.
    pub fn new(rtol: f64, atol: f64) -> Self {
        Self { rtol, atol, max_ulps: None }
    }

    /// Create a new tolerance with ULP bounds.
    pub fn with_ulps(rtol: f64, atol: f64, max_ulps: u32) -> Self {
        Self { rtol, atol, max_ulps: Some(max_ulps) }
    }

    /// Check if two values are within tolerance.
    pub fn is_close(&self, a: f64, b: f64) -> bool {
        let diff = (a - b).abs();
        let max_abs = a.abs().max(b.abs());
        
        // Check absolute tolerance
        if diff <= self.atol {
            return true;
        }
        
        // Check relative tolerance
        if diff <= self.rtol * max_abs {
            return true;
        }
        
        // Check ULP distance if specified
        if let Some(max_ulps) = self.max_ulps {
            if self.ulp_distance(a, b) <= max_ulps {
                return true;
            }
        }
        
        false
    }
    
    /// Calculate ULP (Unit in the Last Place) distance between two floats.
    fn ulp_distance(&self, a: f64, b: f64) -> u32 {
        if a == b {
            return 0;
        }
        
        let a_bits = a.to_bits();
        let b_bits = b.to_bits();
        
        // Handle different signs
        if (a_bits >> 63) != (b_bits >> 63) {
            return u32::MAX; // Infinite distance
        }

        a_bits.abs_diff(b_bits) as u32
    }
}

impl Default for Tolerance {
    fn default() -> Self {
        Self::new(1e-5, 1e-8)
    }
}

/// Numerics configuration for optimization validation.
#[derive(Debug, Clone)]
pub struct NumericsConfig {
    /// Tolerance for FP32 operations.
    pub f32_tolerance: Tolerance,
    /// Tolerance for FP16 operations.
    pub f16_tolerance: Tolerance,
    /// Tolerance for BF16 operations.
    pub f16_bf16_tolerance: Tolerance,
    /// Whether to use strict mode (fail on any deviation).
    pub strict_mode: bool,
    /// Whether to use float64 reference for validation.
    pub use_float64_reference: bool,
}

impl Default for NumericsConfig {
    fn default() -> Self {
        Self {
            f32_tolerance: Tolerance::new(1e-5, 1e-8),
            f16_tolerance: Tolerance::with_ulps(1e-3, 1e-6, 4), // 4 ULPs for FP16
            f16_bf16_tolerance: Tolerance::with_ulps(1e-3, 1e-6, 4), // 4 ULPs for BF16
            strict_mode: false,
            use_float64_reference: false,
        }
    }
}

impl NumericsConfig {
    /// Create a new numerics configuration with default tolerances.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set strict mode.
    pub fn with_strict_mode(mut self, strict: bool) -> Self {
        self.strict_mode = strict;
        self
    }

    /// Enable float64 reference validation.
    pub fn with_float64_reference(mut self, enable: bool) -> Self {
        self.use_float64_reference = enable;
        self
    }

    /// Get tolerance for a specific dtype.
    pub fn tolerance_for(&self, dtype: DType) -> Tolerance {
        match dtype {
            DType::F32 => self.f32_tolerance,
            DType::F16 => self.f16_tolerance,
            DType::BF16 => self.f16_bf16_tolerance,
            DType::F64 => Tolerance::new(1e-10, 1e-12), // Stricter for F64
        }
    }

    /// Validate that two values are within tolerance for a given dtype.
    pub fn validate(&self, a: f64, b: f64, dtype: DType) -> Result<(), NumericsError> {
        let tolerance = self.tolerance_for(dtype);
        
        if tolerance.is_close(a, b) {
            Ok(())
        } else {
            let diff = (a - b).abs();
            Err(NumericsError::ToleranceExceeded {
                dtype,
                value_a: a,
                value_b: b,
                diff,
                rtol: tolerance.rtol,
                atol: tolerance.atol,
            })
        }
    }
}

/// Numerics validation error.
#[derive(Debug, Clone)]
pub enum NumericsError {
    /// Tolerance exceeded for value comparison.
    ToleranceExceeded {
        dtype: DType,
        value_a: f64,
        value_b: f64,
        diff: f64,
        rtol: f64,
        atol: f64,
    },
    /// Shape mismatch in tensor comparison.
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    /// Validation failed in strict mode.
    StrictModeViolation(String),
}

impl fmt::Display for NumericsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NumericsError::ToleranceExceeded { dtype, value_a, value_b, diff, rtol, atol } => {
                write!(
                    f,
                    "Numerics tolerance exceeded for {}: |{} - {}| = {} (rtol={}, atol={})",
                    dtype, value_a, value_b, diff, rtol, atol
                )
            }
            NumericsError::ShapeMismatch { expected, actual } => {
                write!(
                    f,
                    "Shape mismatch: expected {:?}, got {:?}",
                    expected, actual
                )
            }
            NumericsError::StrictModeViolation(msg) => {
                write!(f, "Strict mode violation: {}", msg)
            }
        }
    }
}

impl std::error::Error for NumericsError {}

/// Validation result for tensor comparisons.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether validation passed.
    pub passed: bool,
    /// Maximum absolute difference observed.
    pub max_abs_diff: f64,
    /// Maximum relative difference observed.
    pub max_rel_diff: f64,
    /// Number of elements that exceeded tolerance.
    pub num_violations: usize,
    /// Total number of elements compared.
    pub total_elements: usize,
}

impl ValidationResult {
    /// Create a new validation result.
    pub fn new(
        passed: bool,
        max_abs_diff: f64,
        max_rel_diff: f64,
        num_violations: usize,
        total_elements: usize,
    ) -> Self {
        Self {
            passed,
            max_abs_diff,
            max_rel_diff,
            num_violations,
            total_elements,
        }
    }

    /// Get violation rate as a percentage.
    pub fn violation_rate(&self) -> f64 {
        if self.total_elements == 0 {
            0.0
        } else {
            (self.num_violations as f64 / self.total_elements as f64) * 100.0
        }
    }
}

/// Numerics validator for comparing tensor operations.
pub struct NumericsValidator {
    config: NumericsConfig,
}

impl NumericsValidator {
    /// Create a new validator with default configuration.
    pub fn new() -> Self {
        Self { config: NumericsConfig::default() }
    }

    /// Create a new validator with custom configuration.
    pub fn with_config(config: NumericsConfig) -> Self {
        Self { config }
    }

    /// Validate two arrays of values against numerics policy.
    pub fn validate_arrays(
        &self,
        reference: &[f32],
        test: &[f32],
        dtype: DType,
    ) -> Result<ValidationResult, NumericsError> {
        if reference.len() != test.len() {
            return Err(NumericsError::ShapeMismatch {
                expected: vec![reference.len()],
                actual: vec![test.len()],
            });
        }

        let tolerance = self.config.tolerance_for(dtype);
        let mut max_abs_diff = 0.0f64;
        let mut max_rel_diff = 0.0f64;
        let mut num_violations = 0usize;

        for (_i, (&ref_val, &test_val)) in reference.iter().zip(test.iter()).enumerate() {
            let ref_val_f64 = ref_val as f64;
            let test_val_f64 = test_val as f64;
            let diff = (ref_val_f64 - test_val_f64).abs();
            let max_abs = ref_val_f64.abs().max(test_val_f64.abs());
            let rel_diff = if max_abs > 0.0 { diff / max_abs } else { 0.0 };

            max_abs_diff = max_abs_diff.max(diff);
            max_rel_diff = max_rel_diff.max(rel_diff);

            if !tolerance.is_close(ref_val_f64, test_val_f64) {
                num_violations += 1;
                if self.config.strict_mode {
                    return Err(NumericsError::ToleranceExceeded {
                        dtype,
                        value_a: ref_val_f64,
                        value_b: test_val_f64,
                        diff,
                        rtol: tolerance.rtol,
                        atol: tolerance.atol,
                    });
                }
            }
        }

        let passed = num_violations == 0 || (!self.config.strict_mode && num_violations < reference.len() / 100); // Allow < 1% violations in non-strict mode

        Ok(ValidationResult::new(
            passed,
            max_abs_diff,
            max_rel_diff,
            num_violations,
            reference.len(),
        ))
    }

    /// Get the numerics configuration.
    pub fn config(&self) -> &NumericsConfig {
        &self.config
    }

    /// Set the numerics configuration.
    pub fn set_config(&mut self, config: NumericsConfig) {
        self.config = config;
    }
}

impl Default for NumericsValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tolerance_is_close() {
        let tol = Tolerance::new(1e-5, 1e-8);
        
        // Values within tolerance
        assert!(tol.is_close(1.0, 1.000001));
        assert!(tol.is_close(0.0, 1e-9));
        
        // Values outside tolerance
        assert!(!tol.is_close(1.0, 1.01));
        assert!(!tol.is_close(0.0, 1e-5));
    }

    #[test]
    fn test_dtype_display() {
        assert_eq!(DType::F32.to_string(), "f32");
        assert_eq!(DType::F16.to_string(), "f16");
        assert_eq!(DType::BF16.to_string(), "bf16");
        assert_eq!(DType::F64.to_string(), "f64");
    }

    #[test]
    fn test_numerics_config_default() {
        let config = NumericsConfig::default();
        assert!(!config.strict_mode);
        assert!(!config.use_float64_reference);
    }

    #[test]
    fn test_numerics_config_with_strict_mode() {
        let config = NumericsConfig::new().with_strict_mode(true);
        assert!(config.strict_mode);
    }

    #[test]
    fn test_tolerance_for_dtype() {
        let config = NumericsConfig::default();
        
        let f32_tol = config.tolerance_for(DType::F32);
        let f16_tol = config.tolerance_for(DType::F16);
        
        // F16 should have looser tolerances
        assert!(f16_tol.rtol > f32_tol.rtol);
        assert!(f16_tol.atol > f32_tol.atol);
    }

    #[test]
    fn test_validate_within_tolerance() {
        let config = NumericsConfig::default();
        assert!(config.validate(1.0, 1.000001, DType::F32).is_ok());
    }

    #[test]
    fn test_validate_outside_tolerance() {
        let config = NumericsConfig::default();
        assert!(config.validate(1.0, 1.01, DType::F32).is_err());
    }

    #[test]
    fn test_validation_result_violation_rate() {
        let result = ValidationResult::new(true, 0.001, 0.001, 5, 1000);
        assert_eq!(result.violation_rate(), 0.5);
    }

    #[test]
    fn test_numerics_validator_identical_arrays() {
        let validator = NumericsValidator::new();
        let reference = vec![1.0f32, 2.0, 3.0];
        let test = vec![1.0f32, 2.0, 3.0];
        
        let result = validator.validate_arrays(&reference, &test, DType::F32).unwrap();
        assert!(result.passed);
        assert_eq!(result.num_violations, 0);
    }

    #[test]
    fn test_numerics_validator_small_differences() {
        let validator = NumericsValidator::new();
        let reference = vec![1.0f32, 2.0, 3.0];
        let test = vec![1.000001f32, 2.000001, 3.000001];
        
        let result = validator.validate_arrays(&reference, &test, DType::F32).unwrap();
        assert!(result.passed);
    }

    #[test]
    fn test_numerics_validator_shape_mismatch() {
        let validator = NumericsValidator::new();
        let reference = vec![1.0f32, 2.0, 3.0];
        let test = vec![1.0f32, 2.0];
        
        let result = validator.validate_arrays(&reference, &test, DType::F32);
        assert!(result.is_err());
    }

    #[test]
    fn test_numerics_validator_strict_mode() {
        let config = NumericsConfig::new().with_strict_mode(true);
        let validator = NumericsValidator::with_config(config);
        
        let reference = vec![1.0f32, 2.0, 3.0];
        let test = vec![1.01f32, 2.0, 3.0]; // First element outside tolerance
        
        let result = validator.validate_arrays(&reference, &test, DType::F32);
        assert!(result.is_err());
    }
}
