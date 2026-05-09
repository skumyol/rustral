//! Test harness for validating fused operations against unfused baselines.
//!
//! Provides utilities to compare fused operation results with unfused
//! operation sequences to ensure numerical equivalence within tolerance.

use crate::numerics::{DType, NumericsValidator, ValidationResult};
use std::fmt;

/// Test configuration for fusion validation.
#[derive(Debug, Clone)]
pub struct FusionTestConfig {
    /// Whether to use strict numerics validation.
    pub strict_numerics: bool,
    /// Whether to use float64 reference for validation.
    pub use_float64_reference: bool,
    /// Number of random test cases to generate.
    pub num_random_cases: usize,
    /// Whether to print detailed comparison results.
    pub verbose: bool,
}

impl Default for FusionTestConfig {
    fn default() -> Self {
        Self {
            strict_numerics: false,
            use_float64_reference: false,
            num_random_cases: 10,
            verbose: false,
        }
    }
}

/// Result of a fusion validation test.
#[derive(Debug, Clone)]
pub struct FusionTestResult {
    /// Test name.
    pub test_name: String,
    /// Whether the test passed.
    pub passed: bool,
    /// Numerics validation result.
    pub validation_result: ValidationResult,
    /// Execution time for unfused version (in microseconds).
    pub unfused_time_us: f64,
    /// Execution time for fused version (in microseconds).
    pub fused_time_us: f64,
    /// Speedup factor (unfused_time / fused_time).
    pub speedup: f64,
}

impl FusionTestResult {
    /// Create a new fusion test result.
    pub fn new(
        test_name: String,
        passed: bool,
        validation_result: ValidationResult,
        unfused_time_us: f64,
        fused_time_us: f64,
    ) -> Self {
        let speedup = if fused_time_us > 0.0 {
            unfused_time_us / fused_time_us
        } else {
            1.0
        };
        
        Self {
            test_name,
            passed,
            validation_result,
            unfused_time_us,
            fused_time_us,
            speedup,
        }
    }

    /// Print a summary of the test result.
    pub fn print_summary(&self) {
        let status = if self.passed { "✓ PASS" } else { "✗ FAIL" };
        println!(
            "{}: {} | Speedup: {:.2}x | Violations: {}/{} | Max diff: {:.2e}",
            status,
            self.test_name,
            self.speedup,
            self.validation_result.num_violations,
            self.validation_result.total_elements,
            self.validation_result.max_abs_diff
        );
    }
}

/// Harness for testing fused operations.
pub struct FusionTestHarness {
    config: FusionTestConfig,
    validator: NumericsValidator,
}

impl FusionTestHarness {
    /// Create a new fusion test harness with default configuration.
    pub fn new() -> Self {
        Self {
            config: FusionTestConfig::default(),
            validator: NumericsValidator::new(),
        }
    }

    /// Create a new fusion test harness with custom configuration.
    pub fn with_config(config: FusionTestConfig) -> Self {
        let validator = NumericsValidator::new();
        Self { config, validator }
    }

    /// Run a fusion test comparing unfused and fused implementations.
    ///
    /// # Arguments
    ///
    /// * `test_name` - Name of the test
    /// * `unfused_fn` - Function that produces unfused results
    /// * `fused_fn` - Function that produces fused results
    /// * `dtype` - Data type for numerics validation
    ///
    /// # Returns
    ///
    /// Test result with validation and timing information.
    pub fn run_test<F1, F2>(
        &self,
        test_name: &str,
        unfused_fn: F1,
        fused_fn: F2,
        dtype: DType,
    ) -> FusionTestResult
    where
        F1: FnOnce() -> Vec<f32>,
        F2: FnOnce() -> Vec<f32>,
    {
        // Run unfused implementation and measure time
        let start = std::time::Instant::now();
        let unfused_result = unfused_fn();
        let unfused_time_us = start.elapsed().as_micros() as f64;

        // Run fused implementation and measure time
        let start = std::time::Instant::now();
        let fused_result = fused_fn();
        let fused_time_us = start.elapsed().as_micros() as f64;

        // Validate numerical equivalence
        let validation_result = self
            .validator
            .validate_arrays(&unfused_result, &fused_result, dtype)
            .unwrap_or_else(|_e| ValidationResult {
                passed: false,
                max_abs_diff: f64::MAX,
                max_rel_diff: f64::MAX,
                num_violations: usize::MAX,
                total_elements: unfused_result.len(),
            });

        let passed = validation_result.passed;

        let result = FusionTestResult::new(
            test_name.to_string(),
            passed,
            validation_result,
            unfused_time_us,
            fused_time_us,
        );

        if self.config.verbose {
            result.print_summary();
        }

        result
    }

    /// Run multiple fusion tests and return summary statistics.
    pub fn run_suite(
        &self,
        suite_name: &str,
        test_cases: Vec<(&str, Box<dyn FnOnce() -> Vec<f32>>, Box<dyn FnOnce() -> Vec<f32>>)>,
        dtype: DType,
    ) -> FusionTestSuiteResult {
        let mut results = Vec::new();
        let mut passed = 0;
        let mut total_speedup = 0.0f64;

        for (test_name, unfused_fn, fused_fn) in test_cases {
            // Run unfused implementation and measure time
            let start = std::time::Instant::now();
            let unfused_result = unfused_fn();
            let unfused_time_us = start.elapsed().as_micros() as f64;

            // Run fused implementation and measure time
            let start = std::time::Instant::now();
            let fused_result = fused_fn();
            let fused_time_us = start.elapsed().as_micros() as f64;

            // Validate numerical equivalence
            let validation_result = self
                .validator
                .validate_arrays(&unfused_result, &fused_result, dtype)
                .unwrap_or_else(|_e| ValidationResult {
                    passed: false,
                    max_abs_diff: f64::MAX,
                    max_rel_diff: f64::MAX,
                    num_violations: usize::MAX,
                    total_elements: unfused_result.len(),
                });

            let test_passed = validation_result.passed;
            let speedup = if fused_time_us > 0.0 {
                unfused_time_us / fused_time_us
            } else {
                1.0
            };

            if test_passed {
                passed += 1;
            }
            total_speedup += speedup;

            results.push(FusionTestResult::new(
                test_name.to_string(),
                test_passed,
                validation_result,
                unfused_time_us,
                fused_time_us,
            ));
        }

        let avg_speedup = if !results.is_empty() {
            total_speedup / results.len() as f64
        } else {
            1.0
        };

        FusionTestSuiteResult {
            suite_name: suite_name.to_string(),
            total_tests: results.len(),
            passed_tests: passed,
            avg_speedup,
            results,
        }
    }

    /// Get the test configuration.
    pub fn config(&self) -> &FusionTestConfig {
        &self.config
    }

    /// Set the test configuration.
    pub fn set_config(&mut self, config: FusionTestConfig) {
        self.config = config;
    }
}

impl Default for FusionTestHarness {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a fusion test suite.
#[derive(Debug, Clone)]
pub struct FusionTestSuiteResult {
    /// Suite name.
    pub suite_name: String,
    /// Total number of tests in the suite.
    pub total_tests: usize,
    /// Number of tests that passed.
    pub passed_tests: usize,
    /// Average speedup across all tests.
    pub avg_speedup: f64,
    /// Individual test results.
    pub results: Vec<FusionTestResult>,
}

impl FusionTestSuiteResult {
    /// Print a summary of the suite results.
    pub fn print_summary(&self) {
        println!("\n=== Fusion Test Suite: {} ===", self.suite_name);
        println!("Passed: {}/{}", self.passed_tests, self.total_tests);
        println!("Average speedup: {:.2}x", self.avg_speedup);
        
        if !self.results.is_empty() {
            println!("\nIndividual test results:");
            for result in &self.results {
                result.print_summary();
            }
        }
        println!("============================\n");
    }

    /// Whether all tests in the suite passed.
    pub fn all_passed(&self) -> bool {
        self.passed_tests == self.total_tests
    }
}

impl fmt::Display for FusionTestSuiteResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FusionTestSuiteResult: {} - {}/{} passed, avg speedup: {:.2}x",
            self.suite_name, self.passed_tests, self.total_tests, self.avg_speedup
        )
    }
}

/// Helper function to generate random test data.
pub fn generate_random_data(size: usize, seed: u64) -> Vec<f32> {
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;

    let mut rng = StdRng::seed_from_u64(seed);
    (0..size).map(|_| rng.gen::<f32>()).collect()
}

/// Helper function to generate constant test data.
pub fn generate_constant_data(size: usize, value: f32) -> Vec<f32> {
    vec![value; size]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fusion_test_config_default() {
        let config = FusionTestConfig::default();
        assert!(!config.strict_numerics);
        assert_eq!(config.num_random_cases, 10);
    }

    #[test]
    fn test_fusion_test_result_speedup() {
        let result = FusionTestResult::new(
            "test".to_string(),
            true,
            ValidationResult::new(true, 0.0, 0.0, 0, 100),
            1000.0,
            500.0,
        );
        assert_eq!(result.speedup, 2.0);
    }

    #[test]
    fn test_fusion_test_result_zero_fused_time() {
        let result = FusionTestResult::new(
            "test".to_string(),
            true,
            ValidationResult::new(true, 0.0, 0.0, 0, 100),
            1000.0,
            0.0,
        );
        assert_eq!(result.speedup, 1.0); // Avoid division by zero
    }

    #[test]
    fn test_fusion_test_harness_identical_results() {
        let harness = FusionTestHarness::new();
        let data = vec![1.0f32, 2.0, 3.0, 4.0];

        let result = harness.run_test(
            "identical_test",
            || data.clone(),
            || data.clone(),
            DType::F32,
        );

        assert!(result.passed);
        assert_eq!(result.validation_result.num_violations, 0);
    }

    #[test]
    fn test_fusion_test_harness_different_results() {
        let harness = FusionTestHarness::new();
        let data1 = vec![1.0f32, 2.0, 3.0, 4.0];
        let data2 = vec![1.0f32, 2.0, 3.0, 5.0]; // Last element different

        let result = harness.run_test(
            "different_test",
            || data1.clone(),
            || data2.clone(),
            DType::F32,
        );

        // Should fail due to difference outside tolerance
        assert!(!result.passed);
    }

    #[test]
    fn test_fusion_test_harness_suite() {
        let harness = FusionTestHarness::new();
        let test_cases = vec![
            ("test1", Box::new(|| vec![1.0f32, 2.0]) as Box<dyn FnOnce() -> Vec<f32>>, Box::new(|| vec![1.0f32, 2.0]) as Box<dyn FnOnce() -> Vec<f32>>),
            ("test2", Box::new(|| vec![3.0f32, 4.0]) as Box<dyn FnOnce() -> Vec<f32>>, Box::new(|| vec![3.0f32, 4.0]) as Box<dyn FnOnce() -> Vec<f32>>),
        ];

        let suite_result = harness.run_suite("test_suite", test_cases, DType::F32);
        
        assert_eq!(suite_result.total_tests, 2);
        assert_eq!(suite_result.passed_tests, 2);
        assert!(suite_result.all_passed());
    }

    #[test]
    fn test_generate_random_data() {
        let data = generate_random_data(100, 42);
        assert_eq!(data.len(), 100);
        // Should be deterministic with same seed
        let data2 = generate_random_data(100, 42);
        assert_eq!(data, data2);
    }

    #[test]
    fn test_generate_constant_data() {
        let data = generate_constant_data(10, 5.0);
        assert_eq!(data.len(), 10);
        assert!(data.iter().all(|&x| x == 5.0));
    }

    #[test]
    fn test_fusion_test_suite_result_display() {
        let result = FusionTestSuiteResult {
            suite_name: "test_suite".to_string(),
            total_tests: 10,
            passed_tests: 8,
            avg_speedup: 1.5,
            results: vec![],
        };
        
        let display_str = format!("{}", result);
        assert!(display_str.contains("test_suite"));
        assert!(display_str.contains("8/10"));
        assert!(display_str.contains("1.5"));
    }
}
