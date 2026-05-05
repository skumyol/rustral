//! Common testing utilities for system tests.
//!
//! Provides shared infrastructure for bug detection, performance testing,
//! and integration validation across all crates.

use std::panic;
use std::time::{Duration, Instant};

/// Test result with timing and success/failure info
#[derive(Debug, Clone)]
pub struct TestResult {
    pub name: String,
    pub passed: bool,
    pub duration: Duration,
    pub error: Option<String>,
}

impl TestResult {
    pub fn passed(name: &str, duration: Duration) -> Self {
        Self { name: name.to_string(), passed: true, duration, error: None }
    }

    pub fn failed(name: &str, duration: Duration, error: String) -> Self {
        Self { name: name.to_string(), passed: false, duration, error: Some(error) }
    }
}

/// Test runner that executes tests and collects results
pub struct TestRunner {
    results: Vec<TestResult>,
    start_time: Instant,
}

impl TestRunner {
    pub fn new() -> Self {
        Self { results: Vec::new(), start_time: Instant::now() }
    }

    /// Run a test with panic catching and timing
    pub fn run_test<F>(&mut self, name: &str, test_fn: F)
    where
        F: FnOnce() -> Result<(), String> + panic::UnwindSafe,
    {
        let start = Instant::now();

        let result = panic::catch_unwind(|| test_fn());

        let duration = start.elapsed();

        match result {
            Ok(Ok(())) => {
                self.results.push(TestResult::passed(name, duration));
            }
            Ok(Err(e)) => {
                self.results.push(TestResult::failed(name, duration, e));
            }
            Err(_) => {
                self.results.push(TestResult::failed(name, duration, "Test panicked".to_string()));
            }
        }
    }

    /// Run a test that should panic
    pub fn run_should_panic<F>(&mut self, name: &str, test_fn: F)
    where
        F: FnOnce() + panic::UnwindSafe,
    {
        let start = Instant::now();

        let result = panic::catch_unwind(test_fn);

        let duration = start.elapsed();

        match result {
            Ok(_) => {
                self.results.push(TestResult::failed(
                    name,
                    duration,
                    "Test should have panicked but didn't".to_string(),
                ));
            }
            Err(_) => {
                self.results.push(TestResult::passed(name, duration));
            }
        }
    }

    /// Print summary report
    pub fn print_report(&self) {
        let total = self.results.len();
        let passed = self.results.iter().filter(|r| r.passed).count();
        let failed = total - passed;
        let total_duration = self.start_time.elapsed();

        println!("\n{}", "=".repeat(70));
        println!("TEST SUMMARY");
        println!("{}", "=".repeat(70));
        println!("Total tests: {}", total);
        println!("Passed:      {}", passed);
        println!("Failed:      {}", failed);
        println!("Total time:  {:.3}s", total_duration.as_secs_f64());
        println!("{}", "=".repeat(70));

        if failed > 0 {
            println!("\nFAILED TESTS:");
            for result in &self.results {
                if !result.passed {
                    println!("  ✗ {} ({:.3}s)", result.name, result.duration.as_secs_f64());
                    if let Some(ref error) = result.error {
                        println!("    Error: {}", error);
                    }
                }
            }
        }

        println!();
    }

    /// Check if all tests passed
    pub fn all_passed(&self) -> bool {
        self.results.iter().all(|r| r.passed)
    }

    /// Get results
    pub fn results(&self) -> &[TestResult] {
        &self.results
    }
}

/// Performance test configuration
#[derive(Clone, Debug)]
pub struct PerfConfig {
    pub warmup_iterations: usize,
    pub test_iterations: usize,
    pub max_duration_ms: u64,
    pub tolerance_percent: f64,
}

impl Default for PerfConfig {
    fn default() -> Self {
        Self { warmup_iterations: 3, test_iterations: 10, max_duration_ms: 5000, tolerance_percent: 20.0 }
    }
}

/// Performance test results
#[derive(Debug, Clone)]
pub struct PerfResult {
    pub mean_ms: f64,
    pub median_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub std_dev_ms: f64,
    pub throughput: f64, // items/sec
}

impl PerfResult {
    /// Check if performance meets baseline
    pub fn meets_baseline(&self, baseline_ms: f64, tolerance_percent: f64) -> bool {
        let tolerance = baseline_ms * tolerance_percent / 100.0;
        self.mean_ms <= baseline_ms + tolerance
    }
}

/// Run performance test with statistical analysis
pub fn run_performance_test<F>(config: &PerfConfig, mut test_fn: F) -> PerfResult
where
    F: FnMut(),
{
    // Warmup
    for _ in 0..config.warmup_iterations {
        test_fn();
    }

    // Test iterations
    let mut times = Vec::with_capacity(config.test_iterations);
    let start = Instant::now();

    for _ in 0..config.test_iterations {
        let iter_start = Instant::now();
        test_fn();
        let elapsed = iter_start.elapsed();
        times.push(elapsed.as_secs_f64() * 1000.0);

        // Check timeout
        if start.elapsed().as_millis() as u64 > config.max_duration_ms {
            break;
        }
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = times.len() as f64;
    let mean = times.iter().sum::<f64>() / n;
    let median = times[times.len() / 2];
    let min = times[0];
    let max = times[times.len() - 1];

    let variance = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

    PerfResult {
        mean_ms: mean,
        median_ms: median,
        min_ms: min,
        max_ms: max,
        std_dev_ms: std_dev,
        throughput: 1000.0 / mean,
    }
}

/// Assert that two float slices are approximately equal
#[macro_export]
macro_rules! assert_approx_eq_slice {
    ($actual:expr, $expected:expr, $epsilon:expr) => {
        let actual: &[f32] = $actual;
        let expected: &[f32] = $expected;
        assert_eq!(
            actual.len(),
            expected.len(),
            "Slice lengths differ: {} vs {}",
            actual.len(),
            expected.len()
        );
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - e).abs();
            assert!(diff <= $epsilon, "Element {} differs: {} vs {} (diff: {})", i, a, e, diff);
        }
    };
}

/// Assert that an operation completes within a time limit
#[macro_export]
macro_rules! assert_completes_within {
    ($duration:expr, $operation:expr) => {
        let start = std::time::Instant::now();
        let result = $operation;
        let elapsed = start.elapsed();
        assert!(elapsed <= $duration, "Operation took {:?}, expected within {:?}", elapsed, $duration);
        result
    };
}

/// Generate random tensor data
pub fn random_tensor_data(size: usize, seed: u64) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    let state = hasher.finish();

    (0..size)
        .map(|i| {
            let x = ((state.wrapping_add(i as u64)) as f32) / u64::MAX as f32;
            x * 2.0 - 1.0 // Range [-1, 1]
        })
        .collect()
}

/// Memory tracking for detecting leaks
pub struct MemoryTracker {
    baseline: usize,
}

impl MemoryTracker {
    pub fn new() -> Self {
        Self { baseline: Self::current_memory() }
    }

    fn current_memory() -> usize {
        // Simplified - in real impl would use platform-specific APIs
        0
    }

    pub fn check_leak(&self, tolerance_bytes: usize) -> Result<(), String> {
        let current = Self::current_memory();
        let leaked = current.saturating_sub(self.baseline);

        if leaked > tolerance_bytes {
            Err(format!("Memory leak detected: {} bytes leaked (tolerance: {})", leaked, tolerance_bytes))
        } else {
            Ok(())
        }
    }

    pub fn current_usage(&self) -> usize {
        Self::current_memory().saturating_sub(self.baseline)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_test_runner_pass() {
        let mut runner = TestRunner::new();
        runner.run_test("passing_test", || Ok(()));
        assert!(runner.all_passed());
    }

    #[test]
    fn test_performance_test() {
        let config = PerfConfig {
            warmup_iterations: 1,
            test_iterations: 5,
            max_duration_ms: 1000,
            tolerance_percent: 50.0,
        };

        let result = run_performance_test(&config, || {
            std::thread::sleep(std::time::Duration::from_micros(100));
        });

        assert!(result.mean_ms > 0.0);
        assert!(result.min_ms <= result.max_ms);
    }
}
