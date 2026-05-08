//! Operation Profiler for Cross-Backend Performance Monitoring
//!
//! Provides device-agnostic performance monitoring for tensor operations
//! across all backends. Enables performance regression testing and
//! optimization identification.
//!
//! # Features
//!
//! - Per-operation timing statistics
//! - Cross-backend performance comparison
//! - Automatic performance regression detection
//! - Hot path identification
//!
//! # Example
//!
//! ```rust,ignore
//! use rustral_core::operation_profiler::{OperationProfiler, OperationGuard};
//!
//! let profiler = OperationProfiler::new();
//!
//! {
//!     let _guard = profiler.profile_operation("matmul");
//!     // ... perform operation ...
//! } // Timing automatically recorded
//!
//! profiler.print_report();
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Performance statistics for a single operation.
#[derive(Debug, Clone)]
pub struct OperationStats {
    pub count: usize,
    pub total_duration: Duration,
    pub min_duration: Duration,
    pub max_duration: Duration,
    pub last_duration: Duration,
}

impl OperationStats {
    pub fn new() -> Self {
        Self {
            count: 0,
            total_duration: Duration::ZERO,
            min_duration: Duration::MAX,
            max_duration: Duration::ZERO,
            last_duration: Duration::ZERO,
        }
    }

    pub fn avg_duration(&self) -> Duration {
        if self.count == 0 {
            Duration::ZERO
        } else {
            self.total_duration / self.count as u32
        }
    }

    pub fn record(&mut self, duration: Duration) {
        self.count += 1;
        self.total_duration += duration;
        self.min_duration = self.min_duration.min(duration);
        self.max_duration = self.max_duration.max(duration);
        self.last_duration = duration;
    }
}

impl Default for OperationStats {
    fn default() -> Self {
        Self::new()
    }
}

/// RAII guard for operation profiling.
pub struct OperationGuard {
    profiler: Option<Arc<Mutex<OperationProfiler>>>,
    operation_name: String,
    start_time: Instant,
}

impl OperationGuard {
    pub fn new(profiler: Arc<Mutex<OperationProfiler>>, operation_name: &str) -> Self {
        Self {
            profiler: Some(profiler),
            operation_name: operation_name.to_string(),
            start_time: Instant::now(),
        }
    }
}

impl Drop for OperationGuard {
    fn drop(&mut self) {
        if let Some(ref profiler) = self.profiler {
            let duration = self.start_time.elapsed();
            if let Ok(mut p) = profiler.lock() {
                p.record_operation_internal(&self.operation_name, duration);
            }
        }
    }
}

/// Thread-safe operation profiler.
pub struct OperationProfiler {
    stats: HashMap<String, OperationStats>,
    enabled: bool,
    start_time: Instant,
}

impl OperationProfiler {
    /// Create a new operation profiler.
    pub fn new() -> Self {
        Self {
            stats: HashMap::new(),
            enabled: true,
            start_time: Instant::now(),
        }
    }

    /// Enable profiling.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable profiling.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Check if enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Create an operation guard (RAII).
    pub fn profile_operation(&self, operation_name: &str) -> OperationGuard {
        OperationGuard::new(Arc::new(Mutex::new(self.clone())), operation_name)
    }

    /// Record operation timing internally.
    pub fn record_operation_internal(&mut self, operation_name: &str, duration: Duration) {
        if !self.enabled {
            return;
        }

        let stats = self.stats
            .entry(operation_name.to_string())
            .or_default();
        stats.record(duration);
    }

    /// Get statistics for a specific operation.
    pub fn get_stats(&self, operation_name: &str) -> Option<&OperationStats> {
        self.stats.get(operation_name)
    }

    /// Get all operation statistics.
    pub fn all_stats(&self) -> &HashMap<String, OperationStats> {
        &self.stats
    }

    /// Get the most expensive operations by average duration.
    pub fn most_expensive_ops(&self, limit: usize) -> Vec<(String, Duration)> {
        let mut ops: Vec<_> = self.stats
            .iter()
            .map(|(name, stats)| (name.clone(), stats.avg_duration()))
            .collect();
        ops.sort_by_key(|b| std::cmp::Reverse(b.1));
        ops.into_iter().take(limit).collect()
    }

    /// Get the most frequently called operations.
    pub fn most_frequent_ops(&self, limit: usize) -> Vec<(String, usize)> {
        let mut ops: Vec<_> = self.stats
            .iter()
            .map(|(name, stats)| (name.clone(), stats.count))
            .collect();
        ops.sort_by_key(|b| std::cmp::Reverse(b.1));
        ops.into_iter().take(limit).collect()
    }

    /// Print performance report.
    pub fn print_report(&self) {
        println!("\n{}", "=".repeat(70));
        println!("Operation Profiler Report");
        println!("{}", "=".repeat(70));

        let elapsed = self.start_time.elapsed();
        println!("Profiling Duration: {:.2}s", elapsed.as_secs_f64());
        println!("Total Operations: {}", self.stats.len());
        println!("Total Calls: {}", self.stats.values().map(|s| s.count).sum::<usize>());

        if !self.stats.is_empty() {
            println!("\nMost Expensive Operations (by avg time):");
            let expensive = self.most_expensive_ops(10);
            for (name, avg) in expensive {
                let stats = self.stats.get(&name).unwrap();
                println!(
                    "  {:30} {:10.3}ms (avg) | {:10.3}ms (max) | {:6} calls",
                    name,
                    avg.as_secs_f64() * 1000.0,
                    stats.max_duration.as_secs_f64() * 1000.0,
                    stats.count
                );
            }

            println!("\nMost Frequent Operations:");
            let frequent = self.most_frequent_ops(10);
            for (name, count) in frequent {
                let stats = self.stats.get(&name).unwrap();
                println!(
                    "  {:30} {:6} calls | {:10.3}ms (avg) | {:10.3}ms (total)",
                    name,
                    count,
                    stats.avg_duration().as_secs_f64() * 1000.0,
                    stats.total_duration.as_secs_f64() * 1000.0
                );
            }
        }

        println!("{}", "=".repeat(70));
    }

    /// Export statistics as JSON.
    pub fn export_json(&self, path: &str) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(path)?;

        writeln!(file, "{{")?;
        writeln!(file, "  \"profiling_duration_ms\": {},", elapsed_ms(&self.start_time))?;
        writeln!(file, "  \"operations\": {{")?;

        let mut first = true;
        for (name, stats) in &self.stats {
            if !first {
                writeln!(file, ",")?;
            }
            first = false;

            writeln!(file, "    \"{}\": {{", name)?;
            writeln!(file, "      \"count\": {},", stats.count)?;
            writeln!(file, "      \"total_ms\": {},", stats.total_duration.as_secs_f64() * 1000.0)?;
            writeln!(file, "      \"avg_ms\": {},", stats.avg_duration().as_secs_f64() * 1000.0)?;
            writeln!(file, "      \"min_ms\": {},", stats.min_duration.as_secs_f64() * 1000.0)?;
            writeln!(file, "      \"max_ms\": {},", stats.max_duration.as_secs_f64() * 1000.0)?;
            writeln!(file, "      \"last_ms\": {}", stats.last_duration.as_secs_f64() * 1000.0)?;
            writeln!(file, "    }}")?;
        }

        writeln!(file, "  }}")?;
        writeln!(file, "}}")?;

        Ok(())
    }

    /// Clear all statistics.
    pub fn clear(&mut self) {
        self.stats.clear();
        self.start_time = Instant::now();
    }

    /// Check for performance regression compared to baseline.
    pub fn check_regression(&self, baseline: &OperationProfiler, threshold: f64) -> Vec<String> {
        let mut regressions = Vec::new();

        for (name, current_stats) in &self.stats {
            if let Some(baseline_stats) = baseline.get_stats(name) {
                let current_avg = current_stats.avg_duration().as_secs_f64();
                let baseline_avg = baseline_stats.avg_duration().as_secs_f64();

                if baseline_avg > 0.0 {
                    let ratio = current_avg / baseline_avg;
                    if ratio > (1.0 + threshold) {
                        regressions.push(format!(
                            "{}: {:.2}x slower ({:.3}ms -> {:.3}ms)",
                            name,
                            ratio,
                            baseline_avg * 1000.0,
                            current_avg * 1000.0
                        ));
                    }
                }
            }
        }

        regressions
    }
}

impl Default for OperationProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for OperationProfiler {
    fn clone(&self) -> Self {
        Self {
            stats: self.stats.clone(),
            enabled: self.enabled,
            start_time: self.start_time,
        }
    }
}

/// Helper function to get elapsed milliseconds.
fn elapsed_ms(start: &Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_operation_stats() {
        let mut stats = OperationStats::new();
        assert_eq!(stats.count, 0);
        assert_eq!(stats.avg_duration(), Duration::ZERO);

        stats.record(Duration::from_millis(100));
        stats.record(Duration::from_millis(200));

        assert_eq!(stats.count, 2);
        assert_eq!(stats.avg_duration(), Duration::from_millis(150));
        assert_eq!(stats.min_duration, Duration::from_millis(100));
        assert_eq!(stats.max_duration, Duration::from_millis(200));
    }

    #[test]
    fn test_operation_guard() {
        let profiler = Arc::new(Mutex::new(OperationProfiler::new()));

        {
            let _guard = OperationGuard::new(profiler.clone(), "test_op");
            thread::sleep(Duration::from_millis(10));
        }

        let p = profiler.lock().unwrap();
        let stats = p.get_stats("test_op").unwrap();
        assert_eq!(stats.count, 1);
        assert!(stats.last_duration >= Duration::from_millis(10));
    }

    #[test]
    fn test_profiler_enable_disable() {
        let mut profiler = OperationProfiler::new();
        assert!(profiler.is_enabled());
        profiler.disable();
        assert!(!profiler.is_enabled());
        profiler.enable();
        assert!(profiler.is_enabled());
    }

    #[test]
    fn test_most_expensive_ops() {
        let mut profiler = OperationProfiler::new();
        profiler.record_operation_internal("fast", Duration::from_millis(1));
        profiler.record_operation_internal("slow", Duration::from_millis(100));

        let expensive = profiler.most_expensive_ops(10);
        assert_eq!(expensive.len(), 2);
        assert_eq!(expensive[0].0, "slow");
        assert_eq!(expensive[1].0, "fast");
    }

    #[test]
    fn test_most_frequent_ops() {
        let mut profiler = OperationProfiler::new();
        for _ in 0..10 {
            profiler.record_operation_internal("frequent", Duration::from_millis(1));
        }
        profiler.record_operation_internal("rare", Duration::from_millis(1));

        let frequent = profiler.most_frequent_ops(10);
        assert_eq!(frequent.len(), 2);
        assert_eq!(frequent[0].0, "frequent");
        assert_eq!(frequent[0].1, 10);
    }

    #[test]
    fn test_clear() {
        let mut profiler = OperationProfiler::new();
        profiler.record_operation_internal("test", Duration::from_millis(1));
        assert!(!profiler.stats.is_empty());
        profiler.clear();
        assert!(profiler.stats.is_empty());
    }

    #[test]
    fn test_check_regression() {
        let mut baseline = OperationProfiler::new();
        baseline.record_operation_internal("test", Duration::from_millis(100));

        let mut current = OperationProfiler::new();
        current.record_operation_internal("test", Duration::from_millis(150));

        let regressions = current.check_regression(&baseline, 0.4); // 40% threshold
        assert_eq!(regressions.len(), 1);
        assert!(regressions[0].contains("test"));
    }

    #[test]
    fn test_export_json() {
        let mut profiler = OperationProfiler::new();
        profiler.record_operation_internal("test", Duration::from_millis(100));
        let tmpfile = std::env::temp_dir().join("rustral_op_profile_test.json");
        profiler.export_json(tmpfile.to_str().unwrap()).unwrap();
        std::fs::remove_file(&tmpfile).ok();
    }

    #[test]
    fn test_print_report() {
        let mut profiler = OperationProfiler::new();
        profiler.record_operation_internal("test", Duration::from_millis(100));
        profiler.print_report();
    }

    #[test]
    fn test_clone() {
        let mut profiler = OperationProfiler::new();
        profiler.record_operation_internal("test", Duration::from_millis(100));
        let cloned = profiler.clone();
        assert_eq!(cloned.stats.len(), profiler.stats.len());
    }

    #[test]
    fn test_default() {
        let profiler: OperationProfiler = Default::default();
        assert!(profiler.is_enabled());
    }
}
