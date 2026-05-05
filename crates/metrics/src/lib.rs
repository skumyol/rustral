//! Metrics and Logging for Rustral
//!
//! Provides production-ready metric collection, logging, and export
//! for training and inference workloads.
//!
//! # Features
//!
//! - **Console Logging**: Real-time training metrics to stdout
//! - **TensorBoard Export**: Write summaries for visualization
//! - **Generic Metrics Trait**: Pluggable metric backends
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use rustral_metrics::{MetricsLogger, ConsoleLogger, TensorBoardWriter};
//!
//! let mut logger = MetricsLogger::new();
//! logger.add_backend(Box::new(ConsoleLogger::new()));
//! logger.add_backend(Box::new(TensorBoardWriter::new("./runs")));
//!
//! for epoch in 0..10 {
//!     logger.log_scalar("loss/train", 0.5, epoch);
//!     logger.log_scalar("accuracy/train", 0.95, epoch);
//!     logger.step();
//! }
//! ```

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// A single scalar metric value.
#[derive(Clone, Debug)]
pub struct ScalarMetric {
    pub name: String,
    pub value: f64,
    pub step: u64,
    pub timestamp: u64, // Unix timestamp in seconds
}

/// A histogram metric (for weight distributions, gradients, etc.)
#[derive(Clone, Debug)]
pub struct HistogramMetric {
    pub name: String,
    pub bins: Vec<f64>,
    pub counts: Vec<u64>,
    pub step: u64,
    pub timestamp: u64,
}

/// Backend trait for metric writers.
pub trait MetricsBackend: Send + Sync {
    /// Log a scalar value.
    fn log_scalar(&mut self, metric: &ScalarMetric);

    /// Log a histogram.
    fn log_histogram(&mut self, metric: &HistogramMetric);

    /// Flush any buffered writes.
    fn flush(&mut self);

    /// Backend name.
    fn name(&self) -> &str;
}

/// Simple console logger that prints metrics in a readable format.
pub struct ConsoleLogger {
    enabled: bool,
    scalar_buffer: Vec<ScalarMetric>,
    flush_interval: usize,
}

impl ConsoleLogger {
    pub fn new() -> Self {
        Self { enabled: true, scalar_buffer: Vec::with_capacity(100), flush_interval: 10 }
    }

    pub fn with_flush_interval(mut self, interval: usize) -> Self {
        self.flush_interval = interval;
        self
    }

    pub fn disable(mut self) -> Self {
        self.enabled = false;
        self
    }
}

impl Default for ConsoleLogger {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsBackend for ConsoleLogger {
    fn log_scalar(&mut self, metric: &ScalarMetric) {
        if !self.enabled {
            return;
        }
        self.scalar_buffer.push(metric.clone());
        if self.scalar_buffer.len() >= self.flush_interval {
            self.flush();
        }
    }

    fn log_histogram(&mut self, _metric: &HistogramMetric) {
        // Console doesn't show histograms by default
    }

    fn flush(&mut self) {
        if self.scalar_buffer.is_empty() {
            return;
        }

        // Group by step
        let mut by_step: HashMap<u64, Vec<&ScalarMetric>> = HashMap::new();
        for metric in &self.scalar_buffer {
            by_step.entry(metric.step).or_default().push(metric);
        }

        let mut steps: Vec<u64> = by_step.keys().copied().collect();
        steps.sort();

        for step in steps {
            let metrics = by_step.get(&step).unwrap();
            let mut parts: Vec<String> =
                metrics.iter().map(|m| format!("{}={:.6}", m.name, m.value)).collect();
            parts.sort();

            println!("[step {}] {}", step, parts.join(", "));
        }

        self.scalar_buffer.clear();
    }

    fn name(&self) -> &str {
        "console"
    }
}

/// TensorBoard-compatible writer (simple text-based format for compatibility).
pub struct TensorBoardWriter {
    log_dir: String,
    run_name: String,
    scalars: Vec<ScalarMetric>,
    histograms: Vec<HistogramMetric>,
}

impl TensorBoardWriter {
    pub fn new<P: AsRef<Path>>(log_dir: P) -> Self {
        let log_dir = log_dir.as_ref().to_string_lossy().to_string();
        std::fs::create_dir_all(&log_dir).ok();

        Self {
            log_dir,
            run_name: format!(
                "run_{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
            ),
            scalars: Vec::new(),
            histograms: Vec::new(),
        }
    }

    pub fn with_run_name(mut self, name: &str) -> Self {
        self.run_name = name.to_string();
        self
    }
}

impl MetricsBackend for TensorBoardWriter {
    fn log_scalar(&mut self, metric: &ScalarMetric) {
        self.scalars.push(metric.clone());
    }

    fn log_histogram(&mut self, metric: &HistogramMetric) {
        self.histograms.push(metric.clone());
    }

    fn flush(&mut self) {
        if self.scalars.is_empty() && self.histograms.is_empty() {
            return;
        }

        // Write scalars as simple JSON lines for compatibility
        let filename = format!("{}/{}.jsonl", self.log_dir, self.run_name);
        if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(&filename) {
            for metric in &self.scalars {
                let line = serde_json::json!({
                    "type": "scalar",
                    "name": metric.name,
                    "value": metric.value,
                    "step": metric.step,
                    "timestamp": metric.timestamp,
                });
                writeln!(file, "{}", line).ok();
            }
        }

        self.scalars.clear();
        self.histograms.clear();
    }

    fn name(&self) -> &str {
        "tensorboard"
    }
}

/// Main metrics logger that aggregates multiple backends.
pub struct MetricsLogger {
    backends: Vec<Box<dyn MetricsBackend>>,
    current_step: u64,
}

impl MetricsLogger {
    pub fn new() -> Self {
        Self { backends: Vec::new(), current_step: 0 }
    }

    /// Add a backend.
    pub fn add_backend(&mut self, backend: Box<dyn MetricsBackend>) {
        self.backends.push(backend);
    }

    /// Log a scalar metric.
    pub fn log_scalar(&mut self, name: &str, value: f64) {
        let metric = ScalarMetric {
            name: name.to_string(),
            value,
            step: self.current_step,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        for backend in &mut self.backends {
            backend.log_scalar(&metric);
        }
    }

    /// Log a scalar with explicit step.
    pub fn log_scalar_with_step(&mut self, name: &str, value: f64, step: u64) {
        let metric = ScalarMetric {
            name: name.to_string(),
            value,
            step,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        for backend in &mut self.backends {
            backend.log_scalar(&metric);
        }
    }

    /// Log a histogram.
    pub fn log_histogram(&mut self, name: &str, values: &[f64]) {
        let (bins, counts) = compute_histogram(values, 30);
        let metric = HistogramMetric {
            name: name.to_string(),
            bins,
            counts,
            step: self.current_step,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        for backend in &mut self.backends {
            backend.log_histogram(&metric);
        }
    }

    /// Increment step counter.
    pub fn step(&mut self) {
        self.current_step += 1;
        self.flush();
    }

    /// Set step explicitly.
    pub fn set_step(&mut self, step: u64) {
        self.current_step = step;
    }

    /// Flush all backends.
    pub fn flush(&mut self) {
        for backend in &mut self.backends {
            backend.flush();
        }
    }

    /// Get current step.
    pub fn current_step(&self) -> u64 {
        self.current_step
    }

    /// Training summary at end of epoch/batch.
    pub fn log_training_summary(&mut self, epoch: u64, loss: f64, accuracy: Option<f64>, lr: Option<f64>) {
        self.set_step(epoch);
        self.log_scalar("epoch", epoch as f64);
        self.log_scalar("loss/train", loss);
        if let Some(acc) = accuracy {
            self.log_scalar("accuracy/train", acc);
        }
        if let Some(lr) = lr {
            self.log_scalar("lr", lr);
        }
        self.flush();
    }
}

impl Default for MetricsLogger {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute histogram bins from values.
fn compute_histogram(values: &[f64], num_bins: usize) -> (Vec<f64>, Vec<u64>) {
    if values.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;

    if range == 0.0 {
        return (vec![min], vec![values.len() as u64]);
    }

    let bin_width = range / num_bins as f64;
    let mut counts = vec![0u64; num_bins];

    for &val in values {
        let bin = ((val - min) / bin_width).min((num_bins - 1) as f64) as usize;
        counts[bin] += 1;
    }

    let bins: Vec<f64> = (0..=num_bins).map(|i| min + i as f64 * bin_width).collect();

    (bins, counts)
}

/// Simple progress bar for training loops.
pub struct ProgressBar {
    total: u64,
    current: u64,
    width: usize,
}

impl ProgressBar {
    pub fn new(total: u64) -> Self {
        Self { total, current: 0, width: 40 }
    }

    pub fn update(&mut self, current: u64) {
        self.current = current;
        self.draw();
    }

    pub fn increment(&mut self) {
        self.current += 1;
        self.draw();
    }

    pub fn finish(&mut self) {
        self.current = self.total;
        self.draw();
        println!();
    }

    fn draw(&self) {
        let pct = if self.total > 0 { self.current as f64 / self.total as f64 } else { 0.0 };
        let filled = (pct * self.width as f64) as usize;
        let empty = self.width - filled;

        let bar: String = format!(
            "[{}{}] {:>3.0}% ({}/{})",
            "█".repeat(filled),
            "░".repeat(empty),
            pct * 100.0,
            self.current,
            self.total
        );

        print!("\r{}", bar);
        std::io::stdout().flush().ok();
    }
}

/// Thread-safe metrics logger for multi-threaded training.
pub type ThreadSafeMetricsLogger = Arc<Mutex<MetricsLogger>>;

/// Convenience function to create a thread-safe logger.
pub fn create_logger() -> ThreadSafeMetricsLogger {
    let mut logger = MetricsLogger::new();
    logger.add_backend(Box::new(ConsoleLogger::new()));
    Arc::new(Mutex::new(logger))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_console_logger() {
        let mut logger = ConsoleLogger::new().with_flush_interval(2);
        let metric = ScalarMetric { name: "loss".to_string(), value: 0.5, step: 0, timestamp: 0 };
        logger.log_scalar(&metric);
        logger.flush();
    }

    #[test]
    fn test_tensorboard_writer() {
        let mut logger = TensorBoardWriter::new("/tmp/test_runs");
        let metric = ScalarMetric { name: "accuracy".to_string(), value: 0.95, step: 10, timestamp: 0 };
        logger.log_scalar(&metric);
        logger.flush();
    }

    #[test]
    fn test_metrics_logger() {
        let mut logger = MetricsLogger::new();
        logger.add_backend(Box::new(ConsoleLogger::new().disable()));

        logger.log_scalar("loss", 1.0);
        logger.log_scalar("loss", 0.8);
        logger.step();
        logger.log_scalar("loss", 0.6);
        logger.step();

        assert_eq!(logger.current_step(), 2);
    }

    #[test]
    fn test_histogram() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let (bins, counts) = compute_histogram(&values, 10);

        assert_eq!(bins.len(), 11); // 10 bins = 11 edges
        assert_eq!(counts.len(), 10);

        let total_count: u64 = counts.iter().sum();
        assert_eq!(total_count, 100);
    }

    #[test]
    fn test_progress_bar() {
        let mut pb = ProgressBar::new(100);
        pb.update(50);
        pb.increment();
        pb.finish();
    }

    #[test]
    fn test_console_logger_default() {
        let mut logger = ConsoleLogger::new();
        let metric = ScalarMetric { name: "loss".to_string(), value: 0.5, step: 0, timestamp: 0 };
        logger.log_scalar(&metric);
        logger.log_histogram(&HistogramMetric {
            name: "h".to_string(),
            bins: vec![0.0, 1.0],
            counts: vec![1, 2],
            step: 0,
            timestamp: 0,
        });
        logger.flush();
    }

    #[test]
    fn test_console_logger_disable() {
        let mut logger = ConsoleLogger::new().disable();
        logger.log_scalar(&ScalarMetric { name: "loss".to_string(), value: 0.5, step: 0, timestamp: 0 });
        logger.flush();
    }

    #[test]
    fn test_tensorboard_with_run_name() {
        let logger = TensorBoardWriter::new("/tmp/test_runs").with_run_name("my_run");
        assert_eq!(logger.run_name, "my_run");
    }

    #[test]
    fn test_metrics_logger_with_step() {
        let mut logger = MetricsLogger::new();
        logger.add_backend(Box::new(ConsoleLogger::new().disable()));
        logger.log_scalar_with_step("loss", 0.5, 5);
        assert_eq!(logger.current_step(), 0);
    }

    #[test]
    fn test_metrics_logger_histogram() {
        let mut logger = MetricsLogger::new();
        logger.add_backend(Box::new(ConsoleLogger::new().disable()));
        logger.log_histogram("vals", &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_metrics_logger_training_summary() {
        let mut logger = MetricsLogger::new();
        logger.add_backend(Box::new(ConsoleLogger::new().disable()));
        logger.log_training_summary(1, 0.5, Some(0.9), Some(0.001));
    }

    #[test]
    fn test_histogram_empty() {
        let (bins, counts) = compute_histogram(&[], 10);
        assert!(bins.is_empty());
        assert!(counts.is_empty());
    }

    #[test]
    fn test_histogram_zero_range() {
        let (bins, counts) = compute_histogram(&[5.0, 5.0, 5.0], 10);
        assert_eq!(bins, vec![5.0]);
        assert_eq!(counts, vec![3]);
    }

    #[test]
    fn test_create_logger() {
        let _logger = create_logger();
    }
}
