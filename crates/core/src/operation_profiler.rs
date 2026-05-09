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
//! - CPU/GPU breakdown for device-aware profiling
//! - Shape bucket recording for data-driven optimization decisions
//! - Automatic performance regression detection
//! - Hot path identification
//!
//! # Example
//!
//! ```rust,ignore
//! use rustral_core::operation_profiler::{OperationProfiler, OperationGuard, ProfilingHooks};
//!
//! let hooks = ProfilingHooks {
//!     cpu_gpu_breakdown: true,
//!     per_op_timing: true,
//!     memory_tracking: false,
//!     shape_bucket_recording: true,
//! };
//! let profiler = OperationProfiler::with_hooks(hooks);
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

use serde::{Deserialize, Serialize};

/// Configuration for profiling hooks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfilingHooks {
    /// Enable CPU vs GPU timing breakdown.
    pub cpu_gpu_breakdown: bool,
    /// Enable per-operation timing statistics.
    pub per_op_timing: bool,
    /// Enable memory allocation tracking.
    pub memory_tracking: bool,
    /// Enable shape bucket recording for data-driven optimization.
    pub shape_bucket_recording: bool,
}

impl Default for ProfilingHooks {
    fn default() -> Self {
        Self {
            cpu_gpu_breakdown: false,
            per_op_timing: true,
            memory_tracking: false,
            shape_bucket_recording: false,
        }
    }
}

/// Device type for CPU/GPU breakdown.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceType {
    Cpu,
    Cuda,
    Metal,
    Wgpu,
    Unknown,
}

impl DeviceType {
    /// Detect device type from device name string.
    pub fn from_name(name: &str) -> Self {
        if name.contains("cuda") {
            DeviceType::Cuda
        } else if name.contains("metal") {
            DeviceType::Metal
        } else if name.contains("wgpu") {
            DeviceType::Wgpu
        } else if name.contains("cpu") {
            DeviceType::Cpu
        } else {
            DeviceType::Unknown
        }
    }
}

/// Shape bucket for categorizing tensor shapes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ShapeBucket {
    /// Small dimensions (< 512)
    Small,
    /// Medium dimensions (512-2048)
    Medium,
    /// Large dimensions (2048-8192)
    Large,
    /// Extra large dimensions (> 8192)
    XLarge,
    /// Specific matmul bucket (M, N, K)
    Matmul { m: MatmulDim, n: MatmulDim, k: MatmulDim },
}

/// Matmul dimension bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatmulDim {
    Tiny,   // < 256
    Small,  // 256-512
    Medium, // 512-2048
    Large,  // 2048-8192
    XLarge, // > 8192
}

impl MatmulDim {
    /// Create bucket from dimension size.
    pub fn from_size(size: usize) -> Self {
        if size < 256 {
            MatmulDim::Tiny
        } else if size <= 512 {
            MatmulDim::Small
        } else if size < 2048 {
            MatmulDim::Medium
        } else if size < 8192 {
            MatmulDim::Large
        } else {
            MatmulDim::XLarge
        }
    }
}

impl ShapeBucket {
    /// Create shape bucket from tensor dimensions.
    pub fn from_dims(dims: &[usize]) -> Self {
        if dims.len() == 2 {
            // Assume matmul: [M, K] or [K, N] or [M, N]
            let max_dim = *dims.iter().max().unwrap_or(&0);
            if max_dim < 512 {
                ShapeBucket::Small
            } else if max_dim < 2048 {
                ShapeBucket::Medium
            } else if max_dim < 8192 {
                ShapeBucket::Large
            } else {
                ShapeBucket::XLarge
            }
        } else {
            // For other tensors, use max dimension
            let max_dim = dims.iter().copied().max().unwrap_or(0);
            if max_dim < 512 {
                ShapeBucket::Small
            } else if max_dim < 2048 {
                ShapeBucket::Medium
            } else if max_dim < 8192 {
                ShapeBucket::Large
            } else {
                ShapeBucket::XLarge
            }
        }
    }

    /// Create matmul-specific bucket from (M, N, K) dimensions.
    pub fn from_matmul_dims(m: usize, n: usize, k: usize) -> Self {
        ShapeBucket::Matmul {
            m: MatmulDim::from_size(m),
            n: MatmulDim::from_size(n),
            k: MatmulDim::from_size(k),
        }
    }
}

/// Enhanced performance statistics for a single operation.
#[derive(Debug, Clone)]
pub struct OperationStats {
    pub count: usize,
    pub total_duration: Duration,
    pub min_duration: Duration,
    pub max_duration: Duration,
    pub last_duration: Duration,
    /// CPU timing (if cpu_gpu_breakdown enabled).
    pub cpu_duration: Duration,
    /// GPU timing (if cpu_gpu_breakdown enabled).
    pub gpu_duration: Duration,
    /// Device type for this operation.
    pub device_type: DeviceType,
    /// Shape buckets observed for this operation.
    pub shape_buckets: Vec<ShapeBucket>,
}

impl OperationStats {
    pub fn new(device_type: DeviceType) -> Self {
        Self {
            count: 0,
            total_duration: Duration::ZERO,
            min_duration: Duration::MAX,
            max_duration: Duration::ZERO,
            last_duration: Duration::ZERO,
            cpu_duration: Duration::ZERO,
            gpu_duration: Duration::ZERO,
            device_type,
            shape_buckets: Vec::new(),
        }
    }

    pub fn avg_duration(&self) -> Duration {
        if self.count == 0 {
            Duration::ZERO
        } else {
            self.total_duration / self.count as u32
        }
    }

    pub fn record(&mut self, duration: Duration, cpu_time: Option<Duration>, gpu_time: Option<Duration>) {
        self.count += 1;
        self.total_duration += duration;
        self.min_duration = self.min_duration.min(duration);
        self.max_duration = self.max_duration.max(duration);
        self.last_duration = duration;

        if let Some(cpu) = cpu_time {
            self.cpu_duration += cpu;
        }
        if let Some(gpu) = gpu_time {
            self.gpu_duration += gpu;
        }
    }

    pub fn record_shape_bucket(&mut self, bucket: ShapeBucket) {
        if !self.shape_buckets.contains(&bucket) {
            self.shape_buckets.push(bucket);
        }
    }
}

impl Default for OperationStats {
    fn default() -> Self {
        Self::new(DeviceType::Unknown)
    }
}

/// RAII guard for operation profiling.
pub struct OperationGuard {
    profiler: Option<Arc<Mutex<OperationProfiler>>>,
    operation_name: String,
    start_time: Instant,
    device_type: DeviceType,
    shape_bucket: Option<ShapeBucket>,
    cpu_start: Option<Instant>,
    gpu_start: Option<Instant>,
}

impl OperationGuard {
    pub fn new(profiler: Arc<Mutex<OperationProfiler>>, operation_name: &str) -> Self {
        Self {
            profiler: Some(profiler),
            operation_name: operation_name.to_string(),
            start_time: Instant::now(),
            device_type: DeviceType::Unknown,
            shape_bucket: None,
            cpu_start: None,
            gpu_start: None,
        }
    }

    pub fn with_device_type(mut self, device_type: DeviceType) -> Self {
        self.device_type = device_type;
        self
    }

    pub fn with_shape_bucket(mut self, bucket: ShapeBucket) -> Self {
        self.shape_bucket = Some(bucket);
        self
    }

    pub fn with_cpu_timing(mut self) -> Self {
        self.cpu_start = Some(Instant::now());
        self
    }

    pub fn with_gpu_timing(mut self) -> Self {
        self.gpu_start = Some(Instant::now());
        self
    }
}

impl Drop for OperationGuard {
    fn drop(&mut self) {
        if let Some(ref profiler) = self.profiler {
            let duration = self.start_time.elapsed();
            let cpu_time = self.cpu_start.map(|t| t.elapsed());
            let gpu_time = self.gpu_start.map(|t| t.elapsed());

            if let Ok(mut p) = profiler.lock() {
                p.record_operation_internal(
                    &self.operation_name,
                    duration,
                    cpu_time,
                    gpu_time,
                    self.device_type,
                    self.shape_bucket.clone(),
                );
            }
        }
    }
}

/// Thread-safe operation profiler.
pub struct OperationProfiler {
    stats: HashMap<String, OperationStats>,
    enabled: bool,
    start_time: Instant,
    hooks: ProfilingHooks,
}

/// Stable, machine-readable profiler snapshot for regression checks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProfilerSnapshot {
    pub elapsed_secs: f64,
    pub hooks: ProfilingHooks,
    pub total_calls: usize,
    pub ops: Vec<SnapshotOp>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SnapshotOp {
    pub name: String,
    pub count: usize,
    pub avg_ms: f64,
    pub max_ms: f64,
    pub device_type: DeviceType,
}

impl OperationProfiler {
    /// Create a new operation profiler with default hooks.
    pub fn new() -> Self {
        Self {
            stats: HashMap::new(),
            enabled: true,
            start_time: Instant::now(),
            hooks: ProfilingHooks::default(),
        }
    }

    /// Create a new operation profiler with custom hooks.
    pub fn with_hooks(hooks: ProfilingHooks) -> Self {
        Self {
            stats: HashMap::new(),
            enabled: true,
            start_time: Instant::now(),
            hooks,
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

    /// Get the profiling hooks.
    pub fn hooks(&self) -> ProfilingHooks {
        self.hooks
    }

    /// Create an operation guard (RAII).
    pub fn profile_operation(&self, operation_name: &str) -> OperationGuard {
        OperationGuard::new(Arc::new(Mutex::new(self.clone())), operation_name)
    }

    /// Record operation timing internally.
    pub fn record_operation_internal(
        &mut self,
        operation_name: &str,
        duration: Duration,
        cpu_time: Option<Duration>,
        gpu_time: Option<Duration>,
        device_type: DeviceType,
        shape_bucket: Option<ShapeBucket>,
    ) {
        if !self.enabled {
            return;
        }

        let stats = self
            .stats
            .entry(operation_name.to_string())
            .or_insert_with(|| OperationStats::new(device_type));
        stats.record(duration, cpu_time, gpu_time);

        if let Some(bucket) = shape_bucket {
            stats.record_shape_bucket(bucket);
        }
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
        let mut ops: Vec<_> =
            self.stats.iter().map(|(name, stats)| (name.clone(), stats.avg_duration())).collect();
        ops.sort_by_key(|b| std::cmp::Reverse(b.1));
        ops.into_iter().take(limit).collect()
    }

    /// Get the most frequently called operations.
    pub fn most_frequent_ops(&self, limit: usize) -> Vec<(String, usize)> {
        let mut ops: Vec<_> = self.stats.iter().map(|(name, stats)| (name.clone(), stats.count)).collect();
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
        println!("Profiling Hooks: cpu_gpu={} per_op={} memory={} shapes={}",
                 self.hooks.cpu_gpu_breakdown,
                 self.hooks.per_op_timing,
                 self.hooks.memory_tracking,
                 self.hooks.shape_bucket_recording);

        if !self.stats.is_empty() {
            println!("\nMost Expensive Operations (by avg time):");
            let expensive = self.most_expensive_ops(10);
            for (name, avg) in expensive {
                let stats = self.stats.get(&name).unwrap();
                let device_info = format!("{:?}", stats.device_type);
                let cpu_gpu_info = if self.hooks.cpu_gpu_breakdown && (stats.cpu_duration > Duration::ZERO || stats.gpu_duration > Duration::ZERO) {
                    format!(" | CPU: {:.2}ms GPU: {:.2}ms",
                            stats.cpu_duration.as_secs_f64() * 1000.0,
                            stats.gpu_duration.as_secs_f64() * 1000.0)
                } else {
                    String::new()
                };
                let shape_info = if self.hooks.shape_bucket_recording && !stats.shape_buckets.is_empty() {
                    format!(" | Shapes: {:?}", stats.shape_buckets)
                } else {
                    String::new()
                };
                println!(
                    "  {:30} {:10.3}ms (avg) | {:10.3}ms (max) | {:6} calls | {:?}{}{}",
                    name,
                    avg.as_secs_f64() * 1000.0,
                    stats.max_duration.as_secs_f64() * 1000.0,
                    stats.count,
                    device_info,
                    cpu_gpu_info,
                    shape_info
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

    /// Produce a stable snapshot (top ops by avg time) for regression checks.
    pub fn snapshot(&self, limit: usize) -> ProfilerSnapshot {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let total_calls: usize = self.stats.values().map(|s| s.count).sum();

        let mut ops: Vec<_> = self
            .stats
            .iter()
            .map(|(name, stats)| SnapshotOp {
                name: name.clone(),
                count: stats.count,
                avg_ms: stats.avg_duration().as_secs_f64() * 1000.0,
                max_ms: stats.max_duration.as_secs_f64() * 1000.0,
                device_type: stats.device_type,
            })
            .collect();
        ops.sort_by(|a, b| b.avg_ms.partial_cmp(&a.avg_ms).unwrap_or(std::cmp::Ordering::Equal));
        ops.truncate(limit);

        ProfilerSnapshot { elapsed_secs: elapsed, hooks: self.hooks, total_calls, ops }
    }

    /// Print a JSON snapshot to stdout (stable format).
    pub fn print_snapshot_json(&self, limit: usize) {
        let snap = self.snapshot(limit);
        match serde_json::to_string(&snap) {
            Ok(s) => println!("{s}"),
            Err(e) => eprintln!("failed to serialize profiler snapshot: {e}"),
        }
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
            hooks: self.hooks,
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
        let mut stats = OperationStats::new(DeviceType::Cpu);
        assert_eq!(stats.count, 0);
        assert_eq!(stats.avg_duration(), Duration::ZERO);

        stats.record(Duration::from_millis(100), None, None);
        stats.record(Duration::from_millis(200), None, None);

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
        profiler.record_operation_internal("fast", Duration::from_millis(1), None, None, DeviceType::Cpu, None);
        profiler.record_operation_internal("slow", Duration::from_millis(100), None, None, DeviceType::Cpu, None);

        let expensive = profiler.most_expensive_ops(10);
        assert_eq!(expensive.len(), 2);
        assert_eq!(expensive[0].0, "slow");
        assert_eq!(expensive[1].0, "fast");
    }

    #[test]
    fn test_most_frequent_ops() {
        let mut profiler = OperationProfiler::new();
        for _ in 0..10 {
            profiler.record_operation_internal("frequent", Duration::from_millis(1), None, None, DeviceType::Cpu, None);
        }
        profiler.record_operation_internal("rare", Duration::from_millis(1), None, None, DeviceType::Cpu, None);

        let frequent = profiler.most_frequent_ops(10);
        assert_eq!(frequent.len(), 2);
        assert_eq!(frequent[0].0, "frequent");
        assert_eq!(frequent[0].1, 10);
    }

    #[test]
    fn test_clear() {
        let mut profiler = OperationProfiler::new();
        profiler.record_operation_internal("test", Duration::from_millis(1), None, None, DeviceType::Cpu, None);
        assert!(!profiler.stats.is_empty());
        profiler.clear();
        assert!(profiler.stats.is_empty());
    }

    #[test]
    fn test_check_regression() {
        let mut baseline = OperationProfiler::new();
        baseline.record_operation_internal("test", Duration::from_millis(100), None, None, DeviceType::Cpu, None);

        let mut current = OperationProfiler::new();
        current.record_operation_internal("test", Duration::from_millis(150), None, None, DeviceType::Cpu, None);

        let regressions = current.check_regression(&baseline, 0.4); // 40% threshold
        assert_eq!(regressions.len(), 1);
        assert!(regressions[0].contains("test"));
    }

    #[test]
    fn test_export_json() {
        let mut profiler = OperationProfiler::new();
        profiler.record_operation_internal("test", Duration::from_millis(100), None, None, DeviceType::Cpu, None);
        let tmpfile = std::env::temp_dir().join("rustral_op_profile_test.json");
        profiler.export_json(tmpfile.to_str().unwrap()).unwrap();
        std::fs::remove_file(&tmpfile).ok();
    }

    #[test]
    fn test_print_report() {
        let mut profiler = OperationProfiler::new();
        profiler.record_operation_internal("test", Duration::from_millis(100), None, None, DeviceType::Cpu, None);
        profiler.print_report();
    }

    #[test]
    fn test_clone() {
        let mut profiler = OperationProfiler::new();
        profiler.record_operation_internal("test", Duration::from_millis(100), None, None, DeviceType::Cpu, None);
        let cloned = profiler.clone();
        assert_eq!(cloned.stats.len(), profiler.stats.len());
    }

    #[test]
    fn test_default() {
        let profiler: OperationProfiler = Default::default();
        assert!(profiler.is_enabled());
    }

    #[test]
    fn test_profiling_hooks() {
        let hooks = ProfilingHooks {
            cpu_gpu_breakdown: true,
            per_op_timing: true,
            memory_tracking: false,
            shape_bucket_recording: true,
        };
        let profiler = OperationProfiler::with_hooks(hooks);
        assert!(profiler.hooks().cpu_gpu_breakdown);
        assert!(profiler.hooks().shape_bucket_recording);
    }

    #[test]
    fn test_device_type_detection() {
        assert_eq!(DeviceType::from_name("cuda:0"), DeviceType::Cuda);
        assert_eq!(DeviceType::from_name("cpu"), DeviceType::Cpu);
        assert_eq!(DeviceType::from_name("metal:0"), DeviceType::Metal);
        assert_eq!(DeviceType::from_name("wgpu"), DeviceType::Wgpu);
        assert_eq!(DeviceType::from_name("unknown"), DeviceType::Unknown);
    }

    #[test]
    fn test_shape_bucket_from_dims() {
        assert_eq!(ShapeBucket::from_dims(&[256, 256]), ShapeBucket::Small);
        assert_eq!(ShapeBucket::from_dims(&[1024, 1024]), ShapeBucket::Medium);
        assert_eq!(ShapeBucket::from_dims(&[4096, 4096]), ShapeBucket::Large);
        assert_eq!(ShapeBucket::from_dims(&[10000, 10000]), ShapeBucket::XLarge);
    }

    #[test]
    fn test_matmul_dim_buckets() {
        assert_eq!(MatmulDim::from_size(128), MatmulDim::Tiny);
        assert_eq!(MatmulDim::from_size(256), MatmulDim::Small);
        assert_eq!(MatmulDim::from_size(1024), MatmulDim::Medium);
        assert_eq!(MatmulDim::from_size(4096), MatmulDim::Large);
        assert_eq!(MatmulDim::from_size(10000), MatmulDim::XLarge);
    }

    #[test]
    fn test_shape_bucket_matmul() {
        let bucket = ShapeBucket::from_matmul_dims(512, 1024, 256);
        match bucket {
            ShapeBucket::Matmul { m, n, k } => {
                assert_eq!(m, MatmulDim::Small);
                assert_eq!(n, MatmulDim::Medium);
                assert_eq!(k, MatmulDim::Small);
            }
            _ => panic!("Expected Matmul bucket"),
        }
    }

    #[test]
    fn test_operation_guard_with_device_type() {
        let profiler = Arc::new(Mutex::new(OperationProfiler::new()));
        {
            let _guard = OperationGuard::new(profiler.clone(), "test_op")
                .with_device_type(DeviceType::Cuda)
                .with_cpu_timing()
                .with_gpu_timing();
            thread::sleep(Duration::from_millis(10));
        }

        let p = profiler.lock().unwrap();
        let stats = p.get_stats("test_op").unwrap();
        assert_eq!(stats.count, 1);
        assert_eq!(stats.device_type, DeviceType::Cuda);
    }

    #[test]
    fn test_operation_guard_with_shape_bucket() {
        let profiler = Arc::new(Mutex::new(OperationProfiler::with_hooks(ProfilingHooks {
            cpu_gpu_breakdown: false,
            per_op_timing: true,
            memory_tracking: false,
            shape_bucket_recording: true,
        })));
        {
            let _guard = OperationGuard::new(profiler.clone(), "test_op")
                .with_shape_bucket(ShapeBucket::Medium);
            thread::sleep(Duration::from_millis(10));
        }

        let p = profiler.lock().unwrap();
        let stats = p.get_stats("test_op").unwrap();
        assert_eq!(stats.count, 1);
        assert!(stats.shape_buckets.contains(&ShapeBucket::Medium));
    }
}
