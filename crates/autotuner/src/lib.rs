//! Auto-Tuner for the Rustral
#![allow(dead_code)]
//!
//! Provides automatic kernel configuration search, block size optimization,
//! and algorithm selection for GPU operations. Results are cached for
//! reuse across runs.
//!
//! # CI, determinism, and opt-out
//!
//! - **`TunerConfig::enabled`** — If `false`, [`AutoTuner::tune`](crate::AutoTuner::tune) skips the
//!   search loop and returns a default configuration (`TunerConfig::disabled()`).
//! - **`TunerConfig::ci_mode`** — When `true`, tuning sessions cap how many candidate configs run
//!   (bounded budget). Combine with `enabled: false` in CI when you must not run any search.
//! - **`TunerConfig::ci_safe()`** — Preset with `ci_mode` and tighter time/iteration limits.
//! - **Cache** — With `use_cache`, a cache hit re-benchmarks the stored config so reported timings
//!   are real, not placeholders.
//!
//! # Example
//!
//! ```rust,ignore
//! use rustral_autotuner::{AutoTuner, KernelConfig, TunerConfig};
//!
//! let tuner = AutoTuner::new(TunerConfig::default());
//!
//! // Tune a specific kernel
//! let config = tuner.tune_kernel(
//!     "matmul_f32",
//!     &[1024, 1024, 1024], // Input shapes
//!     || { /* kernel benchmark */ }
//! );
//! ```

mod cache;
mod kernel_config;
mod search;
mod tuner;

pub use cache::{CacheEntry, ConfigCache, TuningCache};
pub use kernel_config::{BlockSize, ConfigSpace, KernelConfig, MatmulAlgorithm, MatmulDimBucket, MatmulShapeBucket, WorkgroupConfig};
pub use search::{BayesianSearch, GridSearch, RandomSearch, SearchStrategy};
pub use tuner::{AutoTuner, TunerConfig, TuningResult, TuningSession};

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// A tunable kernel operation.
pub trait TunableKernel {
    /// Unique identifier for this kernel type.
    fn kernel_id(&self) -> &'static str;

    /// Get the configuration space for this kernel.
    fn config_space(&self) -> ConfigSpace;

    /// Apply a configuration and return a runnable closure.
    fn with_config(&self, config: &KernelConfig) -> Box<dyn FnMut()>;

    /// Get input characteristics that affect tuning.
    fn input_signature(&self) -> Vec<usize>;
}

/// Benchmark a kernel configuration.
pub fn benchmark_kernel<F>(mut kernel: F, warmup: usize, iterations: usize) -> Duration
where
    F: FnMut(),
{
    // Warmup runs
    for _ in 0..warmup {
        kernel();
    }

    // Timed runs
    let start = Instant::now();
    for _ in 0..iterations {
        kernel();
    }
    start.elapsed() / iterations as u32
}

/// Configuration for a specific operation type.
#[derive(Debug, Clone, PartialEq)]
pub enum OpConfig {
    /// Matrix multiplication configuration.
    Matmul { algorithm: MatmulAlgorithm, block_m: usize, block_n: usize, block_k: usize, num_stages: usize },
    /// Convolution configuration.
    Conv2d { algorithm: ConvAlgorithm, tile_size: usize, num_filters: usize },
    /// Reduction configuration.
    Reduce { block_size: usize, items_per_thread: usize, algorithm: ReduceAlgorithm },
    /// Element-wise configuration.
    Elementwise { block_size: usize, items_per_thread: usize, vector_width: usize },
    /// Attention configuration.
    Attention { block_size_m: usize, block_size_n: usize, use_flash: bool },
}

/// Convolution algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ConvAlgorithm {
    /// Direct convolution.
    Direct,
    /// Winograd convolution.
    Winograd,
    /// FFT-based convolution.
    Fft,
    /// Im2Col + GEMM.
    Im2Col,
    /// Implicit GEMM.
    ImplicitGemm,
}

/// Reduction algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ReduceAlgorithm {
    /// Tree-based reduction.
    Tree,
    /// Warp shuffle reduction.
    WarpShuffle,
    /// Atomic reduction.
    Atomic,
    /// Two-pass reduction for large arrays.
    TwoPass,
}

/// Apply a tuned configuration to a kernel.
pub trait ConfigurableKernel {
    /// Apply configuration and return configured kernel.
    fn apply_config(&mut self, config: &OpConfig);

    /// Get current configuration.
    fn current_config(&self) -> Option<&OpConfig>;
}

/// Performance metrics from tuning.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerfMetrics {
    /// Average execution time.
    pub avg_time_us: f64,
    /// Standard deviation.
    pub std_dev_us: f64,
    /// Throughput in GFLOPs or GB/s.
    pub throughput: f64,
    /// Memory bandwidth utilization (%).
    pub bandwidth_pct: f64,
    /// Compute utilization (%).
    pub compute_pct: f64,
}

impl PerfMetrics {
    /// Create metrics from duration and operation characteristics.
    pub fn from_duration(duration: Duration, flops: u64, bytes: u64) -> Self {
        let time_us = duration.as_micros() as f64;
        let time_s = duration.as_secs_f64();

        let throughput_gflops = if time_s > 0.0 { (flops as f64 / time_s) / 1e9 } else { 0.0 };

        // Assume peak bandwidth of 900 GB/s for H100, 600 GB/s for A100
        let peak_bw_gbps = 600.0;
        let achieved_bw_gbps = if time_s > 0.0 { (bytes as f64 / time_s) / 1e9 } else { 0.0 };
        let bandwidth_pct = (achieved_bw_gbps / peak_bw_gbps * 100.0).min(100.0);

        Self {
            avg_time_us: time_us,
            std_dev_us: 0.0, // Would need multiple runs
            throughput: throughput_gflops,
            bandwidth_pct,
            compute_pct: 0.0, // Would need hardware counters
        }
    }

    /// Score this configuration (higher is better).
    pub fn score(&self) -> f64 {
        // Composite score: throughput with bonus for efficiency
        self.throughput * (1.0 + self.bandwidth_pct / 100.0)
    }
}

impl Default for PerfMetrics {
    fn default() -> Self {
        Self { avg_time_us: f64::MAX, std_dev_us: 0.0, throughput: 0.0, bandwidth_pct: 0.0, compute_pct: 0.0 }
    }
}

/// Factory for creating kernels with optimal configurations.
pub struct KernelFactory {
    #[allow(dead_code)]
    tuner: AutoTuner,
    #[allow(dead_code)]
    default_configs: HashMap<String, OpConfig>,
}

impl KernelFactory {
    /// Create a new kernel factory.
    pub fn new(tuner: AutoTuner) -> Self {
        Self { tuner, default_configs: Self::load_default_configs() }
    }

    /// Get or tune a kernel for given input shapes.
    pub fn get_kernel<K: TunableKernel>(&mut self, kernel: K) -> (K, OpConfig) {
        let key = format!("{}:{:?}", kernel.kernel_id(), kernel.input_signature());

        if let Some(config) = self.tuner.get_cached(&key) {
            return (kernel, config);
        }

        // Run tuning session
        let result = self.tuner.tune(&kernel);
        (kernel, result.best_config)
    }

    fn load_default_configs() -> HashMap<String, OpConfig> {
        let mut configs = HashMap::new();

        // Default matmul config
        configs.insert(
            "matmul_default".to_string(),
            OpConfig::Matmul {
                algorithm: MatmulAlgorithm::Tiled,
                block_m: 128,
                block_n: 128,
                block_k: 32,
                num_stages: 2,
            },
        );

        // Default conv config
        configs.insert(
            "conv2d_default".to_string(),
            OpConfig::Conv2d { algorithm: ConvAlgorithm::ImplicitGemm, tile_size: 16, num_filters: 64 },
        );

        configs
    }
}

/// Utility to pre-tune common operations.
pub struct PreTuner {
    #[allow(dead_code)]
    tuner: AutoTuner,
    #[allow(dead_code)]
    common_shapes: Vec<Vec<usize>>,
}

impl PreTuner {
    /// Create a pre-tuner with common ML shapes.
    pub fn new(tuner: AutoTuner) -> Self {
        let common_shapes = vec![
            // BERT/GPT shapes
            vec![512, 768, 768],
            vec![1024, 768, 768],
            vec![2048, 768, 768],
            vec![512, 3072, 768], // FFN expand
            vec![512, 768, 3072], // FFN project
            // ResNet shapes
            vec![64, 64, 224, 224],
            vec![128, 128, 112, 112],
            vec![256, 256, 56, 56],
            // Large model shapes
            vec![8192, 12288, 12288],
            vec![8192, 49152, 12288],
            vec![8192, 12288, 49152],
        ];

        Self { tuner, common_shapes }
    }

    /// Pre-tune all common shapes.
    pub fn pre_tune_all(&mut self) -> HashMap<String, OpConfig> {
        let mut results = HashMap::new();

        for shape in &self.common_shapes {
            let key = format!("matmul:{:?}", shape);
            // Would actually tune here with real kernels
            results.insert(
                key,
                OpConfig::Matmul {
                    algorithm: MatmulAlgorithm::Tiled,
                    block_m: 128,
                    block_n: 128,
                    block_k: 32,
                    num_stages: 2,
                },
            );
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perf_metrics() {
        let duration = Duration::from_micros(100);
        let metrics = PerfMetrics::from_duration(duration, 1_000_000_000, 1_000_000);

        assert!(metrics.avg_time_us >= 100.0);
        assert!(metrics.throughput > 0.0);
    }

    #[test]
    fn test_op_config_matmul() {
        let config = OpConfig::Matmul {
            algorithm: MatmulAlgorithm::Tiled,
            block_m: 128,
            block_n: 128,
            block_k: 32,
            num_stages: 2,
        };

        match config {
            OpConfig::Matmul { block_m, block_n, block_k, .. } => {
                assert_eq!(block_m, 128);
                assert_eq!(block_n, 128);
                assert_eq!(block_k, 32);
            }
            _ => panic!("Expected Matmul config"),
        }
    }

    #[test]
    fn test_kernel_factory() {
        let tuner = AutoTuner::new(TunerConfig::default());
        let factory = KernelFactory::new(tuner);

        assert!(!factory.default_configs.is_empty());
    }

    #[test]
    fn test_benchmark_kernel() {
        let duration = benchmark_kernel(|| std::thread::sleep(Duration::from_micros(100)), 1, 3);
        assert!(duration > Duration::ZERO);
    }

    #[test]
    fn test_perf_metrics_default_and_score() {
        let default = PerfMetrics::default();
        assert_eq!(default.avg_time_us, f64::MAX);
        assert_eq!(default.throughput, 0.0);
        assert_eq!(default.score(), 0.0);

        let metrics = PerfMetrics::from_duration(Duration::from_micros(100), 1_000_000, 1_000_000);
        assert!(metrics.score() > 0.0);
    }

    #[test]
    fn test_op_config_variants() {
        let matmul = OpConfig::Matmul {
            algorithm: MatmulAlgorithm::Tiled,
            block_m: 128,
            block_n: 128,
            block_k: 32,
            num_stages: 2,
        };
        assert_eq!(matmul, matmul);

        let conv = OpConfig::Conv2d { algorithm: ConvAlgorithm::Direct, tile_size: 16, num_filters: 64 };
        assert_eq!(conv, conv);

        let reduce =
            OpConfig::Reduce { block_size: 256, items_per_thread: 8, algorithm: ReduceAlgorithm::Tree };
        assert_eq!(reduce, reduce);

        let elem = OpConfig::Elementwise { block_size: 256, items_per_thread: 4, vector_width: 4 };
        assert_eq!(elem, elem);

        let attn = OpConfig::Attention { block_size_m: 64, block_size_n: 64, use_flash: true };
        assert_eq!(attn, attn);
    }

    #[test]
    fn test_kernel_factory_get_kernel() {
        use crate::kernel_config::ConfigSpace;

        struct DummyKernel;
        impl TunableKernel for DummyKernel {
            fn kernel_id(&self) -> &'static str {
                "test_kernel"
            }
            fn config_space(&self) -> ConfigSpace {
                ConfigSpace::matmul_reduced()
            }
            fn with_config(&self, _config: &KernelConfig) -> Box<dyn FnMut()> {
                Box::new(|| {})
            }
            fn input_signature(&self) -> Vec<usize> {
                vec![64, 64, 64]
            }
        }

        let tuner = AutoTuner::new(TunerConfig::default());
        let mut factory = KernelFactory::new(tuner);
        let (_kernel, config) = factory.get_kernel(DummyKernel);
        // Just verify we got some config back
        let _ = config;
    }

    #[test]
    fn test_pre_tuner() {
        let tuner = AutoTuner::new(TunerConfig::default());
        let mut pre = PreTuner::new(tuner);
        let results = pre.pre_tune_all();
        assert!(!results.is_empty());
    }
}
