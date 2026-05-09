//! Main auto-tuner implementation.

use crate::cache::TuningCache;
use crate::kernel_config::KernelConfig;
use crate::search::{GridSearch, RandomSearch, SearchStrategy};
use crate::{benchmark_kernel, OpConfig, PerfMetrics, TunableKernel};
use std::time::{Duration, Instant};

/// Configuration for the auto-tuner.
#[derive(Debug, Clone)]
pub struct TunerConfig {
    /// Search strategy to use.
    pub strategy: TunerStrategy,
    /// Number of warmup iterations.
    pub warmup_iterations: usize,
    /// Number of timing iterations.
    pub timing_iterations: usize,
    /// Maximum tuning time per kernel.
    pub max_tuning_time: Duration,
    /// Use cached results if available.
    pub use_cache: bool,
    /// Cache results after tuning.
    pub cache_results: bool,
    /// Minimum improvement required to accept new config.
    pub min_improvement_pct: f64,
    /// Verbose output.
    pub verbose: bool,
    /// CI-safe mode: deterministic, cache-only, bounded budget.
    pub ci_mode: bool,
    /// Enable/disable autotuning (opt-out mechanism).
    pub enabled: bool,
    /// Device identifier for cache keying.
    pub device_id: String,
    /// Data type for cache keying.
    pub dtype: String,
}

impl TunerConfig {
    /// Fast tuning configuration.
    pub fn fast() -> Self {
        Self {
            strategy: TunerStrategy::Random { iterations: 10, seed: 42 },
            warmup_iterations: 2,
            timing_iterations: 5,
            max_tuning_time: Duration::from_secs(30),
            use_cache: true,
            cache_results: true,
            min_improvement_pct: 5.0,
            verbose: false,
            ci_mode: false,
            enabled: true,
            device_id: "default".to_string(),
            dtype: "f32".to_string(),
        }
    }

    /// CI-safe tuning configuration (deterministic, cache-only, bounded budget).
    pub fn ci_safe() -> Self {
        Self {
            strategy: TunerStrategy::Random { iterations: 5, seed: 42 }, // Bounded search
            warmup_iterations: 1,
            timing_iterations: 3,
            max_tuning_time: Duration::from_secs(10), // Bounded budget
            use_cache: true,
            cache_results: true,
            min_improvement_pct: 10.0,
            verbose: false,
            ci_mode: true, // Deterministic mode
            enabled: true,
            device_id: "default".to_string(),
            dtype: "f32".to_string(),
        }
    }

    /// Disabled configuration (opt-out mechanism).
    pub fn disabled() -> Self {
        Self {
            strategy: TunerStrategy::Random { iterations: 0, seed: 42 },
            warmup_iterations: 0,
            timing_iterations: 0,
            max_tuning_time: Duration::from_secs(0),
            use_cache: true,
            cache_results: false,
            min_improvement_pct: 0.0,
            verbose: false,
            ci_mode: true,
            enabled: false, // Explicit opt-out
            device_id: "default".to_string(),
            dtype: "f32".to_string(),
        }
    }

    /// Set CI mode (deterministic, bounded budget).
    pub fn with_ci_mode(mut self, ci_mode: bool) -> Self {
        self.ci_mode = ci_mode;
        if ci_mode {
            // Apply CI-safe constraints
            self.strategy = TunerStrategy::Random { iterations: 5, seed: 42 };
            self.max_tuning_time = Duration::from_secs(10);
            self.timing_iterations = 3;
            self.warmup_iterations = 1;
        }
        self
    }

    /// Enable or disable autotuning (opt-out mechanism).
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Set device identifier for cache keying.
    pub fn with_device_id(mut self, device_id: String) -> Self {
        self.device_id = device_id;
        self
    }

    /// Set data type for cache keying.
    pub fn with_dtype(mut self, dtype: String) -> Self {
        self.dtype = dtype;
        self
    }
}

impl Default for TunerConfig {
    fn default() -> Self {
        Self {
            strategy: TunerStrategy::Random { iterations: 50, seed: 42 },
            warmup_iterations: 3,
            timing_iterations: 10,
            max_tuning_time: Duration::from_secs(60),
            use_cache: true,
            cache_results: true,
            min_improvement_pct: 3.0,
            verbose: false,
            ci_mode: false,
            enabled: true,
            device_id: "default".to_string(),
            dtype: "f32".to_string(),
        }
    }
}

/// Search strategy selection.
#[derive(Debug, Clone)]
pub enum TunerStrategy {
    /// Exhaustive grid search.
    Grid,
    /// Random search.
    Random { iterations: usize, seed: u64 },
    /// Evolutionary search.
    Evolutionary { population: usize, generations: usize, seed: u64 },
    /// Bayesian optimization.
    Bayesian { iterations: usize },
}

/// Result of a tuning session.
#[derive(Debug, Clone)]
pub struct TuningResult {
    /// Best configuration found.
    pub best_config: OpConfig,
    /// Performance of best config.
    pub best_time_us: f64,
    /// Default configuration time for comparison.
    pub default_time_us: f64,
    /// Speedup achieved.
    pub speedup: f64,
    /// Number of configurations tried.
    pub configs_tried: usize,
    /// Total tuning time.
    pub tuning_time: Duration,
    /// All results.
    pub all_results: Vec<(OpConfig, PerfMetrics)>,
}

impl TuningResult {
    /// Calculate speedup over default.
    pub fn calculate_speedup(&mut self) {
        if self.default_time_us > 0.0 {
            self.speedup = self.default_time_us / self.best_time_us;
        }
    }

    /// Print tuning summary.
    pub fn print_summary(&self) {
        println!("\n{}", "=".repeat(60));
        println!("Tuning Results");
        println!("{}", "=".repeat(60));
        println!("Configurations tried: {}", self.configs_tried);
        println!("Tuning time: {:?}", self.tuning_time);
        println!("Best time: {:.2} µs", self.best_time_us);
        println!("Default time: {:.2} µs", self.default_time_us);
        println!("Speedup: {:.2}x", self.speedup);
        println!("{}", "=".repeat(60));
    }
}

/// Active tuning session for a specific kernel.
pub struct TuningSession<'a> {
    kernel_id: String,
    input_signature: Vec<usize>,
    tuner: &'a mut AutoTuner,
    start_time: Instant,
    results: Vec<(KernelConfig, PerfMetrics)>,
    best_config: Option<KernelConfig>,
    best_time: f64,
}

impl<'a> TuningSession<'a> {
    /// Create a new tuning session.
    fn new(kernel_id: String, input_signature: Vec<usize>, tuner: &'a mut AutoTuner) -> Self {
        Self {
            kernel_id,
            input_signature,
            tuner,
            start_time: Instant::now(),
            results: Vec::new(),
            best_config: None,
            best_time: f64::MAX,
        }
    }

    /// Run the tuning session.
    pub fn run<K: TunableKernel>(&mut self, kernel: &K) -> TuningResult {
        let space = kernel.config_space();

        // Create search strategy
        let mut strategy: Box<dyn SearchStrategy> = match &self.tuner.config.strategy {
            TunerStrategy::Grid => Box::new(GridSearch::new(&space)),
            TunerStrategy::Random { iterations, seed } => {
                Box::new(RandomSearch::new(space.clone(), *iterations, *seed))
            }
            _ => Box::new(RandomSearch::new(space.clone(), 50, 42)),
        };

        let mut configs_tried = 0;

        // Benchmark default config first
        let default_time = self.benchmark_default(kernel);

        // Search for better configs
        while let Some(config) = strategy.next_config(&space) {
            // Check time budget
            if self.start_time.elapsed() > self.tuner.config.max_tuning_time {
                if self.tuner.config.verbose {
                    println!("Tuning time budget exhausted");
                }
                break;
            }

            configs_tried += 1;

            // Apply config and benchmark
            let metrics = self.benchmark_with_config(kernel, &config);

            // Track results
            strategy.report_result(&config, metrics.avg_time_us);
            self.results.push((config.clone(), metrics));

            if metrics.avg_time_us < self.best_time {
                self.best_time = metrics.avg_time_us;
                self.best_config = Some(config);
            }

            if self.tuner.config.verbose && configs_tried % 10 == 0 {
                println!("Tried {} configs, best: {:.2} µs", configs_tried, self.best_time);
            }
        }

        // Convert to OpConfig result
        let op_results: Vec<(OpConfig, PerfMetrics)> =
            self.results.iter().map(|(kc, pm)| (self.kernel_config_to_op_config(kc), *pm)).collect();

        let best_op_config = self
            .best_config
            .as_ref()
            .map(|kc| self.kernel_config_to_op_config(kc))
            .unwrap_or_else(|| self.default_op_config());

        let mut result = TuningResult {
            best_config: best_op_config,
            best_time_us: self.best_time,
            default_time_us: default_time,
            speedup: 0.0,
            configs_tried,
            tuning_time: self.start_time.elapsed(),
            all_results: op_results,
        };

        result.calculate_speedup();

        // Cache the result
        if self.tuner.config.cache_results {
            if let Some(ref best) = self.best_config {
                let device_id = &self.tuner.config.device_id;
                let dtype = &self.tuner.config.dtype;
                // Try to create shape bucket from input signature for matmul
                let shape_bucket = if self.kernel_id.contains("matmul") && self.input_signature.len() >= 3 {
                    Some(crate::kernel_config::MatmulShapeBucket::from_dims(
                        self.input_signature[0],
                        self.input_signature[1],
                        self.input_signature[2],
                    ))
                } else {
                    None
                };

                self.tuner.cache.store_config(
                    &self.kernel_id,
                    device_id,
                    dtype,
                    shape_bucket.as_ref(),
                    best.clone(),
                    self.best_time,
                );
            }
        }

        result
    }

    /// Benchmark with default configuration.
    fn benchmark_default<K: TunableKernel>(&self, kernel: &K) -> f64 {
        let default_config = KernelConfig::default_for(kernel.kernel_id());
        let metrics = self.benchmark_with_config(kernel, &default_config);
        metrics.avg_time_us
    }

    /// Benchmark a specific configuration.
    fn benchmark_with_config<K: TunableKernel>(&self, kernel: &K, config: &KernelConfig) -> PerfMetrics {
        let mut bench_kernel = kernel.with_config(config);

        let duration = benchmark_kernel(
            || bench_kernel(),
            self.tuner.config.warmup_iterations,
            self.tuner.config.timing_iterations,
        );

        // Estimate FLOPs and bytes
        let (flops, bytes) = estimate_workload(kernel.kernel_id(), &self.input_signature);

        PerfMetrics::from_duration(duration, flops, bytes)
    }

    /// Convert KernelConfig to OpConfig.
    fn kernel_config_to_op_config(&self, config: &KernelConfig) -> OpConfig {
        // Simplified conversion
        use crate::kernel_config::{AlgorithmConfig, MatmulAlgorithm};
        use crate::{ConvAlgorithm as OpConvAlg, ReduceAlgorithm as OpReduceAlg};

        match &config.algorithm {
            AlgorithmConfig::Matmul(MatmulAlgorithm::Tiled) => OpConfig::Matmul {
                algorithm: MatmulAlgorithm::Tiled,
                block_m: config.params.get("tile_m").copied().unwrap_or(128) as usize,
                block_n: config.params.get("tile_n").copied().unwrap_or(128) as usize,
                block_k: config.params.get("tile_k").copied().unwrap_or(8) as usize,
                num_stages: 2,
            },
            AlgorithmConfig::Conv(_) => {
                OpConfig::Conv2d { algorithm: OpConvAlg::ImplicitGemm, tile_size: 16, num_filters: 64 }
            }
            AlgorithmConfig::Reduce(_) => OpConfig::Reduce {
                block_size: config.workgroup.x as usize,
                items_per_thread: config.params.get("items_per_thread").copied().unwrap_or(8) as usize,
                algorithm: OpReduceAlg::Tree,
            },
            _ => OpConfig::Elementwise {
                block_size: config.workgroup.x as usize,
                items_per_thread: config.params.get("items_per_thread").copied().unwrap_or(4) as usize,
                vector_width: config.memory.vector_width,
            },
        }
    }

    /// Get default OpConfig.
    fn default_op_config(&self) -> OpConfig {
        OpConfig::Elementwise { block_size: 256, items_per_thread: 4, vector_width: 4 }
    }
}

/// Main auto-tuner.
pub struct AutoTuner {
    config: TunerConfig,
    cache: TuningCache,
    stats: TunerStats,
}

impl AutoTuner {
    /// Create a new auto-tuner.
    pub fn new(config: TunerConfig) -> Self {
        let cache = if config.use_cache {
            TuningCache::new()
        } else {
            TuningCache::new() // Still create, just won't use
        };

        Self { config, cache, stats: TunerStats::default() }
    }

    /// Tune a kernel for given input shapes.
    pub fn tune<K: TunableKernel>(&mut self, kernel: &K) -> TuningResult {
        let key = format!("{}:{:?}", kernel.kernel_id(), kernel.input_signature());

        // Check cache first
        if self.config.use_cache {
            let device_id = &self.config.device_id;
            let dtype = &self.config.dtype;
            // Try to create shape bucket from input signature for matmul
            let input_sig = kernel.input_signature();
            let shape_bucket = if kernel.kernel_id().contains("matmul") && input_sig.len() >= 3 {
                Some(crate::kernel_config::MatmulShapeBucket::from_dims(
                    input_sig[0],
                    input_sig[1],
                    input_sig[2],
                ))
            } else {
                None
            };

            if let Some(config) = self.cache.get_config(kernel.kernel_id(), device_id, dtype, shape_bucket.as_ref()) {
                if self.config.verbose {
                    println!("Using cached configuration for {}", key);
                }
                // Return cached result (would need to store metrics too)
                return TuningResult {
                    best_config: OpConfig::Elementwise {
                        block_size: config.workgroup.x as usize,
                        items_per_thread: 4,
                        vector_width: config.memory.vector_width,
                    },
                    best_time_us: 0.0, // Unknown from cache
                    default_time_us: 0.0,
                    speedup: 1.0,
                    configs_tried: 0,
                    tuning_time: Duration::ZERO,
                    all_results: Vec::new(),
                };
            }
        }

        // Run tuning session
        let mut session = TuningSession::new(kernel.kernel_id().to_string(), kernel.input_signature(), self);

        let result = session.run(kernel);

        // Update stats
        self.stats.total_tunings += 1;
        self.stats.total_configs_tried += result.configs_tried;
        self.stats.total_tuning_time += result.tuning_time;

        result
    }

    /// Get a cached configuration without running tuning.
    pub fn get_cached(&self, key: &str) -> Option<OpConfig> {
        // Parse key to get kernel and shapes
        let parts: Vec<_> = key.split(':').collect();
        if parts.len() >= 2 {
            let kernel = parts[0];
            let device_id = &self.config.device_id;
            let dtype = &self.config.dtype;
            // Try to parse shapes
            if let Ok(shapes) = serde_json::from_str::<Vec<usize>>(parts[1]) {
                // Try to create shape bucket for matmul
                let shape_bucket = if kernel.contains("matmul") && shapes.len() >= 3 {
                    Some(crate::kernel_config::MatmulShapeBucket::from_dims(shapes[0], shapes[1], shapes[2]))
                } else {
                    None
                };
                return self.cache.get_config(kernel, device_id, dtype, shape_bucket.as_ref()).map(|kc| OpConfig::Elementwise {
                    block_size: kc.workgroup.x as usize,
                    items_per_thread: 4,
                    vector_width: kc.memory.vector_width,
                });
            }
        }
        None
    }

    /// Clear the cache.
    pub fn clear_cache(&mut self) {
        // Would need to add clear method to TuningCache
    }

    /// Get tuner statistics.
    pub fn stats(&self) -> &TunerStats {
        &self.stats
    }

    /// Print tuner statistics.
    pub fn print_stats(&self) {
        self.stats.print();
    }
}

/// Tuner statistics.
#[derive(Debug, Clone, Default)]
pub struct TunerStats {
    pub total_tunings: usize,
    pub total_configs_tried: usize,
    pub total_tuning_time: Duration,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

impl TunerStats {
    /// Print statistics.
    pub fn print(&self) {
        println!("\nAutoTuner Statistics:");
        println!("  Total tuning sessions: {}", self.total_tunings);
        println!("  Total configs tried: {}", self.total_configs_tried);
        println!("  Total tuning time: {:?}", self.total_tuning_time);

        if self.total_tunings > 0 {
            let avg_time = self.total_tuning_time.as_secs_f64() / self.total_tunings as f64;
            println!("  Average tuning time: {:.2}s", avg_time);
        }

        let total_cache = self.cache_hits + self.cache_misses;
        if total_cache > 0 {
            let hit_rate = self.cache_hits as f64 / total_cache as f64 * 100.0;
            println!("  Cache hit rate: {:.1}%", hit_rate);
        }
    }
}

/// Estimate workload characteristics.
fn estimate_workload(kernel_id: &str, input_sig: &[usize]) -> (u64, u64) {
    match kernel_id {
        "matmul" => {
            if input_sig.len() >= 3 {
                let m = input_sig[0] as u64;
                let n = input_sig[1] as u64;
                let k = input_sig[2] as u64;
                let flops = 2 * m * n * k;
                let bytes = 4 * (m * k + k * n + m * n); // f32
                (flops, bytes)
            } else {
                (0, 0)
            }
        }
        "conv2d" => {
            // Simplified: assume common convolution sizes
            let flops = 1_000_000_000u64; // 1 GFLOP default
            let bytes = 100_000_000u64; // 100 MB default
            (flops, bytes)
        }
        _ => (0, 0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_config::ConfigSpace;

    struct TestKernel {
        id: &'static str,
        sig: Vec<usize>,
    }

    impl TunableKernel for TestKernel {
        fn kernel_id(&self) -> &'static str {
            self.id
        }

        fn config_space(&self) -> ConfigSpace {
            ConfigSpace::matmul_reduced()
        }

        fn with_config(&self, _config: &KernelConfig) -> Box<dyn FnMut()> {
            Box::new(|| {
                // Simulate work
                std::thread::sleep(Duration::from_micros(10));
            })
        }

        fn input_signature(&self) -> Vec<usize> {
            self.sig.clone()
        }
    }

    #[test]
    fn test_tuner_config() {
        let fast = TunerConfig::fast();
        assert_eq!(fast.warmup_iterations, 2);
        assert!(fast.use_cache);

        let ci_safe = TunerConfig::ci_safe();
        assert_eq!(ci_safe.timing_iterations, 3);
        assert!(ci_safe.ci_mode);

        let disabled = TunerConfig::disabled();
        assert!(!disabled.enabled);
    }

    #[test]
    fn test_tuner_config_ci_mode() {
        let config = TunerConfig::default().with_ci_mode(true);
        assert!(config.ci_mode);
        assert_eq!(config.max_tuning_time, Duration::from_secs(10));
        assert_eq!(config.timing_iterations, 3);
    }

    #[test]
    fn test_tuner_config_enabled() {
        let enabled = TunerConfig::default().with_enabled(true);
        assert!(enabled.enabled);

        let disabled = TunerConfig::default().with_enabled(false);
        assert!(!disabled.enabled);
    }

    #[test]
    fn test_tuner_config_device_dtype() {
        let config = TunerConfig::default()
            .with_device_id("cuda:0".to_string())
            .with_dtype("f16".to_string());
        assert_eq!(config.device_id, "cuda:0");
        assert_eq!(config.dtype, "f16");
    }

    #[test]
    fn test_tuning_result() {
        let mut result = TuningResult {
            best_config: OpConfig::Elementwise { block_size: 256, items_per_thread: 4, vector_width: 4 },
            best_time_us: 50.0,
            default_time_us: 100.0,
            speedup: 0.0,
            configs_tried: 10,
            tuning_time: Duration::from_secs(1),
            all_results: Vec::new(),
        };

        result.calculate_speedup();
        assert!((result.speedup - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_auto_tuner_creation() {
        let tuner = AutoTuner::new(TunerConfig::fast());
        assert_eq!(tuner.stats().total_tunings, 0);
    }

    #[test]
    fn test_estimate_workload() {
        let (flops, bytes) = estimate_workload("matmul", &[1024, 1024, 1024]);
        assert!(flops > 0);
        assert!(bytes > 0);

        let (flops2, bytes2) = estimate_workload("conv2d", &[64, 64, 224, 224]);
        assert!(flops2 > 0);
        assert!(bytes2 > 0);
    }

    #[test]
    fn test_tuner_config_default() {
        let config = TunerConfig::default();
        assert_eq!(config.warmup_iterations, 3);
        assert_eq!(config.timing_iterations, 10);
        assert!(config.use_cache);
        assert!(config.cache_results);
        assert!(!config.verbose);
    }

    #[test]
    fn test_tuning_result_print_summary() {
        let result = TuningResult {
            best_config: OpConfig::Elementwise { block_size: 256, items_per_thread: 4, vector_width: 4 },
            best_time_us: 50.0,
            default_time_us: 100.0,
            speedup: 2.0,
            configs_tried: 10,
            tuning_time: Duration::from_secs(1),
            all_results: Vec::new(),
        };
        result.print_summary(); // Just verify it doesn't panic
    }

    #[test]
    fn test_auto_tuner_tune_and_cache() {
        let mut config = TunerConfig::fast();
        config.use_cache = false;
        config.cache_results = false;
        let mut tuner = AutoTuner::new(config);

        let kernel = TestKernel { id: "unique_tune_and_cache_test", sig: vec![256, 256, 256] };
        let result = tuner.tune(&kernel);
        assert!(result.speedup >= 0.0);

        // Stats should be updated at least once
        assert!(tuner.stats().total_tunings >= 1);
    }

    #[test]
    fn test_auto_tuner_get_cached() {
        let mut tuner = AutoTuner::new(TunerConfig::fast());
        // Use a unique signature unlikely to be in a stale cache
        let kernel = TestKernel { id: "unique_get_cached_test", sig: vec![9999, 9999, 9999] };

        // Run tuning to populate cache
        let _result = tuner.tune(&kernel);

        // Now should be cached
        let cached = tuner.get_cached("unique_get_cached_test:[9999, 9999, 9999]");
        assert!(cached.is_some());
    }

    #[test]
    fn test_auto_tuner_clear_cache() {
        let mut tuner = AutoTuner::new(TunerConfig::fast());
        // clear_cache doesn't panic
        tuner.clear_cache();
    }

    #[test]
    fn test_auto_tuner_print_stats() {
        let mut tuner = AutoTuner::new(TunerConfig::fast());
        let kernel = TestKernel { id: "matmul", sig: vec![64, 64, 64] };
        let _result = tuner.tune(&kernel);

        // print_stats doesn't panic
        tuner.print_stats();
    }

    #[test]
    fn test_tuner_stats_print() {
        let stats = TunerStats {
            total_tunings: 5,
            total_configs_tried: 100,
            total_tuning_time: Duration::from_secs(30),
            cache_hits: 3,
            cache_misses: 2,
        };
        stats.print(); // Verify no panic
    }

    #[test]
    fn test_estimate_workload_edge_cases() {
        let (flops, bytes) = estimate_workload("matmul", &[10]);
        assert_eq!(flops, 0);
        assert_eq!(bytes, 0);

        let (flops2, bytes2) = estimate_workload("unknown", &[]);
        assert_eq!(flops2, 0);
        assert_eq!(bytes2, 0);
    }

    #[test]
    fn test_tuning_session_with_grid_strategy() {
        let mut tuner = AutoTuner::new(TunerConfig {
            strategy: TunerStrategy::Grid,
            warmup_iterations: 1,
            timing_iterations: 2,
            max_tuning_time: Duration::from_secs(30),
            use_cache: false,
            cache_results: false,
            min_improvement_pct: 1.0,
            verbose: false,
            ci_mode: false,
            enabled: true,
            device_id: "test".to_string(),
            dtype: "f32".to_string(),
        });

        let kernel = TestKernel { id: "matmul", sig: vec![128, 128, 128] };
        let result = tuner.tune(&kernel);
        assert!(result.configs_tried > 0);
    }
}
