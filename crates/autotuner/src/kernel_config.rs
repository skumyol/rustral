//! Kernel configuration types for GPU operations.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Complete kernel configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KernelConfig {
    /// Workgroup/thread block configuration.
    pub workgroup: WorkgroupConfig,
    /// Algorithm selection.
    pub algorithm: AlgorithmConfig,
    /// Memory layout and caching.
    pub memory: MemoryConfig,
    /// Additional parameters.
    pub params: HashMap<String, i32>,
}

impl KernelConfig {
    /// Create a default configuration.
    pub fn default_for(op: &str) -> Self {
        match op {
            "matmul" => Self::default_matmul(),
            "conv2d" => Self::default_conv2d(),
            "reduce" => Self::default_reduce(),
            "elementwise" => Self::default_elementwise(),
            "attention" => Self::default_attention(),
            _ => Self::default(),
        }
    }

    /// Default matmul configuration.
    pub fn default_matmul() -> Self {
        Self {
            workgroup: WorkgroupConfig { x: 16, y: 16, z: 1 },
            algorithm: AlgorithmConfig::Matmul(MatmulAlgorithm::Tiled),
            memory: MemoryConfig {
                use_shared: true,
                shared_size_bytes: 32768, // 32KB
                vector_width: 4,
                cache_config: CacheConfig::PreferL1,
            },
            params: [("tile_m".to_string(), 128), ("tile_n".to_string(), 128), ("tile_k".to_string(), 8)]
                .into_iter()
                .collect(),
        }
    }

    /// Default conv2d configuration.
    pub fn default_conv2d() -> Self {
        Self {
            workgroup: WorkgroupConfig { x: 8, y: 8, z: 4 },
            algorithm: AlgorithmConfig::Conv(ConvAlgorithm::ImplicitGemm),
            memory: MemoryConfig {
                use_shared: true,
                shared_size_bytes: 16384,
                vector_width: 4,
                cache_config: CacheConfig::PreferShared,
            },
            params: [("tile_h".to_string(), 4), ("tile_w".to_string(), 4), ("tile_c".to_string(), 32)]
                .into_iter()
                .collect(),
        }
    }

    /// Default reduction configuration.
    pub fn default_reduce() -> Self {
        Self {
            workgroup: WorkgroupConfig { x: 256, y: 1, z: 1 },
            algorithm: AlgorithmConfig::Reduce(ReduceAlgorithm::Tree),
            memory: MemoryConfig {
                use_shared: true,
                shared_size_bytes: 4096,
                vector_width: 4,
                cache_config: CacheConfig::Default,
            },
            params: [("items_per_thread".to_string(), 8), ("unroll".to_string(), 1)].into_iter().collect(),
        }
    }

    /// Default element-wise configuration.
    pub fn default_elementwise() -> Self {
        Self {
            workgroup: WorkgroupConfig { x: 256, y: 1, z: 1 },
            algorithm: AlgorithmConfig::Elementwise,
            memory: MemoryConfig {
                use_shared: false,
                shared_size_bytes: 0,
                vector_width: 4,
                cache_config: CacheConfig::Default,
            },
            params: [("items_per_thread".to_string(), 4), ("vectorize".to_string(), 1)].into_iter().collect(),
        }
    }

    /// Default attention configuration.
    pub fn default_attention() -> Self {
        Self {
            workgroup: WorkgroupConfig { x: 64, y: 2, z: 1 },
            algorithm: AlgorithmConfig::Attention { use_flash: true, block_size_m: 64, block_size_n: 64 },
            memory: MemoryConfig {
                use_shared: true,
                shared_size_bytes: 49152, // 48KB for SRAM
                vector_width: 4,
                cache_config: CacheConfig::PreferShared,
            },
            params: [("softmax_scale".to_string(), 1), ("use_causal_mask".to_string(), 1)]
                .into_iter()
                .collect(),
        }
    }
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            workgroup: WorkgroupConfig::default(),
            algorithm: AlgorithmConfig::Elementwise,
            memory: MemoryConfig::default(),
            params: HashMap::new(),
        }
    }
}

/// Workgroup/thread block configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorkgroupConfig {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl WorkgroupConfig {
    /// Total threads per workgroup.
    pub fn total_threads(&self) -> u32 {
        self.x * self.y * self.z
    }

    /// Check if this is a valid workgroup size.
    pub fn is_valid(&self) -> bool {
        // Common GPU limits: 1024 threads per workgroup
        self.total_threads() <= 1024 && self.total_threads() > 0
    }
}

impl Default for WorkgroupConfig {
    fn default() -> Self {
        Self { x: 256, y: 1, z: 1 }
    }
}

/// Block size for tiling operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockSize {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

impl BlockSize {
    /// Create a square block.
    pub fn square(size: usize) -> Self {
        Self { m: size, n: size, k: size }
    }

    /// Check if block dimensions are valid.
    pub fn is_valid(&self) -> bool {
        self.m > 0 && self.n > 0 && self.k > 0 && self.m <= 256 && self.n <= 256 && self.k <= 64
    }
}

/// Algorithm configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AlgorithmConfig {
    /// Matrix multiplication.
    Matmul(MatmulAlgorithm),
    /// Convolution.
    Conv(ConvAlgorithm),
    /// Reduction.
    Reduce(ReduceAlgorithm),
    /// Element-wise.
    Elementwise,
    /// Attention.
    Attention { use_flash: bool, block_size_m: usize, block_size_n: usize },
    /// Custom algorithm.
    Custom(String),
}

/// Matrix multiplication algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MatmulAlgorithm {
    /// Naive implementation.
    Naive,
    /// Tiled with shared memory.
    Tiled,
    /// Tiled with vectorized loads.
    TiledVectorized,
    /// Strassen algorithm (for large matrices).
    Strassen,
    /// CUTLASS-style warp-specialized.
    WarpSpecialized,
    /// Tensor core WMMA.
    TensorCoreWmma,
    /// Tensor core MMA (native).
    TensorCoreMma,
}

impl MatmulAlgorithm {
    /// Check if this algorithm requires tensor cores.
    pub fn uses_tensor_cores(&self) -> bool {
        matches!(self, Self::TensorCoreWmma | Self::TensorCoreMma)
    }

    /// Get recommended tile size for this algorithm.
    pub fn recommended_tile(&self) -> BlockSize {
        match self {
            Self::Naive => BlockSize { m: 32, n: 32, k: 1 },
            Self::Tiled => BlockSize { m: 128, n: 128, k: 8 },
            Self::TiledVectorized => BlockSize { m: 128, n: 128, k: 16 },
            Self::Strassen => BlockSize { m: 256, n: 256, k: 64 },
            Self::WarpSpecialized => BlockSize { m: 256, n: 128, k: 64 },
            Self::TensorCoreWmma => BlockSize { m: 128, n: 256, k: 32 },
            Self::TensorCoreMma => BlockSize { m: 256, n: 256, k: 64 },
        }
    }
}

/// Convolution algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

/// Memory configuration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Use shared memory/LDS.
    pub use_shared: bool,
    /// Shared memory size in bytes.
    pub shared_size_bytes: usize,
    /// Vector load/store width.
    pub vector_width: usize,
    /// Cache configuration preference.
    pub cache_config: CacheConfig,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            use_shared: true,
            shared_size_bytes: 16384,
            vector_width: 4,
            cache_config: CacheConfig::Default,
        }
    }
}

/// Cache configuration preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CacheConfig {
    /// No preference.
    Default,
    /// Prefer more L1, less shared.
    PreferL1,
    /// Prefer more shared, less L1.
    PreferShared,
    /// Equal split.
    Equal,
}

/// Configuration space for search.
#[derive(Debug, Clone)]
pub struct ConfigSpace {
    pub workgroup_sizes: Vec<WorkgroupConfig>,
    pub algorithms: Vec<AlgorithmConfig>,
    pub memory_configs: Vec<MemoryConfig>,
    pub param_ranges: HashMap<String, Vec<i32>>,
}

impl ConfigSpace {
    /// Create a full search space for matmul.
    pub fn matmul_full() -> Self {
        let mut workgroup_sizes = Vec::new();
        for x in [8, 16, 32].iter() {
            for y in [8, 16, 32].iter() {
                let wg = WorkgroupConfig { x: *x, y: *y, z: 1 };
                if wg.is_valid() {
                    workgroup_sizes.push(wg);
                }
            }
        }

        let algorithms = vec![
            AlgorithmConfig::Matmul(MatmulAlgorithm::Tiled),
            AlgorithmConfig::Matmul(MatmulAlgorithm::TiledVectorized),
        ];

        let mut param_ranges = HashMap::new();
        param_ranges.insert("tile_m".to_string(), vec![64, 128, 256]);
        param_ranges.insert("tile_n".to_string(), vec![64, 128, 256]);
        param_ranges.insert("tile_k".to_string(), vec![8, 16, 32]);

        Self { workgroup_sizes, algorithms, memory_configs: vec![MemoryConfig::default()], param_ranges }
    }

    /// Create a reduced search space for faster tuning.
    pub fn matmul_reduced() -> Self {
        let workgroup_sizes =
            vec![WorkgroupConfig { x: 16, y: 16, z: 1 }, WorkgroupConfig { x: 32, y: 8, z: 1 }];

        let algorithms = vec![AlgorithmConfig::Matmul(MatmulAlgorithm::Tiled)];

        let mut param_ranges = HashMap::new();
        param_ranges.insert("tile_m".to_string(), vec![128, 256]);
        param_ranges.insert("tile_n".to_string(), vec![128, 256]);
        param_ranges.insert("tile_k".to_string(), vec![16, 32]);

        Self { workgroup_sizes, algorithms, memory_configs: vec![MemoryConfig::default()], param_ranges }
    }

    /// Get total number of configurations.
    pub fn size(&self) -> usize {
        let mut total = self.workgroup_sizes.len() * self.algorithms.len() * self.memory_configs.len();

        for values in self.param_ranges.values() {
            total *= values.len();
        }

        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workgroup_config() {
        let wg = WorkgroupConfig { x: 16, y: 16, z: 1 };
        assert_eq!(wg.total_threads(), 256);
        assert!(wg.is_valid());

        let invalid = WorkgroupConfig { x: 32, y: 32, z: 32 };
        assert!(!invalid.is_valid()); // 32768 > 1024
    }

    #[test]
    fn test_matmul_algorithm() {
        let alg = MatmulAlgorithm::Tiled;
        assert!(!alg.uses_tensor_cores());

        let tc_alg = MatmulAlgorithm::TensorCoreWmma;
        assert!(tc_alg.uses_tensor_cores());

        let tile = tc_alg.recommended_tile();
        assert!(tile.m > 0);
        assert!(tile.n > 0);
        assert!(tile.k > 0);
    }

    #[test]
    fn test_config_space() {
        let space = ConfigSpace::matmul_reduced();
        let size = space.size();
        assert!(size > 0);

        // Check specific combinations
        assert_eq!(space.workgroup_sizes.len(), 2);
        assert!(!space.algorithms.is_empty());
    }

    #[test]
    fn test_kernel_config_default() {
        let config = KernelConfig::default_matmul();
        assert!(config.memory.use_shared);
        assert!(config.workgroup.is_valid());

        if let AlgorithmConfig::Matmul(alg) = config.algorithm {
            assert!(matches!(alg, MatmulAlgorithm::Tiled));
        } else {
            panic!("Expected Matmul algorithm");
        }
    }

    #[test]
    fn test_kernel_config_default_for_all() {
        let matmul = KernelConfig::default_for("matmul");
        assert!(matches!(matmul.algorithm, AlgorithmConfig::Matmul(_)));

        let conv = KernelConfig::default_for("conv2d");
        assert!(matches!(conv.algorithm, AlgorithmConfig::Conv(_)));

        let reduce = KernelConfig::default_for("reduce");
        assert!(matches!(reduce.algorithm, AlgorithmConfig::Reduce(_)));

        let elem = KernelConfig::default_for("elementwise");
        assert!(matches!(elem.algorithm, AlgorithmConfig::Elementwise));

        let attn = KernelConfig::default_for("attention");
        assert!(matches!(attn.algorithm, AlgorithmConfig::Attention { .. }));

        let unknown = KernelConfig::default_for("unknown");
        assert!(matches!(unknown.algorithm, AlgorithmConfig::Elementwise));
    }

    #[test]
    fn test_kernel_config_default_trait() {
        let config = KernelConfig::default();
        assert_eq!(config.workgroup, WorkgroupConfig::default());
        assert!(matches!(config.algorithm, AlgorithmConfig::Elementwise));
        assert!(config.params.is_empty());
    }

    #[test]
    fn test_workgroup_config_default() {
        let wg = WorkgroupConfig::default();
        assert_eq!(wg.x, 256);
        assert_eq!(wg.y, 1);
        assert_eq!(wg.z, 1);
        assert_eq!(wg.total_threads(), 256);
        assert!(wg.is_valid());
    }

    #[test]
    fn test_block_size() {
        let sq = BlockSize::square(64);
        assert_eq!(sq.m, 64);
        assert_eq!(sq.n, 64);
        assert_eq!(sq.k, 64);
        assert!(sq.is_valid());

        let invalid = BlockSize { m: 0, n: 10, k: 10 };
        assert!(!invalid.is_valid());

        let too_large = BlockSize { m: 300, n: 10, k: 10 };
        assert!(!too_large.is_valid());
    }

    #[test]
    fn test_matmul_algorithm_recommended_tiles() {
        for alg in [
            MatmulAlgorithm::Naive,
            MatmulAlgorithm::Tiled,
            MatmulAlgorithm::TiledVectorized,
            MatmulAlgorithm::Strassen,
            MatmulAlgorithm::WarpSpecialized,
            MatmulAlgorithm::TensorCoreWmma,
            MatmulAlgorithm::TensorCoreMma,
        ] {
            let tile = alg.recommended_tile();
            assert!(tile.is_valid(), "{:?} tile invalid", alg);
            assert!(
                alg.uses_tensor_cores()
                    == matches!(alg, MatmulAlgorithm::TensorCoreWmma | MatmulAlgorithm::TensorCoreMma)
            );
        }
    }

    #[test]
    fn test_memory_config_default() {
        let mem = MemoryConfig::default();
        assert!(mem.use_shared);
        assert_eq!(mem.shared_size_bytes, 16384);
        assert_eq!(mem.vector_width, 4);
        assert_eq!(mem.cache_config, CacheConfig::Default);
    }

    #[test]
    fn test_cache_config_variants() {
        let _ = CacheConfig::Default;
        let _ = CacheConfig::PreferL1;
        let _ = CacheConfig::PreferShared;
        let _ = CacheConfig::Equal;
    }

    #[test]
    fn test_config_space_matmul_full() {
        let space = ConfigSpace::matmul_full();
        assert!(!space.workgroup_sizes.is_empty());
        assert!(!space.algorithms.is_empty());
        assert!(space.size() > 0);
    }

    #[test]
    fn test_algorithm_config_variants() {
        let _ = AlgorithmConfig::Matmul(MatmulAlgorithm::Tiled);
        let _ = AlgorithmConfig::Conv(ConvAlgorithm::Direct);
        let _ = AlgorithmConfig::Reduce(ReduceAlgorithm::Tree);
        let _ = AlgorithmConfig::Elementwise;
        let _ = AlgorithmConfig::Attention { use_flash: true, block_size_m: 64, block_size_n: 64 };
        let _ = AlgorithmConfig::Custom("test".to_string());
    }
}
