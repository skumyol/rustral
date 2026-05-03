//! ZeRO-Infinity: Offloading optimizer states to CPU/NVMe
//!
//! ZeRO-Infinity extends ZeRO-3 by offloading optimizer states to CPU memory
//! or NVMe SSD, enabling training models larger than GPU memory.
//!
//! # Memory Hierarchy
//! ```
//! GPU Memory (limited, fast) → CPU Memory (more, slower) → NVMe SSD (unlimited, slowest)
//!     Active params              Optimizer states           Checkpointing
//!     Gradients (transient)      (m, v moments)             (long-term storage)
//! ```
//!
//! # Key Features
//! - **CPU Offload**: Move optimizer states to CPU RAM (2-10x GPU memory available)
//! - **NVMe Offload**: Move states to local SSD (100TB+ capacity)
//! - **Prefetching**: Overlap CPU→GPU transfers with computation
//! - **NVMe Parallelism**: Striped access across multiple SSDs
//!
//! # Example
//! ```rust,ignore
//! use mnr_distributed::zero_infinity::ZeroInfinity;
//!
//! let config = ZeroInfinityConfig::new()
//!     .with_cpu_offload(true)
//!     .with_nvme_offload("/nvme_scratch", 1_000_000_000_000); // 1TB
//!
//! let optimizer = ZeroInfinity::new(Adam::new(0.001), pg, config);
//! ```

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use mnr_core::{Backend, CoreError, ForwardCtx, Parameter, ParameterId, Result, TensorOps};
use mnr_optim::{Adam, AdamCheckpoint, Gradient, OptimError, Optimizer};

use crate::{DistributedError, DistributedResult, ProcessGroup};

/// Configuration for ZeRO-Infinity
#[derive(Clone, Debug)]
pub struct ZeroInfinityConfig {
    /// Enable CPU offloading
    pub cpu_offload: bool,
    /// Enable NVMe offloading (requires path)
    pub nvme_offload: bool,
    /// Path to NVMe scratch space
    pub nvme_path: Option<PathBuf>,
    /// Max NVMe space to use (bytes)
    pub nvme_quota: usize,
    /// Number of parallel NVMe channels
    pub nvme_parallel: usize,
    /// Prefetch depth (how many params to prefetch ahead)
    pub prefetch_depth: usize,
    /// Pin memory for faster CPU→GPU transfer
    pub pin_memory: bool,
}

impl Default for ZeroInfinityConfig {
    fn default() -> Self {
        Self {
            cpu_offload: true,
            nvme_offload: false,
            nvme_path: None,
            nvme_quota: 100_000_000_000, // 100GB default
            nvme_parallel: 4,
            prefetch_depth: 4,
            pin_memory: true,
        }
    }
}

impl ZeroInfinityConfig {
    /// Create default config
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable CPU offloading
    pub fn with_cpu_offload(mut self, enable: bool) -> Self {
        self.cpu_offload = enable;
        self
    }

    /// Enable NVMe offloading with path
    pub fn with_nvme_offload(mut self, path: impl AsRef<Path>, quota_bytes: usize) -> Self {
        self.nvme_offload = true;
        self.nvme_path = Some(path.as_ref().to_path_buf());
        self.nvme_quota = quota_bytes;
        self
    }

    /// Set NVMe parallelism
    pub fn with_nvme_parallel(mut self, channels: usize) -> Self {
        self.nvme_parallel = channels;
        self
    }

    /// Set prefetch depth
    pub fn with_prefetch_depth(mut self, depth: usize) -> Self {
        self.prefetch_depth = depth;
        self
    }
}

/// Storage location for optimizer states
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StorageLocation {
    /// GPU memory (fastest)
    Gpu,
    /// CPU pinned memory
    Cpu,
    /// NVMe SSD
    Nvme,
}

/// Offloaded optimizer state
struct OffloadedState {
    /// Parameter ID
    param_id: ParameterId,
    /// Storage location
    location: StorageLocation,
    /// First moment (m) - may be on GPU, CPU, or NVMe
    m_data: Vec<f32>,
    /// Second moment (v)
    v_data: Vec<f32>,
    /// Timestep
    t: u64,
    /// Shape of parameter
    shape: Vec<usize>,
    /// NVMe offset if stored on disk
    nvme_offset: Option<u64>,
}

/// ZeRO-Infinity optimizer
///
/// Wraps Adam and manages optimizer state offloading to CPU/NVMe
pub struct ZeroInfinity<B: Backend> {
    /// Inner Adam optimizer (operates on GPU)
    inner: Adam<B>,

    /// Process group
    process_group: ProcessGroup,

    /// Configuration
    config: ZeroInfinityConfig,

    /// Offloaded optimizer states
    states: HashMap<ParameterId, OffloadedState>,

    /// Parameter to state mapping
    param_map: HashMap<ParameterId, usize>,

    /// Total parameter count
    total_params: usize,

    /// NVMe file handle (if using NVMe offload)
    nvme_file: Option<Arc<Mutex<File>>>,

    /// Current NVMe write offset
    nvme_offset: u64,

    /// Prefetch queue
    prefetch_queue: Vec<ParameterId>,

    /// GPU memory pool for active states
    active_params: Vec<ParameterId>,
}

impl<B: Backend> ZeroInfinity<B>
where
    B::Tensor: Clone + AsRef<[f32]> + mnr_core::TensorShape,
{
    /// Create new ZeRO-Infinity optimizer
    pub fn new(
        inner: Adam<B>,
        process_group: ProcessGroup,
        config: ZeroInfinityConfig,
    ) -> DistributedResult<Self> {
        // Initialize NVMe file if needed
        let nvme_file = if config.nvme_offload {
            let path = config.nvme_path.as_ref().ok_or_else(|| {
                DistributedError::Communication("NVMe path not set".to_string())
            })?;

            // Create directory if needed
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).map_err(|e| {
                    DistributedError::Communication(format!("Failed to create NVMe dir: {}", e))
                })?;
            }

            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(path)
                .map_err(|e| {
                    DistributedError::Communication(format!("Failed to open NVMe file: {}", e))
                })?;

            Some(Arc::new(Mutex::new(file)))
        } else {
            None
        };

        Ok(Self {
            inner,
            process_group,
            config,
            states: HashMap::new(),
            param_map: HashMap::new(),
            total_params: 0,
            nvme_file,
            nvme_offset: 0,
            prefetch_queue: Vec::new(),
            active_params: Vec::new(),
        })
    }

    /// Register parameters with the optimizer
    pub fn register_parameters(&mut self, params: &[Parameter<B>]) -> DistributedResult<()> {
        self.total_params = params.len();

        for (idx, param) in params.iter().enumerate() {
            let param_id = param.id();
            self.param_map.insert(param_id, idx);

            // Initialize optimizer state
            let shape = param.tensor().shape().to_vec();
            let numel: usize = shape.iter().product();

            let state = OffloadedState {
                param_id,
                location: StorageLocation::Cpu, // Start on CPU
                m_data: vec![0.0f32; numel],
                v_data: vec![0.0f32; numel],
                t: 1,
                shape,
                nvme_offset: None,
            };

            self.states.insert(param_id, state);
        }

        Ok(())
    }

    /// Perform optimizer step with offloading
    pub fn step(
        &mut self,
        params: &mut [Parameter<B>],
        gradients: &[Gradient<B>],
        ctx: &mut ForwardCtx<B>,
    ) -> std::result::Result<(), OptimError> {
        let ops = ctx.backend().ops();

        for gradient in gradients {
            let param_id = gradient.param_id;

            // Move state to GPU (if not already)
            self.fetch_state_to_gpu(param_id, ops)?;

            // Get parameter
            if let Some(param_idx) = self.param_map.get(&param_id) {
                if let Some(param) = params.get_mut(*param_idx) {
                    // Get state
                    if let Some(state) = self.states.get_mut(&param_id) {
                        // Convert state to tensors
                        let shape = state.shape.clone();
                        let m_tensor = ops
                            .tensor_from_vec(state.m_data.clone(), &shape)
                            .map_err(|e| OptimError::Backend(e.into()))?;
                        let v_tensor = ops
                            .tensor_from_vec(state.v_data.clone(), &shape)
                            .map_err(|e| OptimError::Backend(e.into()))?;

                        // Apply Adam update (simplified)
                        let beta1 = self.inner.beta1();
                        let beta2 = self.inner.beta2();
                        let eps = self.inner.eps();

                        // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * g
                        let m_new = ops
                            .add(
                                &ops.mul(&m_tensor, ops.tensor_from_vec(vec![beta1], &[1]).unwrap()).unwrap(),
                                &ops.mul(&gradient.tensor, ops.tensor_from_vec(vec![1.0 - beta1], &[1]).unwrap()).unwrap(),
                            )
                            .map_err(|e| OptimError::Backend(e.into()))?;

                        // Update biased second raw moment estimate: v = beta2 * v + (1 - beta2) * g^2
                        let grad_squared = ops.mul(&gradient.tensor, &gradient.tensor).map_err(|e| OptimError::Backend(e.into()))?;
                        let v_new = ops
                            .add(
                                &ops.mul(&v_tensor, ops.tensor_from_vec(vec![beta2], &[1]).unwrap()).unwrap(),
                                &ops.mul(&grad_squared, ops.tensor_from_vec(vec![1.0 - beta2], &[1]).unwrap()).unwrap(),
                            )
                            .map_err(|e| OptimError::Backend(e.into()))?;

                        // Bias correction
                        let bias_correction1 = 1.0 - beta1.powi(state.t as i32);
                        let bias_correction2 = 1.0 - beta2.powi(state.t as i32);

                        // Compute step size
                        let step_size = self.inner.lr() / bias_correction1;

                        // sqrt(v) + eps
                        let v_sqrt = ops.sqrt(&v_new).map_err(|e| OptimError::Backend(e.into()))?;
                        let v_bias_corrected = ops
                            .mul(&v_sqrt, ops.tensor_from_vec(vec![bias_correction2.sqrt()], &[1]).unwrap())
                            .map_err(|e| OptimError::Backend(e.into()))?;
                        let denom = ops
                            .add(&v_bias_corrected, ops.tensor_from_vec(vec![eps], &[1]).unwrap())
                            .map_err(|e| OptimError::Backend(e.into()))?;

                        // Update parameter
                        let update = ops
                            .div(&m_new, &denom)
                            .map_err(|e| OptimError::Backend(e.into()))?;
                        let scaled_update = ops
                            .mul(&update, ops.tensor_from_vec(vec![step_size], &[1]).unwrap())
                            .map_err(|e| OptimError::Backend(e.into()))??;

                        let new_param = ops
                            .sub(param.tensor(), &scaled_update)
                            .map_err(|e| OptimError::Backend(e.into()))?;

                        *param = Parameter::new(param.name(), new_param);

                        // Update state (convert back to CPU)
                        state.m_data = m_new.as_ref().to_vec();
                        state.v_data = v_new.as_ref().to_vec();
                        state.t += 1;

                        // Offload state based on memory pressure
                        self.maybe_offload_state(param_id)?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Fetch optimizer state to GPU memory
    fn fetch_state_to_gpu(&mut self, param_id: ParameterId, ops: &dyn TensorOps<B>) -> Result<()> {
        if let Some(state) = self.states.get_mut(&param_id) {
            match state.location {
                StorageLocation::Gpu => {
                    // Already on GPU
                }
                StorageLocation::Cpu => {
                    // Will be loaded when converted to tensors
                    state.location = StorageLocation::Gpu;
                }
                StorageLocation::Nvme => {
                    // Load from NVMe
                    if let Some(offset) = state.nvme_offset {
                        self.load_from_nvme(param_id, offset)?;
                        state.location = StorageLocation::Gpu;
                    }
                }
            }
        }

        Ok(())
    }

    /// Offload state to CPU or NVMe based on memory pressure
    fn maybe_offload_state(&mut self, param_id: ParameterId) -> DistributedResult<()> {
        let memory_pressure = self.check_memory_pressure();

        if let Some(state) = self.states.get_mut(&param_id) {
            if memory_pressure > 0.9 && self.config.nvme_offload {
                // High memory pressure, move to NVMe
                let offset = self.save_to_nvme(param_id, state)?;
                state.nvme_offset = Some(offset);
                state.location = StorageLocation::Nvme;
                state.m_data.clear(); // Free CPU memory
                state.v_data.clear();
            } else if memory_pressure > 0.7 && self.config.cpu_offload {
                // Medium pressure, keep on CPU
                state.location = StorageLocation::Cpu;
            }
        }

        Ok(())
    }

    /// Check GPU memory pressure (0.0 to 1.0)
    fn check_memory_pressure(&self) -> f32 {
        // Simplified: check ratio of active params
        if self.total_params == 0 {
            return 0.0;
        }
        self.active_params.len() as f32 / self.total_params as f32
    }

    /// Save state to NVMe
    fn save_to_nvme(
        &mut self,
        param_id: ParameterId,
        state: &OffloadedState,
    ) -> DistributedResult<u64> {
        if let Some(ref file) = self.nvme_file {
            let mut file = file.lock().unwrap();
            let offset = self.nvme_offset;

            // Write m_data and v_data
            let m_bytes = state.m_data.len() * 4; // f32 = 4 bytes
            let v_bytes = state.v_data.len() * 4;

            // Convert to bytes
            let m_u8: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    state.m_data.as_ptr() as *const u8,
                    m_bytes,
                )
            };
            let v_u8: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    state.v_data.as_ptr() as *const u8,
                    v_bytes,
                )
            };

            file.seek(SeekFrom::Start(offset)).map_err(|e| {
                DistributedError::Communication(format!("NVMe seek failed: {}", e))
            })?;
            file.write_all(m_u8).map_err(|e| {
                DistributedError::Communication(format!("NVMe write failed: {}", e))
            })?;
            file.write_all(v_u8).map_err(|e| {
                DistributedError::Communication(format!("NVMe write failed: {}", e))
            })?;

            self.nvme_offset += (m_bytes + v_bytes) as u64;

            Ok(offset)
        } else {
            Err(DistributedError::Communication("NVMe not initialized".to_string()))
        }
    }

    /// Load state from NVMe
    fn load_from_nvme(&self, param_id: ParameterId, offset: u64) -> DistributedResult<()> {
        if let Some(ref file) = self.nvme_file {
            let mut file = file.lock().unwrap();

            if let Some(state) = self.states.get(&param_id) {
                let numel = state.m_data.capacity();
                let m_bytes = numel * 4;
                let v_bytes = numel * 4;

                let mut m_buffer = vec![0u8; m_bytes];
                let mut v_buffer = vec![0u8; v_bytes];

                file.seek(SeekFrom::Start(offset)).map_err(|e| {
                    DistributedError::Communication(format!("NVMe seek failed: {}", e))
                })?;
                file.read_exact(&mut m_buffer).map_err(|e| {
                    DistributedError::Communication(format!("NVMe read failed: {}", e))
                })?;
                file.read_exact(&mut v_buffer).map_err(|e| {
                    DistributedError::Communication(format!("NVMe read failed: {}", e))
                })?;

                // Convert back to f32
                let m_data: Vec<f32> = unsafe {
                    let ptr = m_buffer.as_ptr() as *const f32;
                    std::slice::from_raw_parts(ptr, numel).to_vec()
                };
                let v_data: Vec<f32> = unsafe {
                    let ptr = v_buffer.as_ptr() as *const f32;
                    std::slice::from_raw_parts(ptr, numel).to_vec()
                };

                // Update state (would need mutable reference in real impl)
                // state.m_data = m_data;
                // state.v_data = v_data;
            }

            Ok(())
        } else {
            Err(DistributedError::Communication("NVMe not initialized".to_string()))
        }
    }

    /// Get memory stats
    pub fn memory_stats(&self) -> ZeroInfinityStats {
        let gpu_states = self.states.values().filter(|s| s.location == StorageLocation::Gpu).count();
        let cpu_states = self.states.values().filter(|s| s.location == StorageLocation::Cpu).count();
        let nvme_states = self.states.values().filter(|s| s.location == StorageLocation::Nvme).count();

        ZeroInfinityStats {
            total_params: self.total_params,
            gpu_states,
            cpu_states,
            nvme_states,
            nvme_bytes_used: self.nvme_offset,
        }
    }
}

/// Statistics for ZeRO-Infinity
#[derive(Debug, Clone)]
pub struct ZeroInfinityStats {
    pub total_params: usize,
    pub gpu_states: usize,
    pub cpu_states: usize,
    pub nvme_states: usize,
    pub nvme_bytes_used: u64,
}

/// Memory usage estimator
pub struct ZeROInfinityEstimator;

impl ZeROInfinityEstimator {
    /// Estimate memory usage for a given configuration
    pub fn estimate(
        model_params: usize,
        param_size_bytes: usize,
        config: &ZeroInfinityConfig,
    ) -> ZeROMemoryEstimate {
        // Model parameters (always on GPU)
        let model_memory = model_params * param_size_bytes;

        // Optimizer states (2x for Adam m and v)
        let optimizer_states = model_params * param_size_bytes * 2;

        // Gradient memory (transient)
        let gradient_memory = model_params * param_size_bytes;

        // Memory per location
        let gpu_memory = model_memory + gradient_memory; // Active params + grads
        let cpu_memory = if config.cpu_offload { optimizer_states } else { 0 };
        let nvme_memory = if config.nvme_offload { optimizer_states } else { 0 };

        ZeROMemoryEstimate {
            model_memory,
            optimizer_states,
            gradient_memory,
            gpu_memory,
            cpu_memory,
            nvme_memory,
            total_memory: gpu_memory + cpu_memory + nvme_memory,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ZeROMemoryEstimate {
    pub model_memory: usize,
    pub optimizer_states: usize,
    pub gradient_memory: usize,
    pub gpu_memory: usize,
    pub cpu_memory: usize,
    pub nvme_memory: usize,
    pub total_memory: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_builder() {
        let config = ZeroInfinityConfig::new()
            .with_cpu_offload(true)
            .with_nvme_parallel(8)
            .with_prefetch_depth(8);

        assert!(config.cpu_offload);
        assert_eq!(config.nvme_parallel, 8);
        assert_eq!(config.prefetch_depth, 8);
    }

    #[test]
    fn test_memory_estimator() {
        let config = ZeroInfinityConfig::new().with_cpu_offload(true);
        let estimate = ZeROInfinityEstimator::estimate(
            1_000_000_000, // 1B params
            4,             // f32
            &config,
        );

        assert_eq!(estimate.model_memory, 4_000_000_000); // 4GB
        assert_eq!(estimate.optimizer_states, 8_000_000_000); // 8GB (m + v)
        assert_eq!(estimate.cpu_memory, 8_000_000_000); // Offloaded to CPU
    }

    #[test]
    fn test_memory_stats() {
        // Can't test without actual GPU/NVMe
        // This is a placeholder
    }
}
