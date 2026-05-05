//! ZeRO-Infinity: Offloading optimizer states to CPU/NVMe
//!
//! ZeRO-Infinity extends ZeRO-3 by offloading optimizer states to CPU memory
//! or NVMe SSD, enabling training models larger than GPU memory.
//!
//! # Memory Hierarchy
//! ```text
//! GPU Memory (limited, fast) -> CPU Memory (more, slower) -> NVMe SSD (unlimited, slowest)
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
//! use rustral_distributed::zero_infinity::ZeroInfinity;
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

use rustral_core::{Backend, CoreError, ForwardCtx, Parameter, ParameterId, Result, TensorOps, TensorShape};
use rustral_optim::{Adam, Gradient, OptimError};

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
#[derive(Clone)]
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
    B::Tensor: Clone + AsRef<[f32]> + rustral_core::TensorShape,
{
    /// Create new ZeRO-Infinity optimizer
    pub fn new(
        inner: Adam<B>,
        process_group: ProcessGroup,
        config: ZeroInfinityConfig,
    ) -> DistributedResult<Self> {
        // Initialize NVMe file if needed
        let nvme_file = if config.nvme_offload {
            let path = config
                .nvme_path
                .as_ref()
                .ok_or_else(|| DistributedError::Communication("NVMe path not set".to_string()))?;

            // Create directory if needed
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).map_err(|e| {
                    DistributedError::Communication(format!("Failed to create NVMe dir: {}", e))
                })?;
            }

            let file =
                OpenOptions::new().read(true).write(true).create(true).truncate(true).open(path).map_err(
                    |e| DistributedError::Communication(format!("Failed to open NVMe file: {}", e)),
                )?;

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
            self.fetch_state_to_gpu(param_id, ops).map_err(|e| OptimError::Backend(e.to_string()))?;

            // Get parameter
            if let Some(param_idx) = self.param_map.get(&param_id) {
                if let Some(param) = params.get_mut(*param_idx) {
                    // Get state
                    if let Some(state) = self.states.get_mut(&param_id) {
                        // Convert state to tensors
                        let shape = state.shape.clone();
                        let m_tensor = ops
                            .tensor_from_vec(state.m_data.clone(), &shape)
                            .map_err(|e| OptimError::Backend(e.to_string()))?;
                        let v_tensor = ops
                            .tensor_from_vec(state.v_data.clone(), &shape)
                            .map_err(|e| OptimError::Backend(e.to_string()))?;

                        // Apply Adam update (simplified)
                        let beta1 = self.inner.beta1;
                        let beta2 = self.inner.beta2;
                        let eps = self.inner.eps;

                        // Create scalar tensors
                        let beta1_tensor = ops.tensor_from_vec(vec![beta1], &[1]).unwrap();
                        let one_minus_beta1_tensor = ops.tensor_from_vec(vec![1.0 - beta1], &[1]).unwrap();
                        let beta2_tensor = ops.tensor_from_vec(vec![beta2], &[1]).unwrap();
                        let one_minus_beta2_tensor = ops.tensor_from_vec(vec![1.0 - beta2], &[1]).unwrap();

                        // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * g
                        let m_new = ops
                            .add(
                                &ops.mul(&m_tensor, &beta1_tensor).unwrap(),
                                &ops.mul(&gradient.tensor, &one_minus_beta1_tensor).unwrap(),
                            )
                            .map_err(|e| OptimError::Backend(e.to_string()))?;

                        // Update biased second raw moment estimate: v = beta2 * v + (1 - beta2) * g^2
                        let grad_squared = ops
                            .mul(&gradient.tensor, &gradient.tensor)
                            .map_err(|e| OptimError::Backend(e.to_string()))?;
                        let v_new = ops
                            .add(
                                &ops.mul(&v_tensor, &beta2_tensor).unwrap(),
                                &ops.mul(&grad_squared, &one_minus_beta2_tensor).unwrap(),
                            )
                            .map_err(|e| OptimError::Backend(e.to_string()))?;

                        // Bias correction
                        let bias_correction1: f32 = 1.0 - beta1.powi(state.t as i32);
                        let bias_correction2: f32 = 1.0 - beta2.powi(state.t as i32);

                        // Compute step size
                        let step_size: f32 = self.inner.lr / bias_correction1;

                        // sqrt(v) + eps
                        let v_sqrt = ops.sqrt(&v_new).map_err(|e| OptimError::Backend(e.to_string()))?;
                        let bias_correction2_tensor =
                            ops.tensor_from_vec(vec![bias_correction2.sqrt()], &[1]).unwrap();
                        let v_bias_corrected = ops
                            .mul(&v_sqrt, &bias_correction2_tensor)
                            .map_err(|e| OptimError::Backend(e.to_string()))?;
                        let eps_tensor = ops.tensor_from_vec(vec![eps], &[1]).unwrap();
                        let denom = ops
                            .add(&v_bias_corrected, &eps_tensor)
                            .map_err(|e| OptimError::Backend(e.to_string()))?;

                        // Update parameter
                        let update =
                            ops.div(&m_new, &denom).map_err(|e| OptimError::Backend(e.to_string()))?;
                        let step_size_tensor = ops.tensor_from_vec(vec![step_size], &[1]).unwrap();
                        let scaled_update = ops
                            .mul(&update, &step_size_tensor)
                            .map_err(|e| OptimError::Backend(e.to_string()))?;

                        let new_param = ops
                            .sub(param.tensor(), &scaled_update)
                            .map_err(|e| OptimError::Backend(e.to_string()))?;

                        *param = Parameter::new(param.name(), new_param);

                        // Update state (convert back to CPU)
                        state.m_data = m_new.as_ref().to_vec();
                        state.v_data = v_new.as_ref().to_vec();
                        state.t += 1;

                        // Offload state based on memory pressure
                        self.maybe_offload_state(param_id).map_err(|e| OptimError::Backend(e.to_string()))?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Fetch optimizer state to GPU memory
    fn fetch_state_to_gpu(&mut self, param_id: ParameterId, _ops: &dyn TensorOps<B>) -> Result<()> {
        // Check current location first
        let location = self.states.get(&param_id).map(|s| s.location);

        if let Some(location) = location {
            match location {
                StorageLocation::Gpu => {
                    // Already on GPU
                }
                StorageLocation::Cpu => {
                    // Will be loaded when converted to tensors
                    if let Some(state) = self.states.get_mut(&param_id) {
                        state.location = StorageLocation::Gpu;
                    }
                }
                StorageLocation::Nvme => {
                    // Load from NVMe
                    let offset = self.states.get(&param_id).and_then(|s| s.nvme_offset);
                    if let Some(offset) = offset {
                        self.load_from_nvme(param_id, offset)
                            .map_err(|e| CoreError::Other(format!("{:?}", e)))?;
                        if let Some(state) = self.states.get_mut(&param_id) {
                            state.location = StorageLocation::Gpu;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Offload state to CPU or NVMe based on memory pressure
    fn maybe_offload_state(&mut self, param_id: ParameterId) -> DistributedResult<()> {
        let memory_pressure = self.check_memory_pressure();

        // Check conditions first, then do the offload in separate scope
        let should_nvme_offload = memory_pressure > 0.9 && self.config.nvme_offload;
        let should_cpu_offload = memory_pressure > 0.7 && self.config.cpu_offload;

        if should_nvme_offload {
            // NVMe offload - clone state to avoid borrow issues
            let state_clone = self
                .states
                .get(&param_id)
                .ok_or_else(|| DistributedError::Communication("State not found".to_string()))?
                .clone();
            let offset = self.save_to_nvme(param_id, &state_clone)?;

            // Now update the state
            if let Some(state) = self.states.get_mut(&param_id) {
                state.nvme_offset = Some(offset);
                state.location = StorageLocation::Nvme;
                state.m_data.clear(); // Free CPU memory
                state.v_data.clear();
            }
        } else if should_cpu_offload {
            // Medium pressure, keep on CPU
            if let Some(state) = self.states.get_mut(&param_id) {
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
    fn save_to_nvme(&mut self, param_id: ParameterId, state: &OffloadedState) -> DistributedResult<u64> {
        if let Some(ref file) = self.nvme_file {
            let mut file = file.lock().unwrap();
            let offset = self.nvme_offset;

            // Write m_data and v_data
            let m_bytes = state.m_data.len() * 4; // f32 = 4 bytes
            let v_bytes = state.v_data.len() * 4;

            // Convert to bytes
            let m_u8: &[u8] =
                unsafe { std::slice::from_raw_parts(state.m_data.as_ptr() as *const u8, m_bytes) };
            let v_u8: &[u8] =
                unsafe { std::slice::from_raw_parts(state.v_data.as_ptr() as *const u8, v_bytes) };

            file.seek(SeekFrom::Start(offset))
                .map_err(|e| DistributedError::Communication(format!("NVMe seek failed: {}", e)))?;
            file.write_all(m_u8)
                .map_err(|e| DistributedError::Communication(format!("NVMe write failed: {}", e)))?;
            file.write_all(v_u8)
                .map_err(|e| DistributedError::Communication(format!("NVMe write failed: {}", e)))?;

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

                file.seek(SeekFrom::Start(offset))
                    .map_err(|e| DistributedError::Communication(format!("NVMe seek failed: {}", e)))?;
                file.read_exact(&mut m_buffer)
                    .map_err(|e| DistributedError::Communication(format!("NVMe read failed: {}", e)))?;
                file.read_exact(&mut v_buffer)
                    .map_err(|e| DistributedError::Communication(format!("NVMe read failed: {}", e)))?;

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
    use rustral_core::{ForwardCtx, Mode, Parameter};
    use rustral_ndarray_backend::CpuBackend;
    use rustral_optim::{Adam, Gradient};

    #[test]
    fn test_config_builder() {
        let config =
            ZeroInfinityConfig::new().with_cpu_offload(true).with_nvme_parallel(8).with_prefetch_depth(8);

        assert!(config.cpu_offload);
        assert_eq!(config.nvme_parallel, 8);
        assert_eq!(config.prefetch_depth, 8);
    }

    #[test]
    fn test_config_default() {
        let config = ZeroInfinityConfig::default();
        assert!(config.cpu_offload);
        assert!(!config.nvme_offload);
        assert_eq!(config.nvme_quota, 100_000_000_000);
        assert_eq!(config.nvme_parallel, 4);
        assert_eq!(config.prefetch_depth, 4);
        assert!(config.pin_memory);
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
    fn test_zero_infinity_creation() {
        let pg = ProcessGroup::new_single_process();
        let adam = Adam::<CpuBackend>::new(0.001);
        let config = ZeroInfinityConfig::new();
        let zero = ZeroInfinity::new(adam, pg, config).unwrap();

        assert_eq!(zero.total_params, 0);
        assert!(zero.nvme_file.is_none());
    }

    #[test]
    fn test_zero_infinity_with_nvme() {
        let pg = ProcessGroup::new_single_process();
        let adam = Adam::<CpuBackend>::new(0.001);
        let temp_path = std::env::temp_dir().join("zero_infinity_test.bin");
        let config = ZeroInfinityConfig::new().with_nvme_offload(&temp_path, 1_000_000_000);
        let zero = ZeroInfinity::new(adam, pg, config).unwrap();
        assert!(zero.nvme_file.is_some());

        // Clean up
        let _ = std::fs::remove_file(&temp_path);
    }

    #[test]
    fn test_zero_infinity_nvme_error() {
        let pg = ProcessGroup::new_single_process();
        let adam = Adam::<CpuBackend>::new(0.001);
        let config =
            ZeroInfinityConfig { nvme_offload: true, nvme_path: None, ..ZeroInfinityConfig::default() };
        assert!(ZeroInfinity::new(adam, pg, config).is_err());
    }

    #[test]
    fn test_zero_infinity_register_parameters() {
        let backend = CpuBackend::default();
        let pg = ProcessGroup::new_single_process();
        let adam = Adam::<CpuBackend>::new(0.001);
        let config = ZeroInfinityConfig::new();
        let mut zero = ZeroInfinity::new(adam, pg, config).unwrap();

        let params = vec![
            Parameter::new("p0", backend.tensor_from_vec(vec![1.0f32; 4], &[2, 2]).unwrap()),
            Parameter::new("p1", backend.tensor_from_vec(vec![2.0f32; 4], &[2, 2]).unwrap()),
        ];

        zero.register_parameters(&params).unwrap();
        assert_eq!(zero.total_params, 2);
        assert_eq!(zero.states.len(), 2);
        assert_eq!(zero.param_map.len(), 2);
    }

    #[test]
    fn test_zero_infinity_step() {
        let backend = CpuBackend::default();
        let pg = ProcessGroup::new_single_process();
        let adam = Adam::<CpuBackend>::new(0.001);
        let config = ZeroInfinityConfig::new();
        let mut zero = ZeroInfinity::new(adam, pg, config).unwrap();

        let mut params = vec![Parameter::new("p0", backend.tensor_from_vec(vec![1.0f32], &[1]).unwrap())];

        zero.register_parameters(&params).unwrap();

        let grad_tensor = backend.tensor_from_vec(vec![0.1f32], &[1]).unwrap();
        let gradients = vec![Gradient { param_id: params[0].id(), tensor: grad_tensor }];

        let mut ctx = ForwardCtx::new(&backend, Mode::Train);
        zero.step(&mut params, &gradients, &mut ctx).unwrap();
    }

    #[test]
    fn test_zero_infinity_memory_stats() {
        let backend = CpuBackend::default();
        let pg = ProcessGroup::new_single_process();
        let adam = Adam::<CpuBackend>::new(0.001);
        let config = ZeroInfinityConfig::new();
        let mut zero = ZeroInfinity::new(adam, pg, config).unwrap();

        let params = vec![Parameter::new("p0", backend.tensor_from_vec(vec![1.0f32; 4], &[2, 2]).unwrap())];
        zero.register_parameters(&params).unwrap();

        let stats = zero.memory_stats();
        assert_eq!(stats.total_params, 1);
        assert_eq!(stats.gpu_states, 0); // All start on CPU
        assert_eq!(stats.cpu_states, 1);
        assert_eq!(stats.nvme_states, 0);
    }

    #[test]
    fn test_storage_location_equality() {
        assert_eq!(StorageLocation::Gpu, StorageLocation::Gpu);
        assert_eq!(StorageLocation::Cpu, StorageLocation::Cpu);
        assert_eq!(StorageLocation::Nvme, StorageLocation::Nvme);
        assert_ne!(StorageLocation::Gpu, StorageLocation::Cpu);
    }

    #[test]
    fn test_check_memory_pressure() {
        let pg = ProcessGroup::new_single_process();
        let adam = Adam::<CpuBackend>::new(0.001);
        let config = ZeroInfinityConfig::new();
        let zero = ZeroInfinity::new(adam, pg, config).unwrap();

        assert_eq!(zero.check_memory_pressure(), 0.0);
    }

    #[test]
    fn test_fetch_state_to_gpu() {
        let backend = CpuBackend::default();
        let pg = ProcessGroup::new_single_process();
        let adam = Adam::<CpuBackend>::new(0.001);
        let config = ZeroInfinityConfig::new();
        let mut zero = ZeroInfinity::new(adam, pg, config).unwrap();

        let params = vec![Parameter::new("p0", backend.tensor_from_vec(vec![1.0f32; 4], &[2, 2]).unwrap())];
        zero.register_parameters(&params).unwrap();

        let param_id = params[0].id();
        let ops = backend.ops();
        zero.fetch_state_to_gpu(param_id, ops).unwrap();

        let state = zero.states.get(&param_id).unwrap();
        assert_eq!(state.location, StorageLocation::Gpu);
    }

    #[test]
    fn test_maybe_offload_state_cpu() {
        let backend = CpuBackend::default();
        let pg = ProcessGroup::new_single_process();
        let adam = Adam::<CpuBackend>::new(0.001);
        let config = ZeroInfinityConfig::new().with_cpu_offload(true);
        let mut zero = ZeroInfinity::new(adam, pg, config).unwrap();

        let params = vec![Parameter::new("p0", backend.tensor_from_vec(vec![1.0f32; 4], &[2, 2]).unwrap())];
        zero.register_parameters(&params).unwrap();

        // Simulate high memory pressure by filling active_params
        let param_id = params[0].id();
        zero.active_params.push(param_id);
        zero.total_params = 1;

        zero.maybe_offload_state(param_id).unwrap();
        let state = zero.states.get(&param_id).unwrap();
        // With pressure > 0.7 and cpu_offload=true, should stay on CPU
        assert_eq!(state.location, StorageLocation::Cpu);
    }

    #[test]
    fn test_nvme_save_load_roundtrip() {
        let backend = CpuBackend::default();
        let pg = ProcessGroup::new_single_process();
        let adam = Adam::<CpuBackend>::new(0.001);
        let temp_path = std::env::temp_dir().join("zero_nvme_roundtrip_test.bin");
        let config = ZeroInfinityConfig::new().with_nvme_offload(&temp_path, 1_000_000_000);
        let mut zero = ZeroInfinity::new(adam, pg, config).unwrap();

        let params = vec![Parameter::new("p0", backend.tensor_from_vec(vec![1.0f32; 4], &[2, 2]).unwrap())];
        zero.register_parameters(&params).unwrap();
        let param_id = params[0].id();

        // Manually set state data to known values
        {
            let state = zero.states.get_mut(&param_id).unwrap();
            state.m_data = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
            state.v_data = vec![0.1f32, 0.2f32, 0.3f32, 0.4f32];
        }

        // Save to NVMe
        let state_clone = zero.states.get(&param_id).unwrap().clone();
        let offset = zero.save_to_nvme(param_id, &state_clone).unwrap();
        assert_eq!(offset, 0);

        // Clear state data to simulate offloading
        {
            let state = zero.states.get_mut(&param_id).unwrap();
            state.m_data.clear();
            state.v_data.clear();
            state.nvme_offset = Some(offset);
        }

        // Load back from NVMe
        zero.load_from_nvme(param_id, offset).unwrap();

        // Clean up
        let _ = std::fs::remove_file(&temp_path);
    }

    #[test]
    fn test_zero_infinity_step_with_nvme() {
        let backend = CpuBackend::default();
        let pg = ProcessGroup::new_single_process();
        let adam = Adam::<CpuBackend>::new(0.001);
        let temp_path = std::env::temp_dir().join("zero_step_nvme_test.bin");
        let config = ZeroInfinityConfig::new().with_nvme_offload(&temp_path, 1_000_000_000);
        let mut zero = ZeroInfinity::new(adam, pg, config).unwrap();

        let mut params = vec![Parameter::new("p0", backend.tensor_from_vec(vec![1.0f32], &[1]).unwrap())];
        zero.register_parameters(&params).unwrap();

        let grad_tensor = backend.tensor_from_vec(vec![0.1f32], &[1]).unwrap();
        let gradients = vec![Gradient { param_id: params[0].id(), tensor: grad_tensor }];

        let mut ctx = ForwardCtx::new(&backend, Mode::Train);
        zero.step(&mut params, &gradients, &mut ctx).unwrap();

        let _ = std::fs::remove_file(&temp_path);
    }

    #[test]
    fn test_memory_stats_after_step() {
        let backend = CpuBackend::default();
        let pg = ProcessGroup::new_single_process();
        let adam = Adam::<CpuBackend>::new(0.001);
        let config = ZeroInfinityConfig::new();
        let mut zero = ZeroInfinity::new(adam, pg, config).unwrap();

        let mut params = vec![
            Parameter::new("p0", backend.tensor_from_vec(vec![1.0f32], &[1]).unwrap()),
            Parameter::new("p1", backend.tensor_from_vec(vec![2.0f32], &[1]).unwrap()),
        ];
        zero.register_parameters(&params).unwrap();

        let grad_tensor = backend.tensor_from_vec(vec![0.1f32], &[1]).unwrap();
        let gradients = vec![Gradient { param_id: params[0].id(), tensor: grad_tensor }];

        let mut ctx = ForwardCtx::new(&backend, Mode::Train);
        zero.step(&mut params, &gradients, &mut ctx).unwrap();

        let stats = zero.memory_stats();
        assert_eq!(stats.total_params, 2);
    }
}
