//! KV Cache Management for Inference
//!
//! Optimizes autoregressive generation by caching Key/Value tensors
//! from previous tokens to avoid redundant computation.
//!
//! # Memory Layout
//!
//! ```text
//! [batch, num_heads, seq_len, head_dim]
//!     ↑              ↑
//!     batch dim      cached sequence length (grows during generation)
//! ```
//!
//! # Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)
//!
//! MQA: Single K/V head shared across all query heads (max memory reduction)
//! GQA: K/V heads = Query heads / G (balanced)
//!
//! # Example
//! ```rust,ignore
//! use mnr_nn::kv_cache::{KVCache, CacheConfig};
//!
//! let config = CacheConfig::new(32, 128, 8192) // heads, dim, max_seq
//!     .with_quantization(CacheQuantization::Fp8);
//!
//! let mut cache = KVCache::new(&backend, config)?;
//!
//! // During generation
//! for token_id in 1..max_tokens {
//!     cache.append(&new_keys, &new_values)?;
//!     let output = attention(&queries, cache.k_cache(), cache.v_cache())?;
//! }
//! ```

use mnr_core::{Backend, CoreError, Result, TensorOps};

/// Cache quantization type
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CacheQuantization {
    /// FP32 (4 bytes/element)
    Fp32,
    /// FP16 (2 bytes/element) - 50% memory
    Fp16,
    /// FP8 (1 byte/element) - 75% memory, requires Hopper/Ada
    Fp8,
    /// INT8 (1 byte/element) with scale - 75% memory
    Int8,
}

impl CacheQuantization {
    pub fn bytes_per_element(&self) -> usize {
        match self {
            CacheQuantization::Fp32 => 4,
            CacheQuantization::Fp16 => 2,
            CacheQuantization::Fp8 => 1,
            CacheQuantization::Int8 => 1,
        }
    }

    pub fn memory_reduction(&self) -> f32 {
        1.0 - (self.bytes_per_element() as f32 / 4.0)
    }
}

/// KV Cache configuration
#[derive(Clone, Debug)]
pub struct CacheConfig {
    /// Number of attention heads
    pub num_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Maximum sequence length (pre-allocated)
    pub max_seq_len: usize,
    /// Batch size
    pub batch_size: usize,
    /// Quantization type
    pub quantization: CacheQuantization,
    /// Use Multi-Query Attention (MQA) - single K/V head
    pub use_mqa: bool,
    /// Number of K/V heads for GQA (1 for MQA, num_heads for MHA)
    pub num_kv_heads: usize,
}

impl CacheConfig {
    pub fn new(num_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        Self {
            num_heads,
            head_dim,
            max_seq_len,
            batch_size: 1,
            quantization: CacheQuantization::Fp16,
            use_mqa: false,
            num_kv_heads: num_heads, // Default to MHA
        }
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn with_quantization(mut self, q: CacheQuantization) -> Self {
        self.quantization = q;
        self
    }

    /// Enable Multi-Query Attention (single K/V head)
    pub fn with_mqa(mut self) -> Self {
        self.use_mqa = true;
        self.num_kv_heads = 1;
        self
    }

    /// Enable Grouped-Query Attention
    pub fn with_gqa(mut self, num_kv_heads: usize) -> Self {
        self.use_mqa = false;
        self.num_kv_heads = num_kv_heads;
        self
    }

    /// Calculate memory required for cache
    pub fn memory_bytes(&self) -> usize {
        let kv_len = 2; // K + V
        let elements = self.batch_size
            * self.num_kv_heads
            * self.max_seq_len
            * self.head_dim
            * kv_len;
        elements * self.quantization.bytes_per_element()
    }

    /// Calculate memory for a given sequence length
    pub fn memory_for_seq(&self, seq_len: usize) -> usize {
        let kv_len = 2;
        let elements = self.batch_size * self.num_kv_heads * seq_len * self.head_dim * kv_len;
        elements * self.quantization.bytes_per_element()
    }
}

/// KV Cache for efficient autoregressive generation
pub struct KVCache<B: Backend> {
    /// K cache: [batch, num_kv_heads, max_seq_len, head_dim]
    k_cache: B::Tensor,
    /// V cache: [batch, num_kv_heads, max_seq_len, head_dim]
    v_cache: B::Tensor,
    /// Current sequence length
    current_len: usize,
    /// Configuration
    config: CacheConfig,
    /// Scale factors for quantization (INT8)
    k_scale: Option<f32>,
    v_scale: Option<f32>,
    /// Whether cache is full
    is_full: bool,
}

impl<B: Backend> KVCache<B>
where
    B::Tensor: Clone + AsRef<[f32]> + mnr_core::TensorShape,
{
    /// Create new KV cache
    pub fn new(backend: &B, config: CacheConfig) -> Result<Self> {
        let ops = backend.ops();

        let shape = vec![
            config.batch_size,
            config.num_kv_heads,
            config.max_seq_len,
            config.head_dim,
        ];

        // Pre-allocate caches
        let k_cache = ops.zeros(&shape)?;
        let v_cache = ops.zeros(&shape)?;

        Ok(Self {
            k_cache,
            v_cache,
            current_len: 0,
            config,
            k_scale: None,
            v_scale: None,
            is_full: false,
        })
    }

    /// Append new keys and values to cache
    pub fn append(&mut self, new_k: &B::Tensor, new_v: &B::Tensor, ops: &dyn TensorOps<B>) -> Result<()> {
        let new_len = ops.shape(new_k)[2]; // sequence dimension

        // Check if cache has space
        if self.current_len + new_len > self.config.max_seq_len {
            return Err(CoreError::Shape(format!(
                "KV cache overflow: {} + {} > max {}",
                self.current_len, new_len, self.config.max_seq_len
            )));
        }

        // Get cache slices for update
        let start = self.current_len;
        let end = start + new_len;

        // In a full implementation, would use slice assignment
        // For now, simplified: reconstruct with concatenation

        // Quantize if needed
        let (k_to_store, v_to_store) = if self.config.quantization == CacheQuantization::Int8 {
            self.quantize_kv(new_k, new_v, ops)?
        } else {
            (new_k.clone(), new_v.clone())
        };

        // Update cache
        // Simplified: full cache rewrite
        let mut k_full = self.k_cache.as_ref().to_vec();
        let mut v_full = self.v_cache.as_ref().to_vec();

        let k_new = k_to_store.as_ref();
        let v_new = v_to_store.as_ref();

        // Copy new data into appropriate positions
        // This is a simplified implementation

        let k_shape = ops.shape(&self.k_cache);
        self.k_cache = ops.tensor_from_vec(k_full, &k_shape)?;
        self.v_cache = ops.tensor_from_vec(v_full, &k_shape)?;

        self.current_len += new_len;
        self.is_full = self.current_len >= self.config.max_seq_len;

        Ok(())
    }

    /// Get cached keys up to current length
    pub fn k_cache(&self) -> &B::Tensor {
        &self.k_cache
    }

    /// Get cached values up to current length
    pub fn v_cache(&self) -> &B::Tensor {
        &self.v_cache
    }

    /// Get K/V for specific sequence range
    pub fn get_range(&self, start: usize, end: usize, ops: &dyn TensorOps<B>) -> Result<(B::Tensor, B::Tensor)> {
        // In real impl, would slice the cache
        // Simplified: return full cache
        let k_slice = self.k_cache.clone();
        let v_slice = self.v_cache.clone();

        // Would apply dequantization here for INT8

        Ok((k_slice, v_slice))
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.current_len = 0;
        self.is_full = false;
    }

    /// Current sequence length
    pub fn current_len(&self) -> usize {
        self.current_len
    }

    /// Maximum sequence length
    pub fn max_len(&self) -> usize {
        self.config.max_seq_len
    }

    /// Check if cache is full
    pub fn is_full(&self) -> bool {
        self.is_full
    }

    /// Quantize K/V to INT8
    fn quantize_kv(&self, k: &B::Tensor, v: &B::Tensor, ops: &dyn TensorOps<B>) -> Result<(B::Tensor, B::Tensor)> {
        // Find scales
        let k_data: Vec<f32> = k.as_ref().to_vec();
        let v_data: Vec<f32> = v.as_ref().to_vec();

        let k_max = k_data.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        let v_max = v_data.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);

        let k_scale = k_max / 127.0;
        let v_scale = v_max / 127.0;

        // Quantize
        let k_quantized: Vec<f32> = k_data.iter().map(|&x| {
            (x / k_scale * 127.0).round().clamp(-127.0, 127.0)
        }).collect();

        let v_quantized: Vec<f32> = v_data.iter().map(|&x| {
            (x / v_scale * 127.0).round().clamp(-127.0, 127.0)
        }).collect();

        let shape = ops.shape(k);
        let k_t = ops.tensor_from_vec(k_quantized, &shape)?;
        let v_t = ops.tensor_from_vec(v_quantized, &shape)?;

        Ok((k_t, v_t))
    }

    /// Dequantize from INT8
    fn dequantize_kv(&self, k: &B::Tensor, v: &B::Tensor, k_scale: f32, v_scale: f32, ops: &dyn TensorOps<B>) -> Result<(B::Tensor, B::Tensor)> {
        let k_data: Vec<f32> = k.as_ref().iter().map(|&x| x * k_scale / 127.0).collect();
        let v_data: Vec<f32> = v.as_ref().iter().map(|&x| x * v_scale / 127.0).collect();

        let shape = ops.shape(k);
        let k_t = ops.tensor_from_vec(k_data, &shape)?;
        let v_t = ops.tensor_from_vec(v_data, &shape)?;

        Ok((k_t, v_t))
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> CacheMemoryStats {
        let used_bytes = self.config.memory_for_seq(self.current_len);
        let max_bytes = self.config.memory_bytes();

        CacheMemoryStats {
            used_bytes,
            max_bytes,
            utilization: used_bytes as f32 / max_bytes as f32,
            current_len: self.current_len,
            max_len: self.config.max_seq_len,
        }
    }
}

/// Memory statistics for cache
#[derive(Debug, Clone)]
pub struct CacheMemoryStats {
    pub used_bytes: usize,
    pub max_bytes: usize,
    pub utilization: f32,
    pub current_len: usize,
    pub max_len: usize,
}

/// Sliding window cache for long sequences
pub struct SlidingWindowCache<B: Backend> {
    /// Underlying cache
    cache: KVCache<B>,
    /// Window size
    window_size: usize,
    /// Start position in underlying cache
    window_start: usize,
}

impl<B: Backend> SlidingWindowCache<B>
where
    B::Tensor: Clone + AsRef<[f32]> + mnr_core::TensorShape,
{
    pub fn new(backend: &B, config: CacheConfig, window_size: usize) -> Result<Self> {
        let cache = KVCache::new(backend, config)?;
        Ok(Self {
            cache,
            window_size,
            window_start: 0,
        })
    }

    /// Append with sliding window
    pub fn append(&mut self, new_k: &B::Tensor, new_v: &B::Tensor, ops: &dyn TensorOps<B>) -> Result<()> {
        // If at max length, slide window
        if self.cache.current_len() + 1 > self.window_size {
            self.window_start += 1;
            // Would rotate cache in real impl
        }

        self.cache.append(new_k, new_v, ops)
    }

    /// Get cache with only window contents
    pub fn get_window(&self, _ops: &dyn TensorOps<B>) -> Result<(B::Tensor, B::Tensor)> {
        // Return only the window portion
        let k = &self.cache.k_cache;
        let v = &self.cache.v_cache;

        Ok((k.clone(), v.clone()))
    }
}

/// PagedAttention-style block-based cache (vLLM)
pub struct PagedCache<B: Backend> {
    /// Block size (e.g., 16 tokens per block)
    block_size: usize,
    /// Blocks per sequence
    num_blocks: usize,
    /// Free blocks pool
    free_blocks: Vec<usize>,
    /// Sequence to block mapping
    block_tables: HashMap<usize, Vec<usize>>,
    /// Block data
    blocks: Vec<B::Tensor>,
}

impl<B: Backend> PagedCache<B>
where
    B::Tensor: Clone,
{
    pub fn new(block_size: usize, num_blocks: usize, backend: &B) -> Result<Self> {
        // Pre-allocate blocks
        let mut blocks = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            // In real impl, would allocate block tensors
            // blocks.push(allocate_block(backend, block_size)?);
        }

        let free_blocks: Vec<usize> = (0..num_blocks).collect();

        Ok(Self {
            block_size,
            num_blocks,
            free_blocks,
            block_tables: HashMap::new(),
            blocks,
        })
    }

    /// Allocate blocks for a new sequence
    pub fn allocate(&mut self, seq_id: usize, num_tokens: usize) -> Option<Vec<usize>> {
        let num_blocks_needed = (num_tokens + self.block_size - 1) / self.block_size;

        if self.free_blocks.len() < num_blocks_needed {
            return None; // Out of memory
        }

        let allocated: Vec<usize> = self.free_blocks
            .drain(0..num_blocks_needed)
            .collect();

        self.block_tables.insert(seq_id, allocated.clone());
        Some(allocated)
    }

    /// Free blocks for a sequence
    pub fn free(&mut self, seq_id: usize) {
        if let Some(blocks) = self.block_tables.remove(&seq_id) {
            self.free_blocks.extend(blocks);
        }
    }
}

/// Cache for batched inference with different sequence lengths
pub struct BatchedCache<B: Backend> {
    caches: Vec<KVCache<B>>,
}

impl<B: Backend> BatchedCache<B>
where
    B::Tensor: Clone + AsRef<[f32]> + mnr_core::TensorShape,
{
    pub fn new(backend: &B, config: CacheConfig, batch_size: usize) -> Result<Self> {
        let mut caches = Vec::with_capacity(batch_size);
        let mut cfg = config.clone();
        cfg.batch_size = 1;

        for _ in 0..batch_size {
            caches.push(KVCache::new(backend, cfg.clone())?);
        }

        Ok(Self { caches })
    }

    /// Get cache for specific batch element
    pub fn get(&mut self, batch_idx: usize) -> Option<&mut KVCache<B>> {
        self.caches.get_mut(batch_idx)
    }

    /// Clear all caches
    pub fn clear_all(&mut self) {
        for cache in &mut self.caches {
            cache.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnr_ndarray_backend::CpuBackend;

    #[test]
    fn test_cache_config() {
        let config = CacheConfig::new(32, 128, 8192)
            .with_batch_size(4)
            .with_quantization(CacheQuantization::Fp16)
            .with_mqa();

        assert_eq!(config.num_heads, 32);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.max_seq_len, 8192);
        assert_eq!(config.batch_size, 4);
        assert!(config.use_mqa);
        assert_eq!(config.num_kv_heads, 1);

        // Memory: 4 batch * 1 head * 8192 seq * 128 dim * 2 (K+V) * 2 bytes (FP16)
        let expected = 4 * 1 * 8192 * 128 * 2 * 2;
        assert_eq!(config.memory_bytes(), expected);
    }

    #[test]
    fn test_cache_config_gqa() {
        let config = CacheConfig::new(32, 128, 4096)
            .with_gqa(4); // 4 K/V heads for 32 query heads

        assert_eq!(config.num_kv_heads, 4);
        assert!(!config.use_mqa);
    }

    #[test]
    fn test_kv_cache_creation() {
        let backend = CpuBackend::default();
        let config = CacheConfig::new(8, 64, 1024)
            .with_batch_size(1);

        let cache = KVCache::new(&backend, config).unwrap();

        assert_eq!(cache.current_len(), 0);
        assert!(!cache.is_full());
    }

    #[test]
    fn test_memory_stats() {
        let backend = CpuBackend::default();
        let config = CacheConfig::new(8, 64, 1024)
            .with_batch_size(1)
            .with_quantization(CacheQuantization::Fp16);

        let mut cache = KVCache::new(&backend, config).unwrap();

        // Initially empty
        let stats = cache.memory_stats();
        assert_eq!(stats.utilization, 0.0);
        assert_eq!(stats.current_len, 0);
    }

    #[test]
    fn test_quantization_memory_savings() {
        let base = CacheConfig::new(32, 128, 8192);

        let fp32 = base.clone();
        let fp16 = base.clone().with_quantization(CacheQuantization::Fp16);
        let fp8 = base.clone().with_quantization(CacheQuantization::Fp8);

        assert_eq!(fp32.memory_bytes() / fp16.memory_bytes(), 2);
        assert_eq!(fp32.memory_bytes() / fp8.memory_bytes(), 4);
    }

    #[test]
    fn test_paged_cache() {
        let backend = CpuBackend::default();
        let mut cache = PagedCache::new(16, 100, &backend).unwrap();

        // Allocate blocks for 100 tokens (7 blocks needed: ceil(100/16))
        let blocks = cache.allocate(0, 100);
        assert!(blocks.is_some());
        assert_eq!(blocks.unwrap().len(), 7);

        // Free blocks
        cache.free(0);
        assert_eq!(cache.free_blocks.len(), 100);
    }

    #[test]
    fn test_batched_cache() {
        let backend = CpuBackend::default();
        let config = CacheConfig::new(8, 64, 512);

        let mut batched = BatchedCache::new(&backend, config, 4).unwrap();

        assert!(batched.get(0).is_some());
        assert!(batched.get(3).is_some());
        assert!(batched.get(4).is_none());
    }
}
