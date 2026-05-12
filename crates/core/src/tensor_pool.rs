//! Generic Memory Pooling System for Device-Agnostic Optimization
//!
//! Provides cross-backend memory pooling to reduce allocation overhead
//! and improve performance across all backends.
//!
//! # Features
//!
//! - Shape-based tensor pooling
//! - Configurable pool size limits
//! - Automatic cleanup when pools exceed limits
//! - Thread-safe implementation
//!
//! # Example
//!
//! ```rust,ignore
//! use rustral_core::tensor_pool::{TensorPool, PooledTensor};
//! use rustral_core::Backend;
//!
//! let mut pool = TensorPool::new();
//!
//! // Get a tensor from pool or create new one
//! let tensor = pool.get_or_create(&backend, &[1024, 512])?;
//!
//! // Return tensor to pool when done
//! pool.return_tensor(tensor);
//! ```

use crate::{Backend, Result};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// How the pool should behave across training or inference runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PoolStrategy {
    /// Shape-keyed reuse with existing heuristics (default).
    #[default]
    Standard,
    /// Clear pooled buffers at step boundaries to bound peak memory during training.
    TrainingArena,
    /// Hint that shapes repeat (typical static inference); limits may be tuned tighter later.
    InferenceReuse {
        /// When true, callers promise fixed shapes for the run.
        static_shapes: bool,
    },
}

/// Generic tensor pool for reducing allocation overhead.
pub struct TensorPool<B: Backend> {
    strategy: PoolStrategy,
    pool: HashMap<Vec<usize>, Vec<B::Tensor>>,
    max_size_per_shape: usize,
    max_total_tensors: usize,
}

impl<B: Backend> TensorPool<B> {
    /// Create a new tensor pool with default limits.
    pub fn new() -> Self {
        Self {
            strategy: PoolStrategy::default(),
            pool: HashMap::new(),
            max_size_per_shape: 10,
            max_total_tensors: 100,
        }
    }

    /// Create a pool with an explicit pooling strategy.
    pub fn with_strategy(strategy: PoolStrategy) -> Self {
        Self { strategy, pool: HashMap::new(), max_size_per_shape: 10, max_total_tensors: 100 }
    }

    /// Create a new tensor pool with custom limits.
    pub fn with_limits(max_size_per_shape: usize, max_total_tensors: usize) -> Self {
        Self {
            strategy: PoolStrategy::default(),
            pool: HashMap::new(),
            max_size_per_shape,
            max_total_tensors,
        }
    }

    /// Limits plus strategy.
    pub fn with_limits_and_strategy(
        max_size_per_shape: usize,
        max_total_tensors: usize,
        strategy: PoolStrategy,
    ) -> Self {
        Self { strategy, pool: HashMap::new(), max_size_per_shape, max_total_tensors }
    }

    pub fn strategy(&self) -> PoolStrategy {
        self.strategy
    }

    pub fn set_strategy(&mut self, strategy: PoolStrategy) {
        self.strategy = strategy;
    }

    /// Call between training steps (or when reusing temporaries across a bounded region).
    ///
    /// For [`PoolStrategy::TrainingArena`], this clears pooled tensors so the next step does not
    /// retain allocations from the previous one.
    pub fn begin_step(&mut self) {
        if matches!(self.strategy, PoolStrategy::TrainingArena) {
            self.clear();
        }
    }

    /// Get a tensor from the pool or create a new one.
    pub fn get_or_create(&mut self, backend: &B, shape: &[usize]) -> Result<B::Tensor> {
        if let Some(tensors) = self.pool.get_mut(shape) {
            if let Some(tensor) = tensors.pop() {
                return Ok(tensor);
            }
        }

        // Create new tensor if pool is empty
        backend.ops().zeros(shape)
    }

    /// Return a tensor to the pool for reuse.
    pub fn return_tensor(&mut self, tensor: B::Tensor, ops: &dyn crate::TensorOps<B>) {
        let shape = ops.shape(&tensor);

        // Check if we should add to pool
        if self.should_pool_tensor(&shape) {
            if let Some(tensors) = self.pool.get_mut(&shape) {
                if tensors.len() < self.max_size_per_shape {
                    tensors.push(tensor);
                }
            } else {
                self.pool.insert(shape.clone(), vec![tensor]);
            }
        }

        // Enforce total tensor limit
        self.enforce_total_limit();
    }

    /// Check if a tensor should be pooled based on current state.
    fn should_pool_tensor(&self, shape: &[usize]) -> bool {
        // Don't pool very large tensors (they consume too much memory)
        let total_elements: usize = shape.iter().product();
        if total_elements > 1_000_000 {
            return false;
        }

        // Don't pool very small tensors (overhead outweighs benefit)
        if total_elements < 100 {
            return false;
        }

        true
    }

    /// Enforce total tensor limit by removing oldest tensors.
    fn enforce_total_limit(&mut self) {
        let total_count: usize = self.pool.values().map(|v| v.len()).sum();

        if total_count > self.max_total_tensors {
            // Remove tensors from largest shapes first
            let mut shapes: Vec<_> = self.pool.keys().cloned().collect();
            shapes.sort_by_key(|k| {
                let elements: usize = k.iter().product();
                std::cmp::Reverse(elements)
            });

            let mut removed = 0;
            for shape in shapes {
                if let Some(tensors) = self.pool.get_mut(&shape) {
                    if !tensors.is_empty() {
                        tensors.pop();
                        removed += 1;
                        if total_count - removed <= self.max_total_tensors {
                            break;
                        }
                    }
                }
            }
        }
    }

    /// Clear all pooled tensors.
    pub fn clear(&mut self) {
        self.pool.clear();
    }

    /// Get pool statistics.
    pub fn stats(&self) -> PoolStats {
        let total_tensors: usize = self.pool.values().map(|v| v.len()).sum();
        let total_memory: usize = self
            .pool
            .iter()
            .map(|(shape, tensors)| {
                let elements: usize = shape.iter().product();
                elements * tensors.len() * 4 // 4 bytes per f32
            })
            .sum();

        PoolStats {
            total_tensors,
            total_memory_bytes: total_memory,
            unique_shapes: self.pool.len(),
            max_size_per_shape: self.max_size_per_shape,
            max_total_tensors: self.max_total_tensors,
        }
    }

    /// Print pool statistics.
    pub fn print_stats(&self) {
        let stats = self.stats();
        println!("\nTensor Pool Statistics:");
        println!("  Total tensors: {}", stats.total_tensors);
        println!("  Total memory: {:.2} MB", stats.total_memory_bytes as f64 / 1e6);
        println!("  Unique shapes: {}", stats.unique_shapes);
        println!("  Max per shape: {}", stats.max_size_per_shape);
        println!("  Max total: {}", stats.max_total_tensors);
    }
}

impl<B: Backend> Default for TensorPool<B> {
    fn default() -> Self {
        Self {
            strategy: PoolStrategy::default(),
            pool: HashMap::new(),
            max_size_per_shape: 10,
            max_total_tensors: 100,
        }
    }
}

/// Pool statistics.
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub total_tensors: usize,
    pub total_memory_bytes: usize,
    pub unique_shapes: usize,
    pub max_size_per_shape: usize,
    pub max_total_tensors: usize,
}

/// RAII guard for pooled tensors.
pub struct PooledTensor<B: Backend> {
    tensor: Option<B::Tensor>,
    pool: Option<Arc<Mutex<TensorPool<B>>>>,
    ops: Option<Arc<dyn crate::TensorOps<B> + Send + Sync>>,
}

impl<B: Backend> PooledTensor<B> {
    /// Create a new pooled tensor guard.
    pub fn new(
        tensor: B::Tensor,
        pool: Arc<Mutex<TensorPool<B>>>,
        ops: Arc<dyn crate::TensorOps<B> + Send + Sync>,
    ) -> Self {
        Self { tensor: Some(tensor), pool: Some(pool), ops: Some(ops) }
    }

    /// Get access to the underlying tensor.
    pub fn get(&self) -> &B::Tensor {
        self.tensor.as_ref().expect("Tensor already consumed")
    }

    /// Get mutable access to the underlying tensor.
    pub fn get_mut(&mut self) -> &mut B::Tensor {
        self.tensor.as_mut().expect("Tensor already consumed")
    }

    /// Consume the guard and return the tensor without pooling it.
    pub fn into_inner(mut self) -> B::Tensor {
        self.tensor.take().expect("Tensor already consumed")
    }
}

impl<B: Backend> Drop for PooledTensor<B> {
    fn drop(&mut self) {
        if let (Some(tensor), Some(pool), Some(ops)) = (self.tensor.take(), &self.pool, &self.ops) {
            if let Ok(mut pool) = pool.lock() {
                pool.return_tensor(tensor, ops.as_ref());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_pool_creation() {
        #[derive(Clone)]
        struct MockBackend;
        impl Backend for MockBackend {
            type Tensor = ();
            type Device = ();

            fn device(&self) -> Self::Device {}
            fn ops(&self) -> &dyn crate::TensorOps<Self> {
                unimplemented!()
            }
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
            fn capabilities(&self) -> crate::BackendCapabilities {
                Default::default()
            }
            fn normal_parameter(
                &self,
                _name: &str,
                _shape: &[usize],
                _seed: u64,
                _scale: f32,
            ) -> Result<crate::Parameter<Self>> {
                unimplemented!()
            }
            fn parameter_from_vec(
                &self,
                _name: &str,
                _values: Vec<f32>,
                _shape: &[usize],
            ) -> Result<crate::Parameter<Self>> {
                unimplemented!()
            }
        }

        let pool: TensorPool<MockBackend> = TensorPool::new();
        assert_eq!(pool.stats().total_tensors, 0);
    }

    #[test]
    fn test_pool_limits() {
        #[derive(Clone)]
        struct MockBackend;
        impl Backend for MockBackend {
            type Tensor = ();
            type Device = ();

            fn device(&self) -> Self::Device {}
            fn ops(&self) -> &dyn crate::TensorOps<Self> {
                unimplemented!()
            }
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
            fn capabilities(&self) -> crate::BackendCapabilities {
                Default::default()
            }
            fn normal_parameter(
                &self,
                _name: &str,
                _shape: &[usize],
                _seed: u64,
                _scale: f32,
            ) -> Result<crate::Parameter<Self>> {
                unimplemented!()
            }
            fn parameter_from_vec(
                &self,
                _name: &str,
                _values: Vec<f32>,
                _shape: &[usize],
            ) -> Result<crate::Parameter<Self>> {
                unimplemented!()
            }
        }

        let pool: TensorPool<MockBackend> = TensorPool::with_limits(5, 10);
        assert_eq!(pool.max_size_per_shape, 5);
        assert_eq!(pool.max_total_tensors, 10);
        assert_eq!(pool.strategy(), PoolStrategy::Standard);
    }

    #[test]
    fn training_arena_begin_step_clears() {
        #[derive(Clone)]
        struct MockBackend;
        impl Backend for MockBackend {
            type Tensor = ();
            type Device = ();

            fn device(&self) -> Self::Device {}
            fn ops(&self) -> &dyn crate::TensorOps<Self> {
                unimplemented!()
            }
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
            fn capabilities(&self) -> crate::BackendCapabilities {
                Default::default()
            }
            fn normal_parameter(
                &self,
                _name: &str,
                _shape: &[usize],
                _seed: u64,
                _scale: f32,
            ) -> Result<crate::Parameter<Self>> {
                unimplemented!()
            }
            fn parameter_from_vec(
                &self,
                _name: &str,
                _values: Vec<f32>,
                _shape: &[usize],
            ) -> Result<crate::Parameter<Self>> {
                unimplemented!()
            }
        }

        let mut pool: TensorPool<MockBackend> = TensorPool::with_strategy(PoolStrategy::TrainingArena);
        pool.pool.insert(vec![2, 2], vec![()]);
        pool.begin_step();
        assert!(pool.pool.is_empty());
    }

    #[test]
    fn test_clear() {
        #[derive(Clone)]
        struct MockBackend;
        impl Backend for MockBackend {
            type Tensor = ();
            type Device = ();

            fn device(&self) -> Self::Device {}
            fn ops(&self) -> &dyn crate::TensorOps<Self> {
                unimplemented!()
            }
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
            fn capabilities(&self) -> crate::BackendCapabilities {
                Default::default()
            }
            fn normal_parameter(
                &self,
                _name: &str,
                _shape: &[usize],
                _seed: u64,
                _scale: f32,
            ) -> Result<crate::Parameter<Self>> {
                unimplemented!()
            }
            fn parameter_from_vec(
                &self,
                _name: &str,
                _values: Vec<f32>,
                _shape: &[usize],
            ) -> Result<crate::Parameter<Self>> {
                unimplemented!()
            }
        }

        let mut pool: TensorPool<MockBackend> = TensorPool::new();
        pool.clear();
        assert_eq!(pool.stats().total_tensors, 0);
    }

    #[test]
    fn test_pool_stats() {
        #[derive(Clone)]
        struct MockBackend;
        impl Backend for MockBackend {
            type Tensor = ();
            type Device = ();

            fn device(&self) -> Self::Device {}
            fn ops(&self) -> &dyn crate::TensorOps<Self> {
                unimplemented!()
            }
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
            fn capabilities(&self) -> crate::BackendCapabilities {
                Default::default()
            }
            fn normal_parameter(
                &self,
                _name: &str,
                _shape: &[usize],
                _seed: u64,
                _scale: f32,
            ) -> Result<crate::Parameter<Self>> {
                unimplemented!()
            }
            fn parameter_from_vec(
                &self,
                _name: &str,
                _values: Vec<f32>,
                _shape: &[usize],
            ) -> Result<crate::Parameter<Self>> {
                unimplemented!()
            }
        }

        let pool: TensorPool<MockBackend> = TensorPool::new();
        let stats = pool.stats();
        assert_eq!(stats.total_tensors, 0);
        assert_eq!(stats.total_memory_bytes, 0);
        assert_eq!(stats.unique_shapes, 0);
    }
}
