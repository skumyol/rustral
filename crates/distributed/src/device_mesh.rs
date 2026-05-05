//! Device Mesh for Multi-Dimensional Parallelism
//!
//! Provides a flexible abstraction for mapping computation across
//! multi-dimensional device topologies (for 3D parallelism).
//!
//! # 3D Parallelism Dimensions
//!
//! - **Data Parallel (DP)**: Replicate model, split batch
//! - **Tensor Parallel (TP)**: Split layers across devices  
//! - **Pipeline Parallel (PP)**: Split model stages across devices
//!
//! # Example
//!
//! ```rust,ignore
//! use mnr_distributed::DeviceMesh;
//!
//! // Create 2x4x2 mesh for 16 GPUs
//! // 2 data parallel groups, 4 tensor parallel, 2 pipeline stages
//! let mesh = DeviceMesh::new([2, 4, 2]);
//!
//! // Get the process group for tensor parallelism at my coord
//! let tp_group = mesh.get_process_group(&[1, -1, 0]); // All TP ranks for DP=1, PP=0
//! ```

use std::collections::HashMap;

use mnr_core::Result;

use crate::ProcessGroup;

/// A coordinate in the device mesh.
pub type MeshCoord = Vec<i64>; // -1 means "all" for slicing

/// Multi-dimensional device topology for 3D parallelism.
///
/// Organizes devices into a logical mesh where each dimension
/// corresponds to a different parallelism strategy.
pub struct DeviceMesh {
    /// Shape of the mesh [dp_size, tp_size, pp_size, ...]
    shape: Vec<usize>,
    /// Total number of devices
    world_size: usize,
    /// My rank's coordinate in the mesh
    my_coord: Vec<usize>,
    /// Cached process groups for each slice
    process_groups: HashMap<String, ProcessGroup>,
}

impl DeviceMesh {
    /// Create a new device mesh from the given shape.
    ///
    /// # Arguments
    /// * `shape` - Dimensions [dp, tp, pp, ...]
    /// * `my_rank` - This process's global rank
    pub fn new(shape: &[usize], my_rank: usize) -> Result<Self> {
        let world_size: usize = shape.iter().product();
        
        if my_rank >= world_size {
            return Err(mnr_core::CoreError::InvalidArgument(
                format!("Rank {} out of bounds for mesh with {} devices", my_rank, world_size)
            ));
        }

        let my_coord = Self::rank_to_coord(my_rank, shape);
        
        Ok(Self {
            shape: shape.to_vec(),
            world_size,
            my_coord,
            process_groups: HashMap::new(),
        })
    }

    /// Convert global rank to mesh coordinate.
    fn rank_to_coord(rank: usize, shape: &[usize]) -> Vec<usize> {
        let mut coord = Vec::with_capacity(shape.len());
        let mut remaining = rank;
        
        // Calculate strides
        let mut strides = vec![1usize; shape.len()];
        for i in (0..shape.len()-1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        
        for i in 0..shape.len() {
            coord.push(remaining / strides[i]);
            remaining %= strides[i];
        }
        
        coord
    }

    /// Convert mesh coordinate to global rank.
    fn coord_to_rank(coord: &[usize], shape: &[usize]) -> usize {
        let mut rank = 0;
        let mut stride = 1;
        
        for i in (0..shape.len()).rev() {
            rank += coord[i] * stride;
            stride *= shape[i];
        }
        
        rank
    }

    /// Get all ranks that match the given coordinate pattern.
    ///
    /// # Arguments
    /// * `pattern` - Coordinate pattern where -1 means "any"
    ///
    /// # Example
    /// * `[-1, 2, 3]` - All ranks with tp=2, pp=3 (across all DP)
    /// * `[1, -1, 0]` - All ranks with dp=1, pp=0 (across all TP)
    fn matching_ranks(&self, pattern: &[i64]) -> Vec<usize> {
        assert_eq!(pattern.len(), self.shape.len());
        
        let mut ranks = Vec::new();
        
        // Generate all possible coordinates
        let total_coords: usize = self.shape.iter().product();
        for idx in 0..total_coords {
            let coord = Self::rank_to_coord(idx, &self.shape);
            
            // Check if coord matches pattern
            let matches = pattern.iter().zip(coord.iter()).all(|(p, c)| {
                *p < 0 || *p as usize == *c
            });
            
            if matches {
                ranks.push(idx);
            }
        }
        
        ranks
    }

    /// Get or create a process group for the given slice.
    ///
    /// # Arguments
    /// * `pattern` - Coordinate pattern (use -1 for dimensions to span)
    pub fn get_process_group(&mut self, pattern: &[i64]) -> Result<ProcessGroup> {
        let key = pattern.iter()
            .map(|&x| if x < 0 { "_".to_string() } else { x.to_string() })
            .collect::<Vec<_>>()
            .join(",");
        
        if let Some(pg) = self.process_groups.get(&key) {
            return Ok(pg.clone());
        }
        
        // Find matching ranks
        let ranks = self.matching_ranks(pattern);
        
        if ranks.is_empty() {
            return Err(mnr_core::CoreError::InvalidArgument(
                format!("No ranks match pattern {:?}", pattern)
            ));
        }
        
        // Find my position in this group
        let my_local_rank = ranks.iter()
            .position(|&r| r == self.my_rank())
            .ok_or_else(|| mnr_core::CoreError::InvalidArgument(
                format!("My rank {} not in group for pattern {:?}", self.my_rank(), pattern)
            ))?;
        
        // Create process group (simplified - real impl would use actual comm)
        let pg = ProcessGroup::new_threaded(ranks.len(), my_local_rank)
            .map_err(|e| mnr_core::CoreError::Backend(e.to_string()))?;

        self.process_groups.insert(key, pg.clone());
        Ok(pg)
    }

    /// Get process group for data parallelism (all devices with same TP and PP).
    pub fn get_data_parallel_group(&mut self) -> Result<ProcessGroup> {
        if self.shape.len() < 3 {
            return Err(mnr_core::CoreError::InvalidArgument(
                "Device mesh needs at least 3 dimensions for 3D parallelism".to_string()
            ));
        }
        
        // Pattern: [-1, my_tp, my_pp] - vary DP
        let pattern: Vec<i64> = self.my_coord.iter()
            .enumerate()
            .map(|(i, &c)| if i == 0 { -1 } else { c as i64 })
            .collect();
        
        self.get_process_group(&pattern)
    }

    /// Get process group for tensor parallelism (all devices with same DP and PP).
    pub fn get_tensor_parallel_group(&mut self) -> Result<ProcessGroup> {
        if self.shape.len() < 3 {
            return Err(mnr_core::CoreError::InvalidArgument(
                "Device mesh needs at least 3 dimensions for 3D parallelism".to_string()
            ));
        }
        
        // Pattern: [my_dp, -1, my_pp] - vary TP
        let pattern: Vec<i64> = self.my_coord.iter()
            .enumerate()
            .map(|(i, &c)| if i == 1 { -1 } else { c as i64 })
            .collect();
        
        self.get_process_group(&pattern)
    }

    /// Get process group for pipeline parallelism (all devices with same DP and TP).
    pub fn get_pipeline_parallel_group(&mut self) -> Result<ProcessGroup> {
        if self.shape.len() < 3 {
            return Err(mnr_core::CoreError::InvalidArgument(
                "Device mesh needs at least 3 dimensions for 3D parallelism".to_string()
            ));
        }
        
        // Pattern: [my_dp, my_tp, -1] - vary PP
        let pattern: Vec<i64> = self.my_coord.iter()
            .enumerate()
            .map(|(i, &c)| if i == 2 { -1 } else { c as i64 })
            .collect();
        
        self.get_process_group(&pattern)
    }

    /// Get my rank in the global mesh.
    pub fn my_rank(&self) -> usize {
        Self::coord_to_rank(&self.my_coord, &self.shape)
    }

    /// Get my coordinate in the mesh.
    pub fn my_coord(&self) -> &[usize] {
        &self.my_coord
    }

    /// Get the mesh shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get total world size.
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Get the size of a specific dimension.
    pub fn dim_size(&self, dim: usize) -> usize {
        self.shape.get(dim).copied().unwrap_or(1)
    }

    /// Create a 3D mesh for standard parallelism strategies.
    ///
    /// # Arguments
    /// * `dp_size` - Data parallel size
    /// * `tp_size` - Tensor parallel size  
    /// * `pp_size` - Pipeline parallel size
    /// * `my_rank` - This process's global rank
    pub fn for_3d_parallelism(dp_size: usize, tp_size: usize, pp_size: usize, my_rank: usize) -> Result<Self> {
        Self::new(&[dp_size, tp_size, pp_size], my_rank)
    }
}

/// Automatic parallelism strategy selector.
pub struct ParallelismConfig {
    pub data_parallel: usize,
    pub tensor_parallel: usize,
    pub pipeline_parallel: usize,
}

impl ParallelismConfig {
    /// Create configuration from available GPUs and model size.
    pub fn auto_select(num_gpus: usize, model_params: usize) -> Self {
        // Heuristic: Use TP for large layers, PP for deep models
        let tp_size = if model_params > 1_000_000_000 {
            // Large model: use more tensor parallelism
            (num_gpus / 2).max(1).min(8)
        } else {
            1
        };
        
        let remaining = num_gpus / tp_size;
        
        // Use PP for very deep models, DP otherwise
        let pp_size = if model_params > 10_000_000_000 && remaining >= 2 {
            (remaining / 2).max(1).min(8)
        } else {
            1
        };
        
        let dp_size = remaining / pp_size;
        
        Self {
            data_parallel: dp_size.max(1),
            tensor_parallel: tp_size,
            pipeline_parallel: pp_size,
        }
    }

    /// Get total number of devices needed.
    pub fn total_devices(&self) -> usize {
        self.data_parallel * self.tensor_parallel * self.pipeline_parallel
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rank_coord_conversion() {
        let shape = vec![2, 4, 2]; // 16 devices
        
        // Test rank 0 -> [0, 0, 0]
        let coord0 = DeviceMesh::rank_to_coord(0, &shape);
        assert_eq!(coord0, vec![0, 0, 0]);
        
        // Test rank 7 -> [0, 3, 1]
        let coord7 = DeviceMesh::rank_to_coord(7, &shape);
        assert_eq!(coord7, vec![0, 3, 1]);
        
        // Test rank 8 -> [1, 0, 0]
        let coord8 = DeviceMesh::rank_to_coord(8, &shape);
        assert_eq!(coord8, vec![1, 0, 0]);
        
        // Round-trip test
        for rank in 0..16 {
            let coord = DeviceMesh::rank_to_coord(rank, &shape);
            let back = DeviceMesh::coord_to_rank(&coord, &shape);
            assert_eq!(back, rank);
        }
    }

    #[test]
    fn test_matching_ranks() {
        let mesh = DeviceMesh::new(&[2, 4, 2], 0).unwrap();
        
        // Pattern [-1, 0, 1]: all DP ranks with TP=0, PP=1
        let pattern = vec![-1, 0, 1];
        let ranks = mesh.matching_ranks(&pattern);
        assert_eq!(ranks, vec![1, 9]); // ranks with coords [0,0,1] and [1,0,1]
        
        // Pattern [0, -1, 0]: all TP ranks with DP=0, PP=0
        let pattern2 = vec![0, -1, 0];
        let ranks2 = mesh.matching_ranks(&pattern2);
        assert_eq!(ranks2, vec![0, 2, 4, 6]);
    }

    #[test]
    fn test_mesh_creation_error() {
        let result = DeviceMesh::new(&[2, 4, 2], 16);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_process_group() {
        let mut mesh = DeviceMesh::new(&[2, 2, 2], 0).unwrap();
        
        // Get group for all ranks with same TP and PP as rank 0
        let pg = mesh.get_process_group(&[-1, 0, 0]).unwrap();
        assert_eq!(pg.world_size(), 2); // DP dimension
        
        // Cached: same call should return same group
        let pg2 = mesh.get_process_group(&[-1, 0, 0]).unwrap();
        assert_eq!(pg2.world_size(), 2);
    }

    #[test]
    fn test_get_data_parallel_group() {
        let mut mesh = DeviceMesh::for_3d_parallelism(2, 2, 2, 0).unwrap();
        let pg = mesh.get_data_parallel_group().unwrap();
        assert_eq!(pg.world_size(), 2);
    }

    #[test]
    fn test_get_tensor_parallel_group() {
        let mut mesh = DeviceMesh::for_3d_parallelism(2, 2, 2, 0).unwrap();
        let pg = mesh.get_tensor_parallel_group().unwrap();
        assert_eq!(pg.world_size(), 2);
    }

    #[test]
    fn test_get_pipeline_parallel_group() {
        let mut mesh = DeviceMesh::for_3d_parallelism(2, 2, 2, 0).unwrap();
        let pg = mesh.get_pipeline_parallel_group().unwrap();
        assert_eq!(pg.world_size(), 2);
    }

    #[test]
    fn test_get_group_errors() {
        let mut mesh = DeviceMesh::new(&[2, 2, 2], 0).unwrap();
        
        // Pattern doesn't include rank 0
        let result = mesh.get_process_group(&[1, -1, -1]);
        assert!(result.is_err());

        // Mesh with < 3 dims should fail for DP/TP/PP groups
        let mut small_mesh = DeviceMesh::new(&[2, 2], 0).unwrap();
        assert!(small_mesh.get_data_parallel_group().is_err());
    }

    #[test]
    fn test_mesh_rank_and_shape() {
        let mesh = DeviceMesh::for_3d_parallelism(2, 4, 2, 5).unwrap();
        
        assert_eq!(mesh.world_size(), 16);
        assert_eq!(mesh.shape(), &[2, 4, 2]);
        assert_eq!(mesh.my_coord(), &[0, 2, 1]);
        assert_eq!(mesh.my_rank(), 5);
        assert_eq!(mesh.dim_size(0), 2);
        assert_eq!(mesh.dim_size(1), 4);
        assert_eq!(mesh.dim_size(2), 2);
        assert_eq!(mesh.dim_size(100), 1); // Out of bounds defaults to 1
    }

    #[test]
    fn test_3d_parallelism_config() {
        // Small model: should prefer DP
        let small = ParallelismConfig::auto_select(8, 100_000_000);
        assert_eq!(small.total_devices(), 8);
        assert_eq!(small.tensor_parallel, 1);
        
        // Large model: should use TP and possibly PP
        let large = ParallelismConfig::auto_select(16, 2_000_000_000);
        assert_eq!(large.total_devices(), 16);
        assert!(large.tensor_parallel >= 1);
    }

    #[test]
    fn test_parallelism_config_large_model() {
        let config = ParallelismConfig::auto_select(64, 20_000_000_000);
        assert_eq!(config.total_devices(), 64);
        assert!(config.pipeline_parallel >= 1);
        assert!(config.data_parallel >= 1);
    }

    #[test]
    fn test_mesh_creation() {
        let mesh = DeviceMesh::for_3d_parallelism(2, 4, 2, 5).unwrap();
        
        assert_eq!(mesh.world_size(), 16);
        assert_eq!(mesh.shape(), &[2, 4, 2]);
        assert_eq!(mesh.my_coord(), &[0, 2, 1]); // Rank 5 = [0, 2, 1]
    }
}
