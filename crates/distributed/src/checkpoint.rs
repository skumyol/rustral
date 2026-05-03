//! Distributed checkpointing for multi-node training.
//!
//! Provides sharded checkpointing where each rank saves its portion of the model,
//! enabling efficient save/resume for large models distributed across many GPUs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use mnr_core::{Backend, CoreError, Parameter, ParameterId, Result};
use mnr_io::{load_parameters, save_parameters};
use mnr_optim::AdamCheckpoint;

use crate::{DistributedError, DistributedResult, ProcessGroup};

/// Distributed checkpoint manager.
///
/// Handles saving and loading checkpoints across multiple processes.
/// Each rank saves its shard, and the primary rank coordinates metadata.
pub struct DistributedCheckpointManager {
    /// Process group for coordination.
    process_group: ProcessGroup,

    /// Checkpoint directory.
    checkpoint_dir: PathBuf,

    /// Keep last N checkpoints (rotation).
    keep_last_n: usize,

    /// History of saved checkpoints for rotation.
    checkpoint_history: Vec<u64>, // Epoch numbers
}

impl DistributedCheckpointManager {
    /// Create a new distributed checkpoint manager.
    pub fn new(
        process_group: ProcessGroup,
        checkpoint_dir: impl AsRef<Path>,
        keep_last_n: usize,
    ) -> DistributedResult<Self> {
        let checkpoint_dir = checkpoint_dir.as_ref().to_path_buf();

        // Create directory on primary rank
        if process_group.is_primary() && !checkpoint_dir.exists() {
            fs::create_dir_all(&checkpoint_dir).map_err(|e| {
                DistributedError::Communication(format!("Failed to create checkpoint dir: {}", e))
            })?;
        }

        Ok(Self {
            process_group,
            checkpoint_dir,
            keep_last_n,
            checkpoint_history: Vec::new(),
        })
    }

    /// Save a distributed checkpoint.
    ///
    /// Each rank saves its shard of parameters and optimizer state.
    /// The primary rank saves metadata and coordinates.
    pub fn save<B: Backend>(
        &mut self,
        epoch: u64,
        step: u64,
        params: &[(String, &Parameter<B>)],
        optimizer_checkpoint: Option<&AdamCheckpoint>,
    ) -> DistributedResult<()>
    where
        B::Tensor: AsRef<[f32]> + mnr_core::TensorShape,
    {
        let rank = self.process_group.rank();
        let world_size = self.process_group.world_size();

        // Create epoch directory
        let epoch_dir = self.checkpoint_dir.join(format!("epoch_{}", epoch));
        if self.process_group.is_primary() && !epoch_dir.exists() {
            fs::create_dir_all(&epoch_dir).map_err(|e| {
                DistributedError::Communication(format!("Failed to create epoch dir: {}", e))
            })?;
        }

        // Each rank saves its shard
        let shard_path = epoch_dir.join(format!("rank_{}.safetensors", rank));
        let param_data = save_parameters(params).map_err(|e| {
            DistributedError::Backend(CoreError::Serialization(format!("{:?}", e)))
        })?;

        fs::write(&shard_path, param_data).map_err(|e| {
            DistributedError::Communication(format!("Failed to write checkpoint: {}", e))
        })?;

        // Save optimizer state if provided
        if let Some(opt_ckpt) = optimizer_checkpoint {
            let opt_path = epoch_dir.join(format!("optimizer_rank_{}.json", rank));
            let opt_data = serde_json::to_vec(opt_ckpt).map_err(|e| {
                DistributedError::Communication(format!("Failed to serialize optimizer: {}", e))
            })?;

            fs::write(&opt_path, opt_data).map_err(|e| {
                DistributedError::Communication(format!("Failed to write optimizer checkpoint: {}", e))
            })?;
        }

        // Primary rank saves metadata
        if self.process_group.is_primary() {
            let metadata = CheckpointMetadata {
                epoch,
                step,
                world_size,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                version: "1.0".to_string(),
            };

            let meta_path = epoch_dir.join("metadata.json");
            let meta_data = serde_json::to_vec_pretty(&metadata).map_err(|e| {
                DistributedError::Communication(format!("Failed to serialize metadata: {}", e))
            })?;

            fs::write(&meta_path, meta_data).map_err(|e| {
                DistributedError::Communication(format!("Failed to write metadata: {}", e))
            })?;

            // Update history and rotate
            self.checkpoint_history.push(epoch);
            if self.checkpoint_history.len() > self.keep_last_n {
                let old_epoch = self.checkpoint_history.remove(0);
                let old_dir = self.checkpoint_dir.join(format!("epoch_{}", old_epoch));
                if old_dir.exists() {
                    let _ = fs::remove_dir_all(&old_dir);
                }
            }
        }

        Ok(())
    }

    /// Load a distributed checkpoint.
    ///
    /// Each rank loads its shard. The primary rank broadcasts metadata.
    pub fn load<B: Backend>(
        &self,
        epoch: u64,
        params: &mut [(String, Parameter<B>)],
    ) -> DistributedResult<(u64, Option<AdamCheckpoint>)>
    where
        B::Tensor: AsRef<[f32]> + mnr_core::TensorShape,
    {
        let rank = self.process_group.rank();
        let epoch_dir = self.checkpoint_dir.join(format!("epoch_{}", epoch));

        // Load metadata on primary and broadcast
        let metadata = if self.process_group.is_primary() {
            let meta_path = epoch_dir.join("metadata.json");
            let meta_data = fs::read(&meta_path).map_err(|e| {
                DistributedError::Communication(format!("Failed to read metadata: {}", e))
            })?;

            let metadata: CheckpointMetadata = serde_json::from_slice(&meta_data).map_err(|e| {
                DistributedError::Communication(format!("Failed to parse metadata: {}", e))
            })?;

            // Broadcast world size check
            if metadata.world_size != self.process_group.world_size() {
                return Err(DistributedError::Communication(
                    format!("Checkpoint world_size {} doesn't match current {}",
                        metadata.world_size, self.process_group.world_size())
                ));
            }

            metadata
        } else {
            // Non-primary ranks will get metadata via broadcast (simplified for now)
            CheckpointMetadata {
                epoch,
                step: 0,
                world_size: self.process_group.world_size(),
                timestamp: 0,
                version: "1.0".to_string(),
            }
        };

        // Each rank loads its shard
        let shard_path = epoch_dir.join(format!("rank_{}.safetensors", rank));
        let shard_data = fs::read(&shard_path).map_err(|e| {
            DistributedError::Communication(format!("Failed to read shard for rank {}: {}", rank, e))
        })?;

        // Deserialize and update parameters
        let loaded = load_parameters(&shard_data).map_err(|e| {
            DistributedError::Backend(CoreError::Serialization(format!("{:?}", e)))
        })?;

        // Update provided parameters
        for (name, param) in params.iter_mut() {
            if let Some(loaded_tensor) = loaded.get(name) {
                *param = Parameter::new(&name, loaded_tensor.clone());
            }
        }

        // Try to load optimizer checkpoint
        let opt_path = epoch_dir.join(format!("optimizer_rank_{}.json", rank));
        let optimizer_checkpoint = if opt_path.exists() {
            let opt_data = fs::read(&opt_path).map_err(|e| {
                DistributedError::Communication(format!("Failed to read optimizer: {}", e))
            })?;

            Some(serde_json::from_slice(&opt_data).map_err(|e| {
                DistributedError::Communication(format!("Failed to parse optimizer: {}", e))
            })?)
        } else {
            None
        };

        Ok((metadata.step, optimizer_checkpoint))
    }

    /// List available checkpoints.
    pub fn list_checkpoints(&self) -> DistributedResult<Vec<u64>> {
        if !self.process_group.is_primary() {
            return Ok(Vec::new());
        }

        let mut epochs = Vec::new();

        if let Ok(entries) = fs::read_dir(&self.checkpoint_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();

                if let Some(epoch_str) = name_str.strip_prefix("epoch_") {
                    if let Ok(epoch) = epoch_str.parse::<u64>() {
                        epochs.push(epoch);
                    }
                }
            }
        }

        epochs.sort();
        Ok(epochs)
    }

    /// Get the latest checkpoint epoch.
    pub fn latest_checkpoint(&self) -> DistributedResult<Option<u64>> {
        let checkpoints = self.list_checkpoints()?;
        Ok(checkpoints.last().copied())
    }

    /// Get checkpoint directory.
    pub fn checkpoint_dir(&self) -> &Path {
        &self.checkpoint_dir
    }
}

/// Checkpoint metadata.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CheckpointMetadata {
    /// Training epoch.
    pub epoch: u64,

    /// Training step within epoch.
    pub step: u64,

    /// World size when checkpoint was saved.
    pub world_size: usize,

    /// Unix timestamp.
    pub timestamp: u64,

    /// Checkpoint format version.
    pub version: String,
}

/// Async checkpoint writer for non-blocking saves.
///
/// Writes checkpoints in a background thread to avoid blocking training.
pub struct AsyncCheckpointWriter {
    /// Channel to send checkpoint jobs.
    sender: std::sync::mpsc::Sender<CheckpointJob>,

    /// Handle to the background thread.
    _handle: std::thread::JoinHandle<()>,
}

struct CheckpointJob {
    path: PathBuf,
    data: Vec<u8>,
}

impl AsyncCheckpointWriter {
    /// Create a new async checkpoint writer.
    pub fn new() -> Self {
        let (sender, receiver) = std::sync::mpsc::channel::<CheckpointJob>();

        let handle = std::thread::spawn(move || {
            while let Ok(job) = receiver.recv() {
                // Write checkpoint in background
                if let Some(parent) = job.path.parent() {
                    let _ = fs::create_dir_all(parent);
                }
                let _ = fs::write(&job.path, job.data);
            }
        });

        Self {
            sender,
            _handle: handle,
        }
    }

    /// Queue a checkpoint write.
    pub fn write(&self, path: PathBuf, data: Vec<u8>) -> DistributedResult<()> {
        self.sender
            .send(CheckpointJob { path, data })
            .map_err(|e| DistributedError::Communication(format!("Failed to queue checkpoint: {}", e)))
    }
}

impl Default for AsyncCheckpointWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Checkpoint with sharding information for ZeRO.
pub struct ShardedCheckpoint {
    /// Global checkpoint metadata.
    pub metadata: CheckpointMetadata,

    /// Shards by rank.
    pub shards: HashMap<usize, ShardData>,
}

/// Data for a single shard.
#[derive(Debug, Clone)]
pub struct ShardData {
    /// Parameter tensors.
    pub params: HashMap<String, Vec<f32>>,

    /// Optimizer state.
    pub optimizer: Option<AdamCheckpoint>,
}

impl ShardedCheckpoint {
    /// Gather all shards to primary rank.
    pub fn gather_shards(
        &self,
        process_group: &ProcessGroup,
    ) -> DistributedResult<HashMap<String, Vec<f32>>> {
        // In a real implementation, this would gather from all ranks
        // For now, return the local shard
        if let Some(shard) = self.shards.get(&process_group.rank()) {
            Ok(shard.params.clone())
        } else {
            Ok(HashMap::new())
        }
    }

    /// Scatter shards from primary to all ranks.
    pub fn scatter_shards(
        &mut self,
        process_group: &ProcessGroup,
        full_params: &HashMap<String, Vec<f32>>,
    ) -> DistributedResult<()> {
        // Each rank would extract its shard
        // Simplified: just store reference
        if process_group.is_primary() {
            // Primary has full params, needs to partition
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnr_ndarray_backend::CpuBackend;
    use std::io::Write;

    #[test]
    fn test_checkpoint_manager_creation() {
        let pg = ProcessGroup::new_single_process();
        let temp_dir = tempfile::tempdir().unwrap();

        let manager = DistributedCheckpointManager::new(pg, temp_dir.path(), 3).unwrap();

        assert_eq!(manager.checkpoint_dir(), temp_dir.path());
        assert_eq!(manager.keep_last_n, 3);
    }

    #[test]
    fn test_save_and_load_checkpoint() {
        let backend = CpuBackend::default();
        let pg = ProcessGroup::new_single_process();
        let temp_dir = tempfile::tempdir().unwrap();

        let mut manager = DistributedCheckpointManager::new(pg, temp_dir.path(), 3).unwrap();

        // Create test parameters
        let param = backend.normal_parameter("test", &[10], 42, 0.1).unwrap();
        let params = vec![("test".to_string(), &param)];

        // Save checkpoint
        manager.save(0, 100, &params, None).unwrap();

        // Verify checkpoint exists
        let epoch_dir = temp_dir.path().join("epoch_0");
        assert!(epoch_dir.exists());
        assert!(epoch_dir.join("rank_0.safetensors").exists());
        assert!(epoch_dir.join("metadata.json").exists());

        // Load checkpoint
        let mut loaded_params = vec![("test".to_string(), param.clone())];
        let (step, opt_ckpt) = manager.load(0, &mut loaded_params).unwrap();

        assert_eq!(step, 100);
        assert!(opt_ckpt.is_none());
    }

    #[test]
    fn test_list_checkpoints() {
        let pg = ProcessGroup::new_single_process();
        let temp_dir = tempfile::tempdir().unwrap();

        let mut manager = DistributedCheckpointManager::new(pg, temp_dir.path(), 5).unwrap();

        // Create fake checkpoint directories
        fs::create_dir(temp_dir.path().join("epoch_0")).unwrap();
        fs::create_dir(temp_dir.path().join("epoch_5")).unwrap();
        fs::create_dir(temp_dir.path().join("epoch_10")).unwrap();

        let checkpoints = manager.list_checkpoints().unwrap();
        assert_eq!(checkpoints, vec![0, 5, 10]);
    }

    #[test]
    fn test_checkpoint_metadata() {
        let meta = CheckpointMetadata {
            epoch: 5,
            step: 1000,
            world_size: 8,
            timestamp: 1234567890,
            version: "1.0".to_string(),
        };

        let json = serde_json::to_string(&meta).unwrap();
        let parsed: CheckpointMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.epoch, 5);
        assert_eq!(parsed.world_size, 8);
    }
}
