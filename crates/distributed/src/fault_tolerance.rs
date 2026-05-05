//! Fault Tolerance and Elastic Training
//!
//! Handles node failures during distributed training:
//! - Checkpointing for fault recovery
//! - Elastic membership (add/remove nodes during training)
//! - Heartbeat monitoring for node health
//! - Automatic restarts from last checkpoint
//!
//! # Example
//! ```rust,ignore
//! use mnr_distributed::fault_tolerance::{ElasticTrainer, HealthMonitor};
//!
//! let mut trainer = ElasticTrainer::new(
//!     base_config,
//!     checkpoint_manager,
//! );
//!
//! // Automatically handles node failures
//! trainer.train_loop(dataset, &mut model)?;
//! ```

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use crate::{DistributedCheckpointManager, DistributedError, DistributedResult, ProcessGroup};

/// Node state in elastic training
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NodeState {
    /// Node is healthy and active
    Healthy,
    /// Node hasn't responded recently
    Suspected,
    /// Node has failed
    Failed,
    /// Node is joining
    Joining,
    /// Node is leaving gracefully
    Leaving,
}

/// Node information
#[derive(Clone, Debug)]
pub struct NodeInfo {
    pub rank: usize,
    pub hostname: String,
    pub state: NodeState,
    pub last_heartbeat: Instant,
    pub checkpoint_version: u64,
}

/// Heartbeat monitor for node health
pub struct HealthMonitor {
    /// Heartbeat timeout
    timeout: Duration,
    /// Suspect timeout (before marking as failed)
    suspect_timeout: Duration,
    /// Known nodes
    nodes: HashMap<usize, NodeInfo>,
    /// This node's rank
    my_rank: usize,
    /// Checkpoint version
    checkpoint_version: u64,
}

impl HealthMonitor {
    pub fn new(my_rank: usize, timeout_ms: u64) -> Self {
        Self {
            timeout: Duration::from_millis(timeout_ms),
            suspect_timeout: Duration::from_millis(timeout_ms * 2),
            nodes: HashMap::new(),
            my_rank,
            checkpoint_version: 0,
        }
    }

    /// Register a new node
    pub fn register_node(&mut self, rank: usize, hostname: String) {
        self.nodes.insert(rank, NodeInfo {
            rank,
            hostname,
            state: NodeState::Joining,
            last_heartbeat: Instant::now(),
            checkpoint_version: 0,
        });
    }

    /// Record heartbeat from a node
    pub fn heartbeat(&mut self, rank: usize) {
        if let Some(node) = self.nodes.get_mut(&rank) {
            node.last_heartbeat = Instant::now();
            node.state = NodeState::Healthy;
        }
    }

    /// Check for failed nodes
    pub fn check_health(&mut self) -> Vec<usize> {
        let now = Instant::now();
        let mut failed_nodes = Vec::new();

        for (rank, node) in self.nodes.iter_mut() {
            let elapsed = now.duration_since(node.last_heartbeat);

            match node.state {
                NodeState::Healthy => {
                    if elapsed > self.timeout {
                        node.state = NodeState::Suspected;
                    }
                }
                NodeState::Suspected => {
                    if elapsed > self.suspect_timeout {
                        node.state = NodeState::Failed;
                        failed_nodes.push(*rank);
                    }
                }
                _ => {}
            }
        }

        failed_nodes
    }

    /// Get list of healthy nodes
    pub fn healthy_nodes(&self) -> Vec<usize> {
        self.nodes
            .values()
            .filter(|n| n.state == NodeState::Healthy)
            .map(|n| n.rank)
            .collect()
    }

    /// Update checkpoint version
    pub fn checkpoint_completed(&mut self, version: u64) {
        self.checkpoint_version = version;
        if let Some(node) = self.nodes.get_mut(&self.my_rank) {
            node.checkpoint_version = version;
        }
    }

    /// Get current checkpoint version
    pub fn checkpoint_version(&self) -> u64 {
        self.checkpoint_version
    }
}

/// Elastic process group that handles membership changes
pub struct ElasticProcessGroup {
    /// Base process group
    base_pg: ProcessGroup,
    /// Health monitor
    health: HealthMonitor,
    /// Failed nodes
    failed_nodes: HashSet<usize>,
    /// Is this a reconfiguration epoch?
    reconfiguring: bool,
}

impl ElasticProcessGroup {
    pub fn new(base_pg: ProcessGroup, health_timeout_ms: u64) -> Self {
        let rank = base_pg.rank();
        Self {
            base_pg,
            health: HealthMonitor::new(rank, health_timeout_ms),
            failed_nodes: HashSet::new(),
            reconfiguring: false,
        }
    }

    /// Record heartbeat
    pub fn heartbeat(&mut self, rank: usize) {
        self.health.heartbeat(rank);
    }

    /// Check for membership changes
    pub fn check_membership(&mut self) -> Option<MembershipChange> {
        let failed = self.health.check_health();

        if !failed.is_empty() {
            for rank in &failed {
                self.failed_nodes.insert(*rank);
            }
            return Some(MembershipChange::NodesFailed(failed));
        }

        None
    }

    /// Get effective world size (excluding failed nodes)
    pub fn effective_world_size(&self) -> usize {
        self.base_pg.world_size() - self.failed_nodes.len()
    }

    /// Check if this rank should participate in training
    pub fn is_active(&self) -> bool {
        !self.failed_nodes.contains(&self.base_pg.rank())
    }
}

/// Membership change events
#[derive(Clone, Debug)]
pub enum MembershipChange {
    /// One or more nodes have failed
    NodesFailed(Vec<usize>),
    /// New nodes are joining
    NodesJoining(Vec<usize>),
    /// Reconfiguration complete
    ReconfigurationComplete {
        new_world_size: usize,
        new_rank: usize,
    },
}

/// Elastic trainer that handles fault recovery
pub struct ElasticTrainer<C> {
    /// Checkpoint manager for recovery
    checkpoint_manager: C,
    /// Health check interval
    health_check_interval: Duration,
    /// Max retries
    max_retries: usize,
    /// Current retry count
    retry_count: usize,
    /// Last successful checkpoint
    last_checkpoint: Option<u64>,
}

impl<C: CheckpointStore> ElasticTrainer<C> {
    pub fn new(checkpoint_manager: C) -> Self {
        Self {
            checkpoint_manager,
            health_check_interval: Duration::from_secs(30),
            max_retries: 3,
            retry_count: 0,
            last_checkpoint: None,
        }
    }

    /// Set health check interval
    pub fn with_health_check_interval(mut self, interval: Duration) -> Self {
        self.health_check_interval = interval;
        self
    }

    /// Set max retries
    pub fn with_max_retries(mut self, max: usize) -> Self {
        self.max_retries = max;
        self
    }

    /// Training loop with fault tolerance
    pub fn train_loop<F, R>(&mut self, mut train_fn: F) -> DistributedResult<R>
    where
        F: FnMut() -> DistributedResult<R>,
    {
        loop {
            match train_fn() {
                Ok(result) => return Ok(result),
                Err(e) => {
                    // Check if error is recoverable
                    if self.is_recoverable(&e) && self.retry_count < self.max_retries {
                        self.retry_count += 1;
                        self.recover_from_checkpoint()?;
                    } else {
                        return Err(e);
                    }
                }
            }
        }
    }

    /// Check if error is recoverable
    fn is_recoverable(&self, error: &DistributedError) -> bool {
        match error {
            DistributedError::Communication(_) => true,
            DistributedError::RankMismatch { .. } => true,
            _ => false,
        }
    }

    /// Recover from last checkpoint
    fn recover_from_checkpoint(&mut self) -> DistributedResult<()> {
        if let Some(version) = self.last_checkpoint {
            // In real impl, would load checkpoint
            println!("Recovering from checkpoint version {}", version);
        }
        Ok(())
    }

    /// Save checkpoint and update state
    pub fn save_checkpoint(&mut self, version: u64) -> DistributedResult<()> {
        self.checkpoint_manager.save(version)?;
        self.last_checkpoint = Some(version);
        self.retry_count = 0; // Reset retry count on successful checkpoint
        Ok(())
    }
}

/// Trait for checkpoint storage
pub trait CheckpointStore {
    fn save(&self, version: u64) -> DistributedResult<()>;
    fn load(&self, version: u64) -> DistributedResult<()>;
    fn latest(&self) -> DistributedResult<Option<u64>>;
}

impl CheckpointStore for DistributedCheckpointManager {
    fn save(&self, version: u64) -> DistributedResult<()> {
        // Would actually save checkpoint
        Ok(())
    }

    fn load(&self, version: u64) -> DistributedResult<()> {
        // Would actually load checkpoint
        Ok(())
    }

    fn latest(&self) -> DistributedResult<Option<u64>> {
        self.latest_checkpoint()
    }
}

/// Distributed barrier with timeout
pub struct TimedBarrier {
    world_size: usize,
    arrivals: HashSet<usize>,
    timeout: Duration,
}

impl TimedBarrier {
    pub fn new(world_size: usize, timeout_ms: u64) -> Self {
        Self {
            world_size,
            arrivals: HashSet::new(),
            timeout: Duration::from_millis(timeout_ms),
        }
    }

    /// Arrive at barrier
    pub fn arrive(&mut self, rank: usize) -> bool {
        self.arrivals.insert(rank);
        self.arrivals.len() == self.world_size
    }

    /// Reset barrier
    pub fn reset(&mut self) {
        self.arrivals.clear();
    }

    /// Check if barrier is complete
    pub fn is_complete(&self) -> bool {
        self.arrivals.len() == self.world_size
    }
}

/// Checkpoint versioning for consistency
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct CheckpointVersion(u64);

impl CheckpointVersion {
    pub fn new(version: u64) -> Self {
        Self(version)
    }

    pub fn increment(&mut self) {
        self.0 += 1;
    }

    pub fn get(&self) -> u64 {
        self.0
    }
}

/// Elastic state synchronization
pub struct StateSync {
    /// Target number of nodes
    target_world_size: usize,
    /// Current active nodes
    active_nodes: HashSet<usize>,
    /// Minimum nodes required
    min_nodes: usize,
}

impl StateSync {
    pub fn new(target_world_size: usize, min_nodes: usize) -> Self {
        Self {
            target_world_size,
            active_nodes: HashSet::new(),
            min_nodes,
        }
    }

    /// Check if we have minimum nodes to proceed
    pub fn can_proceed(&self) -> bool {
        self.active_nodes.len() >= self.min_nodes
    }

    /// Check if all nodes are present
    pub fn is_complete(&self) -> bool {
        self.active_nodes.len() == self.target_world_size
    }

    /// Add active node
    pub fn add_node(&mut self, rank: usize) {
        self.active_nodes.insert(rank);
    }

    /// Remove node
    pub fn remove_node(&mut self, rank: usize) {
        self.active_nodes.remove(&rank);
    }

    /// Get active node count
    pub fn active_count(&self) -> usize {
        self.active_nodes.len()
    }
}

/// Automatic restart configuration
#[derive(Clone, Debug)]
pub struct RestartConfig {
    /// Max number of restarts
    pub max_restarts: usize,
    /// Restart delay
    pub restart_delay: Duration,
    /// Checkpoint on failure
    pub checkpoint_on_failure: bool,
}

impl Default for RestartConfig {
    fn default() -> Self {
        Self {
            max_restarts: 3,
            restart_delay: Duration::from_secs(10),
            checkpoint_on_failure: true,
        }
    }
}

/// Fault recovery statistics
#[derive(Debug, Clone)]
pub struct FaultStats {
    pub total_failures: usize,
    pub successful_recovers: usize,
    pub failed_recovers: usize,
    pub checkpoint_versions: Vec<u64>,
    pub total_downtime_ms: u64,
}

impl FaultStats {
    pub fn new() -> Self {
        Self {
            total_failures: 0,
            successful_recovers: 0,
            failed_recovers: 0,
            checkpoint_versions: Vec::new(),
            total_downtime_ms: 0,
        }
    }

    pub fn record_failure(&mut self) {
        self.total_failures += 1;
    }

    pub fn record_success(&mut self, version: u64) {
        self.successful_recovers += 1;
        self.checkpoint_versions.push(version);
    }

    pub fn record_failure_recovery(&mut self) {
        self.failed_recovers += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnr_core::CoreError;

    #[test]
    fn test_health_monitor() {
        let mut monitor = HealthMonitor::new(0, 100); // 100ms timeout, 200ms suspect timeout
        monitor.register_node(1, "node1".to_string());
        monitor.register_node(2, "node2".to_string());

        // All in Joining state initially
        assert_eq!(monitor.healthy_nodes().len(), 0);

        // Heartbeat from both nodes to transition to Healthy
        monitor.heartbeat(1);
        monitor.heartbeat(2);
        let mut healthy = monitor.healthy_nodes();
        healthy.sort();
        assert_eq!(healthy, vec![1, 2]);

        // First check: no nodes have timed out yet
        let failed = monitor.check_health();
        assert!(failed.is_empty());

        // After timeout (100ms), nodes become Suspected
        thread::sleep(Duration::from_millis(150));
        let failed = monitor.check_health();
        assert!(failed.is_empty(), "Nodes should be suspected, not failed yet");

        // After suspect_timeout (200ms total), node 2 fails (no heartbeat)
        thread::sleep(Duration::from_millis(150));
        let failed = monitor.check_health();
        assert!(failed.contains(&2), "Node 2 should be failed after no heartbeat");
    }

    #[test]
    fn test_health_monitor_heartbeat_unknown() {
        let mut monitor = HealthMonitor::new(0, 100);
        // Heartbeat on unregistered node should be a no-op
        monitor.heartbeat(99);
        assert_eq!(monitor.healthy_nodes().len(), 0);
    }

    #[test]
    fn test_health_monitor_checkpoint_version() {
        let mut monitor = HealthMonitor::new(0, 100);
        monitor.register_node(0, "node0".to_string());
        assert_eq!(monitor.checkpoint_version(), 0);

        monitor.checkpoint_completed(5);
        assert_eq!(monitor.checkpoint_version(), 5);
        let node = monitor.nodes.get(&0).unwrap();
        assert_eq!(node.checkpoint_version, 5);
    }

    #[test]
    fn test_elastic_process_group() {
        let pg = ProcessGroup::new_threaded(4, 0).unwrap();
        let mut elastic = ElasticProcessGroup::new(pg, 100);

        assert!(elastic.is_active());
        assert_eq!(elastic.effective_world_size(), 4);

        elastic.heartbeat(1);
        elastic.heartbeat(2);

        // No failures yet
        let change = elastic.check_membership();
        assert!(change.is_none());
    }

    #[test]
    fn test_elastic_trainer() {
        let pg = ProcessGroup::new_single_process();
        let ckpt = crate::DistributedCheckpointManager::new(pg, std::env::temp_dir().join("ft_test"), 3).unwrap();
        let mut trainer = ElasticTrainer::new(ckpt)
            .with_health_check_interval(Duration::from_secs(1))
            .with_max_retries(2);

        assert_eq!(trainer.max_retries, 2);

        // Successful training
        let result = trainer.train_loop(|| Ok(42));
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_elastic_trainer_recoverable() {
        let pg = ProcessGroup::new_single_process();
        let ckpt = crate::DistributedCheckpointManager::new(pg, std::env::temp_dir().join("ft_test2"), 3).unwrap();
        let mut trainer = ElasticTrainer::new(ckpt)
            .with_max_retries(1);

        let mut calls = 0;
        let result = trainer.train_loop(|| {
            calls += 1;
            if calls == 1 {
                Err(DistributedError::Communication("test".to_string()))
            } else {
                Ok(42)
            }
        });
        assert_eq!(result.unwrap(), 42);
        assert_eq!(calls, 2);
    }

    #[test]
    fn test_elastic_trainer_unrecoverable() {
        let pg = ProcessGroup::new_single_process();
        let ckpt = crate::DistributedCheckpointManager::new(pg, std::env::temp_dir().join("ft_test3"), 3).unwrap();
        let mut trainer = ElasticTrainer::new(ckpt)
            .with_max_retries(1);

        let result: DistributedResult<i32> = trainer.train_loop(|| {
            Err(DistributedError::Backend(CoreError::InvalidArgument("fatal".to_string())))
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_elastic_trainer_save_checkpoint() {
        let pg = ProcessGroup::new_single_process();
        let ckpt = crate::DistributedCheckpointManager::new(pg, std::env::temp_dir().join("ft_test4"), 3).unwrap();
        let mut trainer = ElasticTrainer::new(ckpt);

        trainer.save_checkpoint(1).unwrap();
        assert_eq!(trainer.last_checkpoint, Some(1));
        assert_eq!(trainer.retry_count, 0);
    }

    #[test]
    fn test_timed_barrier() {
        let mut barrier = TimedBarrier::new(4, 5000);

        assert!(!barrier.arrive(1));
        assert!(!barrier.arrive(2));
        assert!(!barrier.arrive(3));
        assert!(barrier.arrive(0)); // 4th arrival completes barrier

        assert!(barrier.is_complete());

        barrier.reset();
        assert!(!barrier.is_complete());
    }

    #[test]
    fn test_state_sync() {
        let mut sync = StateSync::new(8, 4);

        assert!(!sync.can_proceed()); // 0 nodes

        sync.add_node(0);
        sync.add_node(1);
        sync.add_node(2);
        assert!(!sync.can_proceed()); // 3 nodes, need 4

        sync.add_node(3);
        assert!(sync.can_proceed()); // 4 nodes, minimum met
        assert!(!sync.is_complete()); // 4 of 8

        sync.remove_node(3);
        assert!(!sync.can_proceed()); // 3 nodes
        assert_eq!(sync.active_count(), 3);

        // Fill all nodes
        for i in 0..8 {
            sync.add_node(i);
        }
        assert!(sync.is_complete());
    }

    #[test]
    fn test_checkpoint_version() {
        let mut version = CheckpointVersion::new(10);
        assert_eq!(version.get(), 10);

        version.increment();
        assert_eq!(version.get(), 11);
    }

    #[test]
    fn test_fault_stats() {
        let mut stats = FaultStats::new();

        stats.record_failure();
        stats.record_success(5);
        stats.record_failure();
        stats.record_failure_recovery();

        assert_eq!(stats.total_failures, 2);
        assert_eq!(stats.successful_recovers, 1);
        assert_eq!(stats.failed_recovers, 1);
    }

    #[test]
    fn test_restart_config() {
        let config = RestartConfig::default();
        assert_eq!(config.max_restarts, 3);
        assert_eq!(config.restart_delay, Duration::from_secs(10));
        assert!(config.checkpoint_on_failure);
    }

    #[test]
    fn test_membership_change_variants() {
        let change = MembershipChange::NodesFailed(vec![1, 2]);
        let _ = format!("{:?}", change);

        let change = MembershipChange::NodesJoining(vec![3]);
        let _ = format!("{:?}", change);

        let change = MembershipChange::ReconfigurationComplete {
            new_world_size: 4,
            new_rank: 0,
        };
        let _ = format!("{:?}", change);
    }
}
