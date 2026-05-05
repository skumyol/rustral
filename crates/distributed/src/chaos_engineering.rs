//! Chaos Engineering for Distributed Training
//!
//! Fault injection framework for testing resilience of distributed
//! training systems. Simulates real-world failure scenarios.
//!
//! # Fault Types
//!
//! - **Network Partition**: Simulates split-brain scenarios
//! - **GPU Failure**: Simulates device OOM or hardware failure
//! - **Slow Nodes**: Simulates stragglers and slow workers
//! - **Checkpoint Corruption**: Tests recovery from bad checkpoints
//! - **Memory Pressure**: Simulates near-OOM conditions
//!
//! # Example
//!
//! ```rust,ignore
//! use rustral_distributed::chaos_engineering::{ChaosMonkey, FaultInjection};
//!
//! let mut chaos = ChaosMonkey::new(42); // seed for reproducibility
//!
//! // Inject random network partitions
//! chaos.add_injection(FaultInjection::NetworkPartition {
//!     target_ranks: vec![2, 3],
//!     duration_ms: 5000,
//! });
//!
//! // Inject GPU failures with 10% probability
//! chaos.add_injection(FaultInjection::GpuFailure {
//!     rank: 1,
//!     failure_type: GpuFailureType::OutOfMemory,
//! });
//!
//! // Run training with chaos
//! let result = chaos.run_with_faults(|| {
//!     trainer.train_step()
//! });
//! ```

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

/// Types of failures that can be injected.
#[derive(Debug, Clone, PartialEq)]
pub enum FaultType {
    /// Network partition (ranks can't communicate)
    NetworkPartition { target_ranks: Vec<usize> },
    /// GPU out of memory
    GpuOutOfMemory { rank: usize },
    /// GPU hardware error
    GpuHardwareError { rank: usize },
    /// Slow worker (high latency)
    SlowWorker { rank: usize, delay_ms: u64 },
    /// Process crash (rank goes down)
    ProcessCrash { rank: usize },
    /// Corrupted checkpoint
    CorruptedCheckpoint { rank: usize, corruption_type: CheckpointCorruption },
    /// Memory leak (gradual OOM)
    MemoryLeak { rank: usize, leak_rate_mb: f64 },
    /// Packet loss (unreliable network)
    PacketLoss { loss_rate: f64 },
}

/// Types of checkpoint corruption.
#[derive(Debug, Clone, PartialEq)]
pub enum CheckpointCorruption {
    /// Random bit flips
    BitFlip { num_flips: usize },
    /// Truncate file
    Truncated { keep_bytes: usize },
    /// Wrong checksum
    WrongChecksum,
    /// Missing keys
    MissingKeys(Vec<String>),
}

/// A single fault injection event.
pub struct FaultInjection {
    pub fault: FaultType,
    /// Probability of triggering (0.0-1.0)
    pub trigger_probability: f64,
    /// Delay before triggering
    pub trigger_delay: Duration,
    /// Duration of fault (None = permanent)
    pub duration: Option<Duration>,
    /// Number of times to trigger (-1 = infinite)
    pub max_triggers: i32,
    /// Current trigger count
    trigger_count: i32,
}

impl FaultInjection {
    pub fn new(fault: FaultType, probability: f64) -> Self {
        Self {
            fault,
            trigger_probability: probability.clamp(0.0, 1.0),
            trigger_delay: Duration::ZERO,
            duration: None,
            max_triggers: 1,
            trigger_count: 0,
        }
    }

    pub fn with_delay(mut self, delay: Duration) -> Self {
        self.trigger_delay = delay;
        self
    }

    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = Some(duration);
        self
    }

    pub fn with_max_triggers(mut self, max: i32) -> Self {
        self.max_triggers = max;
        self
    }

    /// Check if this fault should trigger now.
    pub fn should_trigger(&mut self, rng: &mut StdRng, elapsed: Duration) -> bool {
        if elapsed < self.trigger_delay {
            return false;
        }

        if self.max_triggers >= 0 && self.trigger_count >= self.max_triggers {
            return false;
        }

        if rng.gen::<f64>() < self.trigger_probability {
            self.trigger_count += 1;
            true
        } else {
            false
        }
    }
}

/// Result of a fault injection.
#[derive(Debug, Clone)]
pub struct FaultResult {
    pub fault_type: FaultType,
    pub triggered_at: Instant,
    pub recovered: bool,
    pub recovery_time_ms: u64,
    pub affected_ranks: Vec<usize>,
    pub error_message: Option<String>,
}

/// Chaos monkey for testing system resilience.
pub struct ChaosMonkey {
    rng: StdRng,
    injections: Vec<FaultInjection>,
    start_time: Instant,
    results: Vec<FaultResult>,
    active_faults: Vec<ActiveFault>,
    enabled: bool,
}

/// An actively running fault.
struct ActiveFault {
    fault: FaultType,
    started_at: Instant,
    duration: Option<Duration>,
    affected_ranks: HashSet<usize>,
}

impl ChaosMonkey {
    /// Create a new chaos monkey with given seed.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            injections: Vec::new(),
            start_time: Instant::now(),
            results: Vec::new(),
            active_faults: Vec::new(),
            enabled: true,
        }
    }

    /// Enable chaos testing.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable chaos testing.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Check if chaos is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Add a fault injection.
    pub fn add_injection(&mut self, injection: FaultInjection) {
        self.injections.push(injection);
    }

    /// Add common fault scenarios.
    pub fn add_common_faults(&mut self, world_size: usize) {
        // Random single rank failures
        for rank in 0..world_size {
            self.add_injection(
                FaultInjection::new(
                    FaultType::GpuOutOfMemory { rank },
                    0.01, // 1% chance per check
                )
                .with_duration(Duration::from_secs(5)),
            );

            self.add_injection(
                FaultInjection::new(FaultType::SlowWorker { rank, delay_ms: 500 }, 0.05)
                    .with_duration(Duration::from_secs(10)),
            );
        }

        // Network partition (rare but critical)
        if world_size >= 4 {
            let half = world_size / 2;
            self.add_injection(
                FaultInjection::new(FaultType::NetworkPartition { target_ranks: (0..half).collect() }, 0.001)
                    .with_duration(Duration::from_secs(3)),
            );
        }

        // Packet loss
        self.add_injection(
            FaultInjection::new(FaultType::PacketLoss { loss_rate: 0.1 }, 0.02)
                .with_duration(Duration::from_secs(2)),
        );

        // Memory leak
        self.add_injection(
            FaultInjection::new(FaultType::MemoryLeak { rank: world_size - 1, leak_rate_mb: 100.0 }, 0.005)
                .with_max_triggers(1),
        );

        // Process crash (test elastic recovery)
        self.add_injection(
            FaultInjection::new(FaultType::ProcessCrash { rank: 0 }, 0.003).with_max_triggers(1),
        );
    }

    /// Check and trigger any due faults.
    pub fn tick(&mut self) -> Vec<FaultResult> {
        if !self.enabled {
            return Vec::new();
        }

        let elapsed = self.start_time.elapsed();
        let mut new_results = Vec::new();

        // Check for expired faults
        let now = Instant::now();
        let mut recovered = Vec::new();
        for (i, active) in self.active_faults.iter().enumerate() {
            if let Some(duration) = active.duration {
                if now.duration_since(active.started_at) > duration {
                    recovered.push(i);
                    new_results.push(FaultResult {
                        fault_type: active.fault.clone(),
                        triggered_at: active.started_at,
                        recovered: true,
                        recovery_time_ms: duration.as_millis() as u64,
                        affected_ranks: active.affected_ranks.iter().copied().collect(),
                        error_message: None,
                    });
                }
            }
        }

        // Remove recovered faults (in reverse order to maintain indices)
        for i in recovered.iter().rev() {
            self.active_faults.remove(*i);
        }

        // Check for new fault triggers
        // Collect triggered faults first to avoid borrow checker issues
        let mut triggered_faults = Vec::new();
        for injection in &mut self.injections {
            if injection.should_trigger(&mut self.rng, elapsed) {
                // Clone the needed data
                triggered_faults.push((injection.fault.clone(), injection.duration));
            }
        }

        // Now process triggered faults
        for (fault, duration) in triggered_faults {
            let affected_ranks = self.get_affected_ranks(&fault);

            let active = ActiveFault {
                fault: fault.clone(),
                started_at: now,
                duration,
                affected_ranks: affected_ranks.iter().copied().collect(),
            };

            self.active_faults.push(active);

            new_results.push(FaultResult {
                fault_type: fault.clone(),
                triggered_at: now,
                recovered: false,
                recovery_time_ms: 0,
                affected_ranks: affected_ranks.clone(),
                error_message: Some(format!("Injected: {:?}", fault)),
            });
        }

        self.results.extend(new_results.clone());
        new_results
    }

    /// Get ranks affected by a fault.
    fn get_affected_ranks(&self, fault: &FaultType) -> Vec<usize> {
        match fault {
            FaultType::NetworkPartition { target_ranks } => target_ranks.clone(),
            FaultType::GpuOutOfMemory { rank } => vec![*rank],
            FaultType::GpuHardwareError { rank } => vec![*rank],
            FaultType::SlowWorker { rank, .. } => vec![*rank],
            FaultType::ProcessCrash { rank } => vec![*rank],
            FaultType::CorruptedCheckpoint { rank, .. } => vec![*rank],
            FaultType::MemoryLeak { rank, .. } => vec![*rank],
            FaultType::PacketLoss { .. } => vec![], // All ranks affected
        }
    }

    /// Check if a specific rank has an active fault.
    pub fn is_rank_affected(&self, rank: usize) -> Option<&FaultType> {
        for active in &self.active_faults {
            if active.affected_ranks.contains(&rank) {
                return Some(&active.fault);
            }
        }
        None
    }

    /// Check if communication between two ranks is possible.
    pub fn can_communicate(&self, rank_a: usize, rank_b: usize) -> bool {
        for active in &self.active_faults {
            match &active.fault {
                FaultType::NetworkPartition { target_ranks } => {
                    let a_partitioned = target_ranks.contains(&rank_a);
                    let b_partitioned = target_ranks.contains(&rank_b);
                    // If only one is in the partition, they can't communicate
                    if a_partitioned != b_partitioned {
                        return false;
                    }
                }
                FaultType::ProcessCrash { rank } => {
                    if *rank == rank_a || *rank == rank_b {
                        return false;
                    }
                }
                _ => {}
            }
        }
        true
    }

    /// Get communication delay for a rank (for slow worker simulation).
    pub fn get_delay_ms(&self, rank: usize) -> u64 {
        for active in &self.active_faults {
            match &active.fault {
                FaultType::SlowWorker { rank: r, delay_ms } => {
                    if *r == rank {
                        return *delay_ms;
                    }
                }
                _ => {}
            }
        }
        0
    }

    /// Simulate packet loss check.
    pub fn packet_lost(&mut self, _rank: usize) -> bool {
        for active in &self.active_faults {
            match &active.fault {
                FaultType::PacketLoss { loss_rate } => {
                    if self.rng.gen::<f64>() < *loss_rate {
                        return true;
                    }
                }
                _ => {}
            }
        }
        false
    }

    /// Run a function with fault injection.
    pub fn run_with_faults<F, T>(&mut self, f: F) -> Result<T, String>
    where
        F: FnOnce() -> Result<T, String>,
    {
        // Trigger any due faults
        let _ = self.tick();

        // Run the function
        f()
    }

    /// Get all fault results.
    pub fn results(&self) -> &[FaultResult] {
        &self.results
    }

    /// Get active faults.
    pub fn active_faults(&self) -> Vec<&FaultType> {
        self.active_faults.iter().map(|a| &a.fault).collect()
    }

    /// Print fault summary.
    pub fn print_summary(&self) {
        println!("\n{}", "=".repeat(70));
        println!("Chaos Engineering Fault Summary");
        println!("{}", "=".repeat(70));
        println!("Total Faults Injected: {}", self.results.len());
        println!("Active Faults: {}", self.active_faults.len());
        println!("Seed: {:?}", self.rng);

        if !self.results.is_empty() {
            println!("\nInjected Faults:");
            for (i, result) in self.results.iter().enumerate() {
                let status = if result.recovered { "RECOVERED" } else { "ACTIVE" };
                println!(
                    "  {}: {:?} [{}, affected: {:?}]",
                    i + 1,
                    result.fault_type,
                    status,
                    result.affected_ranks
                );
                if let Some(ref msg) = result.error_message {
                    println!("      {}", msg);
                }
            }
        }

        if !self.active_faults.is_empty() {
            println!("\nCurrently Active:");
            for (i, active) in self.active_faults.iter().enumerate() {
                let duration =
                    active.duration.map(|d| format!("{:?}", d)).unwrap_or_else(|| "permanent".to_string());
                println!("  {}: {:?} (duration: {})", i + 1, active.fault, duration);
            }
        }

        println!("{}", "=".repeat(70));
    }

    /// Generate a test report.
    pub fn generate_report(&self) -> TestReport {
        let mut by_type: HashMap<String, usize> = HashMap::new();
        let mut recovered_count = 0;
        let mut total_affected_ranks = 0;

        for result in &self.results {
            let type_name = format!("{:?}", std::mem::discriminant(&result.fault_type));
            *by_type.entry(type_name).or_insert(0) += 1;

            if result.recovered {
                recovered_count += 1;
            }
            total_affected_ranks += result.affected_ranks.len();
        }

        TestReport {
            total_faults: self.results.len(),
            recovered_faults: recovered_count,
            active_faults: self.active_faults.len(),
            faults_by_type: by_type,
            avg_affected_ranks: if self.results.is_empty() {
                0.0
            } else {
                total_affected_ranks as f64 / self.results.len() as f64
            },
            test_duration_ms: self.start_time.elapsed().as_millis() as u64,
        }
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        self.start_time = Instant::now();
        self.results.clear();
        self.active_faults.clear();
        for injection in &mut self.injections {
            injection.trigger_count = 0;
        }
    }
}

/// Test report from chaos engineering.
#[derive(Debug, Clone)]
pub struct TestReport {
    pub total_faults: usize,
    pub recovered_faults: usize,
    pub active_faults: usize,
    pub faults_by_type: HashMap<String, usize>,
    pub avg_affected_ranks: f64,
    pub test_duration_ms: u64,
}

impl TestReport {
    pub fn print(&self) {
        println!("\nChaos Test Report:");
        println!("  Total Faults: {}", self.total_faults);
        println!("  Recovered: {}", self.recovered_faults);
        println!("  Active: {}", self.active_faults);
        println!("  Avg Affected Ranks: {:.2}", self.avg_affected_ranks);
        println!("  Duration: {}ms", self.test_duration_ms);

        if !self.faults_by_type.is_empty() {
            println!("\n  By Type:");
            for (ftype, count) in &self.faults_by_type {
                println!("    {}: {}", ftype, count);
            }
        }
    }
}

/// Pre-configured chaos scenarios for common tests.
pub struct ChaosScenarios;

impl ChaosScenarios {
    /// Scenario: Single slow worker in large cluster.
    pub fn slow_worker(world_size: usize) -> ChaosMonkey {
        let mut chaos = ChaosMonkey::new(42);
        chaos.add_injection(
            FaultInjection::new(
                FaultType::SlowWorker { rank: world_size / 2, delay_ms: 2000 },
                1.0, // Always trigger
            )
            .with_duration(Duration::from_secs(30)),
        );
        chaos
    }

    /// Scenario: Network partition test.
    pub fn network_partition(world_size: usize) -> ChaosMonkey {
        let mut chaos = ChaosMonkey::new(42);
        let half = world_size / 2;
        chaos.add_injection(
            FaultInjection::new(FaultType::NetworkPartition { target_ranks: (0..half).collect() }, 1.0)
                .with_duration(Duration::from_secs(10)),
        );
        chaos
    }

    /// Scenario: Cascading failures.
    pub fn cascading_failure(world_size: usize) -> ChaosMonkey {
        let mut chaos = ChaosMonkey::new(42);

        // Start with slow worker
        chaos.add_injection(
            FaultInjection::new(FaultType::SlowWorker { rank: 0, delay_ms: 5000 }, 1.0)
                .with_delay(Duration::from_secs(5)),
        );

        // Then OOM on another rank
        chaos.add_injection(
            FaultInjection::new(FaultType::GpuOutOfMemory { rank: 1 }, 1.0)
                .with_delay(Duration::from_secs(15)),
        );

        // Finally network partition
        chaos.add_injection(
            FaultInjection::new(FaultType::NetworkPartition { target_ranks: (2..world_size).collect() }, 1.0)
                .with_delay(Duration::from_secs(30)),
        );

        chaos
    }

    /// Scenario: Random faults for stress testing.
    pub fn random_stress(world_size: usize, seed: u64) -> ChaosMonkey {
        let mut chaos = ChaosMonkey::new(seed);
        chaos.add_common_faults(world_size);
        chaos
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_fault_injection() {
        let injection = FaultInjection::new(
            FaultType::GpuOutOfMemory { rank: 0 },
            1.0, // Always trigger
        )
        .with_duration(Duration::from_secs(5));

        assert_eq!(injection.fault, FaultType::GpuOutOfMemory { rank: 0 });
        assert_eq!(injection.trigger_probability, 1.0);
    }

    #[test]
    fn test_fault_injection_builder() {
        let mut injection = FaultInjection::new(FaultType::SlowWorker { rank: 1, delay_ms: 500 }, 1.0)
            .with_delay(Duration::from_secs(1))
            .with_max_triggers(3);

        assert_eq!(injection.trigger_delay, Duration::from_secs(1));
        assert_eq!(injection.max_triggers, 3);

        // Should not trigger before delay
        let mut rng = StdRng::seed_from_u64(42);
        assert!(!injection.should_trigger(&mut rng, Duration::from_millis(500)));

        // Should trigger after delay
        assert!(injection.should_trigger(&mut rng, Duration::from_secs(2)));
        assert_eq!(injection.trigger_count, 1);

        // Should not trigger after max triggers
        injection.trigger_count = 3;
        assert!(!injection.should_trigger(&mut rng, Duration::from_secs(5)));
    }

    #[test]
    fn test_chaos_monkey_creation() {
        let mut chaos = ChaosMonkey::new(42);
        assert!(chaos.is_enabled());

        chaos.add_common_faults(8);
        assert!(!chaos.injections.is_empty());
    }

    #[test]
    fn test_chaos_monkey_enable_disable() {
        let mut chaos = ChaosMonkey::new(42);
        assert!(chaos.is_enabled());

        chaos.disable();
        assert!(!chaos.is_enabled());

        // tick should return empty when disabled
        let results = chaos.tick();
        assert!(results.is_empty());

        chaos.enable();
        assert!(chaos.is_enabled());
    }

    #[test]
    fn test_network_partition() {
        let mut chaos = ChaosMonkey::new(42);
        chaos.add_injection(FaultInjection::new(
            FaultType::NetworkPartition { target_ranks: vec![0, 1, 2] },
            1.0,
        ));

        // Activate the fault via tick()
        chaos.tick();

        // Check if ranks can communicate
        assert!(!chaos.can_communicate(0, 3)); // Different partitions
        assert!(chaos.can_communicate(3, 4)); // Same partition
    }

    #[test]
    fn test_can_communicate_no_faults() {
        let chaos = ChaosMonkey::new(42);
        assert!(chaos.can_communicate(0, 1));
    }

    #[test]
    fn test_is_rank_affected_none() {
        let chaos = ChaosMonkey::new(42);
        assert!(chaos.is_rank_affected(0).is_none());
    }

    #[test]
    fn test_get_delay_ms() {
        let mut chaos = ChaosMonkey::new(42);
        assert_eq!(chaos.get_delay_ms(0), 0);

        chaos.add_injection(
            FaultInjection::new(FaultType::SlowWorker { rank: 0, delay_ms: 250 }, 1.0)
                .with_duration(Duration::from_secs(5)),
        );
        chaos.tick();

        assert_eq!(chaos.get_delay_ms(0), 250);
        assert_eq!(chaos.get_delay_ms(1), 0);
    }

    #[test]
    fn test_packet_lost() {
        let mut chaos = ChaosMonkey::new(42);
        // No active packet loss
        assert!(!chaos.packet_lost(0));

        chaos.add_injection(
            FaultInjection::new(FaultType::PacketLoss { loss_rate: 1.0 }, 1.0)
                .with_duration(Duration::from_secs(5)),
        );
        chaos.tick();

        // With loss_rate=1.0, all packets should be lost
        assert!(chaos.packet_lost(0));
    }

    #[test]
    fn test_run_with_faults() {
        let mut chaos = ChaosMonkey::new(42);
        let result = chaos.run_with_faults(|| Ok(42));
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_fault_tick() {
        let mut chaos = ChaosMonkey::new(42);
        chaos.add_injection(
            FaultInjection::new(FaultType::SlowWorker { rank: 0, delay_ms: 100 }, 1.0)
                .with_duration(Duration::from_secs(1)),
        );

        // Trigger the fault
        let results = chaos.tick();
        assert!(!results.is_empty());
        assert!(chaos.is_rank_affected(0).is_some());

        // results() should return all fault results
        assert_eq!(chaos.results().len(), results.len());

        // active_faults() should return the active fault types
        assert!(!chaos.active_faults().is_empty());
    }

    #[test]
    fn test_fault_recovery() {
        let mut chaos = ChaosMonkey::new(42);
        chaos.add_injection(
            FaultInjection::new(FaultType::GpuOutOfMemory { rank: 0 }, 1.0)
                .with_duration(Duration::from_millis(50)),
        );

        // Trigger fault
        chaos.tick();
        assert!(chaos.is_rank_affected(0).is_some());

        // Wait for fault to expire
        thread::sleep(Duration::from_millis(100));

        // Next tick should record recovery
        let results = chaos.tick();
        assert!(!results.is_empty());
        assert!(results[0].recovered);

        // Active faults should be empty after recovery
        assert!(chaos.active_faults().is_empty());
    }

    #[test]
    fn test_print_summary() {
        let mut chaos = ChaosMonkey::new(42);
        chaos.add_injection(
            FaultInjection::new(FaultType::GpuOutOfMemory { rank: 0 }, 1.0)
                .with_duration(Duration::from_secs(1)),
        );
        chaos.tick();
        chaos.print_summary();
    }

    #[test]
    fn test_reset() {
        let mut chaos = ChaosMonkey::new(42);
        chaos.add_injection(
            FaultInjection::new(FaultType::GpuOutOfMemory { rank: 0 }, 1.0)
                .with_duration(Duration::from_secs(1)),
        );
        chaos.tick();

        assert!(!chaos.results.is_empty());
        assert!(!chaos.active_faults.is_empty());

        chaos.reset();
        assert!(chaos.results.is_empty());
        assert!(chaos.active_faults.is_empty());
    }

    #[test]
    fn test_scenarios() {
        let slow = ChaosScenarios::slow_worker(8);
        assert!(slow.active_faults().is_empty()); // Not triggered yet

        let partition = ChaosScenarios::network_partition(8);
        assert!(!partition.injections.is_empty());

        let stress = ChaosScenarios::random_stress(16, 123);
        assert!(!stress.injections.is_empty());
    }

    #[test]
    fn test_cascading_failure_scenario() {
        let chaos = ChaosScenarios::cascading_failure(8);
        assert!(!chaos.injections.is_empty());
    }

    #[test]
    fn test_report_generation() {
        let mut chaos = ChaosMonkey::new(42);
        chaos.add_common_faults(4);

        // Manually trigger a fault
        chaos.results.push(FaultResult {
            fault_type: FaultType::GpuOutOfMemory { rank: 0 },
            triggered_at: Instant::now(),
            recovered: true,
            recovery_time_ms: 100,
            affected_ranks: vec![0],
            error_message: Some("OOM injected".to_string()),
        });

        let report = chaos.generate_report();
        assert_eq!(report.total_faults, 1);
        assert_eq!(report.recovered_faults, 1);

        report.print();
    }

    #[test]
    fn test_checkpoint_corruption_variants() {
        let c1 = CheckpointCorruption::BitFlip { num_flips: 3 };
        let c2 = CheckpointCorruption::Truncated { keep_bytes: 100 };
        let c3 = CheckpointCorruption::WrongChecksum;
        let c4 = CheckpointCorruption::MissingKeys(vec!["key1".to_string()]);

        let _ = format!("{:?}", c1);
        let _ = format!("{:?}", c2);
        let _ = format!("{:?}", c3);
        let _ = format!("{:?}", c4);
    }

    #[test]
    fn test_fault_type_variants() {
        let faults = vec![
            FaultType::NetworkPartition { target_ranks: vec![0, 1] },
            FaultType::GpuOutOfMemory { rank: 0 },
            FaultType::GpuHardwareError { rank: 1 },
            FaultType::SlowWorker { rank: 2, delay_ms: 100 },
            FaultType::ProcessCrash { rank: 3 },
            FaultType::CorruptedCheckpoint { rank: 0, corruption_type: CheckpointCorruption::WrongChecksum },
            FaultType::MemoryLeak { rank: 4, leak_rate_mb: 10.0 },
            FaultType::PacketLoss { loss_rate: 0.1 },
        ];

        for fault in faults {
            let _ = format!("{:?}", fault);
        }
    }
}
