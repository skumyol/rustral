//! Memory Profiler for Neural Engine
//!
//! Tracks memory allocations, usage patterns, and provides tools for
//! detecting memory leaks and optimizing memory usage.
//!
//! # Features
//!
//! - Per-tensor allocation tracking
//! - Memory timeline visualization
//! - Peak memory prediction
//! - OOM analysis and prevention
//!
//! # Example
//!
//! ```rust,ignore
//! use rustral_core::memory_profiler::{MemoryProfiler, AllocationTracker};
//!
//! let mut profiler = MemoryProfiler::new();
//!
//! {
//!     let _guard = profiler.track_allocation("tensor_a", 1024 * 1024 * 4);
//!     // ... use tensor ...
//! } // Auto-reported on drop
//!
//! profiler.print_report();
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// A single memory allocation event.
#[derive(Clone, Debug)]
pub struct AllocationEvent {
    pub timestamp: Instant,
    pub size: usize,
    pub tag: String,
    pub is_allocation: bool, // true = alloc, false = free
}

/// Memory snapshot at a point in time.
#[derive(Clone, Debug)]
pub struct MemorySnapshot {
    pub timestamp: Instant,
    pub total_allocated: usize,
    pub peak_allocated: usize,
    pub num_allocations: usize,
    pub by_tag: HashMap<String, usize>,
}

/// Per-tensor allocation tracking.
pub struct AllocationTracker {
    size: usize,
    tag: String,
    profiler: Option<Arc<Mutex<MemoryProfiler>>>,
    freed: bool,
}

impl AllocationTracker {
    pub fn new(profiler: Arc<Mutex<MemoryProfiler>>, size: usize, tag: &str) -> Self {
        // Record allocation
        if let Ok(mut p) = profiler.lock() {
            p.record_allocation_internal(size, tag);
        }

        Self { size, tag: tag.to_string(), profiler: Some(profiler), freed: false }
    }

    /// Get allocation size.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get allocation tag.
    pub fn tag(&self) -> &str {
        &self.tag
    }

    /// Manually free this allocation.
    pub fn free(&mut self) {
        if !self.freed {
            if let Some(ref profiler) = self.profiler {
                if let Ok(mut p) = profiler.lock() {
                    p.record_deallocation_internal(self.size, &self.tag);
                }
            }
            self.freed = true;
            self.profiler = None;
        }
    }
}

impl Drop for AllocationTracker {
    fn drop(&mut self) {
        self.free();
    }
}

/// Thread-safe memory profiler.
pub struct MemoryProfiler {
    events: Vec<AllocationEvent>,
    snapshots: Vec<MemorySnapshot>,
    current_allocated: usize,
    peak_allocated: usize,
    active_allocations: HashMap<String, Vec<usize>>, // tag -> sizes
    total_allocations: usize,
    total_deallocations: usize,
    enabled: bool,
    start_time: Instant,
}

impl MemoryProfiler {
    /// Create a new memory profiler.
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            snapshots: Vec::new(),
            current_allocated: 0,
            peak_allocated: 0,
            active_allocations: HashMap::new(),
            total_allocations: 0,
            total_deallocations: 0,
            enabled: true,
            start_time: Instant::now(),
        }
    }

    /// Enable profiling.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable profiling.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Check if enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Create an allocation tracker (RAII guard).
    pub fn track_allocation(self_arc: &Arc<Mutex<Self>>, tag: &str, size: usize) -> AllocationTracker {
        AllocationTracker::new(self_arc.clone(), size, tag)
    }

    /// Record allocation internally.
    fn record_allocation_internal(&mut self, size: usize, tag: &str) {
        if !self.enabled {
            return;
        }

        self.events.push(AllocationEvent {
            timestamp: Instant::now(),
            size,
            tag: tag.to_string(),
            is_allocation: true,
        });

        self.current_allocated += size;
        if self.current_allocated > self.peak_allocated {
            self.peak_allocated = self.current_allocated;
        }

        self.active_allocations.entry(tag.to_string()).or_default().push(size);

        self.total_allocations += 1;
    }

    /// Record deallocation internally.
    fn record_deallocation_internal(&mut self, size: usize, tag: &str) {
        if !self.enabled {
            return;
        }

        self.events.push(AllocationEvent {
            timestamp: Instant::now(),
            size,
            tag: tag.to_string(),
            is_allocation: false,
        });

        self.current_allocated = self.current_allocated.saturating_sub(size);

        if let Some(sizes) = self.active_allocations.get_mut(tag) {
            if let Some(pos) = sizes.iter().position(|&s| s == size) {
                sizes.remove(pos);
            }
        }

        self.total_deallocations += 1;
    }

    /// Take a memory snapshot.
    pub fn snapshot(&mut self) -> MemorySnapshot {
        let mut by_tag = HashMap::new();
        for (tag, sizes) in &self.active_allocations {
            by_tag.insert(tag.clone(), sizes.iter().sum());
        }

        let snapshot = MemorySnapshot {
            timestamp: Instant::now(),
            total_allocated: self.current_allocated,
            peak_allocated: self.peak_allocated,
            num_allocations: self.total_allocations - self.total_deallocations,
            by_tag,
        };

        self.snapshots.push(snapshot.clone());
        snapshot
    }

    /// Get current memory usage.
    pub fn current_usage(&self) -> usize {
        self.current_allocated
    }

    /// Get peak memory usage.
    pub fn peak_usage(&self) -> usize {
        self.peak_allocated
    }

    /// Check if allocation would exceed limit.
    pub fn would_exceed(&self, additional_bytes: usize, limit: usize) -> bool {
        self.current_allocated + additional_bytes > limit
    }

    /// Predict OOM risk based on current trajectory.
    pub fn predict_oom_risk(&self, available_memory: usize) -> OomRisk {
        let usage_ratio = self.current_allocated as f64 / available_memory as f64;

        match usage_ratio {
            r if r < 0.5 => OomRisk::Low,
            r if r < 0.75 => OomRisk::Medium,
            r if r < 0.9 => OomRisk::High,
            _ => OomRisk::Critical,
        }
    }

    /// Find potential memory leaks (allocations without matching frees).
    pub fn find_leaks(&self) -> Vec<(String, usize)> {
        self.active_allocations
            .iter()
            .filter(|(_, sizes)| !sizes.is_empty())
            .map(|(tag, sizes)| (tag.clone(), sizes.iter().sum()))
            .collect()
    }

    /// Get memory usage by tag.
    pub fn usage_by_tag(&self) -> HashMap<String, usize> {
        self.active_allocations.iter().map(|(tag, sizes)| (tag.clone(), sizes.iter().sum())).collect()
    }

    /// Print memory report.
    pub fn print_report(&self) {
        println!("\n{}", "=".repeat(70));
        println!("Memory Profiler Report");
        println!("{}", "=".repeat(70));
        println!("Current Usage: {:.2} MB", self.current_allocated as f64 / 1e6);
        println!("Peak Usage: {:.2} MB", self.peak_allocated as f64 / 1e6);
        println!("Total Allocations: {}", self.total_allocations);
        println!("Total Deallocations: {}", self.total_deallocations);
        println!("Active Allocations: {}", self.total_allocations - self.total_deallocations);

        let elapsed = self.start_time.elapsed();
        println!("Profiling Duration: {:.2}s", elapsed.as_secs_f64());

        if !self.active_allocations.is_empty() {
            println!("\nActive Allocations by Tag:");
            let mut tags: Vec<_> = self.active_allocations.iter().collect();
            tags.sort_by_key(|b| std::cmp::Reverse(b.1.iter().sum::<usize>()));

            for (tag, sizes) in tags.iter().take(10) {
                let total: usize = sizes.iter().sum();
                println!("  {:30} {:10.2} MB ({} allocations)", tag, total as f64 / 1e6, sizes.len());
            }
        }

        println!("{}", "=".repeat(70));
    }

    /// Export memory timeline as JSON.
    pub fn export_timeline(&self, path: &str) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(path)?;

        writeln!(file, "[")?;

        let mut first = true;
        for event in &self.events {
            if !first {
                writeln!(file, ",")?;
            }
            first = false;

            let event_type = if event.is_allocation { "alloc" } else { "free" };
            let json = serde_json::json!({
                "timestamp_ms": event.timestamp.elapsed().as_millis(),
                "type": event_type,
                "size": event.size,
                "tag": event.tag,
            });
            write!(file, "{}", json)?;
        }

        writeln!(file, "\n]")?;
        Ok(())
    }

    /// Clear all data.
    pub fn clear(&mut self) {
        self.events.clear();
        self.snapshots.clear();
        self.current_allocated = 0;
        self.peak_allocated = 0;
        self.active_allocations.clear();
        self.total_allocations = 0;
        self.total_deallocations = 0;
    }

    /// Get profiling summary.
    pub fn summary(&self) -> MemorySummary {
        MemorySummary {
            current_bytes: self.current_allocated,
            peak_bytes: self.peak_allocated,
            total_allocations: self.total_allocations,
            total_deallocations: self.total_deallocations,
            active_count: self.total_allocations - self.total_deallocations,
            duration: self.start_time.elapsed(),
        }
    }
}

impl Default for MemoryProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// OOM risk level.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OomRisk {
    Low,
    Medium,
    High,
    Critical,
}

impl OomRisk {
    pub fn as_str(&self) -> &'static str {
        match self {
            OomRisk::Low => "low",
            OomRisk::Medium => "medium",
            OomRisk::High => "high",
            OomRisk::Critical => "critical",
        }
    }
}

/// Memory summary statistics.
#[derive(Debug)]
pub struct MemorySummary {
    pub current_bytes: usize,
    pub peak_bytes: usize,
    pub total_allocations: usize,
    pub total_deallocations: usize,
    pub active_count: usize,
    pub duration: Duration,
}

lazy_static::lazy_static! {
    static ref GLOBAL_PROFILER: Arc<Mutex<MemoryProfiler>> = Arc::new(Mutex::new(MemoryProfiler::new()));
}

/// Access global profiler.
pub fn global_profiler() -> Arc<Mutex<MemoryProfiler>> {
    GLOBAL_PROFILER.clone()
}

/// Convenience macro for tracking tensor allocations.
#[macro_export]
macro_rules! track_tensor {
    ($tag:expr, $size:expr) => {
        let _tracker = $crate::memory_profiler::MemoryProfiler::track_allocation(
            &$crate::memory_profiler::global_profiler(),
            $tag,
            $size,
        );
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocation_tracker() {
        let profiler = Arc::new(Mutex::new(MemoryProfiler::new()));

        {
            let tracker = AllocationTracker::new(profiler.clone(), 1024, "test_tensor");
            assert_eq!(tracker.size(), 1024);
            assert_eq!(tracker.tag(), "test_tensor");

            let p = profiler.lock().unwrap();
            assert_eq!(p.current_usage(), 1024);
        }

        let p = profiler.lock().unwrap();
        assert_eq!(p.current_usage(), 0);
        assert_eq!(p.total_allocations, 1);
        assert_eq!(p.total_deallocations, 1);
    }

    #[test]
    fn test_memory_snapshot() {
        let mut profiler = MemoryProfiler::new();

        profiler.record_allocation_internal(1000, "a");
        profiler.record_allocation_internal(2000, "b");

        let snapshot = profiler.snapshot();
        assert_eq!(snapshot.total_allocated, 3000);
        assert_eq!(snapshot.by_tag.get("a"), Some(&1000));
        assert_eq!(snapshot.by_tag.get("b"), Some(&2000));
    }

    #[test]
    fn test_oom_risk() {
        let mut profiler = MemoryProfiler::new();
        profiler.record_allocation_internal(40, "test");

        assert_eq!(profiler.predict_oom_risk(100), OomRisk::Low); // 40% < 0.5
        assert_eq!(profiler.predict_oom_risk(60), OomRisk::Medium); // 66.7% < 0.75
        assert_eq!(profiler.predict_oom_risk(50), OomRisk::High); // 80% >= 0.75, < 0.9
        assert_eq!(profiler.predict_oom_risk(40), OomRisk::Critical); // 100% >= 0.9
    }

    #[test]
    fn test_find_leaks() {
        let mut profiler = MemoryProfiler::new();
        profiler.record_allocation_internal(1000, "leaked");
        // No deallocation - this is a leak

        let leaks = profiler.find_leaks();
        assert_eq!(leaks.len(), 1);
        assert_eq!(leaks[0].0, "leaked");
        assert_eq!(leaks[0].1, 1000);
    }

    #[test]
    fn test_summary() {
        let mut profiler = MemoryProfiler::new();
        profiler.record_allocation_internal(1000, "a");
        profiler.record_deallocation_internal(1000, "a");

        let summary = profiler.summary();
        assert_eq!(summary.current_bytes, 0);
        assert_eq!(summary.peak_bytes, 1000);
        assert_eq!(summary.total_allocations, 1);
        assert_eq!(summary.total_deallocations, 1);
        assert_eq!(summary.active_count, 0);
    }

    #[test]
    fn test_enable_disable() {
        let mut profiler = MemoryProfiler::new();
        assert!(profiler.is_enabled());
        profiler.disable();
        assert!(!profiler.is_enabled());
        profiler.enable();
        assert!(profiler.is_enabled());
    }

    #[test]
    fn test_would_exceed() {
        let mut profiler = MemoryProfiler::new();
        profiler.record_allocation_internal(500, "a");
        assert!(profiler.would_exceed(600, 1000));
        assert!(!profiler.would_exceed(400, 1000));
    }

    #[test]
    fn test_peak_usage() {
        let mut profiler = MemoryProfiler::new();
        assert_eq!(profiler.peak_usage(), 0);
        profiler.record_allocation_internal(1000, "a");
        assert_eq!(profiler.peak_usage(), 1000);
        profiler.record_allocation_internal(500, "b");
        assert_eq!(profiler.peak_usage(), 1500);
    }

    #[test]
    fn test_usage_by_tag() {
        let mut profiler = MemoryProfiler::new();
        profiler.record_allocation_internal(1000, "a");
        profiler.record_allocation_internal(500, "b");
        let usage = profiler.usage_by_tag();
        assert_eq!(usage.get("a"), Some(&1000));
        assert_eq!(usage.get("b"), Some(&500));
    }

    #[test]
    fn test_clear() {
        let mut profiler = MemoryProfiler::new();
        profiler.record_allocation_internal(1000, "a");
        profiler.clear();
        assert_eq!(profiler.current_usage(), 0);
        assert_eq!(profiler.peak_usage(), 0);
        assert_eq!(profiler.total_allocations, 0);
    }

    #[test]
    fn test_oom_risk_as_str() {
        assert_eq!(OomRisk::Low.as_str(), "low");
        assert_eq!(OomRisk::Medium.as_str(), "medium");
        assert_eq!(OomRisk::High.as_str(), "high");
        assert_eq!(OomRisk::Critical.as_str(), "critical");
    }

    #[test]
    fn test_default() {
        let profiler: MemoryProfiler = Default::default();
        assert!(profiler.is_enabled());
    }

    #[test]
    fn test_global_profiler() {
        let gp = global_profiler();
        let _guard = gp.lock().unwrap();
    }

    #[test]
    fn test_track_allocation() {
        let profiler = Arc::new(Mutex::new(MemoryProfiler::new()));
        {
            let tracker = MemoryProfiler::track_allocation(&profiler, "test", 256);
            assert_eq!(tracker.size(), 256);
            assert_eq!(tracker.tag(), "test");
        }
        let p = profiler.lock().unwrap();
        assert_eq!(p.current_usage(), 0);
    }

    #[test]
    fn test_manual_free() {
        let profiler = Arc::new(Mutex::new(MemoryProfiler::new()));
        let mut tracker = AllocationTracker::new(profiler.clone(), 256, "test");
        tracker.free();
        assert!(tracker.tag() == "test");
    }

    #[test]
    fn test_export_timeline() {
        let mut profiler = MemoryProfiler::new();
        profiler.record_allocation_internal(100, "a");
        profiler.record_deallocation_internal(100, "a");
        let tmpfile = std::env::temp_dir().join("rustral_timeline_test.json");
        profiler.export_timeline(tmpfile.to_str().unwrap()).unwrap();
        std::fs::remove_file(&tmpfile).ok();
    }

    #[test]
    fn test_print_report() {
        let mut profiler = MemoryProfiler::new();
        profiler.record_allocation_internal(100, "a");
        profiler.print_report();
    }
}
