//! GPU Profiler for Performance Analysis
//!
//! Provides kernel timing, memory tracking, and bandwidth metrics
//! for optimizing GPU workloads.
//!
//! # Features
//!
//! - Kernel execution timing
//! - Memory allocation tracking
//! - Bandwidth utilization
//! - Timeline export for Chrome tracing
//!
//! # Example
//!
//! ```rust,ignore
//! use mnr_wgpu_backend::profiler::{GpuProfiler, ProfileEvent};
//!
//! let mut profiler = GpuProfiler::new();
//!
//! let event = profiler.start_event("matmul", "compute");
//! // ... run kernel ...
//! profiler.end_event(event);
//!
//! profiler.export_chrome_trace("trace.json");
//! ```

use std::collections::HashMap;
use std::time::Instant;

/// A profiling event.
#[derive(Clone, Debug)]
pub struct ProfileEvent {
    pub name: String,
    pub category: String,
    pub start_time: u64,       // microseconds
    pub end_time: Option<u64>, // microseconds
    pub metadata: HashMap<String, String>,
}

impl ProfileEvent {
    fn new(name: &str, category: &str) -> Self {
        Self {
            name: name.to_string(),
            category: category.to_string(),
            start_time: Self::now_micros(),
            end_time: None,
            metadata: HashMap::new(),
        }
    }

    fn now_micros() -> u64 {
        Instant::now().elapsed().as_micros() as u64
            + std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64
    }

    /// End this event.
    pub fn end(&mut self) {
        self.end_time = Some(Self::now_micros());
    }

    /// Get duration in microseconds.
    pub fn duration_micros(&self) -> Option<u64> {
        self.end_time.map(|end| end.saturating_sub(self.start_time))
    }

    /// Get duration in milliseconds.
    pub fn duration_ms(&self) -> Option<f64> {
        self.duration_micros().map(|us| us as f64 / 1000.0)
    }

    /// Add metadata.
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

/// Memory allocation event.
#[derive(Clone, Debug)]
pub struct MemoryEvent {
    pub timestamp: u64,
    pub allocation_size: i64, // positive = alloc, negative = free
    pub total_allocated: usize,
    pub tag: String,
}

/// GPU Performance Profiler.
pub struct GpuProfiler {
    events: Vec<ProfileEvent>,
    memory_events: Vec<MemoryEvent>,
    current_allocations: HashMap<String, usize>,
    total_allocated: usize,
    peak_allocated: usize,
    enabled: bool,
}

impl GpuProfiler {
    /// Create a new profiler.
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            memory_events: Vec::new(),
            current_allocations: HashMap::new(),
            total_allocated: 0,
            peak_allocated: 0,
            enabled: true,
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

    /// Check if profiling is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Start a profiling event.
    pub fn start_event(&mut self, name: &str, category: &str) -> usize {
        if !self.enabled {
            return 0;
        }

        let event = ProfileEvent::new(name, category);
        let id = self.events.len();
        self.events.push(event);
        id
    }

    /// End a profiling event.
    pub fn end_event(&mut self, event_id: usize) {
        if !self.enabled || event_id >= self.events.len() {
            return;
        }

        self.events[event_id].end();
    }

    /// Record a memory allocation.
    pub fn record_allocation(&mut self, size: usize, tag: &str) {
        if !self.enabled {
            return;
        }

        self.total_allocated += size;
        self.current_allocations.insert(tag.to_string(), size);

        if self.total_allocated > self.peak_allocated {
            self.peak_allocated = self.total_allocated;
        }

        self.memory_events.push(MemoryEvent {
            timestamp: ProfileEvent::now_micros(),
            allocation_size: size as i64,
            total_allocated: self.total_allocated,
            tag: tag.to_string(),
        });
    }

    /// Record a memory deallocation.
    pub fn record_deallocation(&mut self, size: usize, tag: &str) {
        if !self.enabled {
            return;
        }

        self.total_allocated = self.total_allocated.saturating_sub(size);
        self.current_allocations.remove(tag);

        self.memory_events.push(MemoryEvent {
            timestamp: ProfileEvent::now_micros(),
            allocation_size: -(size as i64),
            total_allocated: self.total_allocated,
            tag: tag.to_string(),
        });
    }

    /// Get all completed events.
    pub fn get_events(&self) -> &[ProfileEvent] {
        &self.events
    }

    /// Get events by category.
    pub fn get_events_by_category(&self, category: &str) -> Vec<&ProfileEvent> {
        self.events.iter().filter(|e| e.category == category).collect()
    }

    /// Get summary statistics.
    pub fn summary(&self) -> ProfileSummary {
        let mut category_times: HashMap<String, (u64, usize)> = HashMap::new();

        for event in &self.events {
            if let Some(duration) = event.duration_micros() {
                let entry = category_times.entry(event.category.clone()).or_insert((0, 0));
                entry.0 += duration;
                entry.1 += 1;
            }
        }

        ProfileSummary {
            total_events: self.events.len(),
            completed_events: self.events.iter().filter(|e| e.end_time.is_some()).count(),
            total_memory_allocated: self.peak_allocated,
            current_memory: self.total_allocated,
            category_times,
        }
    }

    /// Export to Chrome trace format.
    pub fn export_chrome_trace(&self, path: &str) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(path)?;

        // Chrome trace format header
        writeln!(file, "[")?;

        let mut first = true;
        for event in &self.events {
            if let Some(end_time) = event.end_time {
                if !first {
                    writeln!(file, ",")?;
                }
                first = false;

                // Chrome trace event format
                let trace_event = serde_json::json!({
                    "name": event.name,
                    "cat": event.category,
                    "ph": "X", // Complete event
                    "ts": event.start_time,
                    "dur": end_time - event.start_time,
                    "pid": 1,
                    "tid": 1,
                });

                write!(file, "{}", trace_event)?;
            }
        }

        // Add memory events as instant events
        for mem_event in &self.memory_events {
            if !first {
                writeln!(file, ",")?;
            }
            first = false;

            let trace_event = serde_json::json!({
                "name": format!("{} {}",
                    if mem_event.allocation_size > 0 { "alloc" } else { "free" },
                    mem_event.tag
                ),
                "cat": "memory",
                "ph": "i", // Instant event
                "ts": mem_event.timestamp,
                "pid": 1,
                "tid": 2,
                "args": {
                    "size": mem_event.allocation_size,
                    "total": mem_event.total_allocated
                }
            });

            write!(file, "{}", trace_event)?;
        }

        writeln!(file, "\n]")?;

        println!("Exported Chrome trace to: {}", path);
        Ok(())
    }

    /// Print summary to console.
    pub fn print_summary(&self) {
        let summary = self.summary();

        println!("\n{}", "=".repeat(60));
        println!("GPU Profile Summary");
        println!("{}", "=".repeat(60));
        println!("Total Events: {} ({} completed)", summary.total_events, summary.completed_events);
        println!("Peak Memory: {:.2} MB", summary.total_memory_allocated as f64 / (1024.0 * 1024.0));
        println!("Current Memory: {:.2} MB", summary.current_memory as f64 / (1024.0 * 1024.0));

        println!("\nTime by Category:");
        for (category, (total_us, count)) in &summary.category_times {
            let avg_us = *total_us as f64 / *count as f64;
            println!(
                "  {:20}: {:8.2} ms total, {:8.2} ms avg ({} calls)",
                category,
                *total_us as f64 / 1000.0,
                avg_us / 1000.0,
                count
            );
        }

        println!("{}", "=".repeat(60));
    }

    /// Clear all events.
    pub fn clear(&mut self) {
        self.events.clear();
        self.memory_events.clear();
        self.current_allocations.clear();
        self.total_allocated = 0;
        self.peak_allocated = 0;
    }
}

impl Default for GpuProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary statistics from profiling.
pub struct ProfileSummary {
    pub total_events: usize,
    pub completed_events: usize,
    pub total_memory_allocated: usize,
    pub current_memory: usize,
    pub category_times: HashMap<String, (u64, usize)>, // (total_us, count)
}

/// Scoped profiler event (RAII).
pub struct ScopedEvent<'a> {
    profiler: &'a mut GpuProfiler,
    event_id: usize,
}

impl<'a> ScopedEvent<'a> {
    pub fn new(profiler: &'a mut GpuProfiler, name: &str, category: &str) -> Self {
        let event_id = profiler.start_event(name, category);
        Self { profiler, event_id }
    }
}

impl<'a> Drop for ScopedEvent<'a> {
    fn drop(&mut self) {
        self.profiler.end_event(self.event_id);
    }
}

/// Macro for scoped profiling.
#[macro_export]
macro_rules! profile_scope {
    ($profiler:expr, $name:expr, $category:expr) => {
        let _scoped_event = $crate::profiler::ScopedEvent::new($profiler, $name, $category);
    };
}

/// Bandwidth calculator for data transfer operations.
pub struct BandwidthCalculator {
    bytes_transferred: u64,
    duration_micros: u64,
}

impl BandwidthCalculator {
    pub fn new(bytes: u64, duration_micros: u64) -> Self {
        Self { bytes_transferred: bytes, duration_micros }
    }

    /// Calculate bandwidth in GB/s.
    pub fn bandwidth_gbps(&self) -> f64 {
        let bytes_per_sec = self.bytes_transferred as f64 * 1_000_000.0 / self.duration_micros as f64;
        bytes_per_sec / (1024.0 * 1024.0 * 1024.0)
    }

    /// Format as human-readable string.
    pub fn format(&self) -> String {
        format!("{:.2} GB/s", self.bandwidth_gbps())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_event() {
        let mut event = ProfileEvent::new("test", "compute");
        std::thread::sleep(std::time::Duration::from_millis(10));
        event.end();

        assert!(event.duration_micros().unwrap() >= 10_000);
        assert!(event.duration_ms().unwrap() >= 10.0);
    }

    #[test]
    fn test_profiler() {
        let mut profiler = GpuProfiler::new();

        let event_id = profiler.start_event("matmul", "compute");
        std::thread::sleep(std::time::Duration::from_millis(5));
        profiler.end_event(event_id);

        let event_id2 = profiler.start_event("memcpy", "memory");
        profiler.record_allocation(1024 * 1024, "tensor_a");
        std::thread::sleep(std::time::Duration::from_millis(2));
        profiler.end_event(event_id2);

        let summary = profiler.summary();
        assert_eq!(summary.total_events, 2);
        assert_eq!(summary.completed_events, 2);
        assert_eq!(summary.current_memory, 1024 * 1024);

        profiler.print_summary();
    }

    #[test]
    fn test_scoped_event() {
        let mut profiler = GpuProfiler::new();
        {
            let _scope = ScopedEvent::new(&mut profiler, "scoped_op", "compute");
            std::thread::sleep(std::time::Duration::from_millis(5));
        }

        let summary = profiler.summary();
        assert_eq!(summary.completed_events, 1);
    }

    #[test]
    fn test_bandwidth_calculator() {
        // 1 GB in 100ms = 10 GB/s
        let calc = BandwidthCalculator::new(1024 * 1024 * 1024, 100_000);
        assert!((calc.bandwidth_gbps() - 10.0).abs() < 0.5);
    }

    #[test]
    fn test_chrome_trace_export() {
        use std::fs;

        let mut profiler = GpuProfiler::new();

        let event_id = profiler.start_event("kernel", "compute");
        profiler.end_event(event_id);

        let path = "/tmp/test_trace.json";
        profiler.export_chrome_trace(path).unwrap();

        let content = fs::read_to_string(path).unwrap();
        assert!(content.contains("kernel"));
        assert!(content.contains("compute"));

        fs::remove_file(path).ok();
    }
}
