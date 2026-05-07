el like our rustral neural eng# rustral-tui

Full-featured terminal dashboard for Rustral training and inference workloads.

Provides a real-time, multi-panel terminal UI (TUI) using `ratatui` and `crossterm`
that displays live training progress, metrics, memory usage, and warnings.

## Features

- **Progress Panel**: Dual progress bars (epochs + batches), ETA, elapsed time, throughput (samples/sec)
- **Metrics Panel**: Live scalar metrics with loss mini-chart (Sparkline)
- **Memory Panel**: Current/peak memory, OOM risk color indicator (green → yellow → red), per-tag allocation breakdown
- **Leak Detection**: Real-time memory leak warnings with tag, bytes, and allocation count
- **Loss Spike Detection**: Detects NaN/Inf and sudden loss divergence
- **Warnings Footer**: Critical alerts for OOM risk, memory leaks, NaN losses
- **JSON Dump**: Export current snapshot as JSON for post-hoc analysis
- **Responsive Layout**: Panels resize automatically with terminal dimensions

## Quick Start

```rust
use std::sync::{Arc, Mutex};
use rustral_tui::{TrainingDashboard, DashboardConfig};

let dashboard = Arc::new(Mutex::new(TrainingDashboard::new(DashboardConfig::default())));

// Launch the TUI in a background thread
let renderer = dashboard.clone();
std::thread::spawn(move || {
    let mut renderer = rustral_tui::DashboardRenderer::new(renderer).unwrap();
    renderer.run().ok();
});

// In your training loop:
{
    let mut db = dashboard.lock().unwrap();
    db.set_epoch(epoch);
    db.set_batch(batch);
    db.record_loss(loss);
    db.record_accuracy(acc);
    db.set_memory_bytes(current_mem);
    db.set_available_memory(total_mem);
}
```

## Example

Run the simulated training demo:

```bash
cargo run -p rustral-tui --example train_dashboard
```

## Integration with Memory Profiler

The dashboard integrates naturally with `rustral_core::memory_profiler::MemoryProfiler`:

```rust
use rustral_tui::TrainingDashboard;
use rustral_core::memory_profiler::MemoryProfiler;

let mut dashboard = TrainingDashboard::new(DashboardConfig::default());
let mut mem_profiler = MemoryProfiler::new();

// After training step:
let summary = mem_profiler.summary();
dashboard.set_memory_bytes(summary.current_bytes);
dashboard.set_peak_memory(summary.peak_bytes);
dashboard.set_allocation_counts(summary.total_allocations, summary.total_deallocations);

// Check for leaks
for (tag, bytes) in mem_profiler.find_leaks() {
    dashboard.report_leak(&tag, bytes, 1);
}

// OOM prediction
let oom = mem_profiler.predict_oom_risk(available_memory);
```

## API Overview

| Method | Description |
|--------|-------------|
| `TrainingDashboard::new(cfg)` | Create dashboard with config |
| `set_epoch(n)` / `set_batch(n)` | Set progress position |
| `set_total_epochs(n)` / `set_total_batches(n)` | Set progress targets |
| `record_loss(v)` / `record_accuracy(v)` / `record_lr(v)` | Feed metrics |
| `set_memory_bytes(v)` / `set_peak_memory(v)` | Feed memory values |
| `set_available_memory(v)` | Set total RAM for OOM ratio |
| `set_memory_by_tag(tags)` | Per-tag allocation breakdown |
| `set_allocation_counts(allocs, deallocs)` | Allocation tracking |
| `report_leak(tag, bytes, count)` | Report detected leak |
| `add_warning(msg)` | Add alert message |
| `dump_metrics()` | Export as JSON string |
| `render(frame)` | Render into ratatui frame |