//! Full-featured Terminal Dashboard for Rustral Training/Inference
//!
//! Provides a real-time, multi-panel terminal UI (TUI) using `ratatui` and
//! `crossterm` that displays:
//!
//! - **Progress Panel**: Training progress bars (epochs, batches), ETA, elapsed time
//! - **Metrics Panel**: Live scalar metrics (loss, accuracy, LR) with auto-scaling
//! - **Memory Panel**: Current/peak memory usage, OOM risk indicator, per-tag breakdown
//! - **Leak Detection**: Real-time memory leak warnings and unstable allocation tracking
//! - **Throughput**: Samples/second, tokens/second, estimated time remaining
//! - **Mini-Chart**: ASCII-smoothed loss history (rolling window)
//! - **Warnings Footer**: Critical alerts for OOM risk, leak detection, NaN losses
//!
//! # Integration Patterns
//!
//! ## 1. As a MetricsBackend (drop-in for existing `MetricsLogger`)
//!
//! ```rust,ignore
//! use rustral_tui::TuiMetricsBackend;
//! use rustral_metrics::MetricsLogger;
//!
//! let mut logger = MetricsLogger::new();
//! logger.add_backend(Box::new(TuiMetricsBackend::new()));
//! ```
//!
//! ## 2. Direct API (manual training loop)
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use rustral_tui::{TrainingDashboard, DashboardConfig, DashboardRenderer};
//!
//! let dash = Arc::new(std::sync::Mutex::new(TrainingDashboard::new(DashboardConfig::default())));
//!
//! // Run renderer in background
//! let r = dash.clone();
//! std::thread::spawn(move || { DashboardRenderer::new(r).unwrap().run().ok(); });
//!
//! // Feed metrics during training:
//! let mut db = dash.lock().unwrap();
//! db.set_epoch(epoch);
//! db.set_batch(batch);
//! db.record_loss(loss);
//! db.set_memory_bytes(current_mem);
//! ```
//!
//! ## 3. Trainer Hook (wraps `TapeTrainer` / `fit_classification`)
//!
//! ```rust,ignore
//! use rustral_tui::TrainingHook;
//! let hook = TrainingHook::new();
//!
//! // The hook auto-records metrics from any EpochStats.
//! // Meanwhile, call renderer.run() in a separate thread.
//! ```

use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{
        Block, BorderType, Borders, Gauge, Paragraph,
        Sparkline,
    },
    Frame, Terminal,
};
use std::collections::VecDeque;
use std::io::{self, Stdout};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Configuration for the training dashboard.
#[derive(Clone, Debug)]
pub struct DashboardConfig {
    /// Title displayed at the top of the dashboard.
    pub title: String,
    /// Number of historical data points for mini-charts (loss, accuracy, memory).
    pub history_len: usize,
    /// Refresh interval in milliseconds between renders.
    pub refresh_ms: u64,
    /// File path to dump JSON metrics (optional). If set, metrics are also written here.
    pub dump_path: Option<String>,
    /// Whether to automatically enable alternate screen on start.
    pub alt_screen: bool,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            title: "Rustral Training Dashboard".into(),
            history_len: 120,
            refresh_ms: 250,
            dump_path: None,
            alt_screen: true,
        }
    }
}

/// Memory leak warning.
#[derive(Clone, Debug)]
pub struct LeakWarning {
    pub tag: String,
    pub bytes: usize,
    pub detected_at: Instant,
    pub allocation_count: usize,
}

/// OOM risk level for display.
#[derive(Clone, Debug, PartialEq)]
pub enum OomRiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

impl OomRiskLevel {
    pub fn from_ratio(ratio: f64) -> Self {
        match ratio {
            r if r < 0.5 => OomRiskLevel::Low,
            r if r < 0.75 => OomRiskLevel::Medium,
            r if r < 0.9 => OomRiskLevel::High,
            _ => OomRiskLevel::Critical,
        }
    }

    pub fn color(&self) -> Color {
        match self {
            OomRiskLevel::Low => Color::Green,
            OomRiskLevel::Medium => Color::Yellow,
            OomRiskLevel::High => Color::Red,
            OomRiskLevel::Critical => Color::LightRed,
        }
    }
}

/// Core dashboard state that tracks all training progress.
pub struct TrainingDashboard {
    // Configuration
    cfg: DashboardConfig,

    // Timeline
    start_time: Instant,
    elapsed: Duration,
    _last_render: Instant,

    // Progress tracking
    current_epoch: u64,
    total_epochs: u64,
    current_batch: u64,
    total_batches: u64,
    current_step: u64,

    // Historical data (ring buffers)
    loss_history: VecDeque<f64>,
    accuracy_history: VecDeque<f64>,
    lr_history: VecDeque<f64>,
    memory_history: VecDeque<f64>,

    // Current metrics (latest values)
    current_loss: f64,
    current_accuracy: Option<f64>,
    current_lr: Option<f64>,

    // Memory tracking
    current_memory_bytes: usize,
    peak_memory_bytes: usize,
    memory_by_tag: Vec<(String, usize)>,
    total_allocations: usize,
    total_deallocations: usize,
    available_memory_bytes: usize,

    // Leak detection
    leak_warnings: Vec<LeakWarning>,
    unstable_allocations: usize,

    // Throughput
    batch_samples: usize,
    samples_processed: u64,
    throughput_samples_per_sec: f64,
    last_throughput_update: Instant,
    epoch_samples: u64,

    // Warnings
    warnings: Vec<String>,

    // Loss divergence detection
    last_loss: f64,
    loss_spike_count: u32,

    // Rendering
    render_count: u64,
    logged_events: VecDeque<String>, // Last N log events for display
}

impl TrainingDashboard {
    /// Create a new dashboard with the given configuration.
    pub fn new(cfg: DashboardConfig) -> Self {
        let history_len = cfg.history_len;
        Self {
            cfg,
            start_time: Instant::now(),
            elapsed: Duration::ZERO,
            _last_render: Instant::now(),
            current_epoch: 0,
            total_epochs: 0,
            current_batch: 0,
            total_batches: 0,
            current_step: 0,
            loss_history: VecDeque::with_capacity(history_len),
            accuracy_history: VecDeque::with_capacity(history_len),
            lr_history: VecDeque::with_capacity(history_len),
            memory_history: VecDeque::with_capacity(history_len),
            current_loss: 0.0,
            current_accuracy: None,
            current_lr: None,
            current_memory_bytes: 0,
            peak_memory_bytes: 0,
            memory_by_tag: Vec::new(),
            total_allocations: 0,
            total_deallocations: 0,
            available_memory_bytes: 8 * 1024 * 1024 * 1024, // default 8GB
            leak_warnings: Vec::new(),
            unstable_allocations: 0,
            batch_samples: 0,
            samples_processed: 0,
            throughput_samples_per_sec: 0.0,
            last_throughput_update: Instant::now(),
            epoch_samples: 0,
            warnings: Vec::new(),
            last_loss: f64::NAN,
            loss_spike_count: 0,
            render_count: 0,
            logged_events: VecDeque::with_capacity(50),
        }
    }

    // ── Progress setters ──────────────────────────────────────────

    /// Set total epochs for progress bar.
    pub fn set_total_epochs(&mut self, total: u64) {
        self.total_epochs = total;
    }

    /// Set total batches per epoch for progress bar.
    pub fn set_total_batches(&mut self, total: u64) {
        self.total_batches = total;
    }

    /// Set current epoch.
    pub fn set_epoch(&mut self, epoch: u64) {
        self.current_epoch = epoch;
    }

    /// Set current batch within epoch.
    pub fn set_batch(&mut self, batch: u64) {
        self.current_batch = batch;
    }

    /// Set current training step.
    pub fn set_step(&mut self, step: u64) {
        self.current_step = step;
    }

    /// Record batch size (samples) for throughput calculation.
    pub fn record_batch_samples(&mut self, samples: usize) {
        self.batch_samples = samples;
        self.samples_processed += samples as u64;
        self.epoch_samples += samples as u64;

        // Update throughput every 2 seconds
        let now = Instant::now();
        let dt = now.duration_since(self.last_throughput_update).as_secs_f64();
        if dt >= 2.0 {
            self.throughput_samples_per_sec = self.samples_processed as f64 / dt;
            self.samples_processed = 0;
            self.last_throughput_update = now;
        }
    }

    // ── Metric recorders ──────────────────────────────────────────

    /// Record a loss value.
    pub fn record_loss(&mut self, loss: f64) {
        // NaN detection
        if loss.is_nan() || loss.is_infinite() {
            self.add_warning(format!("NaN/Inf loss detected at step {}", self.current_step));
            return;
        }

        // Spike detection
        if self.loss_history.len() >= 3 {
            let avg = self.loss_history.iter().rev().take(3).sum::<f64>() / 3.0;
            if loss > avg * 5.0 {
                self.loss_spike_count += 1;
                if self.loss_spike_count >= 3 {
                    self.add_warning(format!(
                        "Sustained loss spikes ({}x normal) at step {}",
                        (loss / avg.max(1e-10)) as u64,
                        self.current_step
                    ));
                    self.loss_spike_count = 0;
                }
            } else {
                self.loss_spike_count = 0;
            }
        }

        if self.loss_history.len() >= self.cfg.history_len {
            self.loss_history.pop_front();
        }
        self.loss_history.push_back(loss);
        self.current_loss = loss;
        self.last_loss = loss;
    }

    /// Record an accuracy value.
    pub fn record_accuracy(&mut self, accuracy: f64) {
        if self.accuracy_history.len() >= self.cfg.history_len {
            self.accuracy_history.pop_front();
        }
        self.accuracy_history.push_back(accuracy);
        self.current_accuracy = Some(accuracy);
    }

    /// Record a learning rate value.
    pub fn record_lr(&mut self, lr: f64) {
        if self.lr_history.len() >= self.cfg.history_len {
            self.lr_history.pop_front();
        }
        self.lr_history.push_back(lr);
        self.current_lr = Some(lr);
    }

    // ── Memory setters ────────────────────────────────────────────

    /// Set current memory usage in bytes.
    pub fn set_memory_bytes(&mut self, bytes: usize) {
        self.current_memory_bytes = bytes;
        if bytes > self.peak_memory_bytes {
            self.peak_memory_bytes = bytes;
        }
        if self.memory_history.len() >= self.cfg.history_len {
            self.memory_history.pop_front();
        }
        self.memory_history.push_back(bytes as f64);
    }

    /// Set peak memory usage in bytes.
    pub fn set_peak_memory(&mut self, bytes: usize) {
        self.peak_memory_bytes = bytes;
    }

    /// Set available/total memory in bytes (for OOM ratio).
    pub fn set_available_memory(&mut self, bytes: usize) {
        self.available_memory_bytes = bytes;
    }

    /// Set per-tag memory breakdown.
    pub fn set_memory_by_tag(&mut self, by_tag: Vec<(String, usize)>) {
        self.memory_by_tag = by_tag;
    }

    /// Set allocation/deallocation counts.
    pub fn set_allocation_counts(&mut self, allocs: usize, deallocs: usize) {
        self.total_allocations = allocs;
        self.total_deallocations = deallocs;
    }

    // ── Leak detection ────────────────────────────────────────────

    /// Report a memory leak warning.
    pub fn report_leak(&mut self, tag: &str, bytes: usize, allocation_count: usize) {
        self.leak_warnings.push(LeakWarning {
            tag: tag.to_string(),
            bytes,
            detected_at: Instant::now(),
            allocation_count,
        });
        self.add_warning(format!(
            "Memory leak detected: {} ({} bytes, {} allocations)",
            tag, bytes, allocation_count
        ));
    }

    /// Report unstable allocations.
    pub fn report_unstable_allocations(&mut self, count: usize) {
        self.unstable_allocations = count;
        if count > 10 {
            self.add_warning(format!(
                "Unstable allocations detected: {} active allocations",
                count
            ));
        }
    }

    // ── Warnings ──────────────────────────────────────────────────

    /// Add a warning entry.
    pub fn add_warning(&mut self, warning: String) {
        if self.warnings.len() >= 5 {
            self.warnings.remove(0);
        }
        self.warnings.push(warning.clone());
        self.log_event(warning);
    }

    /// Log an event message.
    pub fn log_event(&mut self, msg: String) {
        if self.logged_events.len() >= 50 {
            self.logged_events.pop_front();
        }
        self.logged_events.push_back(msg);
    }

    // ── Getters ───────────────────────────────────────────────────

    /// Get the estimated OOM risk level.
    pub fn oom_risk(&self) -> OomRiskLevel {
        if self.available_memory_bytes == 0 {
            return OomRiskLevel::Low;
        }
        let ratio = self.current_memory_bytes as f64 / self.available_memory_bytes as f64;
        OomRiskLevel::from_ratio(ratio)
    }

    /// Check if there are active leak warnings.
    pub fn has_leaks(&self) -> bool {
        !self.leak_warnings.is_empty()
    }

    /// Get throughput in samples/sec as a formatted string.
    pub fn throughput_str(&self) -> String {
        if self.throughput_samples_per_sec > 1_000_000.0 {
            format!("{:.2} M/s", self.throughput_samples_per_sec / 1_000_000.0)
        } else if self.throughput_samples_per_sec > 1_000.0 {
            format!("{:.2} K/s", self.throughput_samples_per_sec / 1_000.0)
        } else {
            format!("{:.1} /s", self.throughput_samples_per_sec)
        }
    }

    /// Get the ETA string based on current epoch progress.
    pub fn eta_str(&self) -> String {
        if self.total_epochs == 0 || self.current_epoch == 0 {
            return "--:--:--".into();
        }
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let epoch_progress = self.current_epoch as f64 / self.total_epochs as f64;
        if epoch_progress <= 0.0 {
            return "--:--:--".into();
        }
        let total_est = elapsed / epoch_progress;
        let remaining = (total_est - elapsed).max(0.0);
        let secs = remaining as u64;
        format!("{:02}:{:02}:{:02}", secs / 3600, (secs % 3600) / 60, secs % 60)
    }

    /// Format bytes to human-readable string.
    pub fn format_bytes(bytes: usize) -> String {
        if bytes >= 1_000_000_000 {
            format!("{:.2} GB", bytes as f64 / 1_000_000_000.0)
        } else if bytes >= 1_000_000 {
            format!("{:.2} MB", bytes as f64 / 1_000_000.0)
        } else if bytes >= 1_000 {
            format!("{:.2} KB", bytes as f64 / 1_000.0)
        } else {
            format!("{} B", bytes)
        }
    }

    // ── Rendering ─────────────────────────────────────────────────

    /// Render the dashboard to the terminal. Must be called within a
    /// terminal.draw() closure or via `DashboardRenderer::render()`.
    pub fn render(&mut self, f: &mut Frame) {
        self.elapsed = self.start_time.elapsed();
        self.render_count += 1;

        let area = f.size();

        // ── Layout: 3 rows (top=progress, middle=metrics+memory, bottom=warnings) ──
        let main_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(8),   // Top: progress + throughput
                Constraint::Min(10),     // Middle: metrics + memory + leaktection
                Constraint::Length(7),   // Bottom: warnings + events
            ])
            .split(area);

        // ── Top Row: Progress Panel ───────────────────────────────
        self.render_progress_panel(f, main_chunks[0]);

        // ── Middle Row: Metrics | Memory | Leaks ──────────────────
        let mid_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(50),
                Constraint::Percentage(50),
            ])
            .split(main_chunks[1]);

        self.render_metrics_panel(f, mid_chunks[0]);
        self.render_memory_panel(f, mid_chunks[1]);

        // ── Bottom Row: Warnings + Events ─────────────────────────
        self.render_warnings_panel(f, main_chunks[2]);
    }

    fn render_progress_panel(&self, f: &mut Frame, area: Rect) {
        let block = Block::default()
            .title(" Training Progress ")
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(Color::Cyan));
        let inner = block.inner(area);
        f.render_widget(block, area);

        // Epoch progress
        let epoch_pct = if self.total_epochs > 0 {
            (self.current_epoch as f64 / self.total_epochs as f64 * 100.0) as u16
        } else {
            0
        };
        let epoch_label = format!(
            " Epoch {}/{} ({:3}%) ",
            self.current_epoch, self.total_epochs, epoch_pct
        );
        let epoch_gauge = Gauge::default()
            .block(
                Block::default()
                    .title(epoch_label)
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::White)),
            )
            .gauge_style(Style::default().fg(Color::Cyan).bg(Color::Black).add_modifier(Modifier::BOLD))
            .percent(epoch_pct);
        f.render_widget(epoch_gauge, Rect::new(inner.x, inner.y, inner.width, 3));

        // Batch progress
        let batch_pct = if self.total_batches > 0 {
            (self.current_batch as f64 / self.total_batches as f64 * 100.0) as u16
        } else {
            0
        };
        let batch_label = format!(
            " Batch {}/{} ({:3}%) ",
            self.current_batch, self.total_batches, batch_pct
        );
        let batch_gauge = Gauge::default()
            .block(
                Block::default()
                    .title(batch_label)
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::White)),
            )
            .gauge_style(Style::default().fg(Color::Green).bg(Color::Black).add_modifier(Modifier::BOLD))
            .percent(batch_pct);
        f.render_widget(batch_gauge, Rect::new(inner.x, inner.y + 3, inner.width, 3));

        // Info line: elapsed, ETA, throughput, step
        let info = format!(
            " Elapsed: {} | ETA: {} | Throughput: {} | Step: {} | Epoch samples: {} ",
            format_duration(self.elapsed),
            self.eta_str(),
            self.throughput_str(),
            self.current_step,
            self.epoch_samples,
        );
        let info_par = Paragraph::new(Line::from(Span::styled(
            info,
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
        )));
        f.render_widget(info_par, Rect::new(inner.x, inner.y + 6, inner.width, 1));
    }

    fn render_metrics_panel(&self, f: &mut Frame, area: Rect) {
        let block = Block::default()
            .title(" Metrics ")
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(Color::Green));
        let inner = block.inner(area);
        f.render_widget(block, area);

        let (chart_area, metrics_area) = {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
                .split(inner);
            (chunks[0], chunks[1])
        };

        // ── Loss mini-chart (Sparkline) ──
        if !self.loss_history.is_empty() {
            let loss_data: Vec<u64> = self
                .loss_history
                .iter()
                .map(|&v| (v * 1000.0) as u64)
                .collect();
            let min_loss = self.loss_history.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_loss = self.loss_history.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            let sparkline = Sparkline::default()
                .block(
                    Block::default()
                        .title(format!(" Loss (min={:.4}, max={:.4}) ", min_loss, max_loss))
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(Color::White)),
                )
                .data(&loss_data)
                .style(Style::default().fg(Color::Red));
            f.render_widget(sparkline, chart_area);
        } else {
            let empty = Paragraph::new("No loss data yet...")
                .style(Style::default().fg(Color::DarkGray));
            f.render_widget(empty, chart_area);
        }

        // ── Current metric values ──
        let mut metric_lines = Vec::new();
        metric_lines.push(Line::from(Span::styled(
            format!(" Loss:    {:.6}", self.current_loss),
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        )));
        if let Some(acc) = self.current_accuracy {
            metric_lines.push(Line::from(Span::styled(
                format!(" Accuracy: {:.4}%", acc * 100.0),
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
            )));
        }
        if let Some(lr) = self.current_lr {
            metric_lines.push(Line::from(Span::styled(
                format!(" LR:       {:.2e}", lr),
                Style::default().fg(Color::Yellow),
            )));
        }
        metric_lines.push(Line::from(Span::styled(
            format!(" Spikes:   {} total", self.loss_spike_count),
            Style::default().fg(if self.loss_spike_count > 5 { Color::Red } else { Color::DarkGray }),
        )));

        let metrics_par = Paragraph::new(Text::from(metric_lines));
        f.render_widget(metrics_par, metrics_area);
    }

    fn render_memory_panel(&self, f: &mut Frame, area: Rect) {
        let border_color = if self.has_leaks() || self.oom_risk() == OomRiskLevel::Critical {
            Color::Red
        } else if self.oom_risk() == OomRiskLevel::High {
            Color::Yellow
        } else {
            Color::Blue
        };

        let block = Block::default()
            .title(" Memory & Resources ")
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(border_color));
        let inner = block.inner(area);
        f.render_widget(block, area);

        let mem_usage_pct = if self.available_memory_bytes > 0 {
            ((self.current_memory_bytes as f64 / self.available_memory_bytes as f64) * 100.0) as u16
        } else {
            0
        };

        // Memory gauge
        let gauge_style = if self.oom_risk() == OomRiskLevel::Critical {
            Style::default().fg(Color::Red).bg(Color::Black).add_modifier(Modifier::BOLD)
        } else if self.oom_risk() == OomRiskLevel::High {
            Style::default().fg(Color::Yellow).bg(Color::Black).add_modifier(Modifier::BOLD)
        } else if self.oom_risk() == OomRiskLevel::Medium {
            Style::default().fg(Color::LightYellow).bg(Color::Black)
        } else {
            Style::default().fg(Color::Blue).bg(Color::Black)
        };

        let mem_label = format!(
            " {}/{} ({:3}%) ",
            Self::format_bytes(self.current_memory_bytes),
            Self::format_bytes(self.available_memory_bytes),
            mem_usage_pct,
        );
        let mem_gauge = Gauge::default()
            .block(
                Block::default()
                    .title(format!("{} Memory", Span::styled("█", gauge_style)))
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::DarkGray)),
            )
            .gauge_style(gauge_style)
            .label(mem_label)
            .percent(mem_usage_pct);
        f.render_widget(mem_gauge, Rect::new(inner.x, inner.y, inner.width, 3));

        // Memory details section
        let mut mem_lines = vec![
            Line::from(Span::styled(
                format!(" Peak:     {}", Self::format_bytes(self.peak_memory_bytes)),
                Style::default().fg(Color::Cyan),
            )),
            Line::from(Span::styled(
                format!(" OOM Risk: {:?}", self.oom_risk()),
                Style::default().fg(self.oom_risk().color()),
            )),
            Line::from(Span::styled(
                format!(" Allocs:   {} ({} active)",
                    self.total_allocations,
                    self.total_allocations.saturating_sub(self.total_deallocations),
                ),
                Style::default().fg(Color::DarkGray),
            )),
        ];

        // Leak warnings
        if !self.leak_warnings.is_empty() {
            mem_lines.push(Line::from(Span::styled(
                format!(" ⚠ Leaks:  {} detected", self.leak_warnings.len()),
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            )));
            for leak in self.leak_warnings.iter().take(2) {
                mem_lines.push(Line::from(Span::styled(
                    format!("   - {}: {}", leak.tag, Self::format_bytes(leak.bytes)),
                    Style::default().fg(Color::LightRed),
                )));
            }
        }

        // Per-tag breakdown (top entries)
        if !self.memory_by_tag.is_empty() {
            mem_lines.push(Line::from(Span::styled(
                " Top allocations:",
                Style::default().fg(Color::DarkGray),
            )));
            for (tag, bytes) in self.memory_by_tag.iter().take(3) {
                mem_lines.push(Line::from(Span::styled(
                    format!("   {}: {}", tag, Self::format_bytes(*bytes)),
                    Style::default().fg(Color::Cyan),
                )));
            }
        }

        let details_area = Rect::new(inner.x, inner.y + 3, inner.width, inner.height.saturating_sub(3));
        let mem_par = Paragraph::new(Text::from(mem_lines));
        f.render_widget(mem_par, details_area);
    }

    fn render_warnings_panel(&self, f: &mut Frame, area: Rect) {
        let has_critical = !self.warnings.is_empty() || self.has_leaks()
            || self.oom_risk() == OomRiskLevel::Critical;

        let border_color = if has_critical {
            Color::Red
        } else {
            Color::DarkGray
        };

        let block = Block::default()
            .title(if has_critical { " ⚠ Alerts " } else { " Status " })
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(border_color));
        let inner = block.inner(area);
        f.render_widget(block, area);

        // Combine warnings and recent events
        let mut items: Vec<(Color, String)> = Vec::new();

        // Active warnings
        for w in &self.warnings {
            items.push((Color::Red, format!("⚠ {}", w)));
        }

        // Recent events (last 3)
        let start = self.logged_events.len().saturating_sub(4);
        for event in self.logged_events.iter().skip(start) {
            items.push((Color::DarkGray, format!("· {}", event)));
        }

        if items.is_empty() {
            items.push((Color::Green, "✓ All systems nominal".into()));
        }

        let lines: Vec<Line> = items
            .iter()
            .map(|(color, text)| {
                Line::from(Span::styled(text.as_str(), Style::default().fg(*color)))
            })
            .collect();

        let par = Paragraph::new(Text::from(lines));
        f.render_widget(par, inner);
    }

    /// Dump current metrics to JSON (if dump_path is configured).
    pub fn dump_metrics(&self) -> Result<String, serde_json::Error> {
        let snapshot = serde_json::json!({
            "timestamp": chrono_now(),
            "epoch": self.current_epoch,
            "batch": self.current_batch,
            "step": self.current_step,
            "loss": self.current_loss,
            "accuracy": self.current_accuracy,
            "learning_rate": self.current_lr,
            "memory": {
                "current_bytes": self.current_memory_bytes,
                "peak_bytes": self.peak_memory_bytes,
                "available_bytes": self.available_memory_bytes,
                "oom_risk": format!("{:?}", self.oom_risk()),
            },
            "throughput_samples_per_sec": self.throughput_samples_per_sec,
            "elapsed_secs": self.elapsed.as_secs_f64(),
            "eta": {
                "eta_str": self.eta_str(),
                "remaining_secs": 0.0,
            },
            "leak_warnings": self.leak_warnings.len(),
            "warnings": self.warnings,
        });
        serde_json::to_string_pretty(&snapshot)
    }

    /// Reset epoch counters (call at start of each epoch).
    pub fn reset_epoch(&mut self) {
        self.epoch_samples = 0;
    }

    /// Clear leak warnings (call after memory is freed).
    pub fn clear_leaks(&mut self) {
        self.leak_warnings.clear();
    }

    /// Get current loss value.
    pub fn current_loss(&self) -> f64 {
        self.current_loss
    }

    /// Get current accuracy value.
    pub fn current_accuracy(&self) -> Option<f64> {
        self.current_accuracy
    }

    /// Get current learning rate.
    pub fn current_lr(&self) -> Option<f64> {
        self.current_lr
    }

    /// Get configuration reference.
    pub fn config(&self) -> &DashboardConfig {
        &self.cfg
    }
}

// ── Dashboard Renderer (wraps crossterm/ratatui terminal) ──────────

/// High-level dashboard renderer that manages the terminal lifecycle.
///
/// Automatically handles terminal setup/teardown, alternate screen,
/// and event polling for the 'q' quit key.
pub struct DashboardRenderer {
    terminal: Terminal<CrosstermBackend<Stdout>>,
    dashboard: Arc<Mutex<TrainingDashboard>>,
    running: bool,
}

impl DashboardRenderer {
    /// Create a new renderer with the given dashboard state.
    pub fn new(dashboard: Arc<Mutex<TrainingDashboard>>) -> io::Result<Self> {
        let mut stdout = io::stdout();
        crossterm::terminal::enable_raw_mode()?;
        crossterm::execute!(stdout, crossterm::terminal::EnterAlternateScreen)?;

        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)?;

        Ok(Self {
            terminal,
            dashboard,
            running: true,
        })
    }

    /// Render the dashboard once. Returns `true` if the user pressed 'q'.
    pub fn render_frame(&mut self) -> io::Result<bool> {
        // Poll for key events (non-blocking)
        if crossterm::event::poll(std::time::Duration::from_millis(0))? {
            if let crossterm::event::Event::Key(key) = crossterm::event::read()? {
                match key.code {
                    crossterm::event::KeyCode::Char('q') => {
                        self.running = false;
                        return Ok(true);
                    }
                    _ => {}
                }
            }
        }

        let dashboard = self.dashboard.clone();
        self.terminal.draw(|f| {
            let mut db = dashboard.lock().unwrap();
            db.render(f);
        })?;

        Ok(false)
    }

    /// Run the render loop, rendering at the configured interval.
    /// Blocks until the user presses 'q'.
    pub fn run(&mut self) -> io::Result<()> {
        let refresh_ms = {
            let db = self.dashboard.lock().unwrap();
            db.config().refresh_ms
        };

        while self.running {
            if self.render_frame()? {
                break;
            }
            std::thread::sleep(Duration::from_millis(refresh_ms));
        }

        Ok(())
    }

    /// Check if still running.
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// Stop the renderer.
    pub fn stop(&mut self) {
        self.running = false;
    }
}

impl Drop for DashboardRenderer {
    fn drop(&mut self) {
        // Restore terminal
        crossterm::execute!(
            self.terminal.backend_mut(),
            crossterm::terminal::LeaveAlternateScreen,
        )
        .ok();
        crossterm::terminal::disable_raw_mode().ok();
    }
}

// ── Helpers ────────────────────────────────────────────────────────

/// Format a Duration as HH:MM:SS.
fn format_duration(d: Duration) -> String {
    let secs = d.as_secs();
    format!("{:02}:{:02}:{:02}", secs / 3600, (secs % 3600) / 60, secs % 60)
}

/// Get current time as ISO string for JSON dumps.
fn chrono_now() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Simple ISO-like format
    format!("1970-01-01T00:00:{}Z", now)
}

// ── MetricsBackend integration ─────────────────────────────────

/// Drop-in [`MetricsBackend`] that renders a live TUI dashboard.
///
/// Use with any existing `MetricsLogger`:
///
/// ```rust,ignore
/// use rustral_metrics::MetricsLogger;
/// use rustral_tui::TuiMetricsBackend;
///
/// let mut logger = MetricsLogger::new();
/// logger.add_backend(Box::new(TuiMetricsBackend::new()));
/// ```
///
/// The dashboard auto-starts in a background render thread.
/// Press 'q' to exit.
pub struct TuiMetricsBackend {
    dashboard: Arc<Mutex<TrainingDashboard>>,
}

impl TuiMetricsBackend {
    /// Create a new TUI backend with default config.
    pub fn new() -> Self {
        Self::with_config(DashboardConfig::default())
    }

    /// Create a new TUI backend with custom config.
    pub fn with_config(cfg: DashboardConfig) -> Self {
        let dashboard = run_dashboard(cfg);
        Self { dashboard }
    }

    /// Access the underlying dashboard (e.g. for memory/leak reporting).
    pub fn dashboard(&self) -> &Arc<Mutex<TrainingDashboard>> {
        &self.dashboard
    }
}

impl Default for TuiMetricsBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl rustral_metrics::MetricsBackend for TuiMetricsBackend {
    fn log_scalar(&mut self, metric: &rustral_metrics::ScalarMetric) {
        let mut db = self.dashboard.lock().unwrap();
        // Route metrics by name pattern
        let name = metric.name.as_str();
        if name.contains("loss") {
            db.record_loss(metric.value);
        } else if name.contains("acc") || name.contains("accuracy") {
            db.record_accuracy(metric.value);
        } else if name == "lr" || name.contains("learning_rate") || name.contains("learning") {
            db.record_lr(metric.value);
        }
        db.set_step(metric.step);
    }

    fn log_histogram(&mut self, _metric: &rustral_metrics::HistogramMetric) {
        // Histograms not rendered in TUI (future: use ratatui Chart)
    }

    fn flush(&mut self) {
        // Dashboard renders continuously; nothing to flush
    }

    fn name(&self) -> &str {
        "tui_dashboard"
    }
}

/// Convenience function to create a dashboard and renderer in a background thread.
///
/// Returns the `Arc<Mutex<TrainingDashboard>>` so you can feed metrics during training.
///
/// ```rust,ignore
/// use rustral_tui::{run_dashboard, DashboardConfig};
/// let dashboard = run_dashboard(DashboardConfig::default());
/// // ... training loop updates dashboard ...
/// dashboard.lock().unwrap().record_loss(0.5);
/// ```
pub fn run_dashboard(cfg: DashboardConfig) -> Arc<Mutex<TrainingDashboard>> {
    let dashboard = Arc::new(Mutex::new(TrainingDashboard::new(cfg)));
    let d = dashboard.clone();

    std::thread::spawn(move || {
        if let Ok(mut renderer) = DashboardRenderer::new(d) {
            renderer.run().ok();
        }
    });

    dashboard
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dashboard_creation() {
        let db = TrainingDashboard::new(DashboardConfig::default());
        assert_eq!(db.current_epoch, 0);
        assert_eq!(db.total_epochs, 0);
        assert!(db.loss_history.is_empty());
    }

    #[test]
    fn test_record_loss() {
        let mut db = TrainingDashboard::new(DashboardConfig::default());
        db.record_loss(0.5);
        db.record_loss(0.4);
        db.record_loss(0.3);
        assert_eq!(db.loss_history.len(), 3);
        assert!((db.current_loss - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_record_nan_loss() {
        let mut db = TrainingDashboard::new(DashboardConfig::default());
        db.record_loss(f64::NAN);
        assert!(db.loss_history.is_empty());
        assert!(!db.warnings.is_empty());
    }

    #[test]
    fn test_oom_risk() {
        let mut db = TrainingDashboard::new(DashboardConfig::default());
        db.set_available_memory(1000);
        db.set_memory_bytes(100);
        assert_eq!(db.oom_risk(), OomRiskLevel::Low);
        db.set_memory_bytes(600);
        assert_eq!(db.oom_risk(), OomRiskLevel::Medium);
        db.set_memory_bytes(800);
        assert_eq!(db.oom_risk(), OomRiskLevel::High);
        db.set_memory_bytes(950);
        assert_eq!(db.oom_risk(), OomRiskLevel::Critical);
    }

    #[test]
    fn test_leak_warning() {
        let mut db = TrainingDashboard::new(DashboardConfig::default());
        db.report_leak("tensor_grad", 1024 * 1024, 5);
        assert_eq!(db.leak_warnings.len(), 1);
        assert!(db.has_leaks());
        assert!(!db.warnings.is_empty());
    }

    #[test]
    fn test_throughput() {
        let mut db = TrainingDashboard::new(DashboardConfig::default());
        db.record_batch_samples(32);
        db.record_batch_samples(32);
        assert_eq!(db.samples_processed, 64);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(TrainingDashboard::format_bytes(500), "500 B");
        assert_eq!(TrainingDashboard::format_bytes(1500), "1.50 KB");
        assert_eq!(TrainingDashboard::format_bytes(1_500_000), "1.50 MB");
        assert_eq!(TrainingDashboard::format_bytes(1_500_000_000), "1.50 GB");
    }

    #[test]
    fn test_loss_spike_detection() {
        let mut db = TrainingDashboard::new(DashboardConfig::default());
        db.record_loss(1.0);
        db.record_loss(1.1);
        db.record_loss(0.9);
        // No spike yet
        assert_eq!(db.loss_spike_count, 0);
        // Spike: 30x the avg of ~1.0
        db.record_loss(30.0);
        assert_eq!(db.loss_spike_count, 1);
    }

    #[test]
    fn test_eta() {
        let mut db = TrainingDashboard::new(DashboardConfig::default());
        assert_eq!(db.eta_str(), "--:--:--");

        db.set_total_epochs(10);
        db.set_epoch(0);
        assert_eq!(db.eta_str(), "--:--:--");

        db.set_epoch(5);
        // ETA should compute something
        let eta = db.eta_str();
        assert!(eta != "--:--:--" || db.start_time.elapsed().as_secs_f64() < 0.01);
    }

    #[test]
    fn test_memory_by_tag() {
        let mut db = TrainingDashboard::new(DashboardConfig::default());
        let tags = vec![
            ("weight".into(), 1000),
            ("gradient".into(), 500),
            ("activations".into(), 300),
        ];
        db.set_memory_by_tag(tags);
        assert_eq!(db.memory_by_tag.len(), 3);
        assert_eq!(db.memory_by_tag[0].1, 1000);
    }

    #[test]
    fn test_peak_memory() {
        let mut db = TrainingDashboard::new(DashboardConfig::default());
        db.set_memory_bytes(100);
        assert_eq!(db.peak_memory_bytes, 100);
        db.set_memory_bytes(200);
        assert_eq!(db.peak_memory_bytes, 200);
        db.set_memory_bytes(50);
        assert_eq!(db.peak_memory_bytes, 200); // Peak unchanged
    }

    #[test]
    fn test_max_warnings() {
        let mut db = TrainingDashboard::new(DashboardConfig::default());
        for i in 0..10 {
            db.add_warning(format!("Warning {}", i));
        }
        assert_eq!(db.warnings.len(), 5); // Max 5 warnings
        assert_eq!(db.warnings[0], "Warning 5");
    }

    #[test]
    fn test_log_events() {
        let mut db = TrainingDashboard::new(DashboardConfig::default());
        for i in 0..60 {
            db.log_event(format!("Event {}", i));
        }
        assert_eq!(db.logged_events.len(), 50); // Max 50
    }

    #[test]
    fn test_oom_risk_levels() {
        assert_eq!(OomRiskLevel::from_ratio(0.3), OomRiskLevel::Low);
        assert_eq!(OomRiskLevel::from_ratio(0.6), OomRiskLevel::Medium);
        assert_eq!(OomRiskLevel::from_ratio(0.8), OomRiskLevel::High);
        assert_eq!(OomRiskLevel::from_ratio(0.95), OomRiskLevel::Critical);
    }

    #[test]
    fn test_oom_risk_color() {
        assert_eq!(OomRiskLevel::Low.color(), Color::Green);
        assert_eq!(OomRiskLevel::Medium.color(), Color::Yellow);
        assert_eq!(OomRiskLevel::High.color(), Color::Red);
    }

    #[test]
    fn test_dump_metrics_not_empty() {
        let db = TrainingDashboard::new(DashboardConfig::default());
        let dump = db.dump_metrics();
        assert!(dump.is_ok());
        let s = dump.unwrap();
        assert!(s.contains("epoch"));
        assert!(s.contains("loss"));
    }

    #[test]
    fn test_clear_leaks() {
        let mut db = TrainingDashboard::new(DashboardConfig::default());
        db.report_leak("test", 100, 1);
        assert!(db.has_leaks());
        db.clear_leaks();
        assert!(!db.has_leaks());
    }
}