//! Native TUI dashboard integration for `TapeTrainer`.
//!
//! When the `tui` feature is enabled, `TapeTrainer::fit()` and
//! `fit_classification()` automatically spawn a live terminal dashboard
//! in a background thread. No user code changes needed — the dashboard
//! just appears when you run any training example.
//!
//! The global dashboard is initialized once (lazily). The trainer feeds
//! epoch/batch/loss/accuracy/step updates into it automatically.

use std::sync::{Arc, Mutex, OnceLock};

use rustral_tui::{DashboardConfig, DashboardRenderer, TrainingDashboard};

/// Global dashboard that trainers push metrics into.
/// Once initialized, all training runs share the same TUI.
static GLOBAL_DASHBOARD: OnceLock<Arc<Mutex<TrainingDashboard>>> = OnceLock::new();

/// Whether the TUI was already activated this process.
static TUI_ACTIVATED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Initialize the global TUI dashboard (called once).
pub fn init_global_dashboard() {
    use std::sync::atomic::Ordering;
    if TUI_ACTIVATED.load(Ordering::Relaxed) {
        return;
    }
    TUI_ACTIVATED.store(true, Ordering::Relaxed);

    let mut cfg = DashboardConfig::default();
    cfg.title = "Rustral Training".into();
    cfg.alt_screen = true;

    let dashboard = Arc::new(Mutex::new(TrainingDashboard::new(cfg)));

    // Set global so trainers can push metrics.
    let _ = GLOBAL_DASHBOARD.set(dashboard.clone());

    // Spawn the renderer in a background thread.
    std::thread::Builder::new()
        .name("tui-dashboard".into())
        .spawn(move || {
            if let Ok(mut renderer) = DashboardRenderer::new(dashboard) {
                let _ = renderer.run();
            }
        })
        .ok();
}

/// Get a reference to the global dashboard (returns None if not initialized).
pub fn dashboard() -> Option<&'static Arc<Mutex<TrainingDashboard>>> {
    GLOBAL_DASHBOARD.get()
}

/// Snapshot of per-epoch training state to push into the dashboard.
pub struct EpochSnapshot {
    pub epoch: u64,
    pub total_epochs: u64,
    pub mean_loss: f32,
    pub accuracy: Option<f32>,
    pub step: u64,
}

impl EpochSnapshot {
    /// Push this snapshot into the global dashboard (no-op if not initialized).
    pub fn push_to_dashboard(&self) {
        if let Some(dash) = dashboard() {
            let mut db = dash.lock().unwrap();
            db.set_total_epochs(self.total_epochs);
            db.set_epoch(self.epoch);
            db.set_step(self.step);
            db.record_loss(self.mean_loss as f64);
            if let Some(acc) = self.accuracy {
                db.record_accuracy(acc as f64);
            }
        }
    }
}