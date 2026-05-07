//! Training Dashboard Example
//!
//! Demonstrates a simulated training loop with the full TUI dashboard.
//! Shows all features: progress bars, live metrics, memory monitoring,
//! leak detection, OOM risk, and throughput tracking.
//!
//! Run with:
//! ```bash
//! cargo run -p rustral-tui --example train_dashboard
//! ```

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use rustral_tui::{DashboardConfig, TrainingDashboard};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── Setup dashboard ───────────────────────────────────────────
    let mut cfg = DashboardConfig::default();
    cfg.title = "Rustral Training Demo".into();
    cfg.history_len = 80;

    let dashboard = Arc::new(Mutex::new(TrainingDashboard::new(cfg)));
    let render_dashboard = dashboard.clone();

    // ── Launch TUI renderer in background thread ──────────────────
    std::thread::spawn(move || {
        use rustral_tui::DashboardRenderer;
        if let Ok(mut renderer) = DashboardRenderer::new(render_dashboard) {
            renderer.run().ok();
        }
    });

    // Give the renderer a moment to start
    std::thread::sleep(Duration::from_millis(500));

    // ── Simulated training configuration ──────────────────────────
    let total_epochs = 50;
    let batches_per_epoch = 100;
    let batch_size = 32;
    let total_steps = total_epochs * batches_per_epoch;

    {
        let mut db = dashboard.lock().unwrap();
        db.set_total_epochs(total_epochs);
        db.set_total_batches(batches_per_epoch);
        db.set_available_memory(8_000_000_000); // 8 GB
    }

    println!("Training started! Press 'q' in the terminal to quit.\n");

    // ── Simulated training loop ───────────────────────────────────
    let mut step = 0u64;
    let start = Instant::now();

    for epoch in 0..total_epochs {
        {
            let mut db = dashboard.lock().unwrap();
            db.set_epoch(epoch);
            db.reset_epoch();
        }

        for batch in 0..batches_per_epoch {
            // Simulate forward/backward pass time
            std::thread::sleep(Duration::from_millis(15));

            // Simulate metrics
            let loss = 2.0 * (-0.05 * step as f64).exp() + 0.1 * rand_f64();
            let accuracy = 0.5 + 0.4 * (1.0 - (-0.04 * step as f64).exp()) + 0.02 * rand_f64();
            let lr = 0.01 * (0.95_f64).powf(step as f64);

            // Simulate memory: grows then plateaus (like real training)
            let base_mem = 256_000_000; // 256 MB base
            let grad_mem = if step > 100 { 512_000_000 } else { step as usize * 5_000_000 };
            let mem = base_mem + grad_mem.min(512_000_000);

            // Simulate memory tags
            let tags = vec![
                ("weights".into(), base_mem / 2),
                ("gradients".into(), grad_mem.min(512_000_000) / 2),
                ("activations".into(), base_mem / 4),
            ];

            // Simulate allocations (growing then stable)
            let total_allocs = step as usize * 10;
            let total_deallocs = if step > 50 { (step - 50) as usize * 10 } else { 0 };

            // Simulate a memory leak around step 500
            let mut leak_warn = None;
            if step == 500 {
                leak_warn = Some(("tensor_cache".to_string(), 50_000_000, 3));
            }

            {
                let mut db = dashboard.lock().unwrap();
                db.set_step(step);
                db.set_batch(batch);
                db.record_loss(loss);
                db.record_accuracy(accuracy);
                db.record_lr(lr);
                db.record_batch_samples(batch_size);
                db.set_memory_bytes(mem);
                db.set_peak_memory(768_000_000);
                db.set_memory_by_tag(tags);
                db.set_allocation_counts(total_allocs, total_deallocs);

                if let Some((tag, bytes, count)) = leak_warn {
                    db.report_leak(&tag, bytes, count);
                    db.add_warning("Tensor cache not freed after backward pass".into());
                }

                // Simulate OOM risk scenario at step 700
                if step == 700 {
                    db.set_memory_bytes(7_200_000_000); // 90% of 8GB
                    db.add_warning("Critical memory pressure, considering gradient checkpointing".into());
                }

                // Simulate NaN loss at step 800
                if step == 800 {
                    db.record_loss(f64::NAN);
                }
            }

            step += 1;
        }
    }

    // ── Training complete ─────────────────────────────────────────
    let elapsed = start.elapsed();
    let mut db = dashboard.lock().unwrap();

    println!("\nTraining completed in {:.1}s", elapsed.as_secs_f64());
    println!("Total steps: {}", step);
    println!("Final loss: {:.6}", db.current_loss());
    println!("Final accuracy: {:.4}%", db.current_accuracy().unwrap_or(0.0) * 100.0);

    // Write final JSON dump
    if let Ok(json) = db.dump_metrics() {
        println!("\nFinal metrics snapshot:\n{}", json);
    }

    // Keep the TUI visible for a moment so the user can see final state
    std::thread::sleep(Duration::from_secs(3));

    Ok(())
}

/// Simple random f64 generator (no external dependency needed).
fn rand_f64() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    (nanos as f64) / 1_000_000_000.0
}