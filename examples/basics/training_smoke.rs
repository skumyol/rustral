//! Serious training smoke test: synthetic classification on **Candle** (CUDA if built with `--features cuda`, else CPU).
//!
//! ```text
//! cargo run -p rustral-examples --bin serious_train
//! cargo run -p rustral-examples --bin serious_train --features cuda
//! ```
//!
//! Runs long enough to demonstrate loss decreasing and optional checkpoint round-trip.

use rustral_candle_backend::CandleBackend;
use rustral_runtime::{SeriousTrainingConfig, SeriousTrainingOutcome, train_synthetic_classification};

fn pick_backend() -> CandleBackend {
    #[cfg(feature = "cuda")]
    {
        match CandleBackend::cuda(0) {
            Ok(b) => {
                eprintln!("Using Candle CUDA device 0.");
                b
            }
            Err(e) => {
                eprintln!("CUDA unavailable ({}), falling back to CPU.", e);
                CandleBackend::cpu()
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("Built without `cuda` feature; using Candle CPU. Rebuild with `--features cuda` for GPU.");
        CandleBackend::cpu()
    }
}

fn main() -> anyhow::Result<()> {
    let backend = pick_backend();

    let checkpoint_dir = std::env::temp_dir().join(format!("rustral_serious_ckpt_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&checkpoint_dir);
    std::fs::create_dir_all(&checkpoint_dir)?;

    let mut cfg = SeriousTrainingConfig::default();
    cfg.epochs = 500;
    cfg.checkpoint_dir = Some(checkpoint_dir.clone());

    println!("Serious training (synthetic classification)");
    println!("==========================================");
    let SeriousTrainingOutcome { loss_start, loss_end, checkpoint_roundtrip_ok } =
        train_synthetic_classification(&backend, cfg)?;

    println!("loss_start = {:.6}", loss_start);
    println!("loss_end   = {:.6}", loss_end);
    println!("checkpoint round-trip: {}", checkpoint_roundtrip_ok);
    assert!(loss_end < loss_start, "training did not reduce loss");
    assert!(checkpoint_roundtrip_ok);

    let _ = std::fs::remove_dir_all(&checkpoint_dir);
    Ok(())
}
