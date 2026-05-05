use crate::common::TestRunner;
use std::process::Command;
use std::time::Instant;

pub fn benchmark_examples_smoke(runner: &mut TestRunner) {
    runner.run_test("perf_examples_smoke", || {
        if std::env::var("RUSTRAL_RUN_EXAMPLE_PERF").ok().as_deref() != Some("1") {
            println!("  Skipped. Set RUSTRAL_RUN_EXAMPLE_PERF=1 to run example binaries.");
            return Ok(());
        }

        // Keep this as a smoke/perf gate, not a full training marathon:
        // - Many examples are educational and may do long loops / heavy allocations.
        // - We only validate they start, run, and finish within a reasonable envelope.
        let examples_dir = repo_root()?.join("examples");

        // Basic binaries
        let bins = ["xor", "train_demo", "mnist", "char_rnn", "serious_train"];
        for bin in bins {
            run_cargo_example(&examples_dir, &["run", "--quiet", "--bin", bin], bin, 60)?;
        }

        // Example targets (cargo --example)
        let exs = [
            "building_blocks",
            "resnet_image_classification",
            "diffusion_model",
            "bert_fine_tuning",
            "gpt_training",
            "moe_training",
            "custom_layer",
        ];
        for ex in exs {
            run_cargo_example(&examples_dir, &["run", "--quiet", "--example", ex], ex, 120)?;
        }

        Ok(())
    });
}

fn run_cargo_example(dir: &std::path::Path, args: &[&str], label: &str, max_secs: u64) -> Result<(), String> {
    let start = Instant::now();
    let output = Command::new("cargo")
        .args(args)
        .current_dir(dir)
        .output()
        .map_err(|e| format!("Failed to spawn cargo for {}: {}", label, e))?;

    let elapsed = start.elapsed();
    if elapsed.as_secs() > max_secs {
        return Err(format!(
            "Example '{}' exceeded {}s (took {:.1}s)",
            label,
            max_secs,
            elapsed.as_secs_f64()
        ));
    }

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Example '{}' failed (exit={:?}):\n{}", label, output.status.code(), stderr));
    }

    println!("  Example '{}' ok in {:.2}s", label, elapsed.as_secs_f64());
    Ok(())
}

fn repo_root() -> Result<std::path::PathBuf, String> {
    // `cargo test --test system_tests` sets CWD to the crate root for this package.
    // But make this robust when run from subdirs.
    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    for p in cwd.ancestors() {
        if p.join("Cargo.toml").exists() && p.join("crates").exists() && p.join("examples").exists() {
            return Ok(p.to_path_buf());
        }
    }
    Err("Could not locate repository root".into())
}
