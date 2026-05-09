//! SST-2 tiny-set overfit (masked pooling + tape training sanity).
//!
//! Run with: `cargo test -p rustral-runtime --features training --test sst2_overfit -- --include-ignored`

#![cfg(feature = "training")]

use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

fn synth_tsv() -> String {
    let mut s = String::from("sentence\tlabel\n");
    // `--overfit-32` truncates to 32 rows; provide 40 so truncation is stable.
    for i in 0..40 {
        let label = if i % 2 == 0 { "1" } else { "0" };
        s.push_str(&format!("repeat token seq {i} words fill token seq {i} longer\t{label}\n"));
    }
    s
}

#[test]
#[ignore]
fn sst2_overfit_32_train_accuracy_high() -> anyhow::Result<()> {
    let tmp = tempfile_named("sst2-overfit")?;
    let cache = tmp.join("cache");
    let out = tmp.join("out");
    fs::create_dir_all(cache.join("datasets/sst2"))?;
    let tsv = synth_tsv();
    fs::write(cache.join("datasets/sst2/train.tsv"), &tsv)?;
    fs::write(cache.join("datasets/sst2/dev.tsv"), &tsv)?;

    let status = Command::new(env!("CARGO"))
        .args([
            "run",
            "--release",
            "-p",
            "rustral-runtime",
            "--features",
            "training",
            "--example",
            "sst2_classifier",
            "--",
            "--overfit-32",
            "--epochs",
            "70",
            "--batch",
            "8",
            "--lr",
            "0.001",
            "--out-dir",
        ])
        .arg(&out)
        .env("RUSTRAL_CACHE_DIR", &cache)
        .env("RUSTRAL_DATASET_OFFLINE", "1")
        .env("RUSTRAL_DATASET_SKIP_CHECKSUM", "1")
        .current_dir(repo_root())
        .status()?;
    assert!(status.success(), "sst2_classifier overfit exited {:?}", status.code());

    let manifest_raw = fs::read_to_string(out.join("manifest.json"))?;
    let v: serde_json::Value = serde_json::from_str(&manifest_raw)?;
    let acc = v["dev_accuracy"].as_f64().expect("dev_accuracy");
    assert!(
        acc >= 0.95,
        "expected train-eval accuracy >= 0.95 after overfit, got {acc} (manifest snippet in CI log)"
    );
    Ok(())
}

fn tempfile_named(prefix: &str) -> std::io::Result<PathBuf> {
    let base = std::env::temp_dir().join(format!(
        "rustral-{prefix}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0)
    ));
    fs::create_dir_all(&base)?;
    Ok(base)
}
