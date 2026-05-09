//! WikiText-2 LM overfit on a tiny token prefix: train perplexity should collapse.
//!
//! Run with: `cargo test -p rustral-runtime --features training --test wikitext2_overfit -- --include-ignored`
//!
//! Note: `dev_perplexity` stays high (train/dev vocab mismatch); we assert on `diagnostics.train_perplexity`.

#![cfg(feature = "training")]

use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

fn synth_corpus() -> String {
    let words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "cat", "sat", "on", "mat", "and",
        "ran", "away", "from", "rain",
    ];
    let mut s = String::new();
    for i in 0..2_000 {
        s.push_str(words[i % words.len()]);
        s.push(' ');
        if i % 13 == 0 {
            s.push('\n');
        }
    }
    s
}

#[test]
#[ignore]
fn wikitext2_overfit_tiny_train_perplexity_low() -> anyhow::Result<()> {
    let tmp = tempfile_named("wt2-overfit")?;
    let cache = tmp.join("cache");
    let out = tmp.join("out");
    let dir = cache.join("datasets/wikitext-2");
    fs::create_dir_all(&dir)?;
    let corpus = synth_corpus();
    fs::write(dir.join("train.txt"), &corpus)?;
    fs::write(dir.join("valid.txt"), &corpus)?;
    fs::write(dir.join("test.txt"), &corpus)?;

    let status = Command::new(env!("CARGO"))
        .args([
            "run",
            "--release",
            "-p",
            "rustral-runtime",
            "--features",
            "training",
            "--example",
            "wikitext2_lm",
            "--",
            "--overfit-tiny",
            "--epochs",
            "100",
            "--train-tokens",
            "48",
            "--block-size",
            "8",
            "--batch",
            "4",
            "--eval-windows",
            "512",
            "--out-dir",
        ])
        .arg(&out)
        .env("RUSTRAL_CACHE_DIR", &cache)
        .env("RUSTRAL_DATASET_OFFLINE", "1")
        .env("RUSTRAL_DATASET_SKIP_CHECKSUM", "1")
        .current_dir(repo_root())
        .status()?;
    assert!(status.success(), "wikitext2_lm overfit exited {:?}", status.code());

    let manifest_raw = fs::read_to_string(out.join("manifest.json"))?;
    let v: serde_json::Value = serde_json::from_str(&manifest_raw)?;
    let train_ppl = v["diagnostics"]["train_perplexity"].as_f64().expect("diagnostics.train_perplexity");
    assert!(train_ppl < 5.0, "expected train subset perplexity < 5 after overfit, got {train_ppl}");
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
