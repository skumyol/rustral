//! Smoke test for the WikiText-2 LM example.
//!
//! Pre-stages tiny synthetic train/valid/test text files and runs the example with
//! `--quick`. Asserts the manifest is JSON-shaped with the expected fields. This test is
//! offline and fast, so it runs by default.

#![cfg(feature = "training")]

use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

fn synth_corpus() -> String {
    let words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "cat", "sat", "on", "mat", "and", "ran", "away", "from", "rain",
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
fn wikitext2_lm_runs_offline_quick() -> anyhow::Result<()> {
    let tmp = tempdir()?;
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
            "--quick",
            "--out-dir",
        ])
        .arg(&out)
        .env("RUSTRAL_CACHE_DIR", &cache)
        .env("RUSTRAL_DATASET_OFFLINE", "1")
        .env("RUSTRAL_DATASET_SKIP_CHECKSUM", "1")
        .current_dir(repo_root())
        .status()?;
    assert!(status.success(), "wikitext2 example exited with {:?}", status.code());

    let manifest = fs::read_to_string(out.join("manifest.json"))?;
    assert!(manifest.contains("\"task\": \"wikitext2_word_lm\""));
    assert!(manifest.contains("\"dev_perplexity\":"));
    assert!(manifest.contains("\"vocab_size\":"));
    assert!(out.join("vocab.txt").exists());

    Ok(())
}

fn tempdir() -> std::io::Result<PathBuf> {
    let base = std::env::temp_dir().join(format!(
        "rustral-wikitext2-smoke-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    fs::create_dir_all(&base)?;
    Ok(base)
}
