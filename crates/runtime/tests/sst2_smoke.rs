//! Smoke test for the SST-2 classifier example.
//!
//! Pre-stages a tiny synthetic SST-2 TSV in a temporary cache dir, runs the example with
//! `--quick`, and asserts the produced manifest is well-formed and contains the expected
//! fields. This test is offline and fast, so it runs by default.

#![cfg(feature = "training")]

use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

fn synth_tsv() -> &'static str {
    "sentence\tlabel\nthis movie was great\t1\nawful boring waste of time\t0\nthe acting was wonderful and inspired\t1\ncompletely terrible\t0\nfun and exciting and amazing\t1\nbad bad bad really bad\t0\nlovely and heartwarming\t1\nawful again ugh\t0\nbrilliant performance throughout\t1\ndisappointing ending\t0\n"
}

#[test]
fn sst2_classifier_runs_offline_quick() -> anyhow::Result<()> {
    let tmp = tempdir()?;
    let cache = tmp.join("cache");
    let out = tmp.join("out");
    fs::create_dir_all(cache.join("datasets/sst2"))?;
    fs::write(cache.join("datasets/sst2/train.tsv"), synth_tsv())?;
    fs::write(cache.join("datasets/sst2/dev.tsv"), synth_tsv())?;

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
            "--quick",
            "--out-dir",
        ])
        .arg(&out)
        .env("RUSTRAL_CACHE_DIR", &cache)
        .env("RUSTRAL_DATASET_OFFLINE", "1")
        .env("RUSTRAL_DATASET_SKIP_CHECKSUM", "1")
        .current_dir(repo_root())
        .status()?;
    assert!(status.success(), "sst2 example exited with {:?}", status.code());

    let manifest = fs::read_to_string(out.join("manifest.json"))?;
    assert!(manifest.contains("\"task\": \"sst2_classifier\""), "task field missing");
    assert!(manifest.contains("\"dev_accuracy\":"), "dev_accuracy missing");
    assert!(manifest.contains("\"git_sha\":"), "git_sha missing");
    assert!(manifest.contains("\"vocab_size\":"), "vocab_size missing");
    assert!(out.join("vocab.txt").exists(), "vocab.txt missing");

    Ok(())
}

fn tempdir() -> std::io::Result<PathBuf> {
    let base = std::env::temp_dir().join(format!(
        "rustral-sst2-smoke-{}-{}",
        std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0)
    ));
    fs::create_dir_all(&base)?;
    Ok(base)
}
