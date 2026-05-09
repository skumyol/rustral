//! WikiText-2 loader (raw, word-level).
//!
//! WikiText-2 is canonically reported in two forms: `wikitext-2-v1` (closed-vocabulary,
//! `<unk>`-replaced) and `wikitext-2-raw-v1` (raw text). We default to the raw variant.
//! Tokenization happens in `crate::tokenizer`, so leaving the source raw lets us treat the
//! tokenizer choice as an experiment knob.
//!
//! Pinned by SHA-256; if upstream rehashes the artifact this code fails loudly.
//!
//! In offline mode (`RUSTRAL_DATASET_OFFLINE=1`) the splits are read from
//! `<cache>/datasets/wikitext-2/{train,valid,test}.txt` which CI smoke tests pre-stage.

#![cfg(feature = "fetch")]

use std::fs;
use std::path::PathBuf;

use crate::fetch::{fetch_url, FetchError};

/// Working WikiText-2 raw v1 zip mirror.
///
/// The original `s3.amazonaws.com/research.metamind.io` URL returns 403 today; this
/// `wikitext.smerity.com` mirror serves the same archive and is referenced by both the
/// PyTorch examples repo and the `datasets` library. The zip contains
/// `wikitext-2-raw/wiki.{train,valid,test}.raw`.
pub const WIKITEXT2_RAW_ZIP_URL: &str = "https://wikitext.smerity.com/wikitext-2-raw-v1.zip";
pub const WIKITEXT2_RAW_ZIP_SHA256: &str = "ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11";

/// Three split files ("train", "valid", "test") loaded as full text.
#[derive(Clone, Debug, Default)]
pub struct WikiText2Splits {
    pub train: String,
    pub valid: String,
    pub test: String,
}

/// Fetch the WikiText-2 raw zip (or read it from cache) and decode the three text splits.
///
/// Pure-text loader: the tokenizer is decoupled from the corpus. In offline mode
/// (`RUSTRAL_DATASET_OFFLINE=1`) splits are read from the cache directory laid out as
/// `<cache>/wikitext-2/{train,valid,test}.txt`. CI smoke tests use this layout.
pub fn load_wikitext2() -> Result<WikiText2Splits, FetchError> {
    let cache_dir = crate::fetch::cache_root().join("datasets").join("wikitext-2");
    let train_txt = cache_dir.join("train.txt");
    let valid_txt = cache_dir.join("valid.txt");
    let test_txt = cache_dir.join("test.txt");
    if train_txt.exists() && valid_txt.exists() && test_txt.exists() {
        return Ok(WikiText2Splits {
            train: fs::read_to_string(&train_txt)?,
            valid: fs::read_to_string(&valid_txt)?,
            test: fs::read_to_string(&test_txt)?,
        });
    }

    let zip_path = fetch_url(WIKITEXT2_RAW_ZIP_URL, WIKITEXT2_RAW_ZIP_SHA256, "wikitext-2")?;
    extract_three_splits(&zip_path, &cache_dir)
}

/// Extract the three split files from the WikiText-2 raw zip into `out_dir` as plain text.
///
/// Implementation note: shells out to the `unzip` CLI to keep deps light. `unzip` is
/// available on macOS and Linux runners by default; if it's missing we surface a clear
/// error pointing to manual extraction. Acceptable for an opt-in `fetch`-feature path.
fn extract_three_splits(zip_path: &PathBuf, out_dir: &PathBuf) -> Result<WikiText2Splits, FetchError> {
    fs::create_dir_all(out_dir)?;
    let status = std::process::Command::new("unzip")
        .arg("-o")
        .arg(zip_path)
        .arg("-d")
        .arg(out_dir)
        .status()
        .map_err(|e| FetchError::Http(format!("failed to invoke `unzip`: {e}")))?;
    if !status.success() {
        return Err(FetchError::Http(
            "`unzip` exited with non-zero status; install unzip or pre-stage the splits".to_string(),
        ));
    }
    // The smerity.com mirror lays out files as `wikitext-2-raw/wiki.{train,valid,test}.raw`.
    // Some legacy mirrors used `wikitext-2-raw-v1/`; try both so we are robust to either.
    let candidates = ["wikitext-2-raw", "wikitext-2-raw-v1"];
    let base =
        candidates.iter().map(|d| out_dir.join(d)).find(|p| p.join("wiki.train.raw").exists()).ok_or_else(
            || {
                FetchError::Http(
                    "WikiText-2 zip did not contain expected `wiki.train.raw` under \
                 `wikitext-2-raw/` or `wikitext-2-raw-v1/`"
                        .to_string(),
                )
            },
        )?;
    let train = fs::read_to_string(base.join("wiki.train.raw"))?;
    let valid = fs::read_to_string(base.join("wiki.valid.raw"))?;
    let test = fs::read_to_string(base.join("wiki.test.raw"))?;
    // Materialise canonical filenames for the offline path on subsequent runs.
    fs::write(out_dir.join("train.txt"), &train)?;
    fs::write(out_dir.join("valid.txt"), &valid)?;
    fs::write(out_dir.join("test.txt"), &test)?;
    Ok(WikiText2Splits { train, valid, test })
}
