//! Stanford Sentiment Treebank-2 (SST-2) loader.
//!
//! Pinned to the HuggingFace `SetFit/sst2` redistribution because the original
//! `dl.fbaipublicfiles.com` GLUE mirror started returning 403s. The SetFit copy is the
//! same binary sentiment classification form (sentence + 0/1 label) that most academic
//! baselines report, distributed as JSONL with a stable URL.
//!
//! Output rows: `(sentence: String, label: u8)` where label is 0 (negative) or 1 (positive).
//!
//! Offline / pre-staged workflows are supported: place `train.jsonl` and `dev.jsonl` under
//! `<cache>/datasets/sst2/` and run with `RUSTRAL_DATASET_OFFLINE=1`. CI smoke tests do
//! exactly that to avoid network round-trips on every PR.
//!
//! NOTE: This module is gated by `feature = "fetch"`.

#![cfg(feature = "fetch")]

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use crate::fetch::{fetch_url, FetchError};

/// Pinned SST-2 mirror.
///
/// HuggingFace `SetFit/sst2` JSONL distribution. SHA-256 verified on download; if upstream
/// rehashes the artifact this code fails loudly so the pin is updated deliberately.
pub const SST2_TRAIN_URL: &str = "https://huggingface.co/datasets/SetFit/sst2/resolve/main/train.jsonl";
pub const SST2_TRAIN_SHA256: &str = "7a4b1cfdd65be1dc48339404db86528bb2427e1d8772860ef838b76b8c38c4a8";

pub const SST2_DEV_URL: &str = "https://huggingface.co/datasets/SetFit/sst2/resolve/main/dev.jsonl";
pub const SST2_DEV_SHA256: &str = "573c3ed18d96aa0a79a6e5980a544b80543317a319f18bd4f1660c16b2f6b939";

// Backwards-compatible alias names so any external caller (or older docs) still resolves.
#[doc(hidden)]
pub const SST2_TRAIN_TSV_URL: &str = SST2_TRAIN_URL;
#[doc(hidden)]
pub const SST2_TRAIN_TSV_SHA256: &str = SST2_TRAIN_SHA256;
#[doc(hidden)]
pub const SST2_DEV_TSV_URL: &str = SST2_DEV_URL;
#[doc(hidden)]
pub const SST2_DEV_TSV_SHA256: &str = SST2_DEV_SHA256;

/// One labelled sentence from SST-2.
#[derive(Clone, Debug)]
pub struct Sst2Example {
    pub sentence: String,
    pub label: u8,
}

/// Fetch (or read from cache) the SST-2 splits and return them as in-memory vectors.
///
/// In offline mode (`RUSTRAL_DATASET_OFFLINE=1`) the JSONL files must already be in the
/// cache directory; otherwise this returns an error. The loader also accepts pre-staged
/// `.tsv` files at the legacy paths for backward compatibility.
pub fn load_sst2() -> Result<(Vec<Sst2Example>, Vec<Sst2Example>), FetchError> {
    // Legacy offline staging: `train.tsv` / `dev.tsv` next to the canonical jsonl. Useful
    // for users who already have the GLUE TSV checked out.
    let cache_dir = crate::fetch::cache_root().join("datasets").join("sst2");
    let legacy_train = cache_dir.join("train.tsv");
    let legacy_dev = cache_dir.join("dev.tsv");
    if legacy_train.exists() && legacy_dev.exists() {
        return Ok((parse_tsv(&legacy_train)?, parse_tsv(&legacy_dev)?));
    }

    let train_path = fetch_url(SST2_TRAIN_URL, SST2_TRAIN_SHA256, "sst2")?;
    let dev_path = fetch_url(SST2_DEV_URL, SST2_DEV_SHA256, "sst2")?;
    Ok((parse_jsonl(&train_path)?, parse_jsonl(&dev_path)?))
}

fn parse_jsonl(path: &PathBuf) -> Result<Vec<Sst2Example>, FetchError> {
    let f = File::open(path)?;
    let mut out = Vec::new();
    for line in BufReader::new(f).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        // Lightweight JSON parsing (no serde_json dep): we only need `text` and `label`.
        // The SetFit/sst2 JSONL is well-formed: each line is a flat object with these keys.
        let Some(text) = extract_string_field(&line, "text") else { continue };
        let Some(label_raw) = extract_int_field(&line, "label") else { continue };
        let label = match label_raw {
            0 => 0u8,
            1 => 1u8,
            _ => continue,
        };
        out.push(Sst2Example { sentence: text, label });
    }
    Ok(out)
}

fn parse_tsv(path: &PathBuf) -> Result<Vec<Sst2Example>, FetchError> {
    let f = File::open(path)?;
    let mut out = Vec::new();
    let mut header_seen = false;
    for line in BufReader::new(f).lines() {
        let line = line?;
        if !header_seen {
            // SST-2 GLUE TSV starts with a `sentence\tlabel` header.
            header_seen = true;
            if line.starts_with("sentence") {
                continue;
            }
        }
        if line.trim().is_empty() {
            continue;
        }
        let mut parts = line.splitn(2, '\t');
        let (Some(sentence), Some(label_str)) = (parts.next(), parts.next()) else {
            continue;
        };
        let label = match label_str.trim() {
            "0" => 0u8,
            "1" => 1u8,
            _ => continue,
        };
        out.push(Sst2Example { sentence: sentence.to_string(), label });
    }
    Ok(out)
}

fn extract_string_field(line: &str, key: &str) -> Option<String> {
    // Find `"key"` then the following `:`, optional whitespace, then `"..."` with simple
    // backslash-escape handling. Adequate for the SetFit/sst2 distribution; if we ever take
    // a richer JSON dataset we should bring in serde_json behind a feature flag.
    let needle = format!("\"{key}\"");
    let start = line.find(&needle)? + needle.len();
    let rest = &line[start..];
    let colon = rest.find(':')? + 1;
    let after_colon = &rest[colon..];
    let quote = after_colon.find('"')? + 1;
    let body = &after_colon[quote..];
    let mut out = String::new();
    let mut chars = body.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next()? {
                '"' => out.push('"'),
                '\\' => out.push('\\'),
                'n' => out.push('\n'),
                't' => out.push('\t'),
                'r' => out.push('\r'),
                other => {
                    out.push('\\');
                    out.push(other);
                }
            }
        } else if c == '"' {
            return Some(out);
        } else {
            out.push(c);
        }
    }
    None
}

fn extract_int_field(line: &str, key: &str) -> Option<i64> {
    let needle = format!("\"{key}\"");
    let start = line.find(&needle)? + needle.len();
    let rest = &line[start..];
    let colon = rest.find(':')? + 1;
    let after_colon = rest[colon..].trim_start();
    let end = after_colon.find(|c: char| !(c.is_ascii_digit() || c == '-')).unwrap_or(after_colon.len());
    after_colon[..end].parse::<i64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_jsonl_extracts_text_and_label() {
        let line = "{\"text\": \"a stirring movie\", \"label\": 1, \"label_text\": \"positive\"}";
        assert_eq!(extract_string_field(line, "text").as_deref(), Some("a stirring movie"));
        assert_eq!(extract_int_field(line, "label"), Some(1));
    }

    #[test]
    fn parse_jsonl_handles_escapes() {
        let line = "{\"text\": \"quote \\\"x\\\" and tab\\there\", \"label\": 0}";
        assert_eq!(extract_string_field(line, "text").as_deref(), Some("quote \"x\" and tab\there"),);
        assert_eq!(extract_int_field(line, "label"), Some(0));
    }
}
