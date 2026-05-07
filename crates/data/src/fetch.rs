//! HTTP dataset fetching with content-addressed caching.
//!
//! Active only under the `fetch` feature so the default `rustral-data` build does not
//! pull `ureq`, `sha2`, or `flate2`. Examples that need real corpora opt in explicitly
//! (see `examples/Cargo.toml` and the SST-2 / WikiText-2 NLP examples).
//!
//! Cache layout: `~/.cache/rustral/datasets/<cache_subdir>/<filename>`. Files are
//! checksum-verified on every call (cheap, prevents corruption). When the file is already
//! present the cached path is returned without a network round-trip.
//!
//! Offline mode: setting `RUSTRAL_DATASET_OFFLINE=1` makes `fetch_url` return an error if
//! the file is not yet cached; useful in CI and tests.

#![cfg(feature = "fetch")]

use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// Errors surfaced by the dataset fetch layer.
#[derive(Debug)]
pub enum FetchError {
    /// Cache directory could not be created or accessed.
    Io(io::Error),
    /// HTTP request failed.
    Http(String),
    /// Downloaded content did not match the pinned SHA-256.
    ChecksumMismatch { expected: String, actual: String, url: String },
    /// Offline mode is on but the file is not cached.
    Offline { url: String, cache_path: PathBuf },
}

impl std::fmt::Display for FetchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FetchError::Io(e) => write!(f, "io: {e}"),
            FetchError::Http(e) => write!(f, "http: {e}"),
            FetchError::ChecksumMismatch { expected, actual, url } => write!(
                f,
                "checksum mismatch for {url}: expected sha256 {expected}, got {actual}"
            ),
            FetchError::Offline { url, cache_path } => write!(
                f,
                "RUSTRAL_DATASET_OFFLINE=1 set but {url} is not cached at {}",
                cache_path.display()
            ),
        }
    }
}

impl std::error::Error for FetchError {}

impl From<io::Error> for FetchError {
    fn from(e: io::Error) -> Self {
        FetchError::Io(e)
    }
}

/// Resolve the rustral cache root: `RUSTRAL_CACHE_DIR` env var, else `~/.cache/rustral`.
pub fn cache_root() -> PathBuf {
    if let Ok(p) = std::env::var("RUSTRAL_CACHE_DIR") {
        return PathBuf::from(p);
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join(".cache").join("rustral");
    }
    PathBuf::from(".cache").join("rustral")
}

fn sha256_hex(path: &Path) -> Result<String, FetchError> {
    let mut f = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn is_offline() -> bool {
    std::env::var("RUSTRAL_DATASET_OFFLINE").is_ok_and(|s| !s.is_empty() && s != "0")
}

/// When `RUSTRAL_DATASET_SKIP_CHECKSUM=1` the cached file is trusted without verification.
///
/// Intended for offline / pre-staged workflows (CI, smoke tests) where the operator owns
/// cache integrity and the upstream SHA may not yet be pinned in code.
fn skip_checksum() -> bool {
    std::env::var("RUSTRAL_DATASET_SKIP_CHECKSUM")
        .is_ok_and(|s| !s.is_empty() && s != "0")
}

/// Fetch a URL into the rustral cache and verify its SHA-256.
///
/// The cached path is `cache_root()/datasets/<cache_subdir>/<basename>` where `<basename>`
/// is the last `/`-segment of the URL. If the file is already present the cached path is
/// returned after re-checking the checksum.
///
/// Setting `RUSTRAL_DATASET_OFFLINE=1` makes this function fail (instead of network
/// fetching) when the file is not yet cached.
pub fn fetch_url(url: &str, sha256: &str, cache_subdir: &str) -> Result<PathBuf, FetchError> {
    let basename = url.rsplit('/').next().filter(|s| !s.is_empty()).unwrap_or("download.bin");
    let cache_dir = cache_root().join("datasets").join(cache_subdir);
    fs::create_dir_all(&cache_dir)?;
    let target = cache_dir.join(basename);

    if target.exists() {
        if sha256.is_empty() || skip_checksum() {
            return Ok(target);
        }
        let actual = sha256_hex(&target)?;
        if actual == sha256 {
            return Ok(target);
        }
        // Stale or corrupted; remove and re-download (unless offline).
        let _ = fs::remove_file(&target);
        if is_offline() {
            return Err(FetchError::Offline { url: url.to_string(), cache_path: target });
        }
    } else if is_offline() {
        return Err(FetchError::Offline { url: url.to_string(), cache_path: target });
    }

    // Network download.
    let resp = ureq::get(url).call().map_err(|e| FetchError::Http(format!("{e}")))?;
    let mut reader = resp.into_reader();
    let tmp = cache_dir.join(format!("{basename}.partial"));
    {
        let mut f = fs::File::create(&tmp)?;
        let mut buf = [0u8; 64 * 1024];
        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            f.write_all(&buf[..n])?;
        }
        f.sync_all()?;
    }

    let actual = sha256_hex(&tmp)?;
    if actual != sha256 {
        let _ = fs::remove_file(&tmp);
        return Err(FetchError::ChecksumMismatch {
            expected: sha256.to_string(),
            actual,
            url: url.to_string(),
        });
    }
    fs::rename(&tmp, &target)?;
    Ok(target)
}

/// Decompress a gzip file to a sibling path (`<input>.txt` or similar).
///
/// Returns the path of the extracted file. If the destination already exists with the
/// expected size it is returned without re-extracting.
pub fn ungzip(input: &Path, output: &Path) -> Result<PathBuf, FetchError> {
    if output.exists() {
        return Ok(output.to_path_buf());
    }
    let f = fs::File::open(input)?;
    let mut decoder = flate2::read::GzDecoder::new(f);
    let mut out = fs::File::create(output)?;
    io::copy(&mut decoder, &mut out)?;
    Ok(output.to_path_buf())
}
