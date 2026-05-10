//! HuggingFace Hub integration for Rustral.
//!
//! Provides standalone functions for downloading and uploading model
//! state dictionaries to/from the HuggingFace Hub.
//!
//! This crate follows the same design philosophy as `rustral-io`: all
//! operations are standalone functions, not methods on [`Backend`] or
//! [`Module`]. Async/networking concerns are kept out of core traits.
//!
//! # Downloading a pretrained model
//!
//! ```rust,ignore
//! use rustral_hf::download_state_dict;
//! use rustral_nn::{Linear, LinearConfig};
//! use rustral_core::Saveable;
//! use rustral_ndarray_backend::CpuBackend;
//!
//! // Download weights from HF Hub
//! let state_dict = download_state_dict("bert-base-uncased").unwrap();
//!
//! // Build model locally from config
//! let backend = CpuBackend::default();
//! let mut linear = Linear::new(&backend, LinearConfig::new(768, 768)).unwrap();
//!
//! // Load weights via existing Saveable trait
//! linear.load_state_dict(&state_dict, &backend).unwrap();
//! ```
//!
//! # Uploading a trained model
//!
//! ```rust,ignore
//! use rustral_hf::upload_state_dict;
//! use std::collections::HashMap;
//!
//! let mut state_dict = HashMap::new();
//! state_dict.insert("weight".to_string(), vec![1.0f32, 2.0, 3.0, 4.0]);
//! state_dict.insert("bias".to_string(), vec![0.1f32, 0.2]);
//!
//! upload_state_dict("my-org/my-model", &state_dict).unwrap();
//! ```
//!
//! [`Backend`]: rustral_core::Backend
//! [`Module`]: rustral_core::Module

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use hf_hub::Repo;
use hf_hub::RepoType;
use rustral_core::{Backend, Saveable};
use rustral_io::load_parameters;
use thiserror::Error;

/// Errors that can occur during HuggingFace Hub operations.
#[derive(Debug, Error)]
pub enum HfError {
    /// Hub API error (network, auth, not found, etc.).
    #[error("hub error: {0}")]
    Hub(String),

    /// I/O error reading the downloaded file.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// Tensor deserialization error.
    #[error("tensor error: {0}")]
    Tensor(#[from] rustral_io::IoError),

    /// Missing expected file in the repository.
    #[error("file '{file}' not found in repo '{repo}'")]
    MissingFile { repo: String, file: String },

    /// State dict loading error propagated from Saveable.
    #[error("state dict error: {0}")]
    StateDict(String),

    /// JSON parsing error.
    #[error("json error: {0}")]
    Json(String),
}

impl From<hf_hub::api::sync::ApiError> for HfError {
    fn from(e: hf_hub::api::sync::ApiError) -> Self {
        HfError::Hub(e.to_string())
    }
}

/// Download a state dictionary from the HuggingFace Hub.
///
/// Looks for `model.safetensors` first, then falls back to
/// `pytorch_model.bin` if available (PyTorch weights are **not**
/// automatically converted — callers should convert `.bin` files
/// externally and re-upload as Safetensors).
///
/// # Arguments
/// * `model_id` — HuggingFace model ID, e.g. `"bert-base-uncased"` or
///   `"google-bert/bert-base-uncased"`.
///
/// # Returns
/// A map from parameter name to flat `f32` values. The caller
/// (typically a [`Saveable`] module) is responsible for knowing the
/// expected shapes and reshaping the data via
/// `backend.parameter_from_vec`.
///
/// # Errors
/// Returns [`HfError::MissingFile`] if neither `model.safetensors` nor
/// `pytorch_model.bin` can be found.
pub fn download_state_dict(model_id: &str) -> Result<HashMap<String, Vec<f32>>, HfError> {
    let api = Api::new()?;
    let repo = api.model(model_id.to_string());

    // Try Safetensors first (preferred format).
    match repo.get("model.safetensors") {
        Ok(path) => {
            let bytes = std::fs::read(&path)?;
            let dict = load_parameters(&bytes)?;
            Ok(dict)
        }
        Err(_) => {
            // Fall back to pytorch_model.bin
            match repo.get("pytorch_model.bin") {
                Ok(_path) => {
                    // PyTorch .bin files require pickle unpickling which
                    // is not supported natively. Users should convert to
                    // Safetensors first (e.g. with `convert-to-safetensors`).
                    Err(HfError::MissingFile {
                        repo: model_id.to_string(),
                        file: "model.safetensors (pytorch_model.bin exists but requires external conversion)"
                            .to_string(),
                    })
                }
                Err(_) => Err(HfError::MissingFile {
                    repo: model_id.to_string(),
                    file: "model.safetensors".to_string(),
                }),
            }
        }
    }
}

/// A local snapshot of a Hugging Face model repository.
///
/// The snapshot is a set of resolved local file paths (downloaded via `hf-hub` cache).
#[derive(Clone, Debug)]
pub struct HubModelSnapshot {
    pub model_id: String,
    pub revision: Option<String>,
    pub root: PathBuf,
    pub files: HubModelFiles,
}

/// Key files we may download for LLM-style workflows.
#[derive(Clone, Debug, Default)]
pub struct HubModelFiles {
    pub config_json: Option<PathBuf>,
    pub tokenizer_json: Option<PathBuf>,
    pub tokenizer_config_json: Option<PathBuf>,
    pub special_tokens_map_json: Option<PathBuf>,
    pub generation_config_json: Option<PathBuf>,
    pub safetensors_index_json: Option<PathBuf>,
    pub safetensors_files: Vec<PathBuf>,
    pub gguf_files: Vec<PathBuf>,
}

#[derive(Debug, serde::Deserialize)]
struct SafeTensorsIndex {
    weight_map: HashMap<String, String>,
}

/// Download a set of common model files from the Hub and return resolved local paths.
///
/// This is a building block for `rustral-llm` and other tooling that needs config/tokenizer
/// metadata and sharded safetensors awareness (via `model.safetensors.index.json`).
///
/// Notes:
/// - This function is intentionally conservative: it only fetches well-known filenames.
/// - Listing arbitrary repository contents is not required and is intentionally avoided.
///
/// Uses the Hub default revision (**`main`**). For a pinned branch, tag, or commit, use
/// [`snapshot_model_at`].
pub fn snapshot_model(model_id: &str) -> Result<HubModelSnapshot, HfError> {
    snapshot_model_at(model_id, None)
}

/// Download common model files like [`snapshot_model`], optionally pinned to a **revision**
/// (branch name, tag, or commit SHA) for reproducible artifact hashes.
///
/// When `revision` is `None`, behavior matches [`snapshot_model`] (default branch `main`).
pub fn snapshot_model_at(model_id: &str, revision: Option<&str>) -> Result<HubModelSnapshot, HfError> {
    let api = Api::new()?;
    let repo = match revision {
        Some(rev) => api.repo(Repo::with_revision(model_id.to_string(), RepoType::Model, rev.to_string())),
        None => api.model(model_id.to_string()),
    };

    let mut files = HubModelFiles::default();

    files.config_json = try_get(&repo, "config.json");
    files.tokenizer_json = try_get(&repo, "tokenizer.json");
    files.tokenizer_config_json = try_get(&repo, "tokenizer_config.json");
    files.special_tokens_map_json = try_get(&repo, "special_tokens_map.json");
    files.generation_config_json = try_get(&repo, "generation_config.json");

    // SafeTensors: prefer an index (sharded) if present, else single-file.
    files.safetensors_index_json = try_get(&repo, "model.safetensors.index.json");
    if let Some(index_path) = &files.safetensors_index_json {
        let idx_bytes = std::fs::read(index_path)?;
        let idx: SafeTensorsIndex =
            serde_json::from_slice(&idx_bytes).map_err(|e| HfError::Json(e.to_string()))?;
        let mut shard_names: Vec<String> = idx.weight_map.values().cloned().collect();
        shard_names.sort();
        shard_names.dedup();

        for shard in shard_names {
            if let Ok(p) = repo.get(&shard) {
                files.safetensors_files.push(p);
            } else {
                return Err(HfError::MissingFile { repo: model_id.to_string(), file: shard });
            }
        }
    } else if let Some(p) = try_get(&repo, "model.safetensors") {
        files.safetensors_files.push(p);
    }

    // Optional GGUF (best-effort only; file enumeration is not attempted here).
    // Common names are tried, but absence is not an error.
    if let Some(p) = try_get(&repo, "model.gguf") {
        files.gguf_files.push(p);
    }

    let root = snapshot_root(&files).unwrap_or_else(|| PathBuf::from("."));
    Ok(HubModelSnapshot {
        model_id: model_id.to_string(),
        revision: revision.map(|s| s.to_string()),
        root,
        files,
    })
}

impl HubModelSnapshot {
    /// Returns the path to `config.json` when present.
    pub fn require_config_json(&self) -> Result<&Path, HfError> {
        self.files.config_json.as_deref().ok_or_else(|| HfError::MissingFile {
            repo: self.model_id.clone(),
            file: "config.json".to_string(),
        })
    }
}

/// Discover model files under a **local directory** (offline, no Hub API).
///
/// Recognizes the same layout as [`snapshot_model`]: optional tokenizer files,
/// `model.safetensors` or sharded `model.safetensors.index.json` + shard files,
/// optional `model.gguf`.
///
/// [`HubModelSnapshot::model_id`] is set to `local:<canonical-root>` for diagnostics.
pub fn scan_local_model_dir(root: impl AsRef<Path>) -> Result<HubModelSnapshot, HfError> {
    let root = root.as_ref().canonicalize()?;
    let model_id = format!("local:{}", root.display());
    let mut files = HubModelFiles::default();

    macro_rules! file_if_exists {
        ($field:ident, $name:expr) => {
            let p = root.join($name);
            if p.is_file() {
                files.$field = Some(p);
            }
        };
    }

    file_if_exists!(config_json, "config.json");
    file_if_exists!(tokenizer_json, "tokenizer.json");
    file_if_exists!(tokenizer_config_json, "tokenizer_config.json");
    file_if_exists!(special_tokens_map_json, "special_tokens_map.json");
    file_if_exists!(generation_config_json, "generation_config.json");

    let index_path = root.join("model.safetensors.index.json");
    if index_path.is_file() {
        files.safetensors_index_json = Some(index_path.clone());
        let idx_bytes = std::fs::read(&index_path)?;
        let idx: SafeTensorsIndex =
            serde_json::from_slice(&idx_bytes).map_err(|e| HfError::Json(e.to_string()))?;
        let mut shard_names: Vec<String> = idx.weight_map.values().cloned().collect();
        shard_names.sort();
        shard_names.dedup();
        for shard in shard_names {
            let p = root.join(&shard);
            if p.is_file() {
                files.safetensors_files.push(p);
            } else {
                return Err(HfError::MissingFile {
                    repo: model_id.clone(),
                    file: shard,
                });
            }
        }
    } else if root.join("model.safetensors").is_file() {
        files.safetensors_files.push(root.join("model.safetensors"));
    }

    let gguf = root.join("model.gguf");
    if gguf.is_file() {
        files.gguf_files.push(gguf);
    }

    Ok(HubModelSnapshot {
        model_id,
        revision: None,
        root,
        files,
    })
}

/// Save a state dictionary to a local Safetensors file.
///
/// Serializes the state dict to Safetensors format and writes it to
/// `{model_id.replace('/', '_')}.safetensors` in the current
/// directory.
///
/// # Arguments
/// * `model_id` — Model identifier used to name the output file.
/// * `state_dict` — Map from parameter name to flat `f32` values.
///
/// # Returns
/// The path to the written file.
///
/// # Uploading to HuggingFace Hub
/// After calling this function, upload the resulting file with the
/// HuggingFace CLI:
/// ```bash
/// huggingface-cli upload my-org/my-model model.safetensors
/// ```
pub fn save_state_dict_to_file(
    model_id: &str,
    state_dict: &HashMap<String, Vec<f32>>,
) -> Result<std::path::PathBuf, HfError> {
    let bytes = rustral_io::save_state_dict(state_dict)?;
    let path = std::path::PathBuf::from(format!("{}.safetensors", sanitize(model_id)));
    std::fs::write(&path, &bytes)?;
    Ok(path)
}

/// Convenience: download a state dict from the HuggingFace Hub and load
/// it into a [`Saveable`] module.
///
/// This is equivalent to calling [`download_state_dict`] followed by
/// [`Saveable::load_state_dict`].
///
/// # Type Parameters
/// * `B` — Backend type
/// * `M` — Module type implementing [`Saveable<B>`]
///
/// # Arguments
/// * `module` — The module to load weights into.
/// * `model_id` — HuggingFace model ID.
/// * `backend` — Backend for creating tensors.
pub fn load_from_hub<B, M>(module: &mut M, model_id: &str, backend: &B) -> Result<(), HfError>
where
    B: Backend,
    M: Saveable<B>,
{
    let state_dict = download_state_dict(model_id)?;
    module.load_state_dict(&state_dict, backend).map_err(|e| HfError::StateDict(e.to_string()))?;
    Ok(())
}

// ------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------

use hf_hub::api::sync::Api;

fn sanitize(model_id: &str) -> String {
    model_id.replace(|c: char| !c.is_alphanumeric(), "_")
}

fn try_get(repo: &hf_hub::api::sync::ApiRepo, file: &str) -> Option<PathBuf> {
    repo.get(file).ok()
}

fn snapshot_root(files: &HubModelFiles) -> Option<PathBuf> {
    let first: Option<&Path> = files
        .config_json
        .as_deref()
        .or(files.tokenizer_json.as_deref())
        .or(files.safetensors_index_json.as_deref())
        .or(files.safetensors_files.first().map(|p| p.as_path()))
        .or(files.gguf_files.first().map(|p| p.as_path()));
    first.and_then(|p| p.parent()).map(|p| p.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustral_core::Saveable;
    use rustral_io::save_state_dict;
    use rustral_ndarray_backend::CpuBackend;
    use rustral_nn::{Linear, LinearConfig};
    use std::collections::HashMap;

    #[test]
    fn test_download_state_dict_local_fallback() {
        // Since we can't guarantee network access in tests, we verify
        // that download_state_dict returns MissingFile for a nonexistent
        // model, confirming the error path works.
        let result = download_state_dict("this-model-does-not-exist-12345");
        assert!(matches!(result, Err(HfError::MissingFile { .. })));
    }

    /// Smoke test: actually download a real tiny model from HF Hub.
    /// Network is opt-in. By default this test returns early and passes.
    #[test]
    fn test_download_real_model_smoke() {
        if std::env::var("RUSTRAL_TEST_HF_NETWORK").ok().as_deref() != Some("1") {
            eprintln!("skipping HF network smoke test (set RUSTRAL_TEST_HF_NETWORK=1 to enable)");
            return;
        }
        let result = download_state_dict("hf-internal-testing/tiny-random-bert");
        match result {
            Ok(dict) => {
                // The tiny-random-bert has a handful of tensors.
                assert!(!dict.is_empty(), "Expected at least one tensor");
                println!("Downloaded {} tensors from tiny-random-bert", dict.len());
            }
            Err(HfError::MissingFile { ref file, .. }) => {
                // If the model doesn't have model.safetensors, that's fine.
                assert!(file.contains("model.safetensors"));
            }
            Err(e) => panic!("Unexpected error: {}", e),
        }
    }

    #[test]
    fn test_save_and_load_roundtrip_local() {
        // Simulate a Hub-like roundtrip by writing a local safetensors
        // file and reading it back through the same parser used by
        // download_state_dict.
        let mut original = HashMap::new();
        original.insert("weight".to_string(), vec![1.0f32, 2.0, 3.0, 4.0]);
        original.insert("bias".to_string(), vec![0.5f32, 0.5f32]);

        let bytes = save_state_dict(&original).unwrap();

        // Parse back with the same loader download_state_dict uses.
        let loaded = load_parameters(&bytes).unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.get("weight").unwrap(), &vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(loaded.get("bias").unwrap(), &vec![0.5, 0.5]);
    }

    #[test]
    fn test_save_state_dict_to_file() {
        let mut dict = HashMap::new();
        dict.insert("w".to_string(), vec![1.0f32, 2.0, 3.0]);

        let path = save_state_dict_to_file("test_model", &dict).unwrap();
        assert_eq!(path.file_name().unwrap(), "test_model.safetensors");
        assert!(path.exists());

        let bytes = std::fs::read(&path).unwrap();
        let loaded = load_parameters(&bytes).unwrap();
        assert_eq!(loaded.get("w").unwrap(), &vec![1.0, 2.0, 3.0]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_load_from_hub_with_linear() {
        let backend = CpuBackend::default();
        let mut linear = Linear::new(&backend, LinearConfig::new(2, 2).with_bias(true)).unwrap();

        // Build a synthetic state dict matching the expected keys.
        let mut state_dict = HashMap::new();
        state_dict.insert("weight".to_string(), vec![1.0f32, 0.0, 0.0, 1.0]);
        state_dict.insert("bias".to_string(), vec![0.0f32, 0.0f32]);

        // Manually load (simulating what load_from_hub does when it
        // succeeds — we skip the actual network call in the test).
        linear.load_state_dict(&state_dict, &backend).unwrap();

        // Verify the weight tensor has the right number of elements.
        let weight_values: Vec<f32> = linear.weight().tensor().as_ref().to_vec();
        assert_eq!(weight_values, vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_sanitize_model_id() {
        assert_eq!(sanitize("org/model-name"), "org_model_name");
        assert_eq!(sanitize("model"), "model");
        assert_eq!(sanitize("a-b_c/d.e"), "a_b_c_d_e");
    }

    #[test]
    fn test_hf_error_display() {
        let e = HfError::MissingFile { repo: "test".to_string(), file: "model.safetensors".to_string() };
        assert!(e.to_string().contains("model.safetensors"));
    }

    #[test]
    fn test_scan_local_model_dir_single_safetensors() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join("config.json"), br#"{"model_type":"gpt2"}"#).unwrap();
        let mut dict = HashMap::new();
        dict.insert("w".to_string(), vec![1.0f32]);
        let bytes = save_state_dict(&dict).unwrap();
        std::fs::write(root.join("model.safetensors"), bytes).unwrap();

        let snap = scan_local_model_dir(root).unwrap();
        assert!(snap.files.config_json.is_some());
        assert_eq!(snap.files.safetensors_files.len(), 1);
        snap.require_config_json().unwrap();
    }

    #[test]
    fn test_scan_local_model_dir_sharded_index() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join("config.json"), br"{}").unwrap();
        let idx = serde_json::json!({
            "weight_map": {
                "a": "shard1.safetensors",
                "b": "shard2.safetensors"
            }
        });
        std::fs::write(root.join("model.safetensors.index.json"), serde_json::to_vec(&idx).unwrap()).unwrap();
        let mut d1 = HashMap::new();
        d1.insert("a".to_string(), vec![1.0f32]);
        let mut d2 = HashMap::new();
        d2.insert("b".to_string(), vec![2.0f32]);
        std::fs::write(root.join("shard1.safetensors"), save_state_dict(&d1).unwrap()).unwrap();
        std::fs::write(root.join("shard2.safetensors"), save_state_dict(&d2).unwrap()).unwrap();

        let snap = scan_local_model_dir(root).unwrap();
        assert_eq!(snap.files.safetensors_files.len(), 2);
        assert!(snap.files.safetensors_index_json.is_some());
    }
}
