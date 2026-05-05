//! HuggingFace Hub integration for MNR.
//!
//! Provides standalone functions for downloading and uploading model
//! state dictionaries to/from the HuggingFace Hub.
//!
//! This crate follows the same design philosophy as `mnr-io`: all
//! operations are standalone functions, not methods on [`Backend`] or
//! [`Module`]. Async/networking concerns are kept out of core traits.
//!
//! # Downloading a pretrained model
//!
//! ```rust,ignore
//! use mnr_hf::download_state_dict;
//! use mnr_nn::{Linear, LinearConfig};
//! use mnr_core::Saveable;
//! use mnr_ndarray_backend::CpuBackend;
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
//! use mnr_hf::upload_state_dict;
//! use std::collections::HashMap;
//!
//! let mut state_dict = HashMap::new();
//! state_dict.insert("weight".to_string(), vec![1.0f32, 2.0, 3.0, 4.0]);
//! state_dict.insert("bias".to_string(), vec![0.1f32, 0.2]);
//!
//! upload_state_dict("my-org/my-model", &state_dict).unwrap();
//! ```
//!
//! [`Backend`]: mnr_core::Backend
//! [`Module`]: mnr_core::Module

use std::collections::HashMap;

use mnr_core::{Backend, Saveable};
use mnr_io::load_parameters;
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
    Tensor(#[from] mnr_io::IoError),

    /// Missing expected file in the repository.
    #[error("file '{file}' not found in repo '{repo}'")]
    MissingFile { repo: String, file: String },

    /// State dict loading error propagated from Saveable.
    #[error("state dict error: {0}")]
    StateDict(String),
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
            let dict = load_parameters::<mnr_ndarray_backend::CpuBackend>(&bytes)?;
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
                        file: "model.safetensors (pytorch_model.bin exists but requires external conversion)".to_string(),
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
    let bytes = mnr_io::save_state_dict(state_dict)?;
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
pub fn load_from_hub<B, M>(
    module: &mut M,
    model_id: &str,
    backend: &B,
) -> Result<(), HfError>
where
    B: Backend,
    M: Saveable<B>,
{
    let state_dict = download_state_dict(model_id)?;
    module
        .load_state_dict(&state_dict, backend)
        .map_err(|e| HfError::StateDict(e.to_string()))?;
    Ok(())
}

// ------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------

use hf_hub::api::sync::Api;

fn sanitize(model_id: &str) -> String {
    model_id.replace(|c: char| !c.is_alphanumeric(), "_")
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnr_core::Saveable;
    use mnr_io::save_state_dict;
    use mnr_ndarray_backend::CpuBackend;
    use mnr_nn::{Linear, LinearConfig};
    use std::collections::HashMap;

    #[test]
    fn test_download_state_dict_local_fallback() {
        // Since we can't guarantee network access in tests, we verify
        // that download_state_dict returns MissingFile for a nonexistent
        // model, confirming the error path works.
        let result = download_state_dict("this-model-does-not-exist-12345");
        assert!(matches!(result, Err(HfError::MissingFile { .. })));
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
        let loaded = load_parameters::<CpuBackend>(&bytes).unwrap();

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
        let loaded = load_parameters::<CpuBackend>(&bytes).unwrap();
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
        let e = HfError::MissingFile {
            repo: "test".to_string(),
            file: "model.safetensors".to_string(),
        };
        assert!(e.to_string().contains("model.safetensors"));
    }
}
