//! Experimental LLM utilities and model-family adapters for Rustral.
//!
//! This crate is intentionally strict and explicit:
//! - No implicit downloads during model forward.
//! - No global mutable state.
//! - Clear errors for unsupported model families / files.

use std::path::{Path, PathBuf};

use thiserror::Error;

pub mod gpt2;

/// Errors surfaced by `rustral-llm`.
#[derive(Debug, Error)]
pub enum LlmError {
    #[error("unsupported backend '{backend}' (supported: ndarray, candle)")]
    UnsupportedBackend { backend: String },

    #[error("missing required file: {0}")]
    MissingFile(String),

    #[error("invalid argument: {0}")]
    InvalidArg(String),

    #[error(transparent)]
    Anyhow(#[from] anyhow::Error),
}

/// A local snapshot of a Hugging Face model repository (paths to key files).
///
/// This is a lightweight mirror of the more complete snapshot type that lives in `rustral-hf`.
#[derive(Clone, Debug)]
pub struct HubSnapshot {
    pub root: PathBuf,
    pub config_json: Option<PathBuf>,
    pub tokenizer_json: Option<PathBuf>,
    pub safetensors_index_json: Option<PathBuf>,
    pub safetensors_files: Vec<PathBuf>,
}

impl HubSnapshot {
    pub fn require_config_json(&self) -> Result<&Path, LlmError> {
        self.config_json.as_deref().ok_or_else(|| LlmError::MissingFile("config.json".to_string()))
    }
}

/// Minimal tokenizer abstraction for LLM workflows.
///
/// This is intentionally narrow; higher-level helpers belong in model-family adapters.
#[derive(Clone)]
pub struct TokenizerHandle {
    #[cfg(feature = "hf-tokenizers")]
    inner: tokenizers::Tokenizer,
}

impl TokenizerHandle {
    #[cfg(feature = "hf-tokenizers")]
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, LlmError> {
        let inner = tokenizers::Tokenizer::from_file(path.as_ref()).map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(Self { inner })
    }

    #[cfg(feature = "hf-tokenizers")]
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, LlmError> {
        let enc = self.inner.encode(text, true).map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(enc.get_ids().iter().map(|&id| id as u32).collect())
    }

    #[cfg(feature = "hf-tokenizers")]
    pub fn decode(&self, ids: &[u32]) -> Result<String, LlmError> {
        let ids_u32: Vec<u32> = ids.to_vec();
        self.inner
            .decode(&ids_u32, true)
            .map_err(|e| anyhow::anyhow!("{e}").into())
    }
}

