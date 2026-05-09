//! Curated model registry for Rustral.
//!
//! The registry is embedded at compile time from [`registry.json`](../registry.json).
//! See the crate README for Hugging Face key mapping notes.

use serde::Deserialize;

/// Top-level registry file (see `registry.json`).
#[derive(Debug, Clone, Deserialize)]
pub struct Registry {
    pub schema_version: u32,
    pub entries: Vec<RegistryEntry>,
}

/// One registry row: Hub id, status, and human notes.
#[derive(Debug, Clone, Deserialize)]
pub struct RegistryEntry {
    pub id: String,
    #[serde(default)]
    pub huggingface_model_id: Option<String>,
    #[serde(default)]
    pub revision: Option<String>,
    pub status: String,
    pub notes: String,
    #[serde(default)]
    pub local_artifact_workflow: Option<String>,
}

const REGISTRY_JSON: &str = include_str!("../registry.json");

/// Parse the built-in registry.
pub fn registry() -> anyhow::Result<Registry> {
    Ok(serde_json::from_str(REGISTRY_JSON)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_parses() {
        let r = registry().unwrap();
        assert!(r.schema_version >= 1);
        assert!(!r.entries.is_empty());
    }

    #[test]
    fn local_workflow_entry_exists() {
        let r = registry().unwrap();
        let has_demo = r.entries.iter().any(|e| {
            e.id == "local_tiny_linear_regression"
                && e.local_artifact_workflow
                    .as_ref()
                    .is_some_and(|s| s.contains("save_linear_artifact"))
        });
        assert!(has_demo, "expected local_tiny_linear_regression with save_linear_artifact workflow");
    }
}
