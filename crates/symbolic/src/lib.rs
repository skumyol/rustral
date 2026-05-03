//! Symbolic utilities for mapping external labels and tokens to numeric ids.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Error type for vocabulary and label operations.
#[derive(Debug, Error)]
pub enum SymbolicError {
    /// Attempted to insert a new token after freezing the vocabulary.
    #[error("vocabulary is frozen; cannot insert token {0}")]
    Frozen(String),

    /// Attempted to retrieve a token id that does not exist.
    #[error("id {0} is out of range")]
    IdOutOfRange(usize),
}

/// Standard result type for symbolic utilities.
pub type Result<T> = std::result::Result<T, SymbolicError>;

/// Bidirectional token/id mapping with a stable unknown token.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Vocabulary {
    token_to_id: HashMap<String, usize>,
    id_to_token: Vec<String>,
    unk_id: usize,
    frozen: bool,
}

impl Vocabulary {
    /// Create a new vocabulary with one unknown token at id `0`.
    pub fn with_specials(unk: impl Into<String>) -> Self {
        let unk = unk.into();
        let mut token_to_id = HashMap::new();
        token_to_id.insert(unk.clone(), 0);
        Self { token_to_id, id_to_token: vec![unk], unk_id: 0, frozen: false }
    }

    /// Insert a token and return its id.
    ///
    /// If the token already exists, returns the existing id. If the vocabulary
    /// is frozen and the token is new, returns an error.
    pub fn insert(&mut self, token: impl Into<String>) -> Result<usize> {
        let token = token.into();
        if let Some(&id) = self.token_to_id.get(&token) {
            return Ok(id);
        }
        if self.frozen {
            return Err(SymbolicError::Frozen(token));
        }
        let id = self.id_to_token.len();
        self.id_to_token.push(token.clone());
        self.token_to_id.insert(token, id);
        Ok(id)
    }

    /// Prevent future insertion of unknown tokens.
    pub fn freeze(&mut self) {
        self.frozen = true;
    }

    /// Return true if the vocabulary is frozen.
    pub fn is_frozen(&self) -> bool {
        self.frozen
    }

    /// Return the number of tokens.
    pub fn len(&self) -> usize {
        self.id_to_token.len()
    }

    /// Return true when there are no tokens.
    ///
    /// A vocabulary created with [`Vocabulary::with_specials`] is never empty.
    pub fn is_empty(&self) -> bool {
        self.id_to_token.is_empty()
    }

    /// Return the id used for unknown tokens.
    pub fn unk_id(&self) -> usize {
        self.unk_id
    }

    /// Return a token id, or `None` if the token is unknown.
    pub fn id(&self, token: &str) -> Option<usize> {
        self.token_to_id.get(token).copied()
    }

    /// Return a token id, falling back to the unknown id when missing.
    pub fn id_or_unk(&self, token: &str) -> usize {
        self.id(token).unwrap_or(self.unk_id)
    }

    /// Return the token string for an id.
    pub fn token(&self, id: usize) -> Result<&str> {
        self.id_to_token.get(id).map(String::as_str).ok_or(SymbolicError::IdOutOfRange(id))
    }

    /// Iterate over tokens in id order.
    pub fn tokens(&self) -> impl Iterator<Item = &str> {
        self.id_to_token.iter().map(String::as_str)
    }
}

/// Result of a label readout prediction.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LabelPrediction {
    /// Predicted label string.
    pub label: String,

    /// Predicted label id.
    pub id: usize,

    /// Score associated with the prediction.
    pub score: f32,
}
