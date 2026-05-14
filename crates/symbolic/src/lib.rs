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

/// Represents a span in a text (start and end token indices).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    pub fn length(&self) -> usize {
        self.end - self.start
    }

    pub fn contains(&self, index: usize) -> bool {
        index >= self.start && index < self.end
    }
}

/// Represents a named entity in a text.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Entity {
    pub span: Span,
    pub label: String,
    pub score: Option<f32>,
}

/// Represents a dependency relation between two tokens.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DependencyEdge {
    pub head: usize,
    pub dependent: usize,
    pub relation: String,
}

/// Optimized structure for representing a dependency parse tree or graph.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DependencyGraph {
    pub nodes: usize,
    pub edges: Vec<DependencyEdge>,
}

impl DependencyGraph {
    pub fn new(nodes: usize) -> Self {
        Self { nodes, edges: Vec::new() }
    }

    pub fn add_edge(&mut self, head: usize, dependent: usize, relation: impl Into<String>) {
        self.edges.push(DependencyEdge { head, dependent, relation: relation.into() });
    }

    /// Return children of a given head node.
    pub fn children(&self, head: usize) -> Vec<usize> {
        self.edges.iter().filter(|e| e.head == head).map(|e| e.dependent).collect()
    }
}

/// A basic token in a document.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Token {
    pub text: String,
    pub id: usize,
    pub span: Span,
    pub pos: Option<String>,
}

/// A sentence within a document.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Sentence {
    pub tokens: Vec<Token>,
    pub dependency_graph: Option<DependencyGraph>,
}

/// A full document with linguistic metadata.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Document {
    pub text: String,
    pub sentences: Vec<Sentence>,
    pub entities: Vec<Entity>,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocabulary_with_specials() {
        let vocab = Vocabulary::with_specials("<unk>");
        assert_eq!(vocab.len(), 1);
        assert_eq!(vocab.unk_id(), 0);
        assert!(!vocab.is_empty());
        assert!(!vocab.is_frozen());
    }

    #[test]
    fn test_vocabulary_insert_and_lookup() {
        let mut vocab = Vocabulary::with_specials("<unk>");
        let id = vocab.insert("hello").unwrap();
        assert_eq!(vocab.id("hello"), Some(id));
        assert_eq!(vocab.token(id).unwrap(), "hello");
        assert_eq!(vocab.id_or_unk("hello"), id);
        assert_eq!(vocab.id_or_unk("missing"), 0);
    }

    #[test]
    fn test_vocabulary_duplicate_insert_returns_same_id() {
        let mut vocab = Vocabulary::with_specials("<unk>");
        let id1 = vocab.insert("hello").unwrap();
        let id2 = vocab.insert("hello").unwrap();
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_vocabulary_freeze() {
        let mut vocab = Vocabulary::with_specials("<unk>");
        vocab.insert("hello").unwrap();
        vocab.freeze();
        assert!(vocab.is_frozen());
        assert!(vocab.insert("world").is_err());
    }

    #[test]
    fn test_vocabulary_token_out_of_range() {
        let vocab = Vocabulary::with_specials("<unk>");
        assert!(vocab.token(99).is_err());
    }

    #[test]
    fn test_vocabulary_tokens_iterator() {
        let mut vocab = Vocabulary::with_specials("<unk>");
        vocab.insert("a").unwrap();
        vocab.insert("b").unwrap();
        let tokens: Vec<&str> = vocab.tokens().collect();
        assert_eq!(tokens, vec!["<unk>", "a", "b"]);
    }

    #[test]
    fn test_label_prediction() {
        let pred = LabelPrediction { label: "cat".to_string(), id: 0, score: 0.9 };
        assert_eq!(pred.label, "cat");
    }

    #[test]
    fn test_span() {
        let span = Span::new(5, 10);
        assert_eq!(span.length(), 5);
        assert!(span.contains(7));
        assert!(!span.contains(10));
    }

    #[test]
    fn test_dependency_graph() {
        let mut graph = DependencyGraph::new(5);
        graph.add_edge(0, 1, "nsubj");
        graph.add_edge(0, 2, "obj");
        let children = graph.children(0);
        assert_eq!(children.len(), 2);
        assert!(children.contains(&1));
        assert!(children.contains(&2));
    }

    #[test]
    fn test_document_structure() {
        let mut doc =
            Document { text: "Bolt is fast.".to_string(), sentences: Vec::new(), entities: Vec::new() };

        let mut sentence = Sentence {
            tokens: vec![
                Token { text: "Bolt".to_string(), id: 0, span: Span::new(0, 4), pos: Some("NNP".into()) },
                Token { text: "is".to_string(), id: 1, span: Span::new(5, 7), pos: Some("VBZ".into()) },
                Token { text: "fast".to_string(), id: 2, span: Span::new(8, 12), pos: Some("JJ".into()) },
                Token { text: ".".to_string(), id: 3, span: Span::new(12, 13), pos: Some(".".into()) },
            ],
            dependency_graph: None,
        };

        let mut graph = DependencyGraph::new(4);
        graph.add_edge(1, 0, "nsubj");
        graph.add_edge(1, 2, "acomp");
        sentence.dependency_graph = Some(graph);

        doc.sentences.push(sentence);
        doc.entities.push(Entity { span: Span::new(0, 4), label: "PER".into(), score: Some(0.99) });

        assert_eq!(doc.sentences[0].tokens.len(), 4);
        assert_eq!(doc.entities[0].label, "PER");
    }
}
