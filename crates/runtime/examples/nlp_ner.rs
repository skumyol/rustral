//! Named Entity Recognition (NER) Task Example
//!
//! Demonstrates high-performance symbolic structures for NER tasks.

use rustral_core::{ForwardCtx, Mode};
use rustral_ndarray_backend::CpuBackend;
use rustral_symbolic::{Document, Entity, Sentence, Span, Token};

fn main() -> anyhow::Result<()> {
    let backend = CpuBackend::default();
    let _ctx = ForwardCtx::new(&backend, Mode::Inference);

    // 1. Setup Document structure (Symbolic)
    let mut doc = Document {
        text: "Bolt is a high-performance system for NLP research at EMNLP.".to_string(),
        sentences: vec![],
        entities: vec![],
    };

    let tokens = vec![
        Token { text: "Bolt".into(), id: 0, span: Span::new(0, 4), pos: Some("NNP".into()) },
        Token { text: "is".into(), id: 1, span: Span::new(5, 7), pos: Some("VBZ".into()) },
        Token { text: "a".into(), id: 2, span: Span::new(8, 9), pos: Some("DT".into()) },
        Token { text: "system".into(), id: 3, span: Span::new(10, 16), pos: Some("NN".into()) },
        Token { text: "for".into(), id: 4, span: Span::new(17, 20), pos: Some("IN".into()) },
        Token { text: "NLP".into(), id: 5, span: Span::new(21, 24), pos: Some("NNP".into()) },
        Token { text: "research".into(), id: 6, span: Span::new(25, 33), pos: Some("NN".into()) },
        Token { text: "at".into(), id: 7, span: Span::new(34, 36), pos: Some("IN".into()) },
        Token { text: "EMNLP".into(), id: 8, span: Span::new(37, 42), pos: Some("NNP".into()) },
        Token { text: ".".into(), id: 9, span: Span::new(42, 43), pos: Some(".".into()) },
    ];

    let sentence = Sentence { tokens, dependency_graph: None };
    doc.sentences.push(sentence);

    // 2. Add identified Entities
    doc.entities.push(Entity { span: Span::new(0, 4), label: "SYSTEM".into(), score: Some(0.98) });
    doc.entities.push(Entity { span: Span::new(37, 42), label: "ORG".into(), score: Some(0.95) });

    println!("--- Rustral NER Task Example ---");
    println!("Document: {}", doc.text);
    println!("Detected Entities:");
    for entity in &doc.entities {
        let text = &doc.text[entity.span.start..entity.span.end];
        println!("  [{}] '{}' (score: {})", entity.label, text, entity.score.unwrap_or(0.0));
    }

    Ok(())
}
