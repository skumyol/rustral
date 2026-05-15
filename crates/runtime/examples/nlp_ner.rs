//! NER (Named Entity Recognition) Task Example
//!
//! Demonstrates using Rustral's native Entity and Span structures for
//! efficient NLP entity extraction and metadata management.

use rustral_core::{ForwardCtx, Mode};
use rustral_ndarray_backend::CpuBackend;
use rustral_symbolic::{Document, Entity, Sentence, Span, Token};

fn main() -> anyhow::Result<()> {
    let backend = CpuBackend::default();
    let mut ctx = ForwardCtx::new(&backend, Mode::Inference);

    println!("--- Rustral NER Task Example ---");

    // 1. Create a Document with sentences and tokens
    let mut doc = Document {
        text: "Bolt is a high-performance system for NLP research at EMNLP.".to_string(),
        sentences: Vec::new(),
        entities: Vec::new(),
    };

    let sentence = Sentence {
        tokens: vec![
            Token { text: "Bolt".to_string(), id: 0, span: Span::new(0, 4), pos: None },
            Token { text: "is".to_string(), id: 1, span: Span::new(5, 7), pos: None },
            Token { text: "a".to_string(), id: 2, span: Span::new(8, 9), pos: None },
            Token { text: "high-performance".to_string(), id: 3, span: Span::new(10, 26), pos: None },
            Token { text: "system".to_string(), id: 4, span: Span::new(27, 33), pos: None },
            Token { text: "for".to_string(), id: 5, span: Span::new(34, 37), pos: None },
            Token { text: "NLP".to_string(), id: 6, span: Span::new(38, 41), pos: None },
            Token { text: "research".to_string(), id: 7, span: Span::new(42, 50), pos: None },
            Token { text: "at".to_string(), id: 8, span: Span::new(51, 53), pos: None },
            Token { text: "EMNLP".to_string(), id: 9, span: Span::new(54, 59), pos: None },
            Token { text: ".".to_string(), id: 10, span: Span::new(59, 60), pos: None },
        ],
        dependency_graph: None,
    };
    doc.sentences.push(sentence);

    // 2. Add identified Entities
    doc.entities.push(Entity {
        span: Span::new(0, 4),
        label: "SYSTEM".into(),
        score: Some(0.98),
    });
    doc.entities.push(Entity {
        span: Span::new(54, 59),
        label: "ORG".into(),
        score: Some(0.95),
    });

    println!("Document: {}", doc.text);
    println!("Detected Entities:");
    for entity in &doc.entities {
        let text = &doc.text[entity.span.start..entity.span.end];
        println!("  [{}] '{}' (score: {:.2})", entity.label, text, entity.score.unwrap_or(0.0));
    }

    Ok(())
}
