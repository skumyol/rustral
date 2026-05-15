//! Dependency Parsing Task Example
//!
//! Demonstrates using Rustral's optimized DependencyGraph for
//! linguistic structural analysis.

use rustral_core::{ForwardCtx, Mode};
use rustral_ndarray_backend::CpuBackend;
use rustral_symbolic::{DependencyGraph, Sentence, Span, Token};

fn main() -> anyhow::Result<()> {
    println!("--- Rustral Dependency Parsing Task Example ---");

    // 1. Define tokens
    let tokens = vec![
        Token { text: "The".to_string(), id: 0, span: Span::new(0, 3), pos: Some("DT".into()) },
        Token { text: "model".to_string(), id: 1, span: Span::new(4, 9), pos: Some("NN".into()) },
        Token { text: "learns".to_string(), id: 2, span: Span::new(10, 16), pos: Some("VBZ".into()) },
        Token { text: "quickly".to_string(), id: 3, span: Span::new(17, 24), pos: Some("RB".into()) },
        Token { text: ".".to_string(), id: 4, span: Span::new(24, 25), pos: Some(".".into()) },
    ];

    // 2. Create Dependency Graph
    let mut graph = DependencyGraph::new(tokens.len());
    graph.add_edge(2, 1, "nsubj");  // learns -> model
    graph.add_edge(1, 0, "det");    // model -> The
    graph.add_edge(2, 3, "advmod"); // learns -> quickly
    graph.add_edge(2, 4, "punct");  // learns -> .

    let sentence = Sentence {
        tokens,
        dependency_graph: Some(graph),
    };

    // 3. Analyze dependencies
    if let Some(ref dg) = sentence.dependency_graph {
        println!("Root token: '{}'", sentence.tokens[2].text);
        let dependents = dg.children(2);
        println!("Dependents of 'learns':");
        for dep_idx in dependents {
            let token = &sentence.tokens[dep_idx];
            println!("  -> '{}' (pos: {:?})", token.text, token.pos);
        }
    }

    Ok(())
}
