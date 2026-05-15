//! NLP Task benchmarks for Rustral (Candle Backend).

use rustral_bench::{samples_to_json, time_runs, Sample};
use rustral_core::{ForwardCtx, Mode};
use rustral_candle_backend::CandleBackend;
use rustral_nn::{TransformerEncoder, TransformerEncoderConfig};
use rustral_symbolic::{Document, Entity, Sentence, Span, Token, DependencyGraph};

const BACKEND: &str = "candle-cpu-rustral";

fn main() {
    let backend = CandleBackend::default();
    let mut samples: Vec<Sample> = Vec::new();

    bench_nlp_pipeline(&backend, 5, 1, &mut samples);

    print!("{}", samples_to_json("rustral-candle-nlp", &samples));
}

fn bench_nlp_pipeline(backend: &CandleBackend, repeats: usize, warmup: usize, out: &mut Vec<Sample>) {
    let d_model = 128;
    let config = TransformerEncoderConfig::new(d_model, 4, 2, 512);
    let encoder = TransformerEncoder::new(backend, config, 1000, 42).unwrap();

    let runs = time_runs(
        || {
            let mut ctx = ForwardCtx::new(backend, Mode::Inference);
            let tokens = vec![
                Token { text: "The".to_string(), id: 0, span: Span::new(0, 3), pos: Some("DT".into()) },
                Token { text: "model".to_string(), id: 1, span: Span::new(4, 9), pos: Some("NN".into()) },
                Token { text: "learns".to_string(), id: 2, span: Span::new(10, 16), pos: Some("VBZ".into()) },
                Token { text: "quickly".to_string(), id: 3, span: Span::new(17, 24), pos: Some("RB".into()) },
                Token { text: ".".to_string(), id: 4, span: Span::new(24, 25), pos: Some(".".into()) },
            ];
            let mut graph = DependencyGraph::new(5);
            graph.add_edge(2, 1, "nsubj");
            let _doc = Document {
                text: "The model learns quickly.".to_string(),
                sentences: vec![Sentence { tokens, dependency_graph: Some(graph) }],
                entities: vec![Entity { span: Span::new(4, 9), label: "COMP".into(), score: Some(0.9) }],
            };
            let input = vec![1usize, 2, 3, 4, 5];
            let _ = encoder.forward(input, &mut ctx).unwrap();
        },
        1,
        5,
    );

    out.push(Sample::cpu_f32(
        "nlp.full_pipeline",
        BACKEND,
        vec![("d_model".into(), d_model.to_string()), ("seq_len".into(), "5".into())],
        runs,
    ));
}
