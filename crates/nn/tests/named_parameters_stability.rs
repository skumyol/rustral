use std::collections::BTreeSet;

use rustral_core::NamedParameters;
use rustral_ndarray_backend::CpuBackend;
use rustral_nn::{chain, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, LinearBuilder, TransformerDecoder,
    TransformerDecoderConfig, TransformerEncoder, TransformerEncoderConfig};

fn collect_names<B: rustral_core::Backend, M: NamedParameters<B>>(m: &M) -> Vec<String> {
    let mut names = Vec::new();
    m.visit_parameters(&mut |name, _p| names.push(name.to_string()));
    names
}

fn assert_stable_names<T: Eq + std::fmt::Debug>(a: Vec<T>, b: Vec<T>) {
    assert_eq!(a, b);
}

#[test]
fn named_parameters_linear_are_stable() {
    let backend = CpuBackend::default();
    let a = LinearBuilder::new(3, 4).with_bias(true).seed(1).build(&backend).unwrap();
    let b = LinearBuilder::new(3, 4).with_bias(true).seed(999).build(&backend).unwrap();
    assert_stable_names(collect_names(&a), collect_names(&b));
}

#[test]
fn named_parameters_embedding_are_stable() {
    let backend = CpuBackend::default();
    let a = Embedding::new(&backend, EmbeddingConfig::new(16, 8), 1).unwrap();
    let b = Embedding::new(&backend, EmbeddingConfig::new(16, 8), 999).unwrap();
    assert_stable_names(collect_names(&a), collect_names(&b));
}

#[test]
fn named_parameters_layer_norm_are_stable() {
    let backend = CpuBackend::default();
    let a = LayerNorm::new(&backend, LayerNormConfig::new(vec![8]).with_eps(1e-5), 1).unwrap();
    let b = LayerNorm::new(&backend, LayerNormConfig::new(vec![8]).with_eps(1e-5), 999).unwrap();
    assert_stable_names(collect_names(&a), collect_names(&b));
}

#[test]
fn named_parameters_sequential2_are_stable_and_hierarchical() {
    let backend = CpuBackend::default();
    let a = chain(
        LinearBuilder::new(3, 4).with_bias(true).seed(1).build(&backend).unwrap(),
        LinearBuilder::new(4, 2).with_bias(true).seed(2).build(&backend).unwrap(),
    );
    let b = chain(
        LinearBuilder::new(3, 4).with_bias(true).seed(10).build(&backend).unwrap(),
        LinearBuilder::new(4, 2).with_bias(true).seed(20).build(&backend).unwrap(),
    );
    let names_a = collect_names(&a);
    let names_b = collect_names(&b);
    assert_stable_names(names_a.clone(), names_b);

    // Cheap hierarchy sanity check: sequential naming should contain a dot.
    assert!(names_a.iter().any(|n| n.contains('.')), "expected hierarchical names, got {names_a:?}");
}

#[test]
fn named_parameters_transformer_encoder_are_stable() {
    let backend = CpuBackend::default();
    let config = TransformerEncoderConfig::new(16, 4, 2, 32).with_max_seq_len(32);
    let a = TransformerEncoder::new(&backend, config.clone(), 64, 1).unwrap();
    let b = TransformerEncoder::new(&backend, config, 64, 999).unwrap();
    assert_stable_names(collect_names(&a), collect_names(&b));
}

#[test]
fn named_parameters_transformer_decoder_are_stable() {
    let backend = CpuBackend::default();
    let config = TransformerDecoderConfig::new(16, 4, 2, 32).with_max_seq_len(32);
    let a = TransformerDecoder::new(&backend, config.clone(), 64, 1).unwrap();
    let b = TransformerDecoder::new(&backend, config, 64, 999).unwrap();
    assert_stable_names(collect_names(&a), collect_names(&b));
}

#[test]
fn named_parameters_have_no_duplicates() {
    let backend = CpuBackend::default();
    let config = TransformerEncoderConfig::new(16, 4, 2, 32).with_max_seq_len(32);
    let m = TransformerEncoder::new(&backend, config, 64, 1).unwrap();
    let names = collect_names(&m);
    let uniq: BTreeSet<String> = names.iter().cloned().collect();
    assert_eq!(uniq.len(), names.len(), "duplicate parameter names detected");
}

