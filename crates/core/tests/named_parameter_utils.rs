use std::collections::BTreeSet;

use rustral_core::{collect_named_parameter_ids, collect_named_parameters, NamedParameters};
use rustral_ndarray_backend::CpuBackend;
use rustral_nn::{chain, LinearBuilder, TransformerEncoder, TransformerEncoderConfig};

#[test]
fn collect_named_parameters_is_deterministic_for_same_structure() {
    let backend = CpuBackend::default();
    let a = chain(
        LinearBuilder::new(3, 4).with_bias(true).seed(1).build(&backend).unwrap(),
        LinearBuilder::new(4, 2).with_bias(true).seed(2).build(&backend).unwrap(),
    );
    let b = chain(
        LinearBuilder::new(3, 4).with_bias(true).seed(10).build(&backend).unwrap(),
        LinearBuilder::new(4, 2).with_bias(true).seed(20).build(&backend).unwrap(),
    );

    let na: Vec<String> = collect_named_parameters(&a).into_iter().map(|(n, _)| n).collect();
    let nb: Vec<String> = collect_named_parameters(&b).into_iter().map(|(n, _)| n).collect();
    assert_eq!(na, nb);
}

#[test]
fn collect_named_parameter_ids_has_unique_names() {
    let backend = CpuBackend::default();
    let cfg = TransformerEncoderConfig::new(16, 4, 2, 32).with_max_seq_len(32);
    let m = TransformerEncoder::new(&backend, cfg, 64, 1).unwrap();

    let id_map = collect_named_parameter_ids(&m);
    assert!(!id_map.is_empty());
    let uniq: BTreeSet<String> = id_map.values().cloned().collect();
    assert_eq!(uniq.len(), id_map.len());
}

#[test]
fn collect_named_parameters_refs_match_visit_parameters_ids() {
    let backend = CpuBackend::default();
    let m = LinearBuilder::new(3, 4).with_bias(true).seed(1).build(&backend).unwrap();

    let refs = collect_named_parameters(&m);
    let mut seen_ids = Vec::new();
    m.visit_parameters(&mut |_n, p| seen_ids.push(p.id()));

    let ref_ids: Vec<_> = refs.into_iter().map(|(_n, r)| r.id).collect();
    assert_eq!(ref_ids, seen_ids);
}

