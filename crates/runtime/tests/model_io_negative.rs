#![cfg(feature = "training")]

use std::collections::HashMap;

use rustral_core::{Backend, ForwardCtx, Mode, Module};
use rustral_ndarray_backend::CpuBackend;
use rustral_nn::{chain, LinearBuilder};
use rustral_runtime::{load_model, save_model};

#[test]
fn load_model_errors_on_missing_key() {
    let backend = CpuBackend::default();
    let model = chain(
        LinearBuilder::new(3, 4).with_bias(true).seed(1).build(&backend).unwrap(),
        LinearBuilder::new(4, 2).with_bias(true).seed(2).build(&backend).unwrap(),
    );

    let bytes = save_model(&model, &backend).unwrap();
    let mut dict: HashMap<String, Vec<f32>> = rustral_io::load_parameters(&bytes).unwrap();
    let victim = dict.keys().next().cloned().expect("state dict must not be empty");
    dict.remove(&victim);
    let broken = rustral_io::save_state_dict(&dict).unwrap();

    let mut model_b = chain(
        LinearBuilder::new(3, 4).with_bias(true).seed(10).build(&backend).unwrap(),
        LinearBuilder::new(4, 2).with_bias(true).seed(20).build(&backend).unwrap(),
    );
    let err = load_model(&mut model_b, &backend, &broken).unwrap_err();
    assert!(err.to_string().contains("missing keys"));
}

#[test]
fn load_model_errors_on_extra_key() {
    let backend = CpuBackend::default();
    let model = chain(
        LinearBuilder::new(3, 4).with_bias(true).seed(1).build(&backend).unwrap(),
        LinearBuilder::new(4, 2).with_bias(true).seed(2).build(&backend).unwrap(),
    );

    let bytes = save_model(&model, &backend).unwrap();
    let mut dict: HashMap<String, Vec<f32>> = rustral_io::load_parameters(&bytes).unwrap();
    dict.insert("extra.unexpected".to_string(), vec![0.0, 1.0, 2.0]);
    let broken = rustral_io::save_state_dict(&dict).unwrap();

    let mut model_b = chain(
        LinearBuilder::new(3, 4).with_bias(true).seed(10).build(&backend).unwrap(),
        LinearBuilder::new(4, 2).with_bias(true).seed(20).build(&backend).unwrap(),
    );
    let err = load_model(&mut model_b, &backend, &broken).unwrap_err();
    assert!(err.to_string().contains("extra keys"));
}

#[test]
fn load_model_errors_on_shape_mismatch() {
    let backend = CpuBackend::default();
    let model = chain(
        LinearBuilder::new(3, 4).with_bias(true).seed(1).build(&backend).unwrap(),
        LinearBuilder::new(4, 2).with_bias(true).seed(2).build(&backend).unwrap(),
    );

    let bytes = save_model(&model, &backend).unwrap();
    let mut dict: HashMap<String, Vec<f32>> = rustral_io::load_parameters(&bytes).unwrap();
    let victim = dict.keys().next().cloned().expect("state dict must not be empty");
    let mut v = dict.remove(&victim).unwrap();
    v.pop(); // wrong length
    dict.insert(victim, v);
    let broken = rustral_io::save_state_dict(&dict).unwrap();

    let mut model_b = chain(
        LinearBuilder::new(3, 4).with_bias(true).seed(10).build(&backend).unwrap(),
        LinearBuilder::new(4, 2).with_bias(true).seed(20).build(&backend).unwrap(),
    );
    let err = load_model(&mut model_b, &backend, &broken).unwrap_err();
    assert!(err.to_string().contains("shape mismatch"));
}

#[test]
fn strict_load_still_preserves_outputs_on_valid_bytes() {
    let backend = CpuBackend::default();
    let model_a = chain(
        LinearBuilder::new(3, 4).with_bias(true).seed(1).build(&backend).unwrap(),
        LinearBuilder::new(4, 2).with_bias(true).seed(2).build(&backend).unwrap(),
    );
    let mut model_b = chain(
        LinearBuilder::new(3, 4).with_bias(true).seed(10).build(&backend).unwrap(),
        LinearBuilder::new(4, 2).with_bias(true).seed(20).build(&backend).unwrap(),
    );

    let input = backend.tensor_from_vec(vec![0.1, 0.2, -0.3], &[3]).unwrap();
    let mut ctx_a = ForwardCtx::new(&backend, Mode::Inference);
    let out_a = model_a.forward(input.clone(), &mut ctx_a).unwrap();
    let out_a_vec = backend.ops().tensor_to_vec(&out_a).unwrap();

    let bytes = save_model(&model_a, &backend).unwrap();
    load_model(&mut model_b, &backend, &bytes).unwrap();

    let mut ctx_b = ForwardCtx::new(&backend, Mode::Inference);
    let out_b = model_b.forward(input, &mut ctx_b).unwrap();
    let out_b_vec = backend.ops().tensor_to_vec(&out_b).unwrap();
    assert_eq!(out_a_vec, out_b_vec);
}

