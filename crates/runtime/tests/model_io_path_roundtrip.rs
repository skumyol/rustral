#![cfg(feature = "training")]

use std::path::PathBuf;

use rustral_core::Module;
use rustral_core::{Backend, ForwardCtx, Mode};
use rustral_ndarray_backend::CpuBackend;
use rustral_nn::{chain, LinearBuilder};
use rustral_runtime::{load_model_from_path, save_model_to_path};

fn tmp_path(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let unique = format!(
        "rustral_{name}_{}_{}.safetensors",
        std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    );
    p.push(unique);
    p
}

#[test]
fn save_load_path_roundtrip_preserves_outputs() {
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

    let path = tmp_path("model_io_roundtrip");
    save_model_to_path(&path, &model_a, &backend).unwrap();
    load_model_from_path(&path, &mut model_b, &backend).unwrap();
    let _ = std::fs::remove_file(&path);

    let mut ctx_b = ForwardCtx::new(&backend, Mode::Inference);
    let out_b = model_b.forward(input, &mut ctx_b).unwrap();
    let out_b_vec = backend.ops().tensor_to_vec(&out_b).unwrap();

    assert_eq!(out_a_vec, out_b_vec);
}
