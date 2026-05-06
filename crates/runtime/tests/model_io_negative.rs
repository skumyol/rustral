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
fn load_model_errors_on_dtype_mismatch() {
    let backend = CpuBackend::default();
    let model = chain(
        LinearBuilder::new(3, 4).with_bias(true).seed(1).build(&backend).unwrap(),
        LinearBuilder::new(4, 2).with_bias(true).seed(2).build(&backend).unwrap(),
    );

    let bytes = save_model(&model, &backend).unwrap();

    // Build a safetensors file where one tensor has i64 dtype.
    // We reuse the original keys so the failure is specifically dtype-related.
    let safe = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let tensors = safe.tensors();
    let (first_name, first_view) = &tensors[0];
    let shape = first_view.shape().to_vec();
    let elem_count: usize = shape.iter().product();
    let i64s: Vec<i64> = (0..elem_count as i64).collect();
    let raw: Vec<u8> = bytemuck::cast_slice(&i64s).to_vec();

    struct I64View {
        shape: Vec<usize>,
        data: Vec<u8>,
    }
    impl safetensors::View for I64View {
        fn dtype(&self) -> safetensors::Dtype {
            safetensors::Dtype::I64
        }
        fn shape(&self) -> &[usize] {
            &self.shape
        }
        fn data(&self) -> std::borrow::Cow<'_, [u8]> {
            std::borrow::Cow::Borrowed(&self.data)
        }
        fn data_len(&self) -> usize {
            self.data.len()
        }
    }

    let mut views: std::collections::HashMap<String, I64View> = std::collections::HashMap::new();
    views.insert(first_name.to_string(), I64View { shape, data: raw });
    let broken = safetensors::serialize(views, &None).unwrap();

    let mut model_b = chain(
        LinearBuilder::new(3, 4).with_bias(true).seed(10).build(&backend).unwrap(),
        LinearBuilder::new(4, 2).with_bias(true).seed(20).build(&backend).unwrap(),
    );

    let err = load_model(&mut model_b, &backend, &broken).unwrap_err();
    assert!(
        err.to_string().to_lowercase().contains("dtype"),
        "expected dtype error, got: {err}"
    );
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

