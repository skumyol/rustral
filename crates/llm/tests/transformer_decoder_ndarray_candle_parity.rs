//! Ndarray (`CpuBackend`) vs Candle (`CandleBackend::cpu`) parity for the GPT-style
//! [`rustral_nn::TransformerDecoder`] forward path used by `Gpt2Decoder`.
//!
//! Weights are copied explicitly by stable `NamedParameters` keys so both stacks run the same tensors.

use std::collections::HashMap;

use rustral_candle_backend::CandleBackend;
use rustral_core::{Backend, ForwardCtx, Mode, NamedParameters, Parameter};
use rustral_ndarray_backend::CpuBackend;
use rustral_nn::{TransformerDecoder, TransformerDecoderConfig};

fn snapshot_decoder_params<B: Backend>(
    model: &TransformerDecoder<B>,
    backend: &B,
) -> HashMap<String, (Vec<f32>, Vec<usize>)>
where
    B::Tensor: Clone,
{
    let mut out = HashMap::new();
    model.visit_parameters(&mut |name: &str, p: &Parameter<B>| {
        let shape = backend.ops().shape(p.tensor());
        let data = backend.ops().tensor_to_vec(p.tensor()).expect("tensor_to_vec");
        out.insert(name.to_string(), (data, shape));
    });
    out
}

fn load_decoder_params<B: Backend>(
    model: &mut TransformerDecoder<B>,
    backend: &B,
    snapshot: &HashMap<String, (Vec<f32>, Vec<usize>)>,
) where
    B::Tensor: Clone,
{
    model.visit_parameters_mut(&mut |name: &str, p: &mut Parameter<B>| {
        if let Some((data, shape)) = snapshot.get(name) {
            let t = backend.ops().tensor_from_vec(data.clone(), shape).expect("tensor_from_vec");
            *p = p.clone().with_tensor(t);
        }
    });
}

fn assert_logits_close(cpu: &[f32], candle: &[f32], atol: f32) {
    assert_eq!(cpu.len(), candle.len(), "logits length mismatch");
    let mut max_abs = 0f32;
    for (i, (a, b)) in cpu.iter().zip(candle.iter()).enumerate() {
        let d = (*a - *b).abs();
        max_abs = max_abs.max(d);
        assert!(d <= atol, "logits differ at {i}: cpu={a} candle={b} diff={d} (max tol {atol})");
    }
}

#[test]
fn transformer_decoder_forward_matches_cpu_on_candle_cpu() {
    let d_model = 32usize;
    let n_head = 4usize;
    let n_layer = 2usize;
    let ff_dim = d_model * 4;
    let vocab = 48usize;
    let max_seq = 64usize;
    let seed = 7u64;

    let mut cfg = TransformerDecoderConfig::new(d_model, n_head, n_layer, ff_dim).with_max_seq_len(max_seq);
    cfg.dropout = 0.0;

    let cpu_backend = CpuBackend::default();
    let candle_backend = CandleBackend::cpu();

    let cpu_dec = TransformerDecoder::new(&cpu_backend, cfg.clone(), vocab, seed).expect("cpu decoder");
    let snap = snapshot_decoder_params(&cpu_dec, &cpu_backend);

    let mut candle_dec =
        TransformerDecoder::new(&candle_backend, cfg, vocab, seed.wrapping_add(999)).expect("candle decoder");
    load_decoder_params(&mut candle_dec, &candle_backend, &snap);

    let candle_snap = snapshot_decoder_params(&candle_dec, &candle_backend);
    assert_eq!(snap.len(), candle_snap.len());
    for (k, (cpu_data, cpu_sh)) in &snap {
        let (can_data, can_sh) = candle_snap.get(k).expect("missing key after load");
        assert_eq!(cpu_sh, can_sh, "{k}");
        assert_eq!(cpu_data, can_data, "{k} weight mismatch after copy");
    }

    let input_ids: Vec<usize> = vec![3, 11, 7, 3];

    let mut ctx_cpu = ForwardCtx::new(&cpu_backend, Mode::Inference);
    let mut ctx_candle = ForwardCtx::new(&candle_backend, Mode::Inference);

    let logits_cpu = cpu_dec.forward(input_ids.clone(), &mut ctx_cpu).expect("cpu forward");
    let logits_candle = candle_dec.forward(input_ids.clone(), &mut ctx_candle).expect("candle forward");

    let cpu_vec = cpu_backend.ops().tensor_to_vec(&logits_cpu).expect("cpu vec");
    let candle_vec = candle_backend.ops().tensor_to_vec(&logits_candle).expect("candle vec");

    // Small transformer + GELU: allow modest f32 divergence between kernels.
    assert_logits_close(&cpu_vec, &candle_vec, 5e-4);

    let tok_cpu = cpu_dec.generate_token(input_ids.clone(), &mut ctx_cpu).expect("cpu token");
    let tok_candle = candle_dec.generate_token(input_ids, &mut ctx_candle).expect("candle token");
    assert_eq!(tok_cpu, tok_candle, "greedy argmax token should match");
}
