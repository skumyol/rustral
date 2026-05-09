//! Tape-friendly transformer encoder block.
//!
//! Built only from primitives that the autodiff `Tape` already supports
//! (`matmul`, `transpose_tape`, `softmax`, `slice_tape`, `concat_tape`, `mul_scalar`,
//! `add`, `relu`, `layer_norm_tape`, plus `Linear`/`Embedding` `TapeModule` impls).
//!
//! Multi-head attention is implemented by computing the full Q/K/V projections once and
//! then extracting each head's `d_head` columns through a `transpose -> slice -> transpose`
//! pattern. Heads are recombined by transposing each head, concatenating along dim 0, and
//! transposing back. This keeps every tape op rank-2 and the backward path well-defined.
//!
//! Single-example shape contract: forward takes `[seq_len, d_model]` and returns the same
//! shape, mirroring how `TapeTrainer::fit_classification` already drives the model
//! one example at a time.
//!
//! Feature-gated by `rustral-nn/autodiff` so this only compiles when the autodiff machinery
//! is available.

use rustral_autodiff::{Tape, TensorId};
use rustral_core::{Backend, ForwardCtx, NamedParameters, Parameter, Result};

use crate::tape::TapeModule;
use crate::{LayerNorm, LayerNormConfig, Linear, LinearBuilder};

/// Configuration for [`TapeTransformerEncoderLayer`].
#[derive(Clone, Debug)]
pub struct TapeTransformerEncoderConfig {
    pub d_model: usize,
    pub num_heads: usize,
    pub ffn_dim: usize,
    pub layer_norm_eps: f32,
    pub bias: bool,
}

impl TapeTransformerEncoderConfig {
    pub fn new(d_model: usize, num_heads: usize, ffn_dim: usize) -> Self {
        assert!(num_heads >= 1, "num_heads must be >= 1");
        assert!(d_model % num_heads == 0, "d_model must be divisible by num_heads");
        Self { d_model, num_heads, ffn_dim, layer_norm_eps: 1e-5, bias: true }
    }
}

/// Multi-head self-attention with tape-driven backward.
///
/// Parameters: `wq`, `wk`, `wv`, `wo` (each a `Linear` of shape
/// `[d_model -> d_model]`). Bias inclusion follows the layer config.
pub struct TapeMultiHeadAttention<B: Backend> {
    pub wq: Linear<B>,
    pub wk: Linear<B>,
    pub wv: Linear<B>,
    pub wo: Linear<B>,
    num_heads: usize,
    d_head: usize,
    inv_sqrt_d_head: f32,
}

impl<B: Backend> TapeMultiHeadAttention<B>
where
    B::Tensor: Clone,
{
    pub fn new(backend: &B, cfg: &TapeTransformerEncoderConfig, seed: u64) -> Result<Self> {
        let d_model = cfg.d_model;
        let bias = cfg.bias;
        let mk = |out: usize, s: u64| -> Result<Linear<B>> {
            LinearBuilder::new(d_model, out).with_bias(bias).seed(s).build(backend)
        };
        let wq = mk(d_model, seed.wrapping_add(11))?;
        let wk = mk(d_model, seed.wrapping_add(13))?;
        let wv = mk(d_model, seed.wrapping_add(17))?;
        let wo = mk(d_model, seed.wrapping_add(19))?;
        let d_head = d_model / cfg.num_heads;
        let inv_sqrt_d_head = 1.0_f32 / (d_head as f32).sqrt();
        Ok(Self { wq, wk, wv, wo, num_heads: cfg.num_heads, d_head, inv_sqrt_d_head })
    }

    pub fn num_heads(&self) -> usize {
        self.num_heads
    }
    pub fn d_head(&self) -> usize {
        self.d_head
    }

    /// Run the attention block into the tape.
    ///
    /// `x` has shape `[seq_len, d_model]`. Optional `mask` has shape `[seq_len, seq_len]` and
    /// is added to the pre-softmax scores (use `-1e9` to suppress positions; `0.0` elsewhere).
    pub fn forward_tape_with_mask(
        &self,
        x: TensorId,
        mask: Option<TensorId>,
        tape: &mut Tape<B>,
        ctx: &mut ForwardCtx<B>,
    ) -> Result<TensorId> {
        let q = self.wq.forward_tape(x, tape, ctx)?;
        let k = self.wk.forward_tape(x, tape, ctx)?;
        let v = self.wv.forward_tape(x, tape, ctx)?;

        // Transpose each to [d_model, seq_len] so we can slice rows == columns of original.
        let q_t = tape.transpose_tape(q, ctx)?;
        let k_t = tape.transpose_tape(k, ctx)?;
        let v_t = tape.transpose_tape(v, ctx)?;

        // Per-head outputs as [d_head, seq_len] tensors so we can stack them with concat(dim=0).
        let mut head_outs_t: Vec<TensorId> = Vec::with_capacity(self.num_heads);
        for h in 0..self.num_heads {
            let lo = h * self.d_head;
            let hi = lo + self.d_head;
            let qh_t = tape.slice_tape(q_t, lo, hi, ctx)?; // [d_head, seq_len]
            let kh_t = tape.slice_tape(k_t, lo, hi, ctx)?;
            let vh_t = tape.slice_tape(v_t, lo, hi, ctx)?;

            // Transpose back to [seq_len, d_head] for the standard attention recipe.
            let qh = tape.transpose_tape(qh_t, ctx)?; // [seq_len, d_head]
            let kh = tape.transpose_tape(kh_t, ctx)?; // [seq_len, d_head]
            let vh = tape.transpose_tape(vh_t, ctx)?; // [seq_len, d_head]

            // scores = qh @ kh^T -> [seq_len, seq_len]
            let kh_tt = tape.transpose_tape(kh, ctx)?; // [d_head, seq_len]
            let scores = tape.matmul(qh, kh_tt, ctx)?;
            let scores = tape.mul_scalar(scores, self.inv_sqrt_d_head, ctx)?;
            let scores = if let Some(m) = mask { tape.add(scores, m, ctx)? } else { scores };

            // weights = softmax(scores) -> [seq_len, seq_len]
            let weights = tape.softmax(scores, ctx)?;

            // head_out = weights @ vh -> [seq_len, d_head]
            let out_h = tape.matmul(weights, vh, ctx)?;

            // Transpose to [d_head, seq_len] so the per-head outputs stack cleanly along dim 0.
            let out_h_t = tape.transpose_tape(out_h, ctx)?;
            head_outs_t.push(out_h_t);
        }

        // Concat heads along dim 0: each is [d_head, seq_len]; stacked is [d_model, seq_len].
        let concat_t =
            if self.num_heads == 1 { head_outs_t[0] } else { tape.concat_tape(&head_outs_t, 0, ctx)? };
        // Transpose back to [seq_len, d_model] for the output projection.
        let concat = tape.transpose_tape(concat_t, ctx)?;

        // Output projection.
        self.wo.forward_tape(concat, tape, ctx)
    }
}

impl<B: Backend> NamedParameters<B> for TapeMultiHeadAttention<B> {
    fn visit_parameters(&self, f: &mut dyn FnMut(&str, &Parameter<B>)) {
        self.wq.visit_parameters(&mut |n, p| f(&format!("wq.{n}"), p));
        self.wk.visit_parameters(&mut |n, p| f(&format!("wk.{n}"), p));
        self.wv.visit_parameters(&mut |n, p| f(&format!("wv.{n}"), p));
        self.wo.visit_parameters(&mut |n, p| f(&format!("wo.{n}"), p));
    }
    fn visit_parameters_mut(&mut self, f: &mut dyn FnMut(&str, &mut Parameter<B>)) {
        self.wq.visit_parameters_mut(&mut |n, p| f(&format!("wq.{n}"), p));
        self.wk.visit_parameters_mut(&mut |n, p| f(&format!("wk.{n}"), p));
        self.wv.visit_parameters_mut(&mut |n, p| f(&format!("wv.{n}"), p));
        self.wo.visit_parameters_mut(&mut |n, p| f(&format!("wo.{n}"), p));
    }
}

/// Position-wise feed-forward block: `Linear -> ReLU -> Linear`.
pub struct TapeFeedForward<B: Backend> {
    pub fc1: Linear<B>,
    pub fc2: Linear<B>,
}

impl<B: Backend> TapeFeedForward<B>
where
    B::Tensor: Clone,
{
    pub fn new(backend: &B, cfg: &TapeTransformerEncoderConfig, seed: u64) -> Result<Self> {
        let bias = cfg.bias;
        let fc1 = LinearBuilder::new(cfg.d_model, cfg.ffn_dim)
            .with_bias(bias)
            .seed(seed.wrapping_add(31))
            .build(backend)?;
        let fc2 = LinearBuilder::new(cfg.ffn_dim, cfg.d_model)
            .with_bias(bias)
            .seed(seed.wrapping_add(37))
            .build(backend)?;
        Ok(Self { fc1, fc2 })
    }
}

impl<B: Backend> TapeModule<B> for TapeFeedForward<B>
where
    B::Tensor: Clone,
{
    fn forward_tape(&self, x: TensorId, tape: &mut Tape<B>, ctx: &mut ForwardCtx<B>) -> Result<TensorId> {
        let h = self.fc1.forward_tape(x, tape, ctx)?;
        let a = tape.relu(h, ctx)?;
        self.fc2.forward_tape(a, tape, ctx)
    }
}

impl<B: Backend> NamedParameters<B> for TapeFeedForward<B> {
    fn visit_parameters(&self, f: &mut dyn FnMut(&str, &Parameter<B>)) {
        self.fc1.visit_parameters(&mut |n, p| f(&format!("fc1.{n}"), p));
        self.fc2.visit_parameters(&mut |n, p| f(&format!("fc2.{n}"), p));
    }
    fn visit_parameters_mut(&mut self, f: &mut dyn FnMut(&str, &mut Parameter<B>)) {
        self.fc1.visit_parameters_mut(&mut |n, p| f(&format!("fc1.{n}"), p));
        self.fc2.visit_parameters_mut(&mut |n, p| f(&format!("fc2.{n}"), p));
    }
}

/// One pre-LN transformer encoder block.
///
/// Forward: `x = x + Attn(LN1(x), mask)` then `x = x + FFN(LN2(x))`.
pub struct TapeTransformerEncoderLayer<B: Backend> {
    pub ln1: LayerNorm<B>,
    pub attn: TapeMultiHeadAttention<B>,
    pub ln2: LayerNorm<B>,
    pub ffn: TapeFeedForward<B>,
    cfg: TapeTransformerEncoderConfig,
}

impl<B: Backend> TapeTransformerEncoderLayer<B>
where
    B::Tensor: Clone,
{
    pub fn new(backend: &B, cfg: TapeTransformerEncoderConfig, seed: u64) -> Result<Self> {
        let ln_cfg = LayerNormConfig::new(vec![cfg.d_model]).with_eps(cfg.layer_norm_eps);
        let ln1 = LayerNorm::new(backend, ln_cfg.clone(), seed.wrapping_add(41))?;
        let attn = TapeMultiHeadAttention::new(backend, &cfg, seed.wrapping_add(43))?;
        let ln2 = LayerNorm::new(backend, ln_cfg, seed.wrapping_add(47))?;
        let ffn = TapeFeedForward::new(backend, &cfg, seed.wrapping_add(53))?;
        Ok(Self { ln1, attn, ln2, ffn, cfg })
    }

    pub fn config(&self) -> &TapeTransformerEncoderConfig {
        &self.cfg
    }

    /// Forward with optional additive attention mask (`[seq_len, seq_len]`).
    pub fn forward_tape_with_mask(
        &self,
        x: TensorId,
        mask: Option<TensorId>,
        tape: &mut Tape<B>,
        ctx: &mut ForwardCtx<B>,
    ) -> Result<TensorId> {
        let n1 = self.ln1.forward_tape(x, tape, ctx)?;
        let a = self.attn.forward_tape_with_mask(n1, mask, tape, ctx)?;
        let x1 = tape.add(x, a, ctx)?;

        let n2 = self.ln2.forward_tape(x1, tape, ctx)?;
        let f = self.ffn.forward_tape(n2, tape, ctx)?;
        tape.add(x1, f, ctx)
    }
}

impl<B: Backend> TapeModule<B> for TapeTransformerEncoderLayer<B>
where
    B::Tensor: Clone,
{
    fn forward_tape(&self, x: TensorId, tape: &mut Tape<B>, ctx: &mut ForwardCtx<B>) -> Result<TensorId> {
        self.forward_tape_with_mask(x, None, tape, ctx)
    }
}

impl<B: Backend> NamedParameters<B> for TapeTransformerEncoderLayer<B> {
    fn visit_parameters(&self, f: &mut dyn FnMut(&str, &Parameter<B>)) {
        self.ln1.visit_parameters(&mut |n, p| f(&format!("ln1.{n}"), p));
        self.attn.visit_parameters(&mut |n, p| f(&format!("attn.{n}"), p));
        self.ln2.visit_parameters(&mut |n, p| f(&format!("ln2.{n}"), p));
        self.ffn.visit_parameters(&mut |n, p| f(&format!("ffn.{n}"), p));
    }
    fn visit_parameters_mut(&mut self, f: &mut dyn FnMut(&str, &mut Parameter<B>)) {
        self.ln1.visit_parameters_mut(&mut |n, p| f(&format!("ln1.{n}"), p));
        self.attn.visit_parameters_mut(&mut |n, p| f(&format!("attn.{n}"), p));
        self.ln2.visit_parameters_mut(&mut |n, p| f(&format!("ln2.{n}"), p));
        self.ffn.visit_parameters_mut(&mut |n, p| f(&format!("ffn.{n}"), p));
    }
}

/// Build a causal `[seq_len, seq_len]` additive mask: `0` on or below diagonal, `-1e9` above.
///
/// The mask is watched onto the tape but has no parameter behind it; the gradient through it
/// is dropped because there are no learnable mask parameters.
pub fn causal_mask_tape<B: Backend>(
    seq_len: usize,
    tape: &mut Tape<B>,
    ctx: &mut ForwardCtx<B>,
) -> Result<TensorId>
where
    B::Tensor: Clone,
{
    let n = seq_len;
    let mut data = vec![0.0_f32; n * n];
    let neg_inf = -1.0e9_f32;
    for i in 0..n {
        for j in (i + 1)..n {
            data[i * n + j] = neg_inf;
        }
    }
    let t = ctx.backend().ops().tensor_from_vec(data, &[n, n])?;
    Ok(tape.watch(t))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustral_core::Mode;
    use rustral_ndarray_backend::CpuBackend;

    fn make_input(backend: &CpuBackend, seq_len: usize, d_model: usize) -> <CpuBackend as Backend>::Tensor {
        let mut data = vec![0.0_f32; seq_len * d_model];
        for (i, v) in data.iter_mut().enumerate() {
            *v = (i as f32 * 0.1).sin();
        }
        backend.ops().tensor_from_vec(data, &[seq_len, d_model]).unwrap()
    }

    #[test]
    fn encoder_layer_forward_shape() {
        let backend = CpuBackend::default();
        let cfg = TapeTransformerEncoderConfig::new(16, 4, 32);
        let layer = TapeTransformerEncoderLayer::new(&backend, cfg, 7).unwrap();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let mut tape = Tape::<CpuBackend>::new();
        let input = tape.watch(make_input(&backend, 8, 16));
        let out = layer.forward_tape(input, &mut tape, &mut ctx).unwrap();
        let out_t = tape.value(out).unwrap();
        let shape = backend.ops().shape(out_t);
        assert_eq!(shape, vec![8, 16]);
    }

    #[test]
    fn causal_mask_zero_on_and_below_diagonal() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let mut tape = Tape::<CpuBackend>::new();
        let m = causal_mask_tape::<CpuBackend>(4, &mut tape, &mut ctx).unwrap();
        let t = tape.value(m).unwrap();
        let v = backend.ops().tensor_to_vec(t).unwrap();
        // Row 0: only col 0 is 0; cols 1..3 are -inf-ish.
        assert!((v[0] - 0.0).abs() < 1e-6);
        assert!(v[1] < -1.0e8);
        // Row 3: all four cols are 0 (j <= i).
        for j in 0..4 {
            assert!((v[3 * 4 + j] - 0.0).abs() < 1e-6);
        }
    }
}
