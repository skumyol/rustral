use rustral_autodiff::{GradExtFromStore, Tape};
use rustral_core::{Backend, ForwardCtx, Mode, Parameter};
use rustral_ndarray_backend::CpuBackend;
use rustral_nn::tape::TapeModule;

fn scalar_from(id: rustral_autodiff::TensorId, tape: &Tape<CpuBackend>, backend: &CpuBackend) -> f32 {
    let t = tape.value(id).expect("tensor missing from tape");
    backend.ops().tensor_to_vec(t).unwrap()[0]
}

fn assert_allclose(a: &[f32], b: &[f32], atol: f32, rtol: f32, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: len mismatch");
    for i in 0..a.len() {
        let diff = (a[i] - b[i]).abs();
        let tol = atol + rtol * b[i].abs().max(a[i].abs());
        assert!(
            diff <= tol,
            "{what}: idx={i} a={} b={} diff={} tol={}",
            a[i],
            b[i],
            diff,
            tol
        );
    }
}

fn finite_diff<F: Fn(&[f32]) -> f32>(base: &[f32], eps: f32, f: F) -> Vec<f32> {
    let mut g = vec![0.0f32; base.len()];
    for i in 0..base.len() {
        let mut plus = base.to_vec();
        let mut minus = base.to_vec();
        plus[i] += eps;
        minus[i] -= eps;
        g[i] = (f(&plus) - f(&minus)) / (2.0 * eps);
    }
    g
}

#[test]
fn finite_difference_matmul_matches_tape() {
    let backend = CpuBackend::default();
    let ops = backend.ops();
    let eps = 1e-3;

    // A:[2,3], B:[3,2]
    let a0 = vec![0.1, -0.2, 0.3, 0.0, 0.5, -0.4];
    let b0 = vec![0.2, -0.1, 0.4, 0.3, -0.2, 0.1];

    // analytic grads via tape
    let mut ctx = ForwardCtx::new(&backend, Mode::Train);
    let mut tape = Tape::<CpuBackend>::new();
    let a = tape.watch(ops.tensor_from_vec(a0.clone(), &[2, 3]).unwrap());
    let b = tape.watch(ops.tensor_from_vec(b0.clone(), &[3, 2]).unwrap());
    let c = tape.matmul(a, b, &mut ctx).unwrap();
    let loss = tape.sum_all_tape(c, &mut ctx).unwrap();
    let grads = tape
        .backward(loss, |data, shape| ops.tensor_from_vec(data, shape), ops)
        .unwrap();
    let ga = ops.tensor_to_vec(grads.get(&a).unwrap()).unwrap();
    let gb = ops.tensor_to_vec(grads.get(&b).unwrap()).unwrap();

    // numeric grad for A
    let gna = finite_diff(&a0, eps, |avec| {
        let mut ctx = ForwardCtx::new(&backend, Mode::Train);
        let mut tape = Tape::<CpuBackend>::new();
        let a = tape.watch(ops.tensor_from_vec(avec.to_vec(), &[2, 3]).unwrap());
        let b = tape.watch(ops.tensor_from_vec(b0.clone(), &[3, 2]).unwrap());
        let c = tape.matmul(a, b, &mut ctx).unwrap();
        let loss = tape.sum_all_tape(c, &mut ctx).unwrap();
        scalar_from(loss, &tape, &backend)
    });
    assert_allclose(&ga, &gna, 1e-2, 1e-2, "matmul dA");

    // numeric grad for B
    let gnb = finite_diff(&b0, eps, |bvec| {
        let mut ctx = ForwardCtx::new(&backend, Mode::Train);
        let mut tape = Tape::<CpuBackend>::new();
        let a = tape.watch(ops.tensor_from_vec(a0.clone(), &[2, 3]).unwrap());
        let b = tape.watch(ops.tensor_from_vec(bvec.to_vec(), &[3, 2]).unwrap());
        let c = tape.matmul(a, b, &mut ctx).unwrap();
        let loss = tape.sum_all_tape(c, &mut ctx).unwrap();
        scalar_from(loss, &tape, &backend)
    });
    assert_allclose(&gb, &gnb, 1e-2, 1e-2, "matmul dB");
}

#[test]
fn finite_difference_linear_weight_and_bias_match_tape() {
    let backend = CpuBackend::default();
    let ops = backend.ops();
    let eps = 1e-3;

    let lin = rustral_nn::LinearBuilder::new(2, 3).with_bias(true).seed(0).build(&backend).unwrap();
    let w0 = ops.tensor_to_vec(lin.weight().tensor()).unwrap();
    let b0 = ops.tensor_to_vec(lin.bias().unwrap().tensor()).unwrap();

    let x = ops.tensor_from_vec(vec![0.5, -0.25], &[1, 2]).unwrap();

    // analytic grads
    let mut ctx = ForwardCtx::new(&backend, Mode::Train);
    let mut tape = Tape::<CpuBackend>::new();
    let xid = tape.watch(x);
    let yid = lin.forward_tape(xid, &mut tape, &mut ctx).unwrap();
    let loss = tape.sum_all_tape(yid, &mut ctx).unwrap();
    let param_map = tape.param_map().clone();
    let grads = tape
        .backward(loss, |data, shape| ops.tensor_from_vec(data, shape), ops)
        .unwrap();

    let gw = lin.weight().gradient_from_store(&grads, &param_map).unwrap();
    let gb = lin.bias().unwrap().gradient_from_store(&grads, &param_map).unwrap();
    let gw = ops.tensor_to_vec(gw).unwrap();
    let gb = ops.tensor_to_vec(gb).unwrap();

    // numeric W grad
    let gnw = finite_diff(&w0, eps, |wvec| {
        let mut ctx = ForwardCtx::new(&backend, Mode::Train);
        let mut tape = Tape::<CpuBackend>::new();
        let w = Parameter::new("w", ops.tensor_from_vec(wvec.to_vec(), &[3, 2]).unwrap());
        let b = Parameter::new("b", ops.tensor_from_vec(b0.clone(), &[3]).unwrap());
        let lin = rustral_nn::Linear::from_parameters(
            rustral_nn::LinearConfig::new(2, 3).with_bias(true),
            w,
            Some(b),
        );
        let x = tape.watch(ops.tensor_from_vec(vec![0.5, -0.25], &[1, 2]).unwrap());
        let out = lin.forward_tape(x, &mut tape, &mut ctx).unwrap();
        let loss = tape.sum_all_tape(out, &mut ctx).unwrap();
        scalar_from(loss, &tape, &backend)
    });
    assert_allclose(&gw, &gnw, 1e-2, 1e-2, "linear dW");

    // numeric b grad
    let gnb = finite_diff(&b0, eps, |bvec| {
        let mut ctx = ForwardCtx::new(&backend, Mode::Train);
        let mut tape = Tape::<CpuBackend>::new();
        let w = Parameter::new("w", ops.tensor_from_vec(w0.clone(), &[3, 2]).unwrap());
        let b = Parameter::new("b", ops.tensor_from_vec(bvec.to_vec(), &[3]).unwrap());
        let lin = rustral_nn::Linear::from_parameters(
            rustral_nn::LinearConfig::new(2, 3).with_bias(true),
            w,
            Some(b),
        );
        let x = tape.watch(ops.tensor_from_vec(vec![0.5, -0.25], &[1, 2]).unwrap());
        let out = lin.forward_tape(x, &mut tape, &mut ctx).unwrap();
        let loss = tape.sum_all_tape(out, &mut ctx).unwrap();
        scalar_from(loss, &tape, &backend)
    });
    assert_allclose(&gb, &gnb, 1e-2, 1e-2, "linear db");
}

#[test]
fn finite_difference_embedding_table_matches_tape_on_used_rows() {
    let backend = CpuBackend::default();
    let ops = backend.ops();
    let eps = 1e-3;

    // table: [4, 3], ids: [0, 2]
    let table0 = vec![
        0.1, 0.2, -0.1, //
        -0.2, 0.0, 0.3, //
        0.05, -0.05, 0.1, //
        0.2, -0.1, 0.0, //
    ];
    let ids = vec![0.0f32, 2.0f32];

    let table = Parameter::new("table", ops.tensor_from_vec(table0.clone(), &[4, 3]).unwrap());

    // analytic
    let mut ctx = ForwardCtx::new(&backend, Mode::Train);
    let mut tape = Tape::<CpuBackend>::new();
    let ids_t = tape.watch(ops.tensor_from_vec(ids.clone(), &[2]).unwrap());
    let out = tape.gather_rows_tape(&table, ids_t, &mut ctx).unwrap();
    let loss = tape.sum_all_tape(out, &mut ctx).unwrap();
    let param_map = tape.param_map().clone();
    let grads = tape
        .backward(loss, |data, shape| ops.tensor_from_vec(data, shape), ops)
        .unwrap();
    let gt = table.gradient_from_store(&grads, &param_map).unwrap();
    let gt = ops.tensor_to_vec(gt).unwrap();

    // numeric grad for table values
    let gnt = finite_diff(&table0, eps, |tvec| {
        let table = Parameter::new("table", ops.tensor_from_vec(tvec.to_vec(), &[4, 3]).unwrap());
        let mut ctx = ForwardCtx::new(&backend, Mode::Train);
        let mut tape = Tape::<CpuBackend>::new();
        let ids_t = tape.watch(ops.tensor_from_vec(ids.clone(), &[2]).unwrap());
        let out = tape.gather_rows_tape(&table, ids_t, &mut ctx).unwrap();
        let loss = tape.sum_all_tape(out, &mut ctx).unwrap();
        scalar_from(loss, &tape, &backend)
    });

    // Only rows 0 and 2 are used; other rows should have ~0 grad.
    for r in [1usize, 3usize] {
        for j in 0..3 {
            assert!(gt[r * 3 + j].abs() <= 1e-5);
        }
    }
    assert_allclose(&gt, &gnt, 1e-2, 1e-2, "embedding dTable");
}

#[test]
fn finite_difference_layer_norm_gamma_beta_match_tape() {
    let backend = CpuBackend::default();
    let ops = backend.ops();
    let eps = 1e-3;

    let x0 = vec![1.0, 2.0, 3.0, 4.0]; // [2,2] feature_count=2 groups=2
    let g0 = vec![1.0, 1.5];
    let b0 = vec![0.0, 0.1];

    let gamma = Parameter::new("gamma", ops.tensor_from_vec(g0.clone(), &[2]).unwrap());
    let beta = Parameter::new("beta", ops.tensor_from_vec(b0.clone(), &[2]).unwrap());

    // analytic
    let mut ctx = ForwardCtx::new(&backend, Mode::Train);
    let mut tape = Tape::<CpuBackend>::new();
    let x = tape.watch(ops.tensor_from_vec(x0.clone(), &[2, 2]).unwrap());
    let y = tape.layer_norm_tape(x, &gamma, &beta, 1e-5, 2, &mut ctx).unwrap();
    let loss = tape.sum_all_tape(y, &mut ctx).unwrap();
    let param_map = tape.param_map().clone();
    let grads = tape
        .backward(loss, |data, shape| ops.tensor_from_vec(data, shape), ops)
        .unwrap();
    let gg = ops.tensor_to_vec(gamma.gradient_from_store(&grads, &param_map).unwrap()).unwrap();
    let gb = ops.tensor_to_vec(beta.gradient_from_store(&grads, &param_map).unwrap()).unwrap();

    // numeric gamma
    let gng = finite_diff(&g0, eps, |gvec| {
        let gamma = Parameter::new("gamma", ops.tensor_from_vec(gvec.to_vec(), &[2]).unwrap());
        let beta = Parameter::new("beta", ops.tensor_from_vec(b0.clone(), &[2]).unwrap());
        let mut ctx = ForwardCtx::new(&backend, Mode::Train);
        let mut tape = Tape::<CpuBackend>::new();
        let x = tape.watch(ops.tensor_from_vec(x0.clone(), &[2, 2]).unwrap());
        let y = tape.layer_norm_tape(x, &gamma, &beta, 1e-5, 2, &mut ctx).unwrap();
        let loss = tape.sum_all_tape(y, &mut ctx).unwrap();
        scalar_from(loss, &tape, &backend)
    });
    assert_allclose(&gg, &gng, 1e-2, 1e-2, "layer_norm dGamma");

    // numeric beta
    let gnb = finite_diff(&b0, eps, |bvec| {
        let gamma = Parameter::new("gamma", ops.tensor_from_vec(g0.clone(), &[2]).unwrap());
        let beta = Parameter::new("beta", ops.tensor_from_vec(bvec.to_vec(), &[2]).unwrap());
        let mut ctx = ForwardCtx::new(&backend, Mode::Train);
        let mut tape = Tape::<CpuBackend>::new();
        let x = tape.watch(ops.tensor_from_vec(x0.clone(), &[2, 2]).unwrap());
        let y = tape.layer_norm_tape(x, &gamma, &beta, 1e-5, 2, &mut ctx).unwrap();
        let loss = tape.sum_all_tape(y, &mut ctx).unwrap();
        scalar_from(loss, &tape, &backend)
    });
    assert_allclose(&gb, &gnb, 1e-2, 1e-2, "layer_norm dBeta");
}

#[test]
fn finite_difference_cross_entropy_logits_match_tape() {
    let backend = CpuBackend::default();
    let ops = backend.ops();
    let eps = 1e-3;

    // logits: [2,3], target indices [2,0]
    let logits0 = vec![1.0, 2.0, 3.0, 0.0, -1.0, 2.0];
    let target = vec![2.0, 0.0];

    // analytic
    let mut ctx = ForwardCtx::new(&backend, Mode::Train);
    let mut tape = Tape::<CpuBackend>::new();
    let logits = tape.watch(ops.tensor_from_vec(logits0.clone(), &[2, 3]).unwrap());
    let tgt = tape.watch(ops.tensor_from_vec(target.clone(), &[2]).unwrap());
    let loss = tape.cross_entropy_loss(logits, tgt, &mut ctx).unwrap();
    let grads = tape
        .backward(loss, |data, shape| ops.tensor_from_vec(data, shape), ops)
        .unwrap();
    let gl = ops.tensor_to_vec(grads.get(&logits).unwrap()).unwrap();

    // numeric
    let gnl = finite_diff(&logits0, eps, |lvec| {
        let mut ctx = ForwardCtx::new(&backend, Mode::Train);
        let mut tape = Tape::<CpuBackend>::new();
        let logits = tape.watch(ops.tensor_from_vec(lvec.to_vec(), &[2, 3]).unwrap());
        let tgt = tape.watch(ops.tensor_from_vec(target.clone(), &[2]).unwrap());
        let loss = tape.cross_entropy_loss(logits, tgt, &mut ctx).unwrap();
        scalar_from(loss, &tape, &backend)
    });

    assert_allclose(&gl, &gnl, 2e-2, 2e-2, "cross_entropy dLogits");
}

