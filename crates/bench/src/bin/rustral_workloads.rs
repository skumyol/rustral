//! Rustral workload runner.
//!
//! Runs a fixed set of workloads (matmul, attention, conv2d, mlp_train_step,
//! optimizer_step, transformer encoder forward, decoder prefill+decode, kv-cache
//! prefill+decode, model save/load throughput) with a controlled number of repeats,
//! then prints a single JSON document to stdout in the unified benchmark schema
//! (`benchmarks/SCHEMA.md`, schema_version 2.0.0).
//!
//! Usage:
//!   cargo run --release -p rustral-bench --bin rustral_workloads -- --repeats 5 --warmup 1
//!   cargo run --release -p rustral-bench --bin rustral_workloads -- --profile heavy
//!     # heavy profile enables 100M-param optimizer step instead of the default 10M
//!     # and the larger 50M-param save/load workload.

use std::env;

use rustral_autodiff::Tape;
use rustral_bench::{samples_to_json, time_runs, time_train_step, Sample};
use rustral_core::{Backend, ForwardCtx, Mode, Module, NamedParameters, Parameter};
use rustral_ndarray_backend::CpuBackend;
use rustral_nn::{
    CacheConfig, Conv2d, Conv2dConfig, KVCache, MultiHeadAttention, SelfAttentionConfig,
    TransformerDecoder, TransformerDecoderConfig, TransformerEncoder, TransformerEncoderConfig,
};
use rustral_optim::{Adam, Gradient, Optimizer, Sgd};
use rustral_runtime::{load_model, save_model};

const BACKEND: &str = "ndarray-cpu";

fn parse_arg(args: &[String], name: &str, default: usize) -> usize {
    for w in args.windows(2) {
        if w[0] == name {
            if let Ok(v) = w[1].parse::<usize>() {
                return v;
            }
        }
    }
    default
}

fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

fn parse_string_arg<'a>(args: &'a [String], name: &str, default: &'a str) -> &'a str {
    for w in args.windows(2) {
        if w[0] == name {
            return &w[1];
        }
    }
    default
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let repeats = parse_arg(&args, "--repeats", 5);
    let warmup = parse_arg(&args, "--warmup", 1);
    let profile = parse_string_arg(&args, "--profile", "default");
    let heavy = profile == "heavy" || has_flag(&args, "--heavy");

    let backend = CpuBackend::default();
    let mut samples: Vec<Sample> = Vec::new();

    bench_matmul(&backend, repeats, warmup, &mut samples);
    bench_attention(&backend, repeats, warmup, &mut samples);
    bench_conv2d(&backend, repeats, warmup, &mut samples);
    // NOTE: lstm_forward and a tape-integrated lstm_lm_train_step (Track K, K4) are
    // intentionally still skipped in this binary. The existing criterion bench
    // (`crates/bench/benches/lstm_forward.rs`) panics with a ShapeMismatch because
    // `LstmCell` creates `wx` as `[input_dim, 4*hidden_dim]` while the CPU `linear` op
    // expects weight shape `[out, in]`. Fixing the LstmCell weight convention is tracked
    // separately; both `lstm_forward` and `lstm_lm_train_step` will be promoted once the
    // weight layout is reconciled. See BENCHMARKS.md "Workload coverage" for the gap.
    bench_mlp_train_step(&backend, repeats, warmup, &mut samples);
    bench_optimizer_step(&backend, repeats, warmup, heavy, &mut samples);
    bench_transformer_encoder_forward(&backend, repeats, warmup, &mut samples);
    bench_decoder_prefill_decode(&backend, repeats, warmup, &mut samples);
    bench_kv_cache_prefill_decode(&backend, repeats, warmup, &mut samples);
    bench_save_load_throughput(&backend, repeats, warmup, heavy, &mut samples);

    print!("{}", samples_to_json("rustral", &samples));
}

fn bench_matmul(backend: &CpuBackend, repeats: usize, warmup: usize, out: &mut Vec<Sample>) {
    let ops = backend.ops();
    for &(m, k, n) in &[(128usize, 128, 128), (256, 256, 256), (512, 512, 512)] {
        let a = backend.tensor_from_vec(vec![1.0f32; m * k], &[m, k]).unwrap();
        let b = backend.tensor_from_vec(vec![1.0f32; k * n], &[k, n]).unwrap();
        let runs = time_runs(
            || {
                let _ = ops.matmul(&a, &b).unwrap();
            },
            warmup,
            repeats,
        );
        out.push(Sample::cpu_f32(
            "matmul",
            BACKEND,
            vec![("m".into(), m.to_string()), ("k".into(), k.to_string()), ("n".into(), n.to_string())],
            runs,
        ));
    }
}

fn bench_attention(backend: &CpuBackend, repeats: usize, warmup: usize, out: &mut Vec<Sample>) {
    for &(name, d_model, heads, seq_len) in
        &[("small", 64usize, 4usize, 32usize), ("medium", 256, 8, 128)]
    {
        let config = SelfAttentionConfig::new(d_model, heads);
        let mha = MultiHeadAttention::new(backend, config, 42).unwrap();
        let input = backend
            .tensor_from_vec(vec![1.0f32; seq_len * d_model], &[1, seq_len, d_model])
            .unwrap();
        let runs = time_runs(
            || {
                let mut ctx = ForwardCtx::new(backend, Mode::Inference);
                let _ = mha.forward(input.clone(), &mut ctx).unwrap();
            },
            warmup,
            repeats,
        );
        out.push(Sample::cpu_f32(
            format!("attention.{name}"),
            BACKEND,
            vec![
                ("d_model".into(), d_model.to_string()),
                ("heads".into(), heads.to_string()),
                ("seq_len".into(), seq_len.to_string()),
            ],
            runs,
        ));
    }
}

fn bench_conv2d(backend: &CpuBackend, repeats: usize, warmup: usize, out: &mut Vec<Sample>) {
    // Standard NCHW conv2d shapes that match PyTorch / Candle semantics
    // (filter shape `[out, in, kH, kW]` with input in_channels matching filter in_channels).
    let configs: [(&str, [usize; 4], [usize; 4]); 3] = [
        ("small", [1, 1, 28, 28], [6, 1, 5, 5]),         // MNIST first layer
        ("medium", [4, 16, 32, 32], [16, 16, 3, 3]),
        ("large", [8, 64, 64, 64], [64, 64, 3, 3]),
    ];
    for &(name, input_shape, filter_shape) in &configs {
        let total: usize = input_shape.iter().product();
        let input = backend.tensor_from_vec(vec![1.0f32; total], &input_shape).unwrap();
        let out_channels = filter_shape[0];
        let kernel_h = filter_shape[2];
        let kernel_w = filter_shape[3];
        let conv = Conv2d::new(backend, Conv2dConfig::new(out_channels, kernel_h, kernel_w)).unwrap();
        let runs = time_runs(
            || {
                let mut ctx = ForwardCtx::new(backend, Mode::Inference);
                let _ = conv.forward(input.clone(), &mut ctx).unwrap();
            },
            warmup,
            repeats,
        );
        out.push(Sample::cpu_f32(
            format!("conv2d.{name}"),
            BACKEND,
            vec![
                ("batch".into(), input_shape[0].to_string()),
                ("in_channels".into(), input_shape[1].to_string()),
                ("h".into(), input_shape[2].to_string()),
                ("w".into(), input_shape[3].to_string()),
                ("out_channels".into(), out_channels.to_string()),
                ("kernel_h".into(), kernel_h.to_string()),
                ("kernel_w".into(), kernel_w.to_string()),
            ],
            runs,
        ));
    }
}

/// MLP train-step micro-benchmark: forward + backward + Adam step on a 2-layer MLP.
///
/// Times only the hot loop; no host-side probes inside the timed window.
fn bench_mlp_train_step(backend: &CpuBackend, repeats: usize, warmup: usize, out: &mut Vec<Sample>) {
    let ops = backend.ops();
    let batch = 32usize;
    let in_dim = 128usize;
    let hidden = 256usize;
    let out_dim = 64usize;

    // Initialize parameters with small constant values; benchmark does not require convergence.
    let w1 = backend.tensor_from_vec(vec![0.01f32; in_dim * hidden], &[in_dim, hidden]).unwrap();
    let w2 = backend.tensor_from_vec(vec![0.01f32; hidden * out_dim], &[hidden, out_dim]).unwrap();
    let mut params = vec![Parameter::new("w1", w1), Parameter::new("w2", w2)];

    // Static input + target used every step (not ideal for learning, fine for timing).
    let x = backend.tensor_from_vec(vec![1.0f32; batch * in_dim], &[batch, in_dim]).unwrap();
    let y = backend.tensor_from_vec(vec![0.5f32; batch * out_dim], &[batch, out_dim]).unwrap();

    let mut adam = Adam::<CpuBackend>::new(1e-3);
    let total_params: u64 = (in_dim * hidden + hidden * out_dim) as u64;

    let runs = time_train_step(
        || {
            let mut ctx = ForwardCtx::new(backend, Mode::Train);
            let mut tape = Tape::<CpuBackend>::new();
            let w1_id = tape.watch_parameter(&params[0]);
            let w2_id = tape.watch_parameter(&params[1]);
            let x_id = tape.watch(x.clone());
            let y_id = tape.watch(y.clone());

            // Forward: relu(x @ w1) @ w2; loss: MSE.
            let h = tape.matmul(x_id, w1_id, &mut ctx).unwrap();
            let h = tape.relu(h, &mut ctx).unwrap();
            let yhat = tape.matmul(h, w2_id, &mut ctx).unwrap();
            let loss = tape.mse_loss(yhat, y_id, &mut ctx).unwrap();

            // Backward.
            let param_map = tape.param_map().clone();
            let make_ones =
                |data: Vec<f32>, shape: &[usize]| ops.tensor_from_vec(data, shape);
            let grads_store = tape.backward(loss, make_ones, ops).unwrap();

            let mut grads: Vec<Gradient<CpuBackend>> = Vec::with_capacity(params.len());
            for p in params.iter() {
                if let Some(tid) = param_map.get(&p.id()) {
                    if let Some(g) = grads_store.get(tid) {
                        grads.push(Gradient { param_id: p.id(), tensor: g.clone() });
                    }
                }
            }

            // Optimizer step (mutates params in place via with_tensor).
            adam.step(&mut params, &grads, &mut ctx).unwrap();
        },
        warmup,
        repeats,
    );

    out.push(
        Sample::cpu_f32(
            "mlp_train_step",
            BACKEND,
            vec![
                ("batch".into(), batch.to_string()),
                ("in_dim".into(), in_dim.to_string()),
                ("hidden".into(), hidden.to_string()),
                ("out_dim".into(), out_dim.to_string()),
                ("optimizer".into(), "adam".into()),
            ],
            runs,
        )
        .with_model_params(total_params),
    );
}

/// Pure `Optimizer::step` micro-benchmark.
///
/// Default scale is 10M parameters; `--profile heavy` (or `--heavy`) bumps to 100M.
/// Reports both Adam and SGD steps.
fn bench_optimizer_step(
    backend: &CpuBackend,
    repeats: usize,
    warmup: usize,
    heavy: bool,
    out: &mut Vec<Sample>,
) {
    // Distribute the parameter budget across a handful of slabs to mimic real models with
    // multiple parameter tensors rather than one giant one.
    let total: usize = if heavy { 100_000_000 } else { 10_000_000 };
    let slabs: usize = 8;
    let slab_size: usize = total / slabs;

    let mut params: Vec<Parameter<CpuBackend>> = Vec::with_capacity(slabs);
    let mut grads: Vec<Gradient<CpuBackend>> = Vec::with_capacity(slabs);
    for i in 0..slabs {
        let t = backend.tensor_from_vec(vec![0.01f32; slab_size], &[slab_size]).unwrap();
        let g = backend.tensor_from_vec(vec![0.01f32; slab_size], &[slab_size]).unwrap();
        let p = Parameter::new(format!("slab_{i}"), t);
        grads.push(Gradient { param_id: p.id(), tensor: g });
        params.push(p);
    }

    {
        let mut sgd = Sgd::new(1e-3);
        let runs = time_runs(
            || {
                let mut ctx = ForwardCtx::new(backend, Mode::Train);
                sgd.step(&mut params, &grads, &mut ctx).unwrap();
            },
            warmup,
            repeats,
        );
        out.push(
            Sample::cpu_f32(
                "optimizer_step.sgd",
                BACKEND,
                vec![
                    ("total_params".into(), total.to_string()),
                    ("slabs".into(), slabs.to_string()),
                    ("profile".into(), if heavy { "heavy".into() } else { "default".into() }),
                ],
                runs,
            )
            .with_model_params(total as u64),
        );
    }

    {
        let mut adam = Adam::<CpuBackend>::new(1e-3);
        let runs = time_runs(
            || {
                let mut ctx = ForwardCtx::new(backend, Mode::Train);
                adam.step(&mut params, &grads, &mut ctx).unwrap();
            },
            warmup,
            repeats,
        );
        out.push(
            Sample::cpu_f32(
                "optimizer_step.adam",
                BACKEND,
                vec![
                    ("total_params".into(), total.to_string()),
                    ("slabs".into(), slabs.to_string()),
                    ("profile".into(), if heavy { "heavy".into() } else { "default".into() }),
                ],
                runs,
            )
            .with_model_params(total as u64),
        );
    }
}

/// Transformer encoder forward (P4 K1, small profile).
///
/// 2 layers, d_model=128, n_heads=4, ff_dim=512, seq_len=128, vocab_size=1024.
///
/// Note: forward-only. The full forward+backward+optimizer.step ("encoder train step")
/// is gated on a tape-integrated `MultiHeadAttention` / `TransformerEncoderLayer`. Today
/// only `Linear`, `Embedding`, and `LayerNorm` implement `TapeModule`, so wiring a
/// faithful tape-based attention block requires more groundwork than this benchmark
/// phase covers. The forward number lets us track encoder-cost trends; backward will
/// fold in once the tape integration lands.
fn bench_transformer_encoder_forward(
    backend: &CpuBackend,
    repeats: usize,
    warmup: usize,
    out: &mut Vec<Sample>,
) {
    let d_model = 128usize;
    let num_heads = 4usize;
    let num_layers = 2usize;
    let ff_dim = 512usize;
    let seq_len = 128usize;
    let vocab_size = 1024usize;

    let config = TransformerEncoderConfig::new(d_model, num_heads, num_layers, ff_dim)
        .with_max_seq_len(seq_len);
    let encoder = TransformerEncoder::<CpuBackend>::new(backend, config, vocab_size, 42).unwrap();
    let total_params = count_named_params(&encoder, backend);

    // Token IDs as Vec<usize>; the encoder reshapes internally to [1, seq_len, d_model].
    let input: Vec<usize> = (0..seq_len).map(|i| i % vocab_size).collect();

    let runs = time_runs(
        || {
            let mut ctx = ForwardCtx::new(backend, Mode::Inference);
            let _ = encoder.forward(input.clone(), &mut ctx).unwrap();
        },
        warmup,
        repeats,
    );

    out.push(
        Sample::cpu_f32(
            "transformer_encoder.forward",
            BACKEND,
            vec![
                ("d_model".into(), d_model.to_string()),
                ("num_heads".into(), num_heads.to_string()),
                ("num_layers".into(), num_layers.to_string()),
                ("ff_dim".into(), ff_dim.to_string()),
                ("seq_len".into(), seq_len.to_string()),
                ("vocab".into(), vocab_size.to_string()),
            ],
            runs,
        )
        .with_model_params(total_params),
    );
}

/// Decoder prefill vs per-token decode throughput (P4 K2).
///
/// Uses [`TransformerDecoder::forward`] without an external KV cache, so each step
/// recomputes attention over the full prefix. We report two samples:
/// - `decoder.prefill`: forward over `prefill_len = 64` tokens (one call).
/// - `decoder.decode_step`: average forward time over a `prefill_len + decode_steps`
///   long context (single call per timed iteration). Cache-accelerated decode lives in
///   the K3 micro-bench below.
fn bench_decoder_prefill_decode(
    backend: &CpuBackend,
    repeats: usize,
    warmup: usize,
    out: &mut Vec<Sample>,
) {
    let d_model = 128usize;
    let num_heads = 4usize;
    let num_layers = 2usize;
    let ff_dim = 512usize;
    let max_seq_len = 256usize;
    let vocab_size = 1024usize;
    let prefill_len = 64usize;
    let decode_total_len = prefill_len + 64; // 128 tokens total context for the "decode" step.

    let config = TransformerDecoderConfig::new(d_model, num_heads, num_layers, ff_dim)
        .with_max_seq_len(max_seq_len);
    let decoder = TransformerDecoder::<CpuBackend>::new(backend, config, vocab_size, 42).unwrap();
    let total_params = count_named_params(&decoder, backend);

    let prefill_input: Vec<usize> = (0..prefill_len).map(|i| i % vocab_size).collect();
    let decode_input: Vec<usize> = (0..decode_total_len).map(|i| i % vocab_size).collect();

    // Prefill: one forward over `prefill_len` tokens.
    let prefill_runs = time_runs(
        || {
            let mut ctx = ForwardCtx::new(backend, Mode::Inference);
            let _ = decoder.forward(prefill_input.clone(), &mut ctx).unwrap();
        },
        warmup,
        repeats,
    );
    out.push(
        Sample::cpu_f32(
            "decoder.prefill",
            BACKEND,
            vec![
                ("d_model".into(), d_model.to_string()),
                ("num_heads".into(), num_heads.to_string()),
                ("num_layers".into(), num_layers.to_string()),
                ("seq_len".into(), prefill_len.to_string()),
                ("vocab".into(), vocab_size.to_string()),
            ],
            prefill_runs,
        )
        .with_model_params(total_params),
    );

    // Decode: forward over a prefill+decode-length context, simulating cost of a single
    // additional decode step without a KV cache (the no-cache baseline).
    let decode_runs = time_runs(
        || {
            let mut ctx = ForwardCtx::new(backend, Mode::Inference);
            let _ = decoder.forward(decode_input.clone(), &mut ctx).unwrap();
        },
        warmup,
        repeats,
    );
    out.push(
        Sample::cpu_f32(
            "decoder.decode_step.no_cache",
            BACKEND,
            vec![
                ("d_model".into(), d_model.to_string()),
                ("num_heads".into(), num_heads.to_string()),
                ("num_layers".into(), num_layers.to_string()),
                ("seq_len".into(), decode_total_len.to_string()),
                ("vocab".into(), vocab_size.to_string()),
            ],
            decode_runs,
        )
        .with_model_params(total_params),
    );
}

/// KV cache prefill vs decode latency (P4 K3).
///
/// Times two access patterns directly against [`KVCache::append`]:
/// - `kv_cache.prefill`: a single append of `prefill_len = 64` tokens.
/// - `kv_cache.decode`: the time to append one token (single timed iteration appends 1).
///
/// Both runs use the same cache shape (batch=1, num_heads=4, max_seq_len=256, head_dim=32).
/// The cache is reset between iterations so each run measures the same starting state.
fn bench_kv_cache_prefill_decode(
    backend: &CpuBackend,
    repeats: usize,
    warmup: usize,
    out: &mut Vec<Sample>,
) {
    let num_heads = 4usize;
    let head_dim = 32usize;
    let max_seq_len = 256usize;
    let prefill_len = 64usize;

    let cfg = CacheConfig::new(num_heads, head_dim, max_seq_len);

    // Prefill: append a `prefill_len`-token chunk to a fresh cache each time.
    let prefill_elems = num_heads * prefill_len * head_dim;
    let prefill_k = backend
        .tensor_from_vec(
            vec![1.0f32; prefill_elems],
            &[1, num_heads, prefill_len, head_dim],
        )
        .unwrap();
    let prefill_v = backend
        .tensor_from_vec(
            vec![1.0f32; prefill_elems],
            &[1, num_heads, prefill_len, head_dim],
        )
        .unwrap();
    let ops = backend.ops();
    let prefill_runs = time_runs(
        || {
            let mut cache = KVCache::<CpuBackend>::new(backend, cfg.clone()).unwrap();
            cache.append(&prefill_k, &prefill_v, ops).unwrap();
        },
        warmup,
        repeats,
    );
    out.push(Sample::cpu_f32(
        "kv_cache.prefill",
        BACKEND,
        vec![
            ("num_heads".into(), num_heads.to_string()),
            ("head_dim".into(), head_dim.to_string()),
            ("max_seq_len".into(), max_seq_len.to_string()),
            ("prefill_len".into(), prefill_len.to_string()),
        ],
        prefill_runs,
    ));

    // Decode: append a single token at a time to a fresh cache. Measures the per-token
    // append cost separately from prefill.
    let decode_elems = num_heads * head_dim;
    let decode_k = backend
        .tensor_from_vec(vec![1.0f32; decode_elems], &[1, num_heads, 1, head_dim])
        .unwrap();
    let decode_v = backend
        .tensor_from_vec(vec![1.0f32; decode_elems], &[1, num_heads, 1, head_dim])
        .unwrap();
    let decode_runs = time_runs(
        || {
            let mut cache = KVCache::<CpuBackend>::new(backend, cfg.clone()).unwrap();
            cache.append(&decode_k, &decode_v, ops).unwrap();
        },
        warmup,
        repeats,
    );
    out.push(Sample::cpu_f32(
        "kv_cache.decode_step",
        BACKEND,
        vec![
            ("num_heads".into(), num_heads.to_string()),
            ("head_dim".into(), head_dim.to_string()),
            ("max_seq_len".into(), max_seq_len.to_string()),
            ("token_len".into(), "1".into()),
        ],
        decode_runs,
    ));
}

/// Save / load roundtrip throughput (P4 K5).
///
/// Builds a synthetic ~50M-parameter "model" out of plain `Parameter` slabs (the same
/// shape pattern used by `bench_optimizer_step`) and exercises [`save_model`] +
/// [`load_model`] from `rustral-runtime::model_io`. Throughput numbers and absolute MB
/// are encoded into the workload `params` so dashboards can derive MB/s outside the JSON.
///
/// Default scale is ~50M params (~200 MB at f32); `--profile heavy` keeps the same
/// shape so it stays a stable trend signal — bumping it would change the byte volume.
fn bench_save_load_throughput(
    backend: &CpuBackend,
    repeats: usize,
    warmup: usize,
    _heavy: bool,
    out: &mut Vec<Sample>,
) {
    // ~50M params = 50 slabs × 1M each.
    let slabs: usize = 50;
    let slab_size: usize = 1_000_000;
    let total: usize = slabs * slab_size;
    let mut model = SyntheticModel::<CpuBackend>::new(backend, slabs, slab_size).unwrap();

    // Pre-serialize once outside the timed window so the load run can use a known buffer
    // without picking up save overhead.
    let saved_bytes = save_model(&model, backend).unwrap();
    let bytes_total = saved_bytes.len();

    let save_runs = time_runs(
        || {
            let _ = save_model(&model, backend).unwrap();
        },
        warmup,
        repeats,
    );
    out.push(
        Sample::cpu_f32(
            "model_io.save",
            BACKEND,
            vec![
                ("slabs".into(), slabs.to_string()),
                ("slab_size".into(), slab_size.to_string()),
                ("total_params".into(), total.to_string()),
                ("bytes".into(), bytes_total.to_string()),
            ],
            save_runs,
        )
        .with_model_params(total as u64),
    );

    let load_runs = time_runs(
        || {
            load_model(&mut model, backend, &saved_bytes).unwrap();
        },
        warmup,
        repeats,
    );
    out.push(
        Sample::cpu_f32(
            "model_io.load",
            BACKEND,
            vec![
                ("slabs".into(), slabs.to_string()),
                ("slab_size".into(), slab_size.to_string()),
                ("total_params".into(), total.to_string()),
                ("bytes".into(), bytes_total.to_string()),
            ],
            load_runs,
        )
        .with_model_params(total as u64),
    );
}

/// Simple synthetic model: N independent slabs of f32 parameters, each visited under a
/// stable name so [`save_model`] / [`load_model`] roundtrip cleanly.
struct SyntheticModel<B: Backend> {
    params: Vec<Parameter<B>>,
}

impl<B: Backend> SyntheticModel<B>
where
    B::Tensor: Clone,
{
    fn new(backend: &B, slabs: usize, slab_size: usize) -> rustral_core::Result<Self> {
        let mut params = Vec::with_capacity(slabs);
        for i in 0..slabs {
            let t = backend
                .ops()
                .tensor_from_vec(vec![0.01f32; slab_size], &[slab_size])?;
            params.push(Parameter::new(format!("slab_{i}"), t));
        }
        Ok(Self { params })
    }
}

impl<B: Backend> NamedParameters<B> for SyntheticModel<B> {
    fn visit_parameters(&self, f: &mut dyn FnMut(&str, &Parameter<B>)) {
        for p in &self.params {
            f(p.name(), p);
        }
    }
    fn visit_parameters_mut(&mut self, f: &mut dyn FnMut(&str, &mut Parameter<B>)) {
        for p in &mut self.params {
            let name = p.name().to_string();
            f(&name, p);
        }
    }
}

/// Walk a `NamedParameters` model and sum element counts across every parameter tensor.
fn count_named_params<M: NamedParameters<CpuBackend>>(model: &M, backend: &CpuBackend) -> u64 {
    let ops = backend.ops();
    let mut total: u64 = 0;
    model.visit_parameters(&mut |_name, p| {
        let shape = ops.shape(p.tensor());
        let n: usize = shape.iter().product();
        total = total.saturating_add(n as u64);
    });
    total
}
