//! Candle-direct baseline workload runner.
//!
//! Mirrors a subset of the workloads in `rustral_workloads` but executed against
//! `candle-core` tensors directly (no Rustral abstraction layer). Output is the same JSON
//! schema (v2.0.0) so the orchestrator can join Rustral and Candle results into a single
//! table.
//!
//! Coverage parity:
//!   - matmul: yes
//!   - attention: yes (manual QK^T softmax V)
//!   - conv2d:  yes (`Tensor::conv2d`)
//!   - lstm_forward: skipped (would require candle-nn LSTM; not added on purpose to keep
//!     the candle-direct surface dependency-free)
//!   - mlp_train_step / optimizer_step: skipped (training APIs require candle-nn)
//!
//! Usage:
//!   cargo run --release -p rustral-bench --bin candle_workloads -- --repeats 5 --warmup 1

use std::env;

use candle_core::{DType, Device, Tensor};
use rustral_bench::{samples_to_json, time_runs, Sample};

const BACKEND: &str = "candle-cpu";

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

fn main() {
    let args: Vec<String> = env::args().collect();
    let repeats = parse_arg(&args, "--repeats", 5);
    let warmup = parse_arg(&args, "--warmup", 1);

    let device = Device::Cpu;
    let mut samples: Vec<Sample> = Vec::new();

    bench_matmul(&device, repeats, warmup, &mut samples);
    bench_attention(&device, repeats, warmup, &mut samples);
    bench_conv2d(&device, repeats, warmup, &mut samples);

    print!("{}", samples_to_json("candle", &samples));
}

fn bench_matmul(device: &Device, repeats: usize, warmup: usize, out: &mut Vec<Sample>) {
    for &(m, k, n) in &[(128usize, 128, 128), (256, 256, 256), (512, 512, 512)] {
        let a = Tensor::ones((m, k), DType::F32, device).unwrap();
        let b = Tensor::ones((k, n), DType::F32, device).unwrap();
        let runs = time_runs(
            || {
                let _ = a.matmul(&b).unwrap();
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

fn bench_attention(device: &Device, repeats: usize, warmup: usize, out: &mut Vec<Sample>) {
    for &(name, d_model, heads, seq_len) in &[
        ("small", 64usize, 4usize, 32usize),
        ("medium", 256, 8, 128),
    ] {
        let q = Tensor::ones((1, seq_len, d_model), DType::F32, device).unwrap();
        let k = Tensor::ones((1, seq_len, d_model), DType::F32, device).unwrap();
        let v = Tensor::ones((1, seq_len, d_model), DType::F32, device).unwrap();
        let runs = time_runs(
            || {
                let kt = k.transpose(1, 2).unwrap();
                let scores = q.matmul(&kt).unwrap();
                let scaled = (scores * (1.0f64 / (d_model as f64).sqrt())).unwrap();
                let max = scaled.max_keepdim(2).unwrap();
                let shifted = scaled.broadcast_sub(&max).unwrap();
                let exp = shifted.exp().unwrap();
                let sum = exp.sum_keepdim(2).unwrap();
                let probs = exp.broadcast_div(&sum).unwrap();
                let _ = probs.matmul(&v).unwrap();
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

fn bench_conv2d(device: &Device, repeats: usize, warmup: usize, out: &mut Vec<Sample>) {
    // Mirror the rustral_workloads conv2d sizes (filter shape `[out, in, kH, kW]`).
    let configs: [(&str, [usize; 4], [usize; 4]); 3] = [
        ("small", [1, 1, 28, 28], [6, 1, 5, 5]),
        ("medium", [4, 16, 32, 32], [16, 16, 3, 3]),
        ("large", [8, 64, 64, 64], [64, 64, 3, 3]),
    ];
    for &(name, input_shape, filter_shape) in &configs {
        let input = Tensor::ones(&input_shape, DType::F32, device).unwrap();
        let kernel = Tensor::ones(&filter_shape, DType::F32, device).unwrap();
        let runs = time_runs(
            || {
                // padding=0, stride=1, dilation=1, groups=1
                let _ = input.conv2d(&kernel, 0, 1, 1, 1).unwrap();
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
                ("out_channels".into(), filter_shape[0].to_string()),
                ("kernel_h".into(), filter_shape[2].to_string()),
                ("kernel_w".into(), filter_shape[3].to_string()),
            ],
            runs,
        ));
    }
}
