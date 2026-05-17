//! Tabular Task benchmarks for Rustral.
use rustral_bench::{samples_to_json, time_runs, Sample};
use rustral_core::Backend;
use rustral_ndarray_backend::CpuBackend;

const BACKEND: &str = "ndarray-cpu";

fn main() {
    let backend = CpuBackend::default();
    let mut samples: Vec<Sample> = Vec::new();
    bench_knn(&backend, 5, 1, &mut samples);
    bench_histogram(&backend, 5, 1, &mut samples);
    println!("{}", samples_to_json("rustral-tabular", &samples));
}

fn bench_knn(backend: &CpuBackend, repeats: usize, warmup: usize, out: &mut Vec<Sample>) {
    let ops = backend.ops();
    let num_items = 1000;
    let dim = 128;
    let ref_data = backend.tensor_from_vec(vec![0.1f32; num_items * dim], &[num_items, dim]).unwrap();
    let query = backend.tensor_from_vec(vec![0.2f32; dim], &[1, dim]).unwrap();
    let runs = time_runs(
        || {
            let mut distances = Vec::with_capacity(num_items);
            for i in 0..num_items {
                let item = ops.slice(&ref_data, i, i + 1).unwrap();
                let d = ops.dist_l2(&query, &item).unwrap();
                distances.push(ops.tensor_element(&d, 0).unwrap());
            }
            let dist_tensor = backend.tensor_from_vec(distances, &[num_items]).unwrap();
            let _ = ops.topk(&dist_tensor, 10, 0, false).unwrap();
        },
        warmup,
        repeats,
    );
    out.push(Sample::cpu_f32(
        "tabular.knn",
        BACKEND,
        vec![("num_items".into(), num_items.to_string())],
        runs,
    ));
}

fn bench_histogram(backend: &CpuBackend, repeats: usize, warmup: usize, out: &mut Vec<Sample>) {
    let ops = backend.ops();
    let n = 100_000;
    let data = backend.tensor_from_vec((0..n).map(|i| (i % 100) as f32).collect(), &[n]).unwrap();
    let runs = time_runs(
        || {
            let _ = ops.bincount(&data, 100).unwrap();
        },
        warmup,
        repeats,
    );
    out.push(Sample::cpu_f32("tabular.bincount", BACKEND, vec![("n".into(), n.to_string())], runs));
}
