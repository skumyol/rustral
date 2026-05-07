use std::collections::HashSet;
use std::process::Command;

use serde_json::Value;

fn run_bin(bin_env: &str) -> Value {
    let exe = std::env::var(bin_env).unwrap_or_else(|_| panic!("missing {bin_env}; cargo did not build bin"));
    let out = Command::new(exe)
        .args(["--repeats", "1", "--warmup", "0"])
        .output()
        .expect("failed to run benchmark binary");
    assert!(
        out.status.success(),
        "benchmark binary failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout)
        .unwrap_or_else(|e| panic!("binary did not emit valid JSON: {e}\n{}", String::from_utf8_lossy(&out.stdout)))
}

fn sample_names(doc: &Value) -> HashSet<String> {
    doc.get("samples")
        .and_then(Value::as_array)
        .expect("samples must be an array")
        .iter()
        .map(|s| {
            s.get("name")
                .and_then(Value::as_str)
                .expect("sample.name must be a string")
                .to_string()
        })
        .collect()
}

fn assert_schema_v2_envelope(doc: &Value, expected_suite: &str) {
    assert_eq!(
        doc.get("suite").and_then(Value::as_str),
        Some(expected_suite),
        "suite mismatch"
    );
    assert_eq!(
        doc.get("schema_version").and_then(Value::as_str),
        Some("2.0.0"),
        "schema version mismatch"
    );
    let machine = doc.get("machine").expect("missing machine object");
    for key in ["os", "arch", "hostname", "rustc", "commit", "features"] {
        assert!(machine.get(key).is_some(), "machine.{key} missing");
    }
    let samples = doc
        .get("samples")
        .and_then(Value::as_array)
        .expect("samples must be array");
    assert!(!samples.is_empty(), "samples must not be empty");
    for sample in samples {
        for key in [
            "name",
            "backend",
            "device",
            "dtype",
            "model_params",
            "params",
            "runs_ms",
            "mean_ms",
            "std_ms",
            "min_ms",
            "max_ms",
            "p50_ms",
        ] {
            assert!(sample.get(key).is_some(), "sample.{key} missing");
        }
    }
}

#[test]
fn rustral_workloads_emits_schema_and_expected_workloads() {
    let doc = run_bin("CARGO_BIN_EXE_rustral_workloads");
    assert_schema_v2_envelope(&doc, "rustral");
    let names = sample_names(&doc);
    for required in [
        "matmul",
        "attention.small",
        "conv2d.small",
        "mlp_train_step",
        "optimizer_step.sgd",
        "optimizer_step.adam",
        "transformer_encoder.forward",
        "decoder.prefill",
        "decoder.decode_step.no_cache",
        "kv_cache.prefill",
        "kv_cache.decode_step",
        "model_io.save",
        "model_io.load",
    ] {
        assert!(names.contains(required), "missing workload: {required}");
    }
}

#[test]
fn candle_workloads_emits_schema_and_expected_workloads() {
    let doc = run_bin("CARGO_BIN_EXE_candle_workloads");
    assert_schema_v2_envelope(&doc, "candle");
    let names = sample_names(&doc);
    for required in ["matmul", "attention.small", "conv2d.small"] {
        assert!(names.contains(required), "missing workload: {required}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn rustral_workloads_cuda_emits_schema_when_opted_in() {
    if std::env::var("RUSTRAL_TEST_GPU").ok().as_deref() != Some("1") {
        eprintln!("skipping CUDA workload binary (set RUSTRAL_TEST_GPU=1 to enable)");
        return;
    }
    let doc = run_bin("CARGO_BIN_EXE_rustral_workloads_cuda");
    assert_schema_v2_envelope(&doc, "rustral-cuda");
    // Spot-check: at least one known workload.
    let names = sample_names(&doc);
    assert!(names.contains("matmul"), "missing workload: matmul");
}

#[cfg(feature = "metal")]
#[test]
fn rustral_workloads_metal_emits_schema_when_opted_in() {
    if std::env::var("RUSTRAL_TEST_GPU").ok().as_deref() != Some("1") {
        eprintln!("skipping Metal workload binary (set RUSTRAL_TEST_GPU=1 to enable)");
        return;
    }
    let doc = run_bin("CARGO_BIN_EXE_rustral_workloads_metal");
    assert_schema_v2_envelope(&doc, "rustral-metal");
    let names = sample_names(&doc);
    assert!(names.contains("matmul"), "missing workload: matmul");
}
