//! Shared utilities for the Rustral benchmark harness binaries.
//!
//! These helpers let `rustral_workloads`, `candle_workloads`, and any future
//! GPU-specific binaries emit results in the unified JSON schema described in
//! [`benchmarks/SCHEMA.md`](../../../benchmarks/SCHEMA.md). Schema version is
//! `2.0.0`.
//!
//! The harness orchestrator (`scripts/bench/run_all.py`) collects these JSON
//! emissions from multiple suites and produces a unified summary table.

use std::time::{Duration, Instant};

/// Schema version emitted in every suite document.
pub const SCHEMA_VERSION: &str = "2.0.0";

/// Result of a single workload run.
#[derive(Clone, Debug)]
pub struct Sample {
    pub name: String,
    pub backend: String,
    /// Logical device, e.g. `"cpu"`, `"cuda:0"`, `"metal:0"`, `"wgpu:0"`.
    pub device: String,
    /// Element dtype, e.g. `"f32"`, `"f16"`, `"bf16"`.
    pub dtype: String,
    /// Optional model parameter count (only meaningful for model-level workloads).
    pub model_params: Option<u64>,
    pub params: Vec<(String, String)>,
    pub runs_ms: Vec<f64>,
}

impl Sample {
    /// Build a CPU/f32 sample without a model parameter count (most micro-benches).
    pub fn cpu_f32(
        name: impl Into<String>,
        backend: impl Into<String>,
        params: Vec<(String, String)>,
        runs_ms: Vec<f64>,
    ) -> Self {
        Self {
            name: name.into(),
            backend: backend.into(),
            device: "cpu".to_string(),
            dtype: "f32".to_string(),
            model_params: None,
            params,
            runs_ms,
        }
    }

    /// Set the model parameter count and return self.
    pub fn with_model_params(mut self, params: u64) -> Self {
        self.model_params = Some(params);
        self
    }

    /// Set the device label and return self.
    pub fn with_device(mut self, device: impl Into<String>) -> Self {
        self.device = device.into();
        self
    }

    pub fn mean_ms(&self) -> f64 {
        if self.runs_ms.is_empty() {
            return 0.0;
        }
        self.runs_ms.iter().sum::<f64>() / self.runs_ms.len() as f64
    }

    pub fn std_ms(&self) -> f64 {
        if self.runs_ms.len() < 2 {
            return 0.0;
        }
        let m = self.mean_ms();
        let var = self.runs_ms.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (self.runs_ms.len() - 1) as f64;
        var.sqrt()
    }

    pub fn min_ms(&self) -> f64 {
        self.runs_ms.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    pub fn max_ms(&self) -> f64 {
        self.runs_ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    pub fn p50_ms(&self) -> f64 {
        if self.runs_ms.is_empty() {
            return 0.0;
        }
        let mut sorted = self.runs_ms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted[sorted.len() / 2]
    }
}

/// Run a closure `repeats` times after a small warmup; collect per-run wall time in ms.
pub fn time_runs<F: FnMut()>(mut f: F, warmup: usize, repeats: usize) -> Vec<f64> {
    for _ in 0..warmup {
        f();
    }
    let mut out = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let t = Instant::now();
        f();
        out.push(duration_ms(t.elapsed()));
    }
    out
}

/// Time a closure that runs one full forward+backward+optimizer step.
///
/// Identical to [`time_runs`] today, but exposed under an explicit name so callers can
/// signal intent ("this is a train_step, not a forward-only micro-bench") and so that
/// future gating (e.g. NVTX ranges, per-step memory probes) only flips behavior in this
/// path.
///
/// IMPORTANT for GPU workloads: do not call host-bound operations such as `tensor_to_vec`
/// inside `step`. Loss/accuracy probes belong outside the timed region.
pub fn time_train_step<F: FnMut()>(step: F, warmup: usize, repeats: usize) -> Vec<f64> {
    time_runs(step, warmup, repeats)
}

#[inline]
fn duration_ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// Resolve hostname portably. Falls back to `HOSTNAME`/`COMPUTERNAME` env vars then `"unknown"`.
fn detect_hostname() -> String {
    if let Ok(out) = std::process::Command::new("hostname").output() {
        if let Ok(s) = String::from_utf8(out.stdout) {
            let trimmed = s.trim();
            if !trimmed.is_empty() {
                return trimmed.to_string();
            }
        }
    }
    if let Ok(s) = std::env::var("HOSTNAME") {
        if !s.is_empty() {
            return s;
        }
    }
    if let Ok(s) = std::env::var("COMPUTERNAME") {
        if !s.is_empty() {
            return s;
        }
    }
    "unknown".to_string()
}

fn detect_features() -> Vec<String> {
    // Surfaced from the harness orchestrator when known.
    let raw = std::env::var("RUSTRAL_BENCH_FEATURES").unwrap_or_default();
    raw.split(',').map(str::trim).filter(|s| !s.is_empty()).map(|s| s.to_string()).collect()
}

/// Serialize a list of samples to the unified harness JSON schema (v2).
pub fn samples_to_json(suite: &str, samples: &[Sample]) -> String {
    let mut s = String::new();
    s.push_str("{\n");
    s.push_str(&format!("  \"suite\": \"{}\",\n", json_escape(suite)));
    s.push_str(&format!("  \"schema_version\": \"{}\",\n", SCHEMA_VERSION));
    s.push_str("  \"machine\": ");
    s.push_str(&machine_metadata_json());
    s.push_str(",\n");
    s.push_str("  \"samples\": [\n");
    for (i, smp) in samples.iter().enumerate() {
        s.push_str("    {\n");
        s.push_str(&format!("      \"name\": \"{}\",\n", json_escape(&smp.name)));
        s.push_str(&format!("      \"backend\": \"{}\",\n", json_escape(&smp.backend)));
        s.push_str(&format!("      \"device\": \"{}\",\n", json_escape(&smp.device)));
        s.push_str(&format!("      \"dtype\": \"{}\",\n", json_escape(&smp.dtype)));
        match smp.model_params {
            Some(n) => s.push_str(&format!("      \"model_params\": {},\n", n)),
            None => s.push_str("      \"model_params\": null,\n"),
        }
        s.push_str("      \"params\": {");
        let mut first = true;
        for (k, v) in &smp.params {
            if !first {
                s.push_str(", ");
            }
            first = false;
            s.push_str(&format!("\"{}\": \"{}\"", json_escape(k), json_escape(v)));
        }
        s.push_str("},\n");
        s.push_str("      \"runs_ms\": [");
        for (j, r) in smp.runs_ms.iter().enumerate() {
            if j > 0 {
                s.push_str(", ");
            }
            s.push_str(&format!("{:.6}", r));
        }
        s.push_str("],\n");
        s.push_str(&format!(
            "      \"mean_ms\": {:.6}, \"std_ms\": {:.6}, \"min_ms\": {:.6}, \"max_ms\": {:.6}, \"p50_ms\": {:.6}\n",
            smp.mean_ms(),
            smp.std_ms(),
            smp.min_ms(),
            smp.max_ms(),
            smp.p50_ms(),
        ));
        s.push_str("    }");
        if i + 1 < samples.len() {
            s.push(',');
        }
        s.push('\n');
    }
    s.push_str("  ]\n");
    s.push_str("}\n");
    s
}

fn machine_metadata_json() -> String {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let rustc = std::env::var("RUSTC_VERSION").unwrap_or_else(|_| "unknown".to_string());
    let commit = std::env::var("GIT_SHA").unwrap_or_else(|_| "unknown".to_string());
    let hostname = detect_hostname();
    let features = detect_features();
    let mut s = String::new();
    s.push('{');
    s.push_str(&format!(" \"os\": \"{}\",", json_escape(os)));
    s.push_str(&format!(" \"arch\": \"{}\",", json_escape(arch)));
    s.push_str(&format!(" \"hostname\": \"{}\",", json_escape(&hostname)));
    s.push_str(&format!(" \"rustc\": \"{}\",", json_escape(&rustc)));
    s.push_str(&format!(" \"commit\": \"{}\",", json_escape(&commit)));
    s.push_str(" \"features\": [");
    for (i, f) in features.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        s.push_str(&format!("\"{}\"", json_escape(f)));
    }
    s.push_str("] }");
    s
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}
