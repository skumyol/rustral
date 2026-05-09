use rustral_core::Tolerance;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum BackendFlavor {
    Cpu,
    Candle,
    Wgpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum OpFamily {
    Elementwise,
    Matmul,
    Softmax,
    LayerNorm,
}

pub fn default_tolerance(backend: BackendFlavor, op: OpFamily) -> Tolerance {
    match (backend, op) {
        // CPU is our tightest baseline.
        (BackendFlavor::Cpu, OpFamily::Elementwise) => Tolerance::new(1e-6, 1e-7),
        (BackendFlavor::Cpu, OpFamily::Matmul) => Tolerance::new(1e-5, 1e-6),
        (BackendFlavor::Cpu, OpFamily::Softmax) => Tolerance::new(1e-5, 1e-6),
        (BackendFlavor::Cpu, OpFamily::LayerNorm) => Tolerance::new(1e-5, 1e-6),

        // Candle may use GPU kernels; allow slightly looser defaults.
        (BackendFlavor::Candle, OpFamily::Elementwise) => Tolerance::new(1e-5, 1e-6),
        (BackendFlavor::Candle, OpFamily::Matmul) => Tolerance::new(1e-4, 1e-5),
        (BackendFlavor::Candle, OpFamily::Softmax) => Tolerance::new(1e-4, 1e-5),
        (BackendFlavor::Candle, OpFamily::LayerNorm) => Tolerance::new(1e-4, 1e-5),

        // WGPU kernels are expected to be correct, but may differ slightly by backend/driver.
        (BackendFlavor::Wgpu, OpFamily::Elementwise) => Tolerance::new(1e-5, 1e-6),
        (BackendFlavor::Wgpu, OpFamily::Matmul) => Tolerance::new(1e-4, 1e-5),
        (BackendFlavor::Wgpu, OpFamily::Softmax) => Tolerance::new(5e-4, 5e-5),
        (BackendFlavor::Wgpu, OpFamily::LayerNorm) => Tolerance::new(5e-4, 5e-5),
    }
}

pub fn assert_allclose(name: &str, got: &[f32], expected: &[f32], tol: Tolerance) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{name}: length mismatch got={} expected={}",
        got.len(),
        expected.len()
    );

    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        if !g.is_finite() || !e.is_finite() {
            panic!("{name}: non-finite at idx={i} got={g} expected={e}");
        }
        let diff = (g - e).abs() as f64;
        let max_abs = (g.abs().max(e.abs())) as f64;
        let ok = diff <= tol.atol || diff <= tol.rtol * max_abs;
        if !ok {
            panic!(
                "{name}: mismatch at idx={i} got={g} expected={e} diff={diff} (rtol={} atol={})",
                tol.rtol,
                tol.atol
            );
        }
    }
}

