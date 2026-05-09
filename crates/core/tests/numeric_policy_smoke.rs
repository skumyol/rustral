mod common;

use common::numeric::{assert_allclose, default_tolerance, BackendFlavor, OpFamily};

#[test]
fn tolerance_smoke_allclose_passes_for_identical() {
    let tol = default_tolerance(BackendFlavor::Cpu, OpFamily::Elementwise);
    let a = vec![1.0f32, 2.0, 3.0];
    let b = vec![1.0f32, 2.0, 3.0];
    assert_allclose("identical", &a, &b, tol);
}

