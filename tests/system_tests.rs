//! System Integration Tests
//!
//! Comprehensive test suite covering:
//! - Bug regression tests
//! - Performance benchmarks
//! - Edge cases and boundary conditions
//! - Stress tests
//! - Module integration tests
//!
//! Run with: `cargo test --test system_tests`

mod bug_regression;
mod common;
mod edge_cases;
mod integration;
mod performance;
mod stress;

use common::TestRunner;

/// Run all system tests and report results
#[test]
fn run_all_system_tests() {
    println!("\n{}", "=".repeat(80));
    println!("  NEURAL ENGINE SYSTEM TEST SUITE");
    println!("  Bug Detection | Performance | Integration | Edge Cases | Stress");
    println!("{}", "=".repeat(80));

    let mut runner = TestRunner::new();

    // Bug Regression Tests
    println!("\n--- BUG REGRESSION TESTS ---\n");
    bug_regression::run_all(&mut runner);

    // Edge Case Tests
    println!("\n--- EDGE CASE & BOUNDARY TESTS ---\n");
    edge_cases::run_all(&mut runner);

    // Integration Tests
    println!("\n--- MODULE INTEGRATION TESTS ---\n");
    integration::run_all(&mut runner);

    // Performance Tests
    println!("\n--- PERFORMANCE BENCHMARK TESTS ---\n");
    performance::run_all(&mut runner);

    // Stress Tests
    println!("\n--- STRESS & LOAD TESTS ---\n");
    stress::run_all(&mut runner);

    // Final Report
    runner.print_report();
    assert!(runner.all_passed(), "Some system tests failed!");
}

/// Quick smoke test for CI
#[test]
fn smoke_test() {
    let mut runner = TestRunner::new();

    runner.run_test("smoke_core_module", || {
        // Quick sanity check that core modules load
        use mnr_core::{ForwardCtx, Mode};
        use mnr_ndarray_backend::CpuBackend;

        let backend = CpuBackend::default();
        let _ctx = ForwardCtx::new(&backend, Mode::Inference);
        Ok(())
    });

    runner.run_test("smoke_nn_module", || {
        use mnr_ndarray_backend::CpuBackend;
        use mnr_nn::{Linear, LinearConfig};

        let backend = CpuBackend::default();
        let _linear = Linear::new(&backend, LinearConfig::new(10, 5)).unwrap();
        Ok(())
    });

    runner.run_test("smoke_data_module", || {
        use mnr_data::DataLoaderConfig;
        // Verify data module is available
        let _config = DataLoaderConfig { batch_size: 32, ..Default::default() };
        Ok(())
    });

    assert!(runner.all_passed());
}
