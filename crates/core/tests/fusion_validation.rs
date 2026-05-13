//! Tests for fusion validation using the fusion test harness.
//!
//! Validates that fused operations produce numerically equivalent results
//! to unfused operation sequences within tolerance.

use rustral_core::{generate_random_data, DType, FusionTestHarness};

#[test]
fn test_fusion_matmul_bias_relu_validation() {
    use rustral_core::Backend;
    use rustral_ndarray_backend::CpuBackend;

    let backend = CpuBackend::default();
    let harness = FusionTestHarness::new();

    // Generate test data with simple shapes
    let input_data = generate_random_data(3 * 2, 42); // [2, 3]
    let input = backend.tensor_from_vec(input_data.clone(), &[2, 3]).unwrap();

    let weight_data = generate_random_data(4 * 3, 43); // [3, 4]
    let weight = backend.tensor_from_vec(weight_data, &[3, 4]).unwrap();

    let bias_data = generate_random_data(2 * 4, 44); // [2, 4]
    let bias = backend.tensor_from_vec(bias_data, &[2, 4]).unwrap(); // Match matmul output shape

    // Unfused implementation
    let unfused_fn = || {
        let ops = backend.ops();
        let h = ops.matmul(&input, &weight).unwrap();
        let h = ops.add(&h, &bias).unwrap();
        let h = ops.relu(&h).unwrap();
        ops.tensor_to_vec(&h).unwrap()
    };

    // Fused implementation (using FusionOptimizer)
    let fused_fn = || {
        use rustral_core::{FusionOptimizer, Parameter};
        let weight_param = Parameter::new("weight", weight.clone());
        let bias_param = Parameter::new("bias", bias.clone());
        let optimizer = FusionOptimizer::new(backend.clone());
        let h = optimizer.matmul_bias_relu(&input, &weight_param, &bias_param).unwrap();
        backend.ops().tensor_to_vec(&h).unwrap()
    };

    let result = harness.run_test("matmul_bias_relu", unfused_fn, fused_fn, DType::F32);

    // The fused implementation should produce identical results to unfused
    // (since the current implementation is just a sequence fallback)
    assert!(result.passed);
    println!("Matmul+Bias+ReLU fusion test: speedup={:.2}x", result.speedup);
}

#[test]
fn test_fusion_matmul_bias_gelu_validation() {
    use rustral_core::Backend;
    use rustral_ndarray_backend::CpuBackend;

    let backend = CpuBackend::default();
    let harness = FusionTestHarness::new();

    // Generate test data with simple shapes
    let input_data = generate_random_data(3 * 2, 45); // [2, 3]
    let input = backend.tensor_from_vec(input_data.clone(), &[2, 3]).unwrap();

    let weight_data = generate_random_data(4 * 3, 46); // [3, 4]
    let weight = backend.tensor_from_vec(weight_data, &[3, 4]).unwrap();

    let bias_data = generate_random_data(2 * 4, 47); // [2, 4]
    let bias = backend.tensor_from_vec(bias_data, &[2, 4]).unwrap(); // Match matmul output shape

    // Unfused implementation
    let unfused_fn = || {
        let ops = backend.ops();
        let h = ops.matmul(&input, &weight).unwrap();
        let h = ops.add(&h, &bias).unwrap();
        let h = ops.gelu(&h).unwrap();
        ops.tensor_to_vec(&h).unwrap()
    };

    // Fused implementation (using FusionOptimizer)
    let fused_fn = || {
        use rustral_core::{FusionOptimizer, Parameter};
        let weight_param = Parameter::new("weight", weight.clone());
        let bias_param = Parameter::new("bias", bias.clone());
        let optimizer = FusionOptimizer::new(backend.clone());
        let h = optimizer.matmul_bias_gelu(&input, &weight_param, &bias_param).unwrap();
        backend.ops().tensor_to_vec(&h).unwrap()
    };

    let result = harness.run_test("matmul_bias_gelu", unfused_fn, fused_fn, DType::F32);

    // The fused implementation should produce identical results to unfused
    assert!(result.passed);
    println!("Matmul+Bias+GELU fusion test: speedup={:.2}x", result.speedup);
}

#[test]
fn test_fusion_matmul_bias_validation() {
    use rustral_core::Backend;
    use rustral_ndarray_backend::CpuBackend;

    let backend = CpuBackend::default();
    let harness = FusionTestHarness::new();

    // Generate test data with simple shapes
    let input_data = generate_random_data(3 * 2, 48); // [2, 3]
    let input = backend.tensor_from_vec(input_data.clone(), &[2, 3]).unwrap();

    let weight_data = generate_random_data(4 * 3, 49); // [3, 4]
    let weight = backend.tensor_from_vec(weight_data, &[3, 4]).unwrap();

    let bias_data = generate_random_data(2 * 4, 50); // [2, 4]
    let bias = backend.tensor_from_vec(bias_data, &[2, 4]).unwrap(); // Match matmul output shape

    // Unfused implementation
    let unfused_fn = || {
        let ops = backend.ops();
        let h = ops.matmul(&input, &weight).unwrap();
        let h = ops.add(&h, &bias).unwrap();
        ops.tensor_to_vec(&h).unwrap()
    };

    // Fused implementation (using FusionOptimizer)
    let fused_fn = || {
        use rustral_core::{FusionOptimizer, Parameter};
        let weight_param = Parameter::new("weight", weight.clone());
        let bias_param = Parameter::new("bias", bias.clone());
        let optimizer = FusionOptimizer::new(backend.clone());
        let h = optimizer.matmul_bias(&input, &weight_param, &bias_param).unwrap();
        backend.ops().tensor_to_vec(&h).unwrap()
    };

    let result = harness.run_test("matmul_bias", unfused_fn, fused_fn, DType::F32);

    // The fused implementation should produce identical results to unfused
    assert!(result.passed);
    println!("Matmul+Bias fusion test: speedup={:.2}x", result.speedup);
}
