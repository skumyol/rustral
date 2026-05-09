//! WGPU Backend Correctness Tests
//!
//! This test suite validates the correctness of WGPU backend operations
//! including kernel knobs (workgroup size, vectorization) and environment
//! variable controls.
//!
//! Environment Variables:
//! - RUSTRAL_REQUIRE_GPU=1: Fail fast if GPU initialization fails
//! - RUSTRAL_WGPU_WORKGROUP=128|256: Control workgroup size (default: 256)
//! - RUSTRAL_WGPU_VECTORIZED=1: Enable vectorized kernels (default: disabled)
//! - RUSTRAL_WGPU_MATMUL_TILE=8|16: Control matmul tile size (default: heuristic)

use rustral_core::Parameter;

fn get_backend() -> Option<rustral_wgpu_backend::WgpuBackend> {
    match rustral_wgpu_backend::WgpuBackend::new_sync() {
        Ok(b) => Some(b),
        Err(e) => {
            let require = std::env::var("RUSTRAL_REQUIRE_GPU").as_deref() == Ok("1");
            if require {
                panic!("RUSTRAL_REQUIRE_GPU=1 but WGPU backend init failed: {e:?}");
            }
            println!("Skipping wgpu test - GPU init failed: {e:?}");
            None
        }
    }
}

#[test]
fn test_wgpu_error_display() {
    use rustral_wgpu_backend::WgpuError;
    
    let err = WgpuError::NoAdapter;
    assert_eq!(err.to_string(), "no suitable GPU adapter found");

    let err = WgpuError::Shader("syntax error".to_string());
    assert_eq!(err.to_string(), "shader compilation failed: syntax error");
}

#[test]
fn test_wgpu_error_debug() {
    use rustral_wgpu_backend::WgpuError;
    
    let err = WgpuError::NoAdapter;
    let debug = format!("{:?}", err);
    assert!(debug.contains("NoAdapter"));
}

#[test]
fn test_wgpu_backend_creation() {
    let backend = match get_backend() {
        Some(b) => b,
        None => return,
    };

    let tensor = backend.tensor_from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let data = backend.to_vec(&tensor);
    assert_eq!(data, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_dropout_inference_identity() {
    let backend = match get_backend() {
        Some(b) => b,
        None => return,
    };

    let input = backend.tensor_from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();

    // Inference mode: dropout should be identity
    let output = backend.ops().dropout(&input, 0.5, false).unwrap();
    let output_data = backend.to_vec(&output);

    // In inference mode, output should equal input (inverted dropout scales during training)
    assert_eq!(
        output_data,
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        "Dropout in inference mode should be identity"
    );
}

#[test]
fn test_dropout_training_effect() {
    let backend = match get_backend() {
        Some(b) => b,
        None => return,
    };

    // Large tensor for statistical test
    let size = 10000usize;
    let input_data: Vec<f32> = (0..size).map(|_i| 1.0f32).collect();
    let input = backend.tensor_from_vec(input_data.clone(), &[size]).unwrap();

    // Apply dropout with p=0.5
    let output = backend.ops().dropout(&input, 0.5, true).unwrap();
    let output_data = backend.to_vec(&output);

    // With inverted dropout (scale = 1/(1-p) = 2.0), expected values are:
    // - 0.0 (dropped, probability p=0.5)
    // - 2.0 (kept and scaled, probability 1-p=0.5)
    // Check that we have both zeros and scaled values
    let zeros = output_data.iter().filter(|&&v| v == 0.0).count();
    let scaled = output_data.iter().filter(|&&v| v == 2.0).count();
    let total = zeros + scaled;

    assert_eq!(total, size, "All values should be either 0 or 2.0");

    // With p=0.5, expect roughly 50% dropped (allow ±10% for randomness)
    let expected_zeros = size / 2;
    let tolerance = size / 10;
    assert!(
        zeros >= expected_zeros.saturating_sub(tolerance) && zeros <= expected_zeros + tolerance,
        "Expected ~{} zeros, got {} (p=0.5, n={})",
        expected_zeros,
        zeros,
        size
    );

    println!("Dropout test: {} zeros, {} scaled (of {} total)", zeros, scaled, size);
}

#[test]
fn test_dropout_zero_probability() {
    let backend = match get_backend() {
        Some(b) => b,
        None => return,
    };

    let input = backend.tensor_from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();

    // p=0 means no dropout
    let output = backend.ops().dropout(&input, 0.0, true).unwrap();
    let output_data = backend.to_vec(&output);

    // With p=0, scale=1.0, so output equals input
    assert_eq!(output_data, vec![1.0, 2.0, 3.0, 4.0, 5.0], "Dropout with p=0 should be identity");
}

#[test]
fn test_dropout_one_probability() {
    let backend = match get_backend() {
        Some(b) => b,
        None => return,
    };

    let input = backend.tensor_from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();

    // p=1 means everything is dropped
    let output = backend.ops().dropout(&input, 1.0, true).unwrap();
    let output_data = backend.to_vec(&output);

    // With p=1, everything is zeroed (or undefined, but should be finite)
    assert!(output_data.iter().all(|&v| v.is_finite()), "All values should be finite");
}

#[test]
fn test_wgpu_matmul() {
    let backend = match get_backend() {
        Some(b) => b,
        None => return,
    };

    let a = backend.tensor_from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = backend.tensor_from_vec(vec![5.0f32, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

    let c = backend.ops().matmul(&a, &b).unwrap();
    let data = backend.to_vec(&c);

    // [1 2; 3 4] * [5 6; 7 8] = [19 22; 43 50]
    assert_eq!(data, vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn test_wgpu_element_wise_ops() {
    let backend = match get_backend() {
        Some(b) => b,
        None => return,
    };

    let a = backend.tensor_from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
    let b = backend.tensor_from_vec(vec![5.0f32, 4.0, 3.0, 2.0], &[4]).unwrap();

    // add
    let c = backend.ops().add(&a, &b).unwrap();
    assert_eq!(backend.to_vec(&c), vec![6.0, 6.0, 6.0, 6.0]);

    // mul
    let c = backend.ops().mul(&a, &b).unwrap();
    assert_eq!(backend.to_vec(&c), vec![5.0, 8.0, 9.0, 8.0]);

    // sub
    let c = backend.ops().sub(&a, &b).unwrap();
    assert_eq!(backend.to_vec(&c), vec![-4.0, -2.0, 0.0, 2.0]);

    // div
    let c = backend.ops().div(&a, &b).unwrap();
    assert_eq!(backend.to_vec(&c), vec![0.2, 0.5, 1.0, 2.0]);

    // maximum
    let c = backend.ops().maximum(&a, &b).unwrap();
    assert_eq!(backend.to_vec(&c), vec![5.0, 4.0, 3.0, 4.0]);
}

#[test]
fn test_wgpu_unary_ops() {
    let backend = match get_backend() {
        Some(b) => b,
        None => return,
    };

    let a = backend.tensor_from_vec(vec![-1.0f32, 0.0, 1.0, 2.0], &[4]).unwrap();

    // relu
    let c = backend.ops().relu(&a).unwrap();
    assert_eq!(backend.to_vec(&c), vec![0.0, 0.0, 1.0, 2.0]);

    // neg
    let c = backend.ops().neg(&a).unwrap();
    assert_eq!(backend.to_vec(&c), vec![1.0, 0.0, -1.0, -2.0]);

    // sigmoid at 0 should be 0.5
    let c = backend.ops().sigmoid(&backend.tensor_from_vec(vec![0.0f32], &[1]).unwrap()).unwrap();
    assert!((backend.to_vec(&c)[0] - 0.5).abs() < 1e-5);

    // tanh at 0 should be 0
    let c = backend.ops().tanh(&backend.tensor_from_vec(vec![0.0f32], &[1]).unwrap()).unwrap();
    assert!(backend.to_vec(&c)[0].abs() < 1e-5);
}

#[test]
fn test_wgpu_scalar_ops() {
    let backend = match get_backend() {
        Some(b) => b,
        None => return,
    };

    let a = backend.tensor_from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();

    // add_scalar
    let c = backend.ops().add_scalar(&a, 10.0).unwrap();
    assert_eq!(backend.to_vec(&c), vec![11.0, 12.0, 13.0, 14.0]);

    // mul_scalar
    let c = backend.ops().mul_scalar(&a, 2.0).unwrap();
    assert_eq!(backend.to_vec(&c), vec![2.0, 4.0, 6.0, 8.0]);

    // gt_scalar
    let c = backend.ops().gt_scalar(&a, 2.5).unwrap();
    assert_eq!(backend.to_vec(&c), vec![0.0, 0.0, 1.0, 1.0]);
}

#[test]
fn test_wgpu_transpose() {
    let backend = match get_backend() {
        Some(b) => b,
        None => return,
    };

    let a = backend.tensor_from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

    let c = backend.ops().transpose(&a).unwrap();
    let data = backend.to_vec(&c);

    // [1 2 3; 4 5 6] transposed = [1 4; 2 5; 3 6]
    assert_eq!(data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_wgpu_gather_rows() {
    let backend = match get_backend() {
        Some(b) => b,
        None => return,
    };

    // Table: 4 rows x 3 cols
    let table_data = vec![
        1.0f32, 2.0, 3.0, // row 0
        4.0, 5.0, 6.0, // row 1
        7.0, 8.0, 9.0, // row 2
        10.0, 11.0, 12.0, // row 3
    ];
    let table_tensor = backend.tensor_from_vec(table_data, &[4, 3]).unwrap();
    let table_param = Parameter::new("table", table_tensor);

    // Gather rows [0, 2, 1]
    let gathered = backend.ops().gather_rows(&table_param, &[0, 2, 1]).unwrap();
    let data = backend.to_vec(&gathered);

    // Expected: row0, row2, row1
    assert_eq!(data, vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0, 4.0, 5.0, 6.0,]);
    assert_eq!(gathered.shape, vec![3, 3]);
}

#[test]
fn test_wgpu_linear() {
    let backend = match get_backend() {
        Some(b) => b,
        None => return,
    };

    // input: [2, 3]
    let input = backend.tensor_from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    // weight: [2, 3] -> out=2, in=3
    let weight = backend.tensor_from_vec(vec![1.0f32, 0.0, 1.0, 0.0, 1.0, 1.0], &[2, 3]).unwrap();
    let weight_param = Parameter::new("w", weight);
    // bias: [2]
    let bias = backend.tensor_from_vec(vec![1.0f32, 2.0], &[2]).unwrap();
    let bias_param = Parameter::new("b", bias);

    let output = backend.ops().linear(&input, &weight_param, Some(&bias_param)).unwrap();
    let data = backend.to_vec(&output);

    // row0: [1,2,3] dot [1,0,1]=4, [0,1,1]=5 -> +bias [1,2] = [5,7]
    // row1: [4,5,6] dot [1,0,1]=10, [0,1,1]=11 -> +bias [1,2] = [11,13]
    assert_eq!(data, vec![5.0, 7.0, 11.0, 13.0]);
    assert_eq!(output.shape, vec![2, 2]);
}

#[test]
fn test_wgpu_add_row_vector() {
    let backend = match get_backend() {
        Some(b) => b,
        None => return,
    };

    let a = backend.tensor_from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let row = backend.tensor_from_vec(vec![10.0f32, 20.0, 30.0], &[3]).unwrap();

    let c = backend.ops().add_row_vector(&a, &row).unwrap();
    let data = backend.to_vec(&c);

    assert_eq!(data, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
}

#[test]
fn test_wgpu_softmax_2d() {
    let backend = match get_backend() {
        Some(b) => b,
        None => return,
    };

    let a = backend.tensor_from_vec(vec![1.0f32, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3]).unwrap();

    let c = backend.ops().softmax(&a).unwrap();
    let data = backend.to_vec(&c);

    // Each row should sum to 1
    let row0_sum: f32 = data[0..3].iter().sum();
    let row1_sum: f32 = data[3..6].iter().sum();
    assert!((row0_sum - 1.0).abs() < 1e-5, "row0 sum = {}", row0_sum);
    assert!((row1_sum - 1.0).abs() < 1e-5, "row1 sum = {}", row1_sum);
    // Both rows are identical input, so output should match
    assert_eq!(data[0..3], data[3..6]);
}

#[test]
fn test_wgpu_log_softmax_2d() {
    let backend = match get_backend() {
        Some(b) => b,
        None => return,
    };

    let a = backend.tensor_from_vec(vec![1.0f32, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3]).unwrap();

    let c = backend.ops().log_softmax(&a).unwrap();
    let data = backend.to_vec(&c);

    // exp(log_softmax) should sum to 1 per row
    let row0_sum: f32 = data[0..3].iter().map(|&v| v.exp()).sum();
    let row1_sum: f32 = data[3..6].iter().map(|&v| v.exp()).sum();
    assert!((row0_sum - 1.0).abs() < 1e-5, "row0 sum = {}", row0_sum);
    assert!((row1_sum - 1.0).abs() < 1e-5, "row1 sum = {}", row1_sum);
}

#[test]
fn test_wgpu_tensor_element() {
    let backend = match get_backend() {
        Some(b) => b,
        None => return,
    };

    let a = backend.tensor_from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();

    assert_eq!(backend.ops().tensor_element(&a, 0).unwrap(), 1.0);
    assert_eq!(backend.ops().tensor_element(&a, 2).unwrap(), 3.0);
    assert_eq!(backend.ops().tensor_element(&a, 3).unwrap(), 4.0);
}

#[test]
fn test_wgpu_tensor_to_vec() {
    let backend = match get_backend() {
        Some(b) => b,
        None => return,
    };

    let a = backend.tensor_from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let data = backend.ops().tensor_to_vec(&a).unwrap();
    assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
}

// Correctness harness for kernel knobs
#[test]
fn test_workgroup_size_knob() {
    let backend = match get_backend() {
        Some(b) => b,
        None => return,
    };

    // Test that workgroup size environment variable is respected
    // This is a smoke test - the actual kernel selection happens internally
    let a = backend.tensor_from_vec(vec![1.0f32; 1000], &[1000]).unwrap();
    let b = backend.tensor_from_vec(vec![2.0f32; 1000], &[1000]).unwrap();
    
    let c = backend.ops().add(&a, &b).unwrap();
    let data = backend.to_vec(&c);
    
    // Regardless of workgroup size, result should be correct
    assert_eq!(data, vec![3.0f32; 1000]);
    
    println!("Workgroup size knob test passed with RUSTRAL_WGPU_WORKGROUP={:?}", 
             std::env::var("RUSTRAL_WGPU_WORKGROUP"));
}

#[test]
fn test_vectorization_knob() {
    let backend = match get_backend() {
        Some(b) => b,
        None => return,
    };

    // Test that vectorization environment variable is respected
    let a = backend.tensor_from_vec(vec![1.0f32; 1000], &[1000]).unwrap();
    let b = backend.tensor_from_vec(vec![2.0f32; 1000], &[1000]).unwrap();
    
    let c = backend.ops().mul(&a, &b).unwrap();
    let data = backend.to_vec(&c);
    
    // Regardless of vectorization, result should be correct
    assert_eq!(data, vec![2.0f32; 1000]);
    
    println!("Vectorization knob test passed with RUSTRAL_WGPU_VECTORIZED={:?}", 
             std::env::var("RUSTRAL_WGPU_VECTORIZED"));
}

#[test]
fn test_matmul_tile_knob() {
    let backend = match get_backend() {
        Some(b) => b,
        None => return,
    };

    // Test that matmul tile size environment variable is respected
    let a = backend.tensor_from_vec(vec![1.0f32; 64 * 64], &[64, 64]).unwrap();
    let b = backend.tensor_from_vec(vec![1.0f32; 64 * 64], &[64, 64]).unwrap();
    
    let c = backend.ops().matmul(&a, &b).unwrap();
    let data = backend.to_vec(&c);
    
    // Regardless of tile size, result should be correct
    // Each element should be 64.0 (sum of 64 ones)
    assert!(data.iter().all(|&v| (v - 64.0).abs() < 1e-3));
    
    println!("Matmul tile knob test passed with RUSTRAL_WGPU_MATMUL_TILE={:?}", 
             std::env::var("RUSTRAL_WGPU_MATMUL_TILE"));
}

#[test]
fn test_kernel_knob_combination() {
    let backend = match get_backend() {
        Some(b) => b,
        None => return,
    };

    // Test that all kernel knobs work together
    let a = backend.tensor_from_vec(vec![1.0f32; 128], &[128]).unwrap();
    let b = backend.tensor_from_vec(vec![2.0f32; 128], &[128]).unwrap();
    
    let c = backend.ops().add(&a, &b).unwrap();
    let d = backend.ops().mul(&c, &backend.tensor_from_vec(vec![3.0f32; 128], &[128]).unwrap()).unwrap();
    let data = backend.to_vec(&d);
    
    // (1 + 2) * 3 = 9
    assert_eq!(data, vec![9.0f32; 128]);
    
    println!("Combined kernel knobs test passed");
}
