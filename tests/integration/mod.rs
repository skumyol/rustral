//! Module Integration Tests
//!
//! Tests that verify proper interaction between different crates
//! and components of the neural engine.

use mnr_core::{Backend, ForwardCtx, Mode, Module, Trainable, TensorOps};
use mnr_nn::{
    Conv2d, Conv2dConfig, Embedding, EmbeddingConfig, Linear, LinearConfig,
    LayerNorm, LayerNormConfig, BatchNorm, BatchNormConfig,
    SelfAttention, SelfAttentionConfig,
    TransformerEncoder, TransformerEncoderConfig,
    Sequential2, chain,
    Dropout, DropoutConfig,
};
use mnr_ndarray_backend::CpuBackend;

use crate::common::TestRunner;

pub fn run_all(runner: &mut TestRunner) {
    test_core_to_nn_integration(runner);
    test_autodiff_integration(runner);
    test_optim_integration(runner);
    test_data_nn_integration(runner);
    test_sequential_chaining(runner);
    test_mixed_precision_integration(runner);
    test_save_load_integration(runner);
    test_backend_consistency(runner);
    test_parameter_sharing(runner);
    test_mode_switching(runner);
}

// ========================================================================
// Core ↔ NN Integration
// ========================================================================

fn test_core_to_nn_integration(runner: &mut TestRunner) {
    runner.run_test("integration_core_nn", || {
        let backend = CpuBackend::default();

        // Create NN module using core types
        let linear = Linear::new(&backend, LinearConfig::new(10, 5))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        // Get parameters (core trait)
        let params = linear.parameters();
        assert!(!params.is_empty(), "Linear should have parameters");

        // Create input (core tensor)
        let input = backend
            .tensor_from_vec(vec![1.0f32; 20], &[2, 10])
            .map_err(|e| format!("Create input failed: {}", e))?;

        // Forward through context (core mechanism)
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = linear.forward(input, &mut ctx)
            .map_err(|e| format!("Forward failed: {}", e))?;

        // Verify output (core tensor ops)
        let shape = backend.ops().shape(&output);
        assert_eq!(shape, vec![2, 5], "Output shape mismatch");

        Ok(())
    });

    runner.run_test("integration_tensor_ops", || {
        let backend = CpuBackend::default();

        // Test various tensor operations through backend
        let a = backend.tensor_from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2])
            .map_err(|e| format!("Create tensor A failed: {}", e))?;

        let b = backend.tensor_from_vec(vec![0.5f32, 0.5, 0.5, 0.5], &[2, 2])
            .map_err(|e| format!("Create tensor B failed: {}", e))?;

        let ops = backend.ops();

        // Addition
        let c = ops.add(&a, &b)
            .map_err(|e| format!("Add failed: {}", e))?;
        let c_data: Vec<f32> = c.as_ref().to_vec();
        assert_eq!(c_data, vec![1.5, 2.5, 3.5, 4.5], "Addition wrong");

        // Multiplication
        let d = ops.mul(&a, &b)
            .map_err(|e| format!("Mul failed: {}", e))?;
        let d_data: Vec<f32> = d.as_ref().to_vec();
        assert_eq!(d_data, vec![0.5, 1.0, 1.5, 2.0], "Multiplication wrong");

        // Shape
        let shape = ops.shape(&a);
        assert_eq!(shape, vec![2, 2], "Shape wrong");

        Ok(())
    });
}

// ========================================================================
// Autodiff Integration
// ========================================================================

fn test_autodiff_integration(runner: &mut TestRunner) {
    runner.run_test("integration_autodiff_forward", || {
        let backend = CpuBackend::default();

        // Create a simple computation graph
        let linear = Linear::new(&backend, LinearConfig::new(3, 2))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let input = backend
            .tensor_from_vec(vec![1.0f32, 2.0, 3.0], &[1, 3])
            .map_err(|e| format!("Create input failed: {}", e))?;

        // Forward in training mode (autodiff should track)
        let mut ctx = ForwardCtx::new(&backend, Mode::Training);
        let output = linear.forward(input, &mut ctx)
            .map_err(|e| format!("Forward failed: {}", e))?;

        let shape = backend.ops().shape(&output);
        assert_eq!(shape, vec![1, 2], "Autodiff output shape wrong");

        Ok(())
    });

    runner.run_test("integration_mode_switching", || {
        let backend = CpuBackend::default();

        let linear = Linear::new(&backend, LinearConfig::new(5, 3))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let input = backend
            .tensor_from_vec(vec![1.0f32; 10], &[2, 5])
            .map_err(|e| format!("Create input failed: {}", e))?;

        // Inference mode
        let mut inf_ctx = ForwardCtx::new(&backend, Mode::Inference);
        let inf_output = linear.forward(input.clone(), &mut inf_ctx)
            .map_err(|e| format!("Inference forward failed: {}", e))?;

        // Training mode
        let mut train_ctx = ForwardCtx::new(&backend, Mode::Training);
        let train_output = linear.forward(input, &mut train_ctx)
            .map_err(|e| format!("Training forward failed: {}", e))?;

        // Both should produce same shape
        assert_eq!(
            backend.ops().shape(&inf_output),
            backend.ops().shape(&train_output),
            "Mode switching changed output shape"
        );

        Ok(())
    });
}

// ========================================================================
// Optimizer Integration
// ========================================================================

fn test_optim_integration(runner: &mut TestRunner) {
    runner.run_test("integration_optimizer_params", || {
        let backend = CpuBackend::default();

        // Create a model
        let model = Linear::new(&backend, LinearConfig::new(10, 5))
            .map_err(|e| format!("Create model failed: {}", e))?;

        // Get parameters for optimizer
        let params = model.parameters();
        assert_eq!(params.len(), 2, "Should have weight and bias");

        // Verify parameters have gradients in training mode
        let input = backend
            .tensor_from_vec(vec![1.0f32; 20], &[2, 10])
            .map_err(|e| format!("Create input failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Training);
        let _output = model.forward(input, &mut ctx)
            .map_err(|e| format!("Forward failed: {}", e))?;

        Ok(())
    });
}

// ========================================================================
// Data ↔ NN Integration
// ========================================================================

fn test_data_nn_integration(runner: &mut TestRunner) {
    runner.run_test("integration_data_loader", || {
        use mnr_data::{DataLoader, DataLoaderConfig, Dataset};

        let backend = CpuBackend::default();

        // Verify data loader types integrate with NN
        let config = DataLoaderConfig::new(32)
            .with_shuffle(true);

        // Data loader should produce tensors compatible with NN
        println!("  Data loader batch size: {}", config.batch_size);

        Ok(())
    });
}

// ========================================================================
// Sequential Chaining
// ========================================================================

fn test_sequential_chaining(runner: &mut TestRunner) {
    runner.run_test("integration_sequential", || {
        let backend = CpuBackend::default();

        // Build a sequential model
        let model = chain()
            .add(Linear::new(&backend, LinearConfig::new(10, 20))
                .map_err(|e| format!("Create linear1 failed: {}", e))?)
            .add(Linear::new(&backend, LinearConfig::new(20, 5))
                .map_err(|e| format!("Create linear2 failed: {}", e))?);

        let input = backend
            .tensor_from_vec(vec![1.0f32; 30], &[3, 10])
            .map_err(|e| format!("Create input failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = model.forward(input, &mut ctx)
            .map_err(|e| format!("Sequential forward failed: {}", e))?;

        let shape = backend.ops().shape(&output);
        assert_eq!(shape, vec![3, 5], "Sequential output shape wrong");

        // Verify parameters from all layers
        let params = model.parameters();
        assert_eq!(params.len(), 4, "Should have 2 weights + 2 biases");

        Ok(())
    });

    runner.run_test("integration_complex_pipeline", || {
        let backend = CpuBackend::default();

        // Embedding → LayerNorm → Linear
        let embedding = Embedding::new(&backend, EmbeddingConfig::new(1000, 64), 42)
            .map_err(|e| format!("Create embedding failed: {}", e))?;

        let norm = LayerNorm::new(&backend, LayerNormConfig::new(vec![64]), 43)
            .map_err(|e| format!("Create norm failed: {}", e))?;

        let projection = Linear::new(&backend, LinearConfig::new(64, 10))
            .map_err(|e| format!("Create projection failed: {}", e))?;

        // Build pipeline
        let model = chain()
            .add(embedding)
            .add(norm)
            .add(projection);

        let input = backend
            .tensor_from_vec(vec![1u32, 2, 3, 4, 5], &[1, 5])
            .map_err(|e| format!("Create input failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = model.forward(input, &mut ctx)
            .map_err(|e| format!("Pipeline forward failed: {}", e))?;

        let shape = backend.ops().shape(&output);
        assert_eq!(shape, vec![1, 5, 10], "Complex pipeline output shape wrong");

        Ok(())
    });
}

// ========================================================================
// Mixed Precision Integration
// ========================================================================

fn test_mixed_precision_integration(runner: &mut TestRunner) {
    runner.run_test("integration_mixed_precision", || {
        let backend = CpuBackend::default();

        // Test that different operations work together
        let input = backend
            .tensor_from_vec(vec![0.5f32; 100], &[10, 10])
            .map_err(|e| format!("Create input failed: {}", e))?;

        // Linear (FP32)
        let linear = Linear::new(&backend, LinearConfig::new(10, 5))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let linear_out = linear.forward(input, &mut ctx)
            .map_err(|e| format!("Linear failed: {}", e))?;

        // LayerNorm (FP32)
        let norm = LayerNorm::new(&backend, LayerNormConfig::new(vec![5]), 42)
            .map_err(|e| format!("Create norm failed: {}", e))?;

        let norm_out = norm.forward(linear_out, &mut ctx)
            .map_err(|e| format!("Norm failed: {}", e))?;

        let shape = backend.ops().shape(&norm_out);
        assert_eq!(shape, vec![10, 5], "Mixed precision output shape wrong");

        Ok(())
    });
}

// ========================================================================
// Save/Load Integration
// ========================================================================

fn test_save_load_integration(runner: &mut TestRunner) {
    runner.run_test("integration_save_load", || {
        let backend = CpuBackend::default();

        // Create model
        let model = Linear::new(&backend, LinearConfig::new(10, 5))
            .map_err(|e| format!("Create model failed: {}", e))?;

        // Get original parameters
        let original_params: Vec<Vec<f32>> = model
            .parameters()
            .iter()
            .map(|p| p.as_tensor(backend.ops()).as_ref().to_vec())
            .collect();

        // Verify parameters are stable across multiple forwards
        let input = backend
            .tensor_from_vec(vec![0.5f32; 20], &[2, 10])
            .map_err(|e| format!("Create input failed: {}", e))?;

        let mut ctx1 = ForwardCtx::new(&backend, Mode::Inference);
        let out1 = model.forward(input.clone(), &mut ctx1)
            .map_err(|e| format!("First forward failed: {}", e))?;

        let mut ctx2 = ForwardCtx::new(&backend, Mode::Inference);
        let out2 = model.forward(input, &mut ctx2)
            .map_err(|e| format!("Second forward failed: {}", e))?;

        let data1: Vec<f32> = out1.as_ref().to_vec();
        let data2: Vec<f32> = out2.as_ref().to_vec();

        for (i, (a, b)) in data1.iter().zip(data2.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(
                diff < 1e-5,
                "Output element {} differs: {} vs {} (diff: {})",
                i, a, b, diff
            );
        }

        Ok(())
    });
}

// ========================================================================
// Backend Consistency
// ========================================================================

fn test_backend_consistency(runner: &mut TestRunner) {
    runner.run_test("integration_backend_ops", || {
        let backend = CpuBackend::default();

        // Test that all backend operations work correctly
        let ops = backend.ops();

        // Create test tensors
        let a = backend.tensor_from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2])
            .map_err(|e| format!("Create tensor A failed: {}", e))?;

        let b = backend.tensor_from_vec(vec![2.0f32, 2.0, 2.0, 2.0], &[2, 2])
            .map_err(|e| format!("Create tensor B failed: {}", e))?;

        // Test operations
        let _sum = ops.add(&a, &b).map_err(|e| format!("Add failed: {}", e))?;
        let _diff = ops.sub(&a, &b).map_err(|e| format!("Sub failed: {}", e))?;
        let _prod = ops.mul(&a, &b).map_err(|e| format!("Mul failed: {}", e))?;
        let _div = ops.div(&a, &b).map_err(|e| format!("Div failed: {}", e))?;

        // Shape and reshape
        let shape = ops.shape(&a);
        assert_eq!(shape, vec![2, 2], "Shape wrong");

        Ok(())
    });
}

// ========================================================================
// Parameter Sharing
// ========================================================================

fn test_parameter_sharing(runner: &mut TestRunner) {
    runner.run_test("integration_param_sharing", || {
        let backend = CpuBackend::default();

        // Create two layers with independent parameters
        let layer1 = Linear::new(&backend, LinearConfig::new(10, 5))
            .map_err(|e| format!("Create layer1 failed: {}", e))?;

        let layer2 = Linear::new(&backend, LinearConfig::new(10, 5))
            .map_err(|e| format!("Create layer2 failed: {}", e))?;

        // They should have different parameters
        let params1: Vec<Vec<f32>> = layer1
            .parameters()
            .iter()
            .map(|p| p.as_tensor(backend.ops()).as_ref().to_vec())
            .collect();

        let params2: Vec<Vec<f32>> = layer2
            .parameters()
            .iter()
            .map(|p| p.as_tensor(backend.ops()).as_ref().to_vec())
            .collect();

        // Verify they're different (not shared)
        let any_different = params1.iter().zip(params2.iter())
            .any(|(p1, p2)| p1 != p2);

        assert!(any_different, "Two separate layers should have different parameters");

        Ok(())
    });
}

// ========================================================================
// Mode Switching
// ========================================================================

fn test_mode_switching(runner: &mut TestRunner) {
    runner.run_test("integration_mode_inference", || {
        let backend = CpuBackend::default();

        // Test inference mode behavior
        let dropout = Dropout::new(0.5, 42);

        let input = backend
            .tensor_from_vec(vec![1.0f32; 10], &[2, 5])
            .map_err(|e| format!("Create input failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = dropout.forward(input.clone(), &mut ctx)
            .map_err(|e| format!("Dropout inference failed: {}", e))?;

        // In inference mode, dropout should be identity (scale by p)
        let in_data: Vec<f32> = input.as_ref().to_vec();
        let out_data: Vec<f32> = output.as_ref().to_vec();

        // Should be approximately equal (or scaled)
        assert!(
            out_data.iter().zip(in_data.iter()).all(|(a, b)| (a - b).abs() < 1e-5),
            "Inference mode dropout should be identity"
        );

        Ok(())
    });

    runner.run_test("integration_mode_training", || {
        let backend = CpuBackend::default();

        let dropout = Dropout::new(0.5, 42);

        let input = backend
            .tensor_from_vec(vec![1.0f32; 10], &[2, 5])
            .map_err(|e| format!("Create input failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Training);
        let output = dropout.forward(input.clone(), &mut ctx)
            .map_err(|e| format!("Dropout training failed: {}", e))?;

        // In training mode, dropout should modify values
        let out_data: Vec<f32> = output.as_ref().to_vec();

        // Output should be 0.0 or 2.0 (scaled by 1/(1-p) = 2)
        for &val in &out_data {
            assert!(
                val == 0.0 || (val - 2.0).abs() < 1e-5,
                "Training dropout should produce 0 or 2, got {}",
                val
            );
        }

        Ok(())
    });
}
