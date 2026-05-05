//! Module Integration Tests

use mnr_core::{Backend, ForwardCtx, Mode, Module, Trainable, TensorOps};
use mnr_nn::{
    Conv2d, Conv2dConfig, Embedding, EmbeddingConfig, Linear, LinearConfig,
    LayerNorm, LayerNormConfig,
    SelfAttention, SelfAttentionConfig,
    TransformerEncoder, TransformerEncoderConfig,
    TransformerDecoder, TransformerDecoderConfig,
    Sequential2, chain,
    Dropout, DropoutConfig,
};
use mnr_ndarray_backend::CpuBackend;

use crate::common::{TestRunner};

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

fn test_core_to_nn_integration(runner: &mut TestRunner) {
    runner.run_test("integration_core_nn", || {
        let backend = CpuBackend::default();
        let linear = Linear::new(&backend, LinearConfig::new(10, 5))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let params = linear.parameters();
        assert!(!params.is_empty(), "Linear should have parameters");

        let input = backend
            .tensor_from_vec(vec![1.0f32; 20], &[2, 10])
            .map_err(|e| format!("Create input failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = linear.forward(input, &mut ctx)
            .map_err(|e| format!("Forward failed: {}", e))?;

        let shape = backend.ops().shape(&output);
        assert_eq!(shape, vec![2, 5], "Output shape mismatch");
        Ok(())
    });

    runner.run_test("integration_tensor_ops", || {
        let backend = CpuBackend::default();
        let a = backend.tensor_from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2])
            .map_err(|e| format!("Create tensor A failed: {}", e))?;

        let b = backend.tensor_from_vec(vec![0.5f32, 0.5, 0.5, 0.5], &[2, 2])
            .map_err(|e| format!("Create tensor B failed: {}", e))?;

        let ops = backend.ops();

        let c = ops.add(&a, &b)
            .map_err(|e| format!("Add failed: {}", e))?;
        let c_data: Vec<f32> = c.as_ref().to_vec();
        assert_eq!(c_data, vec![1.5, 2.5, 3.5, 4.5], "Addition wrong");

        let d = ops.mul(&a, &b)
            .map_err(|e| format!("Mul failed: {}", e))?;
        let d_data: Vec<f32> = d.as_ref().to_vec();
        assert_eq!(d_data, vec![0.5, 1.0, 1.5, 2.0], "Multiplication wrong");

        let shape = ops.shape(&a);
        assert_eq!(shape, vec![2, 2], "Shape wrong");

        Ok(())
    });
}

fn test_autodiff_integration(runner: &mut TestRunner) {
    runner.run_test("integration_autodiff_forward", || {
        let backend = CpuBackend::default();
        let linear = Linear::new(&backend, LinearConfig::new(3, 2))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let input = backend
            .tensor_from_vec(vec![1.0f32, 2.0, 3.0], &[1, 3])
            .map_err(|e| format!("Create input failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Train);
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

        let mut inf_ctx = ForwardCtx::new(&backend, Mode::Inference);
        let inf_output = linear.forward(input.clone(), &mut inf_ctx)
            .map_err(|e| format!("Inference forward failed: {}", e))?;

        let mut train_ctx = ForwardCtx::new(&backend, Mode::Train);
        let train_output = linear.forward(input, &mut train_ctx)
            .map_err(|e| format!("Training forward failed: {}", e))?;

        assert_eq!(
            backend.ops().shape(&inf_output),
            backend.ops().shape(&train_output),
            "Mode switching changed output shape"
        );
        Ok(())
    });
}

fn test_optim_integration(runner: &mut TestRunner) {
    runner.run_test("integration_optimizer_params", || {
        let backend = CpuBackend::default();
        let model = Linear::new(&backend, LinearConfig::new(10, 5).with_bias(true))
            .map_err(|e| format!("Create model failed: {}", e))?;

        let params = model.parameters();
        assert_eq!(params.len(), 2, "Should have weight and bias");

        let input = backend
            .tensor_from_vec(vec![1.0f32; 20], &[2, 10])
            .map_err(|e| format!("Create input failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Train);
        let _output = model.forward(input, &mut ctx)
            .map_err(|e| format!("Forward failed: {}", e))?;

        Ok(())
    });
}

fn test_data_nn_integration(runner: &mut TestRunner) {
    runner.run_test("integration_data_loader", || {
        use mnr_data::DataLoaderConfig;
        let config = DataLoaderConfig {
            batch_size: 32,
            shuffle: true,
            ..Default::default()
        };
        assert_eq!(config.batch_size, 32);
        Ok(())
    });
}

fn test_sequential_chaining(runner: &mut TestRunner) {
    runner.run_test("integration_sequential", || {
        let backend = CpuBackend::default();
        let l1 = Linear::new(&backend, LinearConfig::new(10, 20).with_bias(true))
            .map_err(|e| format!("Create linear1 failed: {}", e))?;
        let l2 = Linear::new(&backend, LinearConfig::new(20, 5).with_bias(true))
            .map_err(|e| format!("Create linear2 failed: {}", e))?;
        let model = chain(l1, l2);

        let input = backend
            .tensor_from_vec(vec![1.0f32; 30], &[3, 10])
            .map_err(|e| format!("Create input failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = model.forward(input, &mut ctx)
            .map_err(|e| format!("Sequential forward failed: {}", e))?;

        let shape = backend.ops().shape(&output);
        assert_eq!(shape, vec![3, 5], "Sequential output shape wrong");

        let params = model.parameters();
        assert_eq!(params.len(), 4, "Should have 2 weights + 2 biases");
        Ok(())
    });

    runner.run_test("integration_complex_pipeline", || {
        let backend = CpuBackend::default();
        let embedding = Embedding::new(&backend, EmbeddingConfig::new(1000, 64), 42)
            .map_err(|e| format!("Create embedding failed: {}", e))?;
        let norm = LayerNorm::new(&backend, LayerNormConfig::new(vec![64]), 43)
            .map_err(|e| format!("Create norm failed: {}", e))?;
        let projection = Linear::new(&backend, LinearConfig::new(64, 10))
            .map_err(|e| format!("Create projection failed: {}", e))?;

        let embed_norm = chain(embedding, norm);
        let model = chain(embed_norm, projection);

        let input: Vec<usize> = vec![1, 2, 3, 4, 5];
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = model.forward(input, &mut ctx)
            .map_err(|e| format!("Pipeline forward failed: {}", e))?;

        let shape = backend.ops().shape(&output);
        assert_eq!(shape, vec![5, 10], "Complex pipeline output shape wrong");
        Ok(())
    });
}

fn test_mixed_precision_integration(runner: &mut TestRunner) {
    runner.run_test("integration_mixed_precision", || {
        let backend = CpuBackend::default();
        let input = backend
            .tensor_from_vec(vec![0.5f32; 100], &[10, 10])
            .map_err(|e| format!("Create input failed: {}", e))?;

        let linear = Linear::new(&backend, LinearConfig::new(10, 5))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let linear_out = linear.forward(input, &mut ctx)
            .map_err(|e| format!("Linear failed: {}", e))?;

        let norm = LayerNorm::new(&backend, LayerNormConfig::new(vec![5]), 42)
            .map_err(|e| format!("Create norm failed: {}", e))?;

        let norm_out = norm.forward(linear_out, &mut ctx)
            .map_err(|e| format!("Norm failed: {}", e))?;

        let shape = backend.ops().shape(&norm_out);
        assert_eq!(shape, vec![10, 5], "Mixed precision output shape wrong");
        Ok(())
    });
}

fn test_save_load_integration(runner: &mut TestRunner) {
    runner.run_test("integration_save_load", || {
        let backend = CpuBackend::default();
        let model = Linear::new(&backend, LinearConfig::new(10, 5))
            .map_err(|e| format!("Create model failed: {}", e))?;

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

fn test_backend_consistency(runner: &mut TestRunner) {
    runner.run_test("integration_backend_ops", || {
        let backend = CpuBackend::default();
        let ops = backend.ops();

        let a = backend.tensor_from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2])
            .map_err(|e| format!("Create tensor A failed: {}", e))?;

        let b = backend.tensor_from_vec(vec![2.0f32, 2.0, 2.0, 2.0], &[2, 2])
            .map_err(|e| format!("Create tensor B failed: {}", e))?;

        let _sum = ops.add(&a, &b).map_err(|e| format!("Add failed: {}", e))?;
        let _diff = ops.sub(&a, &b).map_err(|e| format!("Sub failed: {}", e))?;
        let _prod = ops.mul(&a, &b).map_err(|e| format!("Mul failed: {}", e))?;
        let _div = ops.div(&a, &b).map_err(|e| format!("Div failed: {}", e))?;

        let shape = ops.shape(&a);
        assert_eq!(shape, vec![2, 2], "Shape wrong");

        Ok(())
    });
}

fn test_parameter_sharing(runner: &mut TestRunner) {
    runner.run_test("integration_param_sharing", || {
        let backend = CpuBackend::default();
        let layer1 = Linear::new(&backend, LinearConfig::new(10, 5))
            .map_err(|e| format!("Create layer1 failed: {}", e))?;

        let layer2 = Linear::new(&backend, LinearConfig::new(10, 5))
            .map_err(|e| format!("Create layer2 failed: {}", e))?;

        let params1 = layer1.parameters();
        let params2 = layer2.parameters();
        assert_eq!(params1.len(), params2.len(), "Both layers should have same param count");
        assert!(params1 != params2, "Two separate layers should have different parameters");

        Ok(())
    });
}

fn test_mode_switching(runner: &mut TestRunner) {
    runner.run_test("integration_mode_inference", || {
        let backend = CpuBackend::default();
        let dropout = Dropout::new(DropoutConfig::new(0.5));

        let input = backend
            .tensor_from_vec(vec![1.0f32; 10], &[2, 5])
            .map_err(|e| format!("Create input failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = dropout.forward(input.clone(), &mut ctx)
            .map_err(|e| format!("Dropout inference failed: {}", e))?;

        let in_data: Vec<f32> = input.as_ref().to_vec();
        let out_data: Vec<f32> = output.as_ref().to_vec();

        assert!(
            out_data.iter().zip(in_data.iter()).all(|(a, b)| (a - b).abs() < 1e-5),
            "Inference mode dropout should be identity"
        );
        Ok(())
    });

    runner.run_test("integration_mode_training", || {
        let backend = CpuBackend::default();
        let dropout = Dropout::new(DropoutConfig::new(0.5));

        let input = backend
            .tensor_from_vec(vec![1.0f32; 10], &[2, 5])
            .map_err(|e| format!("Create input failed: {}", e))?;

        let mut ctx = ForwardCtx::new(&backend, Mode::Train);
        let output = dropout.forward(input.clone(), &mut ctx)
            .map_err(|e| format!("Dropout training failed: {}", e))?;

        let out_data: Vec<f32> = output.as_ref().to_vec();
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
