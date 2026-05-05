//! Integration tests for distributed training.
//!
//! Tests data parallelism, tensor parallelism, ZeRO, and checkpointing.

use std::fs;
use std::thread;

use mnr_core::{Backend, ForwardCtx, Mode, TensorOps};
use mnr_distributed::{
    DataParallelTrainer, DistributedCheckpointManager, ParallelStyle, ProcessGroup, TensorParallelLinear,
    ZeRoMemoryStats, ZeroOptimizer,
};
use mnr_ndarray_backend::CpuBackend;
use mnr_optim::{Adam, Gradient, Optimizer};

/// Test single-process data parallel trainer.
#[test]
fn test_data_parallel_single_process() {
    let backend = CpuBackend::default();
    let pg = ProcessGroup::new_single_process();
    let optimizer = Adam::<CpuBackend>::new(0.001);

    let mut trainer = DataParallelTrainer::new(pg, optimizer);

    // Create some test parameters
    let param = backend.normal_parameter("test", &[10], 42, 0.1).unwrap();
    let mut params = vec![param.clone()];

    // Dummy loss function
    let backend_ref = backend.clone();
    let mut loss_fn = |_sample: &&[f32], _ctx: &mut ForwardCtx<CpuBackend>| {
        let grad_tensor = backend_ref.ops().zeros(&[10]).unwrap();
        let gradients = vec![Gradient { param_id: param.id(), tensor: grad_tensor }];
        Ok((1.0f32, gradients))
    };

    // Run a training step
    let samples: Vec<&[f32]> = vec![&[1.0; 10], &[2.0; 10], &[3.0; 10]];
    let mut ctx = ForwardCtx::new(&backend, Mode::Train);

    let loss = trainer.step(&mut params, &samples, &mut loss_fn, &mut ctx).unwrap();

    // Loss should be averaged across the single process
    assert!(loss >= 0.0);
}

/// Test multi-threaded data parallelism.
#[test]
fn test_data_parallel_threaded() {
    let backend = CpuBackend::default();
    let world_size = 4;

    let handles: Vec<_> = (0..world_size)
        .map(|rank| {
            let pg = ProcessGroup::new_threaded(world_size, rank).unwrap();
            let optimizer = Adam::<CpuBackend>::new(0.001);
            let mut trainer = DataParallelTrainer::new(pg, optimizer);
            let backend_thread = backend.clone();

            thread::spawn(move || {
                let param = backend_thread
                    .normal_parameter(&format!("rank_{}", rank), &[10], rank as u64, 0.1)
                    .unwrap();
                let mut params = vec![param.clone()];

                let samples: Vec<&[f32]> = vec![&[1.0; 10], &[2.0; 10]];
                let mut ctx = ForwardCtx::new(&backend_thread, Mode::Train);

                let mut loss_fn = |_sample: &&[f32], _ctx: &mut ForwardCtx<CpuBackend>| {
                    let grad_tensor = backend_thread.ops().zeros(&[10]).unwrap();
                    let gradients = vec![Gradient { param_id: param.id(), tensor: grad_tensor }];
                    Ok((1.0f32, gradients))
                };

                trainer.step(&mut params, &samples, &mut loss_fn, &mut ctx).unwrap();
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
}

/// Test ZeRO memory statistics.
#[test]
fn test_zero_memory_stats() {
    // 1B parameters, f32 (4 bytes), 8 GPUs
    let stats = ZeRoMemoryStats::zero1(1_000_000_000, 4, 8);

    // Should show significant memory savings
    assert!(stats.memory_saved_percent > 30.0);
    assert_eq!(stats.total_params_mb, (1_000_000_000i64 * 4) as f32 / (1024.0 * 1024.0));

    // ZeRO-1: optimizer states are sharded 8 ways
    let expected_optimizer_mb = stats.total_params_mb * 2.0 / 8.0;
    assert!((stats.local_optimizer_states_mb - expected_optimizer_mb).abs() < 0.1);
}

/// Test ZeRO-2 memory statistics.
#[test]
fn test_zero2_memory_stats() {
    let stats = ZeRoMemoryStats::zero2(1_000_000_000, 4, 8);

    // ZeRO-2 should save more memory than ZeRO-1
    let stats1 = ZeRoMemoryStats::zero1(1_000_000_000, 4, 8);
    assert!(stats.memory_saved_percent > stats1.memory_saved_percent);

    // Gradients are sharded in ZeRO-2
    assert!(stats.local_gradients_mb < stats1.local_gradients_mb);
}

/// Test process group creation.
#[test]
fn test_process_group_creation() {
    // Single process
    let pg_single = ProcessGroup::new_single_process();
    assert_eq!(pg_single.rank(), 0);
    assert_eq!(pg_single.world_size(), 1);
    assert!(pg_single.is_primary());

    // Threaded
    let pg_threaded = ProcessGroup::new_threaded(8, 3).unwrap();
    assert_eq!(pg_threaded.rank(), 3);
    assert_eq!(pg_threaded.world_size(), 8);
    assert!(!pg_threaded.is_primary());

    // Invalid rank
    assert!(ProcessGroup::new_threaded(8, 10).is_err());
}

/// Test tensor parallel linear layer creation.
#[test]
fn test_tensor_parallel_linear_creation() {
    let backend = CpuBackend::default();
    let pg = ProcessGroup::new_single_process();

    // Column parallel
    let col_parallel = TensorParallelLinear::column_parallel(64, 128, &pg, &backend).unwrap();
    assert!(matches!(col_parallel.parallel_style(), ParallelStyle::ColumnParallel));

    // Row parallel
    let row_parallel = TensorParallelLinear::row_parallel(128, 64, &pg, &backend).unwrap();
    assert!(matches!(row_parallel.parallel_style(), ParallelStyle::RowParallel));

    // Invalid dimensions (not divisible by world_size)
    let pg_multi = ProcessGroup::new_threaded(8, 0).unwrap();
    assert!(TensorParallelLinear::column_parallel(64, 127, &pg_multi, &backend).is_err());
}

/// Test distributed checkpoint manager.
#[test]
fn test_distributed_checkpoint_manager() {
    let temp_dir = tempfile::tempdir().unwrap();
    let pg = ProcessGroup::new_single_process();

    let _manager = DistributedCheckpointManager::new(pg, temp_dir.path(), 3).unwrap();

    // Check directory was created
    assert!(temp_dir.path().exists());
}

/// Test ZeRO optimizer wrapping.
#[test]
fn test_zero_optimizer_creation() {
    let _backend = CpuBackend::default();
    let pg = ProcessGroup::new_single_process();
    let adam = Adam::<CpuBackend>::new(0.001);

    let zero = ZeroOptimizer::new(adam, pg, 100);

    assert_eq!(zero.total_params(), 100);
    assert_eq!(zero.process_group().world_size(), 1);
}

/// Test all-reduce operation.
#[test]
fn test_all_reduce_operation() {
    let pg = ProcessGroup::new_single_process();

    let mut data = vec![1.0f32, 2.0, 3.0];
    pg.all_reduce_sum("test", &mut data).unwrap();

    // With single process, data should be unchanged
    assert_eq!(data, vec![1.0, 2.0, 3.0]);
}

/// End-to-end distributed training simulation.
#[test]
fn test_e2e_distributed_training() {
    let backend = CpuBackend::default();
    let pg = ProcessGroup::new_single_process();

    // Create a simple model
    let param = backend.normal_parameter("weight", &[10], 42, 0.1).unwrap();
    let bias = backend.normal_parameter("bias", &[10], 0, 0.0).unwrap();
    let mut params = vec![param, bias];

    // Create optimizer and trainer
    let optimizer = Adam::<CpuBackend>::new(0.01);
    let mut trainer = DataParallelTrainer::new(pg, optimizer);

    // Training data
    let samples: Vec<Vec<f32>> = (0..100).map(|i| vec![i as f32; 10]).collect();
    let samples_refs: Vec<&[f32]> = samples.iter().map(|s| s.as_slice()).collect();

    // Training loop
    let param0_id = params[0].id();
    let param1_id = params[1].id();
    for _epoch in 0..3 {
        let mut ctx = ForwardCtx::new(&backend, Mode::Train);

        // Pre-compute parameter tensors for the closure
        let weight_tensor = params[0].tensor().clone();
        let bias_tensor = params[1].tensor().clone();
        let backend_ref = backend.clone();

        let mut loss_fn = |_sample: &&[f32], _ctx: &mut ForwardCtx<CpuBackend>| {
            // Forward: y = x * w + b (simplified)
            let output = backend_ref.ops().add(&weight_tensor, &bias_tensor).unwrap();
            let target = backend_ref.ops().zeros(&[10]).unwrap();
            let diff = backend_ref.ops().sub(&output, &target).unwrap();
            let squared = backend_ref.ops().mul(&diff, &diff).unwrap();

            // Compute mean
            let data: Vec<f32> = squared.as_ref().to_vec();
            let loss: f32 = data.iter().sum::<f32>() / data.len() as f32;

            // Simple gradients (just zeros for this test)
            let grad0 = Gradient { param_id: param0_id, tensor: backend_ref.ops().zeros(&[10]).unwrap() };
            let grad1 = Gradient { param_id: param1_id, tensor: backend_ref.ops().zeros(&[10]).unwrap() };

            Ok((loss, vec![grad0, grad1]))
        };

        let loss = trainer.step(&mut params, &samples_refs, &mut loss_fn, &mut ctx).unwrap();
        assert!(loss >= 0.0, "Loss should be non-negative");
    }
}

/// Test gradient accumulation.
#[test]
fn test_gradient_accumulation() {
    use mnr_distributed::GradientAccumulator;

    let backend = CpuBackend::default();
    let pg = ProcessGroup::new_single_process();

    let mut acc = GradientAccumulator::new(pg);

    // Create some dummy gradients
    let param_id = mnr_core::ParameterId::fresh();
    let grad_tensor = backend.ops().tensor_from_vec(vec![1.0f32; 10], &[10]).unwrap();
    let gradients = vec![Gradient { param_id, tensor: grad_tensor }];

    // Accumulate gradients twice
    acc.accumulate(&gradients, backend.ops()).unwrap();
    acc.accumulate(&gradients, backend.ops()).unwrap();

    assert_eq!(acc.steps(), 2);
}

/// Test checkpoint saving and loading.
#[test]
fn test_checkpoint_save_load() {
    let temp_dir = tempfile::tempdir().unwrap();
    let backend = CpuBackend::default();
    let pg = ProcessGroup::new_single_process();

    let mut manager = DistributedCheckpointManager::new(pg.clone(), temp_dir.path(), 3).unwrap();

    // Create test parameters
    let param = backend.normal_parameter("test_param", &[10], 42, 0.1).unwrap();
    let params = vec![("test_param".to_string(), &param)];

    // Save checkpoint
    manager.save(0, 100, &params, None).unwrap();

    // Verify files exist
    let epoch_dir = temp_dir.path().join("epoch_0");
    assert!(epoch_dir.exists());
    assert!(epoch_dir.join("rank_0.safetensors").exists());
    assert!(epoch_dir.join("metadata.json").exists());

    // Load checkpoint
    let mut loaded_params = vec![("test_param".to_string(), param.clone())];
    let (step, opt_ckpt) = manager.load(0, &mut loaded_params).unwrap();

    assert_eq!(step, 100);
    assert!(opt_ckpt.is_none());
}

/// Test list checkpoints.
#[test]
fn test_list_checkpoints() {
    let temp_dir = tempfile::tempdir().unwrap();
    let pg = ProcessGroup::new_single_process();

    let manager = DistributedCheckpointManager::new(pg, temp_dir.path(), 5).unwrap();

    // Create fake checkpoint directories
    fs::create_dir(temp_dir.path().join("epoch_0")).unwrap();
    fs::create_dir(temp_dir.path().join("epoch_5")).unwrap();
    fs::create_dir(temp_dir.path().join("epoch_10")).unwrap();

    let checkpoints = manager.list_checkpoints().unwrap();
    assert_eq!(checkpoints, vec![0, 5, 10]);

    let latest = manager.latest_checkpoint().unwrap();
    assert_eq!(latest, Some(10));
}

/// Test async checkpoint writer.
#[test]
fn test_async_checkpoint_writer() {
    use mnr_distributed::AsyncCheckpointWriter;

    let temp_dir = tempfile::tempdir().unwrap();
    let writer = AsyncCheckpointWriter::new();

    let path = temp_dir.path().join("test.ckpt");
    let data = vec![1, 2, 3, 4, 5];

    writer.write(path.clone(), data).unwrap();

    // Give the background thread time to write
    std::thread::sleep(std::time::Duration::from_millis(100));

    // Note: The file may or may not exist depending on timing
    // This is mainly testing that the writer doesn't panic
}

/// Benchmark memory savings with different ZeRO configurations.
#[test]
fn test_memory_savings_comparison() {
    // Simulate a 7B parameter model (like LLaMA-7B)
    let params = 7_000_000_000usize;
    let bytes_per_param = 4; // f32

    // No ZeRO (baseline)
    let baseline_mb = (params * bytes_per_param) as f32 / (1024.0 * 1024.0);

    // ZeRO-1: optimizer states sharded across 8 GPUs
    let zero1 = ZeRoMemoryStats::zero1(params, bytes_per_param, 8);

    // ZeRO-2: optimizer states + gradients sharded
    let zero2 = ZeRoMemoryStats::zero2(params, bytes_per_param, 8);

    // Verify memory savings increase with more aggressive sharding
    assert!(
        zero2.memory_saved_percent > zero1.memory_saved_percent,
        "ZeRO-2 should save more memory than ZeRO-1"
    );

    // Verify we actually save memory
    assert!(zero1.memory_saved_percent > 0.0, "ZeRO should save memory");
    assert!(zero2.memory_saved_percent > 0.0, "ZeRO should save memory");

    println!("Baseline: {:.2} MB per GPU", baseline_mb);
    println!("ZeRO-1: saves {:.1}%", zero1.memory_saved_percent);
    println!("ZeRO-2: saves {:.1}%", zero2.memory_saved_percent);
}
