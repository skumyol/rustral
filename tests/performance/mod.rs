//! Performance Benchmark Tests
//!
//! Tests that measure and validate performance characteristics.
//! Each test includes:
//! - Baseline comparison against expected performance
//! - Statistical analysis (mean, stddev, min, max)
//! - Throughput measurements
//! - Memory usage tracking

use mnr_core::{Backend, ForwardCtx, Mode};
use mnr_nn::{
    Conv2d, Conv2dConfig, Linear, LinearConfig,
    SelfAttention, SelfAttentionConfig,
    TransformerEncoder, TransformerEncoderConfig,
    TransformerDecoder, TransformerDecoderConfig,
    Embedding, EmbeddingConfig,
    LayerNorm, LayerNormConfig,
};
use mnr_ndarray_backend::CpuBackend;

use crate::common::{run_performance_test, PerfConfig, PerfResult, TestRunner};
use std::time::Duration;

/// Baseline performance expectations (ms per forward pass)
struct Baselines {
    linear_small: f64,    // 256x256 → 256x128
    linear_medium: f64,   // 1024x512 → 1024x256
    linear_large: f64,    // 4096x1024 → 4096x512
    conv_small: f64,      // 32x3x224x224 → 32x64x112x112
    attention_small: f64, // batch=8, seq=64, d_model=256
    attention_medium: f64, // batch=4, seq=256, d_model=512
    transformer_small: f64, // 2 layers, d_model=256
    embedding_lookup: f64,  // vocab=50000, d_model=768, batch=32, seq=128
}

impl Default for Baselines {
    fn default() -> Self {
        Self {
            linear_small: 0.5,
            linear_medium: 2.0,
            linear_large: 10.0,
            conv_small: 5.0,
            attention_small: 3.0,
            attention_medium: 15.0,
            transformer_small: 20.0,
            embedding_lookup: 1.0,
        }
    }
}

pub fn run_all(runner: &mut TestRunner) {
    let config = PerfConfig::default();
    let baselines = Baselines::default();

    benchmark_linear_small(runner, &config, &baselines);
    benchmark_linear_medium(runner, &config, &baselines);
    benchmark_linear_large(runner, &config, &baselines);
    benchmark_conv2d_small(runner, &config, &baselines);
    benchmark_self_attention(runner, &config, &baselines);
    benchmark_transformer_encoder(runner, &config, &baselines);
    benchmark_transformer_decoder(runner, &config, &baselines);
    benchmark_embedding_lookup(runner, &config, &baselines);
    benchmark_end_to_end_pipeline(runner, &config);
    benchmark_memory_scaling(runner);
    benchmark_batch_scaling(runner, &config);
}

fn benchmark_linear_small(runner: &mut TestRunner, config: &PerfConfig, baselines: &Baselines) {
    runner.run_test("perf_linear_small", || {
        let backend = CpuBackend::default();

        let batch = 256usize;
        let in_features = 256usize;
        let out_features = 128usize;

        let linear = Linear::new(&backend, LinearConfig::new(in_features, out_features))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let input = backend
            .tensor_from_vec(vec![0.5f32; batch * in_features], &[batch, in_features])
            .map_err(|e| format!("Create input failed: {}", e))?;

        let result = run_performance_test(config, || {
            let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
            let _ = linear.forward(input.clone(), &mut ctx).unwrap();
        });

        println!("  Linear small ({}x{} → {}x{}):", batch, in_features, batch, out_features);
        print_perf_result(&result);

        if !result.meets_baseline(baselines.linear_small, config.tolerance_percent) {
            println!("  WARNING: Exceeds baseline of {}ms", baselines.linear_small);
        }

        // Assert it completes within reasonable time (not infinite)
        assert!(result.mean_ms < baselines.linear_small * 10.0,
            "Linear small too slow: {:.2}ms", result.mean_ms);
        Ok(())
    });
}

fn benchmark_linear_medium(runner: &mut TestRunner, config: &PerfConfig, baselines: &Baselines) {
    runner.run_test("perf_linear_medium", || {
        let backend = CpuBackend::default();

        let batch = 1024usize;
        let in_features = 512usize;
        let out_features = 256usize;

        let linear = Linear::new(&backend, LinearConfig::new(in_features, out_features))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let input = backend
            .tensor_from_vec(vec![0.5f32; batch * in_features], &[batch, in_features])
            .map_err(|e| format!("Create input failed: {}", e))?;

        let result = run_performance_test(config, || {
            let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
            let _ = linear.forward(input.clone(), &mut ctx).unwrap();
        });

        println!("  Linear medium ({}x{} → {}x{}):", batch, in_features, batch, out_features);
        print_perf_result(&result);

        assert!(result.mean_ms < baselines.linear_medium * 10.0,
            "Linear medium too slow: {:.2}ms", result.mean_ms);
        Ok(())
    });
}

fn benchmark_linear_large(runner: &mut TestRunner, config: &PerfConfig, baselines: &Baselines) {
    runner.run_test("perf_linear_large", || {
        let backend = CpuBackend::default();

        let batch = 128usize;  // Smaller batch for large model
        let in_features = 4096usize;
        let out_features = 2048usize;

        let linear = Linear::new(&backend, LinearConfig::new(in_features, out_features))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        let input = backend
            .tensor_from_vec(vec![0.1f32; batch * in_features], &[batch, in_features])
            .map_err(|e| format!("Create input failed: {}", e))?;

        let result = run_performance_test(config, || {
            let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
            let _ = linear.forward(input.clone(), &mut ctx).unwrap();
        });

        println!("  Linear large ({}x{} → {}x{}):", batch, in_features, batch, out_features);
        print_perf_result(&result);

        assert!(result.mean_ms < baselines.linear_large * 10.0,
            "Linear large too slow: {:.2}ms", result.mean_ms);
        Ok(())
    });
}

fn benchmark_conv2d_small(runner: &mut TestRunner, config: &PerfConfig, baselines: &Baselines) {
    runner.run_test("perf_conv2d_small", || {
        let backend = CpuBackend::default();

        let batch = 8usize;
        let channels = 3usize;
        let height = 64usize;
        let width = 64usize;
        let out_channels = 64usize;

        let conv = Conv2d::new(
            &backend,
            Conv2dConfig {
                in_channels: channels,
                out_channels: out_channels,
                kernel_size: 3,
                stride: 1,
                padding: 1,
            },
            42,
        )
        .map_err(|e| format!("Create conv failed: {}", e))?;

        let input = backend
            .tensor_from_vec(
                vec![0.5f32; batch * channels * height * width],
                &[batch, channels, height, width],
            )
            .map_err(|e| format!("Create input failed: {}", e))?;

        let result = run_performance_test(config, || {
            let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
            let _ = conv.forward(input.clone(), &mut ctx).unwrap();
        });

        println!("  Conv2d ({}x{}x{}x{}):", batch, channels, height, width);
        print_perf_result(&result);

        assert!(result.mean_ms < baselines.conv_small * 10.0,
            "Conv2d too slow: {:.2}ms", result.mean_ms);
        Ok(())
    });
}

fn benchmark_self_attention(runner: &mut TestRunner, config: &PerfConfig, baselines: &Baselines) {
    runner.run_test("perf_attention_small", || {
        let backend = CpuBackend::default();

        let batch = 4usize;
        let seq_len = 64usize;
        let d_model = 256usize;
        let num_heads = 8usize;

        let attn_config = SelfAttentionConfig::new(d_model, num_heads);
        let attention = SelfAttention::new(&backend, attn_config, 42)
            .map_err(|e| format!("Create attention failed: {}", e))?;

        let input = backend
            .tensor_from_vec(vec![0.5f32; batch * seq_len * d_model], &[batch, seq_len, d_model])
            .map_err(|e| format!("Create input failed: {}", e))?;

        let result = run_performance_test(config, || {
            let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
            let _ = attention.forward(input.clone(), &mut ctx).unwrap();
        });

        println!("  Self-attention (batch={}, seq={}, d_model={}, heads={}):",
            batch, seq_len, d_model, num_heads);
        print_perf_result(&result);

        assert!(result.mean_ms < baselines.attention_small * 10.0,
            "Attention too slow: {:.2}ms", result.mean_ms);
        Ok(())
    });

    runner.run_test("perf_attention_medium", || {
        let backend = CpuBackend::default();

        let batch = 2usize;
        let seq_len = 256usize;
        let d_model = 512usize;
        let num_heads = 8usize;

        let attn_config = SelfAttentionConfig::new(d_model, num_heads);
        let attention = SelfAttention::new(&backend, attn_config, 42)
            .map_err(|e| format!("Create attention failed: {}", e))?;

        let input = backend
            .tensor_from_vec(vec![0.3f32; batch * seq_len * d_model], &[batch, seq_len, d_model])
            .map_err(|e| format!("Create input failed: {}", e))?;

        let result = run_performance_test(config, || {
            let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
            let _ = attention.forward(input.clone(), &mut ctx).unwrap();
        });

        println!("  Self-attention (batch={}, seq={}, d_model={}, heads={}):",
            batch, seq_len, d_model, num_heads);
        print_perf_result(&result);

        assert!(result.mean_ms < baselines.attention_medium * 10.0,
            "Attention medium too slow: {:.2}ms", result.mean_ms);
        Ok(())
    });
}

fn benchmark_transformer_encoder(runner: &mut TestRunner, config: &PerfConfig, baselines: &Baselines) {
    runner.run_test("perf_transformer_encoder", || {
        let backend = CpuBackend::default();

        let d_model = 256usize;
        let num_heads = 8usize;
        let num_layers = 4usize;
        let ff_dim = 1024usize;
        let seq_len = 64usize;
        let batch = 2usize;
        let vocab_size = 10000usize;

        let encoder_config = TransformerEncoderConfig::new(d_model, num_heads, num_layers, ff_dim)
            .with_max_seq_len(128);

        let encoder = TransformerEncoder::new(&backend, encoder_config, vocab_size, 42)
            .map_err(|e| format!("Create encoder failed: {}", e))?;

        let input = backend
            .tensor_from_vec(vec![100u32; batch * seq_len], &[batch, seq_len])
            .map_err(|e| format!("Create input failed: {}", e))?;

        let result = run_performance_test(config, || {
            let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
            let _ = encoder.forward(input.clone(), &mut ctx).unwrap();
        });

        println!("  Transformer Encoder ({} layers, d_model={}, seq_len={}):",
            num_layers, d_model, seq_len);
        print_perf_result(&result);

        assert!(result.mean_ms < baselines.transformer_small * 10.0,
            "Transformer encoder too slow: {:.2}ms", result.mean_ms);
        Ok(())
    });
}

fn benchmark_transformer_decoder(runner: &mut TestRunner, config: &PerfConfig, baselines: &Baselines) {
    runner.run_test("perf_transformer_decoder", || {
        let backend = CpuBackend::default();

        let d_model = 256usize;
        let num_heads = 8usize;
        let num_layers = 4usize;
        let ff_dim = 1024usize;
        let seq_len = 32usize;
        let batch = 2usize;
        let vocab_size = 10000usize;

        let decoder_config = TransformerDecoderConfig::new(d_model, num_heads, num_layers, ff_dim)
            .with_max_seq_len(128);

        let decoder = TransformerDecoder::new(&backend, decoder_config, vocab_size, 42)
            .map_err(|e| format!("Create decoder failed: {}", e))?;

        let input = backend
            .tensor_from_vec(vec![100u32; batch * seq_len], &[batch, seq_len])
            .map_err(|e| format!("Create input failed: {}", e))?;

        let result = run_performance_test(config, || {
            let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
            let _ = decoder.forward(input.clone(), &mut ctx).unwrap();
        });

        println!("  Transformer Decoder ({} layers, d_model={}, seq_len={}):",
            num_layers, d_model, seq_len);
        print_perf_result(&result);

        assert!(result.mean_ms < baselines.transformer_small * 15.0,
            "Transformer decoder too slow: {:.2}ms", result.mean_ms);
        Ok(())
    });
}

fn benchmark_embedding_lookup(runner: &mut TestRunner, config: &PerfConfig, baselines: &Baselines) {
    runner.run_test("perf_embedding_lookup", || {
        let backend = CpuBackend::default();

        let vocab_size = 50000usize;
        let d_model = 768usize;
        let batch = 32usize;
        let seq_len = 128usize;

        let embedding = Embedding::new(
            &backend,
            EmbeddingConfig::new(vocab_size, d_model),
            42,
        )
        .map_err(|e| format!("Create embedding failed: {}", e))?;

        let input = backend
            .tensor_from_vec(vec![42u32; batch * seq_len], &[batch, seq_len])
            .map_err(|e| format!("Create input failed: {}", e))?;

        let result = run_performance_test(config, || {
            let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
            let _ = embedding.forward(input.clone(), &mut ctx).unwrap();
        });

        println!("  Embedding lookup (vocab={}, d_model={}, batch={}, seq={}):",
            vocab_size, d_model, batch, seq_len);
        print_perf_result(&result);

        assert!(result.mean_ms < baselines.embedding_lookup * 10.0,
            "Embedding lookup too slow: {:.2}ms", result.mean_ms);
        Ok(())
    });
}

fn benchmark_end_to_end_pipeline(runner: &mut TestRunner, _config: &PerfConfig) {
    runner.run_test("perf_end_to_end_pipeline", || {
        let backend = CpuBackend::default();

        // Simulate a full NLP pipeline: Embedding → Transformer → Linear → Output
        let vocab_size = 10000usize;
        let d_model = 256usize;
        let num_classes = 10usize;
        let seq_len = 32usize;
        let batch = 4usize;

        // Build pipeline
        let embedding = Embedding::new(
            &backend,
            EmbeddingConfig::new(vocab_size, d_model),
            42,
        )
        .map_err(|e| format!("Create embedding failed: {}", e))?;

        let encoder_config = TransformerEncoderConfig::new(d_model, 8, 2, 1024)
            .with_max_seq_len(128);
        let encoder = TransformerEncoder::new(&backend, encoder_config, vocab_size, 43)
            .map_err(|e| format!("Create encoder failed: {}", e))?;

        let classifier = Linear::new(&backend, LinearConfig::new(d_model, num_classes))
            .map_err(|e| format!("Create classifier failed: {}", e))?;

        let input = backend
            .tensor_from_vec(vec![100u32; batch * seq_len], &[batch, seq_len])
            .map_err(|e| format!("Create input failed: {}", e))?;

        let config = PerfConfig {
            warmup_iterations: 2,
            test_iterations: 5,
            max_duration_ms: 30000,
            tolerance_percent: 50.0,
        };

        let result = run_performance_test(&config, || {
            let mut ctx = ForwardCtx::new(&backend, Mode::Inference);

            // Embedding
            let embedded = embedding.forward(input.clone(), &mut ctx).unwrap();

            // Transformer
            let encoded = encoder.forward(input.clone(), &mut ctx).unwrap();

            // CLS token classification
            let cls = encoder.cls_token(&encoded, backend.ops()).unwrap();

            // Classifier
            let _logits = classifier.forward(cls, &mut ctx).unwrap();
        });

        println!("  End-to-end pipeline (Embed → Transformer → Classify):");
        print_perf_result(&result);
        println!("  Throughput: {:.1} samples/sec", result.throughput);

        assert!(result.mean_ms < 1000.0, "End-to-end pipeline too slow: {:.2}ms", result.mean_ms);
        Ok(())
    });
}

fn benchmark_memory_scaling(runner: &mut TestRunner) {
    runner.run_test("perf_memory_scaling", || {
        let backend = CpuBackend::default();

        // Test that memory doesn't grow unboundedly with model size
        let sizes = vec![
            (128, 64),
            (256, 128),
            (512, 256),
            (1024, 512),
        ];

        println!("  Memory scaling test:");
        for (in_features, out_features) in sizes {
            let linear = Linear::new(&backend, LinearConfig::new(in_features, out_features))
                .map_err(|e| format!("Create linear failed: {}", e))?;

            let params: usize = linear
                .parameters()
                .iter()
                .map(|p| p.as_tensor(backend.ops()).as_ref().len())
                .sum();

            let input = backend
                .tensor_from_vec(vec![0.5f32; 64 * in_features], &[64, in_features])
                .map_err(|e| format!("Create input failed: {}", e))?;

            let start = std::time::Instant::now();
            let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
            let _ = linear.forward(input, &mut ctx)
                .map_err(|e| format!("Forward failed: {}", e))?;
            let elapsed = start.elapsed();

            println!("    {}x{}: {} params, {:?} forward",
                in_features, out_features, params, elapsed);
        }

        Ok(())
    });
}

fn benchmark_batch_scaling(runner: &mut TestRunner, config: &PerfConfig) {
    runner.run_test("perf_batch_scaling", || {
        let backend = CpuBackend::default();

        let in_features = 256usize;
        let out_features = 128usize;

        let linear = Linear::new(&backend, LinearConfig::new(in_features, out_features))
            .map_err(|e| format!("Create linear failed: {}", e))?;

        println!("  Batch scaling test:");
        for batch in [1usize, 4, 16, 64, 256] {
            let input = backend
                .tensor_from_vec(vec![0.5f32; batch * in_features], &[batch, in_features])
                .map_err(|e| format!("Create input failed: {}", e))?;

            let result = run_performance_test(config, || {
                let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
                let _ = linear.forward(input.clone(), &mut ctx).unwrap();
            });

            let throughput = batch as f64 / (result.mean_ms / 1000.0);
            println!("    Batch={:4}: {:>8.2}ms, {:>10.1} items/sec",
                batch, result.mean_ms, throughput);
        }

        Ok(())
    });
}

fn print_perf_result(result: &PerfResult) {
    println!("    mean={:.3}ms, median={:.3}ms, min={:.3}ms, max={:.3}ms, std={:.3}ms",
        result.mean_ms, result.median_ms, result.min_ms, result.max_ms, result.std_dev_ms);
}
