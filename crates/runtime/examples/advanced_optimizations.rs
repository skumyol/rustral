//! Comprehensive demonstration of Rustral's advanced optimization capabilities.
//!
//! This example shows all the optimization features implemented:
//! - Mixed precision training with capability detection
//! - Memory layout optimization (NHWC vs NCHW)
//! - Gradient checkpointing for memory efficiency
//! - Operation fusion for performance
//! - Flash Attention 2 support (trait-based)
//! - Quantization support (trait-based)
//!
//! All optimizations are backend-agnostic and use capability detection
//! for automatic adaptation across different backends.

use rustral_core::{Backend, ConvLayout, TrainingDtype};
use rustral_ndarray_backend::CpuBackend;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Rustral Advanced Optimization Capabilities Demo ===\n");

    let backend = CpuBackend::default();

    // 1. Mixed Precision Training
    println!("--- 1. Mixed Precision Training ---");
    demonstrate_mixed_precision(&backend)?;

    // 2. Memory Layout Optimization
    println!("\n--- 2. Memory Layout Optimization ---");
    demonstrate_memory_layout(&backend)?;

    // 3. Gradient Checkpointing
    println!("\n--- 3. Gradient Checkpointing ---");
    demonstrate_gradient_checkpointing()?;

    // 4. Operation Fusion
    println!("\n--- 4. Operation Fusion ---");
    demonstrate_fusion(&backend)?;

    // 5. Flash Attention Support
    println!("\n--- 5. Flash Attention Support ---");
    demonstrate_flash_attention(&backend)?;

    // 6. Quantization Support
    println!("\n--- 6. Quantization Support ---");
    demonstrate_quantization(&backend)?;

    println!("\n=== Summary ===");
    println!("All optimizations are backend-agnostic and use capability detection.");
    println!("Model code can automatically adapt to backend capabilities.");
    println!("This enables maximum performance across CPU, CUDA, Metal, and WGPU backends.");

    Ok(())
}

/// Demonstrate mixed precision training capability detection.
fn demonstrate_mixed_precision<B: Backend>(backend: &B) -> Result<(), Box<dyn std::error::Error>> {
    let capabilities = backend.capabilities();

    println!("Device: {:?}", backend.device());
    println!("Supports mixed precision: {}", capabilities.supports_mixed_precision);
    println!("Recommended training dtype: {:?}", capabilities.recommended_training_dtype);
    println!("Fast FP16 tensor cores: {}", capabilities.supports_fast_fp16_tensor_cores);

    let training_dtype = if capabilities.supports_mixed_precision {
        capabilities.recommended_training_dtype
    } else {
        TrainingDtype::F32
    };

    println!("Selected training dtype: {:?}", training_dtype);

    match training_dtype {
        TrainingDtype::F16 => println!("✓ Using FP16 for ~50% memory savings and 2-4x speedup"),
        TrainingDtype::Bf16 => println!("✓ Using BF16 for ~50% memory savings and 2-3x speedup"),
        TrainingDtype::F32 => println!("✓ Using F32 (baseline precision)"),
    }

    Ok(())
}

/// Demonstrate memory layout optimization.
fn demonstrate_memory_layout<B: Backend>(backend: &B) -> Result<(), Box<dyn std::error::Error>> {
    let capabilities = backend.capabilities();

    println!("Preferred conv layout: {:?}", capabilities.preferred_conv_layout);
    println!("Supports strided layouts: {}", capabilities.supports_strided_layouts);
    println!("Supports packed layouts: {}", capabilities.supports_packed_layouts);

    match capabilities.preferred_conv_layout {
        ConvLayout::NCHW => println!("✓ Using NCHW layout (optimal for CUDA/PyTorch)"),
        ConvLayout::NHWC => println!("✓ Using NHWC layout (optimal for Metal/TensorFlow)"),
    }

    if capabilities.supports_packed_layouts {
        println!("✓ Can use tensor core optimized packed layouts");
    }

    Ok(())
}

/// Demonstrate gradient checkpointing memory savings.
fn demonstrate_gradient_checkpointing() -> Result<(), Box<dyn std::error::Error>> {
    // Example: 24-layer transformer, 768 hidden size, 512 sequence length, batch size 8
    // Calculate memory savings manually (simplified version of MemoryStats)
    let num_layers = 24;
    let hidden_size = 768;
    let seq_length = 512;
    let batch_size = 8;
    let checkpoint_frequency = 2;
    let dtype_bytes = 4; // f32 = 4 bytes

    let activations_per_layer = 4; // Q, K, V, attention output
    let activation_size = hidden_size * seq_length * batch_size * dtype_bytes;

    let without_checkpointing =
        num_layers as f32 * activations_per_layer as f32 * activation_size as f32 / (1024.0 * 1024.0);

    let checkpointed_layers = num_layers / checkpoint_frequency;
    let non_checkpointed_layers = num_layers - checkpointed_layers;

    let with_checkpointing = (checkpointed_layers as f32 * activation_size as f32
        + non_checkpointed_layers as f32 * activations_per_layer as f32 * activation_size as f32)
        / (1024.0 * 1024.0);

    let saved = without_checkpointing - with_checkpointing;
    let reduction = (saved / without_checkpointing) * 100.0;
    let extra_overhead = checkpointed_layers as f32 / num_layers as f32;

    println!("Model configuration: 24 layers, 768 hidden, 512 seq_len, batch 8");
    println!("Memory without checkpointing: {:.2} MB", without_checkpointing);
    println!("Memory with checkpointing: {:.2} MB", with_checkpointing);
    println!("Memory saved: {:.2} MB ({:.1}%)", saved, reduction);
    println!("Extra compute overhead: {:.1}%", extra_overhead * 100.0);

    if reduction > 20.0 {
        println!("✓ Significant memory reduction achieved");
    }

    Ok(())
}

/// Demonstrate operation fusion support.
fn demonstrate_fusion<B: Backend>(backend: &B) -> Result<(), Box<dyn std::error::Error>> {
    let has_fusion = backend.fusion_ops().is_some();

    println!("Fusion operations support: {}", has_fusion);

    if has_fusion {
        println!("✓ Backend supports fused operations (Linear+Bias, Linear+Bias+ReLU, etc.)");
        println!("  - Reduces kernel launches");
        println!("  - Eliminates intermediate tensors");
        println!("  - Improves memory bandwidth utilization");
    } else {
        println!("✗ No fusion support (will use individual operations)");
    }

    Ok(())
}

/// Demonstrate Flash Attention support.
fn demonstrate_flash_attention<B: Backend>(backend: &B) -> Result<(), Box<dyn std::error::Error>> {
    let has_flash_attention = backend.attention_ops().is_some();

    println!("Flash Attention support: {}", has_flash_attention);

    if has_flash_attention {
        println!("✓ Backend supports Flash Attention 2");
        println!("  - O(seq_len) memory complexity vs O(seq_len^2) for standard attention");
        println!("  - Enables training long-context models");
        println!("  - Tiled computation with online softmax");
    } else {
        println!("✗ No Flash Attention support (will use standard attention)");
        println!("  - Standard attention has O(seq_len^2) memory complexity");
    }

    Ok(())
}

/// Demonstrate quantization support.
fn demonstrate_quantization<B: Backend>(backend: &B) -> Result<(), Box<dyn std::error::Error>> {
    let has_quantization = backend.quantization_ops().is_some();

    println!("Quantization support: {}", has_quantization);

    if has_quantization {
        println!("✓ Backend supports INT8 quantization");
        println!("  - 4x memory reduction for quantized weights");
        println!("  - Faster inference on hardware with INT8 support");
        println!("  - Symmetric quantization with scale factors");
    } else {
        println!("✗ No quantization support (will use FP32)");
        println!("  - Full precision weights and activations");
    }

    Ok(())
}
