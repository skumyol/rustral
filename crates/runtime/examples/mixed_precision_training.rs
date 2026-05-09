//! Mixed precision training example demonstrating backend-agnostic capability detection.
//!
//! This example shows how to:
//! - Detect if a backend supports mixed precision training
//! - Get the recommended training dtype for a backend
//! - Use capability detection to adapt training strategy
//!
//! The design is backend-agnostic: the same code works across all backends,
//! with automatic fallback when mixed precision is not supported.

use rustral_core::{Backend, TrainingDtype};
use rustral_ndarray_backend::CpuBackend;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Mixed Precision Training Capability Detection ===\n");

    // Test CPU backend (no mixed precision support)
    println!("--- CPU Backend (Reference) ---");
    let cpu_backend = CpuBackend::default();
    demonstrate_mixed_precision(&cpu_backend)?;

    println!("\n=== Summary ===");
    println!("Mixed precision training is detected automatically via backend capabilities.");
    println!("Training code can adapt dtype selection based on backend support.");
    println!("CPU backends fall back to F32, GPU backends use F16/BF16 for speed.");

    Ok(())
}

/// Demonstrate mixed precision capability detection for a backend.
fn demonstrate_mixed_precision<B: Backend>(backend: &B) -> Result<(), Box<dyn std::error::Error>> {
    let capabilities = backend.capabilities();

    println!("Device: {:?}", backend.device());
    println!("Supports mixed precision: {}", capabilities.supports_mixed_precision);
    println!("Recommended training dtype: {:?}", capabilities.recommended_training_dtype);
    println!("Fast FP16 tensor cores: {}", capabilities.supports_fast_fp16_tensor_cores);

    // Simulate training strategy selection based on capabilities
    let training_dtype = if capabilities.supports_mixed_precision {
        capabilities.recommended_training_dtype
    } else {
        TrainingDtype::F32 // Fallback to F32
    };

    println!("Selected training dtype: {:?}", training_dtype);

    // Show memory savings estimate
    match training_dtype {
        TrainingDtype::F16 => {
            println!("Memory savings: ~50% compared to F32");
            println!("Expected speedup: 2-4x on tensor cores");
        }
        TrainingDtype::Bf16 => {
            println!("Memory savings: ~50% compared to F32");
            println!("Expected speedup: 2-3x on tensor cores");
            println!("Better numerical stability than FP16");
        }
        TrainingDtype::F32 => {
            println!("Memory savings: None (baseline)");
            println!("Expected speedup: Baseline");
        }
    }

    // Demonstrate automatic fallback pattern
    println!("\nAutomatic fallback pattern:");
    if capabilities.supports_mixed_precision {
        println!("✓ Using mixed precision training");
        println!("✓ Forward pass in {:?}", training_dtype);
        println!("✓ Gradient accumulation in {:?}", training_dtype);
        println!("✓ Parameter updates in F32 (master weights)");
    } else {
        println!("✗ Mixed precision not supported");
        println!("✓ Falling back to full precision (F32)");
        println!("✓ All operations in F32");
    }

    Ok(())
}
