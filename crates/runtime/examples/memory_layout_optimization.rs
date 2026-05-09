//! Memory layout optimization example demonstrating backend-agnostic layout detection.
//!
//! This example shows how to:
//! - Detect the preferred convolution memory layout (NHWC vs NCHW)
//! - Check support for strided and packed memory layouts
//! - Use capability detection to adapt data layout strategy
//!
//! The design is backend-agnostic: the same code works across all backends,
//! with automatic adaptation based on backend preferences.

use rustral_core::{Backend, ConvLayout};
use rustral_ndarray_backend::CpuBackend;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Memory Layout Optimization Capability Detection ===\n");

    // Test CPU backend
    println!("--- CPU Backend (Reference) ---");
    let cpu_backend = CpuBackend::default();
    demonstrate_memory_layout(&cpu_backend)?;

    println!("\n=== Summary ===");
    println!("Memory layout optimization is detected automatically via backend capabilities.");
    println!("Model code can adapt data layout based on backend preferences.");
    println!("CUDA backends prefer NCHW, Metal backends prefer NHWC.");

    Ok(())
}

/// Demonstrate memory layout capability detection for a backend.
fn demonstrate_memory_layout<B: Backend>(backend: &B) -> Result<(), Box<dyn std::error::Error>> {
    let capabilities = backend.capabilities();

    println!("Device: {:?}", backend.device());
    println!("Preferred conv layout: {:?}", capabilities.preferred_conv_layout);
    println!("Supports strided layouts: {}", capabilities.supports_strided_layouts);
    println!("Supports packed layouts: {}", capabilities.supports_packed_layouts);
    println!("Prefers contiguous: {}", capabilities.prefers_contiguous);

    // Show layout selection strategy
    println!("\nLayout selection strategy:");
    match capabilities.preferred_conv_layout {
        ConvLayout::NCHW => {
            println!("✓ Using NCHW layout [batch, channels, height, width]");
            println!("  - Better for CUDA/PyTorch compatibility");
            println!("  - Efficient for channel-wise operations");
        }
        ConvLayout::NHWC => {
            println!("✓ Using NHWC layout [batch, height, width, channels]");
            println!("  - Better for Metal/TensorFlow compatibility");
            println!("  - More cache-friendly for spatial operations");
        }
    }

    // Show strided layout usage
    if capabilities.supports_strided_layouts {
        println!("\n✓ Strided layouts supported");
        println!("  - Can use views without copying");
        println!("  - Efficient for slicing and transpose operations");
    } else {
        println!("\n✗ Strided layouts not supported");
        println!("  - Will need to copy data for layout transformations");
    }

    // Show packed layout usage
    if capabilities.supports_packed_layouts {
        println!("\n✓ Packed layouts supported");
        println!("  - Can use tensor core optimized layouts");
        println!("  - Better performance for matrix multiplication");
    } else {
        println!("\n✗ Packed layouts not supported");
        println!("  - Using standard contiguous layouts");
    }

    // Show automatic adaptation pattern
    println!("\nAutomatic adaptation pattern:");
    println!("Model code checks backend.capabilities().preferred_conv_layout");
    println!("and adapts tensor layout accordingly:");
    println!("```rust");
    println!("let layout = backend.capabilities().preferred_conv_layout;");
    println!("match layout {{");
    println!("    ConvLayout::NCHW => tensor.reshape([batch, channels, h, w]),");
    println!("    ConvLayout::NHWC => tensor.reshape([batch, h, w, channels]),");
    println!("}}");
    println!("```");

    Ok(())
}
