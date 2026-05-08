//! Backend-Agnostic Model Demonstration
//!
//! This example demonstrates Rustral's plug-and-play backend architecture.
//! The same model definition works seamlessly across different backends (CPU, CUDA, Metal)
//! with automatic fallback for unsupported features like operation fusion.

use rustral_core::{Backend, ForwardCtx, Mode, Module, Result};
use rustral_ndarray_backend::CpuBackend;
use rustral_nn::{Linear, LinearConfig, LinearReLU, LinearGELU};

fn main() -> Result<()> {
    println!("=== Backend-Agnostic Model Demonstration ===\n");

    // Define a simple model using fused layers
    // This model definition is completely backend-agnostic
    fn create_model<B: Backend>(backend: &B) -> Result<(LinearReLU<B>, LinearGELU<B>, Linear<B>)> {
        let config_relu = LinearConfig {
            in_dim: 128,
            out_dim: 256,
            bias: true,
        };
        
        let config_gelu = LinearConfig {
            in_dim: 256,
            out_dim: 128,
            bias: true,
        };
        
        let config_linear = LinearConfig {
            in_dim: 128,
            out_dim: 64,
            bias: true,
        };

        let layer_relu = LinearReLU::new(backend, config_relu)?;
        let layer_gelu = LinearGELU::new(backend, config_gelu)?;
        let layer_linear = Linear::new(backend, config_linear)?;

        Ok((layer_relu, layer_gelu, layer_linear))
    }

    // Test with CPU backend (no fusion support)
    println!("1. Testing with CPU Backend (no fusion support):");
    println!("   The model will automatically fall back to unfused operations\n");
    
    let cpu_backend = CpuBackend::default();
    let (cpu_relu, cpu_gelu, cpu_linear) = create_model(&cpu_backend)?;
    
    let mut cpu_ctx = ForwardCtx::new(&cpu_backend, Mode::Inference);
    let cpu_input = cpu_backend.tensor_from_vec(vec![1.0f32; 1 * 128], &[1, 128])?;
    
    // Check if CPU backend supports fusion
    let has_fusion = cpu_backend.fusion_ops().is_some();
    println!("   CPU backend fusion support: {}", has_fusion);
    
    let cpu_relu_out = cpu_relu.forward(cpu_input.clone(), &mut cpu_ctx)?;
    println!("   LinearReLU output shape: {:?}", cpu_backend.ops().shape(&cpu_relu_out));
    
    let cpu_gelu_out = cpu_gelu.forward(cpu_relu_out.clone(), &mut cpu_ctx)?;
    println!("   LinearGELU output shape: {:?}", cpu_backend.ops().shape(&cpu_gelu_out));
    
    let cpu_linear_out = cpu_linear.forward(cpu_gelu_out, &mut cpu_ctx)?;
    println!("   Linear output shape: {:?}", cpu_backend.ops().shape(&cpu_linear_out));
    println!("   ✓ CPU backend execution successful (automatic fallback)\n");

    // Test with Candle backend (has fusion support)
    #[cfg(feature = "cuda")]
    {
        println!("2. Testing with Candle CUDA Backend (has fusion support):");
        println!("   The model will use fused operations when available\n");
        
        use rustral_candle_backend::CandleBackend;
        
        let cuda_backend = CandleBackend::cuda(0).unwrap_or_else(|_| CandleBackend::cpu());
        let (cuda_relu, cuda_gelu, cuda_linear) = create_model(&cuda_backend)?;
        
        let mut cuda_ctx = ForwardCtx::new(&cuda_backend, Mode::Inference);
        let cuda_input = cuda_backend.tensor_from_vec(vec![1.0f32; 1 * 128], &[1, 128])?;
        
        // Check if CUDA backend supports fusion
        let has_fusion = cuda_backend.fusion_ops().is_some();
        println!("   CUDA backend fusion support: {}", has_fusion);
        
        let cuda_relu_out = cuda_relu.forward(cuda_input.clone(), &mut cuda_ctx)?;
        println!("   LinearReLU output shape: {:?}", cuda_backend.ops().shape(&cuda_relu_out));
        
        let cuda_gelu_out = cuda_gelu.forward(cuda_relu_out.clone(), &mut cuda_ctx)?;
        println!("   LinearGELU output shape: {:?}", cuda_backend.ops().shape(&cuda_gelu_out));
        
        let cuda_linear_out = cuda_linear.forward(cuda_gelu_out, &mut cuda_ctx)?;
        println!("   Linear output shape: {:?}", cuda_backend.ops().shape(&cuda_linear_out));
        println!("   ✓ CUDA backend execution successful (fusion used if available)\n");
    }

    #[cfg(feature = "metal")]
    {
        println!("3. Testing with Candle Metal Backend (Apple Silicon):");
        println!("   The model will use fused operations when available\n");
        
        use rustral_candle_backend::CandleBackend;
        
        let metal_backend = CandleBackend::metal(0).unwrap_or_else(|_| CandleBackend::cpu());
        let (metal_relu, metal_gelu, metal_linear) = create_model(&metal_backend)?;
        
        let mut metal_ctx = ForwardCtx::new(&metal_backend, Mode::Inference);
        let metal_input = metal_backend.tensor_from_vec(vec![1.0f32; 1 * 128], &[1, 128])?;
        
        // Check if Metal backend supports fusion
        let has_fusion = metal_backend.fusion_ops().is_some();
        println!("   Metal backend fusion support: {}", has_fusion);
        
        let metal_relu_out = metal_relu.forward(metal_input.clone(), &mut metal_ctx)?;
        println!("   LinearReLU output shape: {:?}", metal_backend.ops().shape(&metal_relu_out));
        
        let metal_gelu_out = metal_gelu.forward(metal_relu_out.clone(), &mut metal_ctx)?;
        println!("   LinearGELU output shape: {:?}", metal_backend.ops().shape(&metal_gelu_out));
        
        let metal_linear_out = metal_linear.forward(metal_gelu_out, &mut metal_ctx)?;
        println!("   Linear output shape: {:?}", metal_backend.ops().shape(&metal_linear_out));
        println!("   ✓ Metal backend execution successful (fusion used if available)\n");
    }

    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    {
        println!("Note: To test with CUDA or Metal backends, run with:");
        println!("  cargo run -p rustral-runtime --example backend_agnostic_fusion --features cuda");
        println!("  cargo run -p rustral-runtime --example backend_agnostic_fusion --features metal\n");
    }

    println!("=== Key Design Principles ===");
    println!("1. Single Model Definition: The same model code works across all backends");
    println!("2. Automatic Fallback: Layers automatically use fusion when available, fall back otherwise");
    println!("3. No Backend-Specific Code: Model implementations don't need to know the backend type");
    println!("4. Capability Detection: Backends report their capabilities for runtime adaptation");
    println!("5. Plug-and-Play: Switch backends by changing one line of initialization code");

    Ok(())
}
