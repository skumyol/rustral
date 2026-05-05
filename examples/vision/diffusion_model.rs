//! Diffusion Model Example
//!
//! Demonstrates the concept of diffusion for image generation.
//!
//! Run with: `cargo run --example diffusion_model`

use mnr_core::{Backend, ForwardCtx, Mode, Module};
use mnr_ndarray_backend::CpuBackend;
use mnr_nn::{Linear, LinearConfig};

fn main() {
    println!("Diffusion Model Example");
    println!("=======================\n");

    let backend = CpuBackend::default();
    let ops = backend.ops();

    println!("Diffusion Process Concept:");
    println!("-------------------------");

    // Simulate adding noise to an image
    let image = backend.tensor_from_vec(vec![0.5f32; 16], &[4, 4]).unwrap();
    println!("Original image shape: {:?}", ops.shape(&image));

    // Add noise (simulate diffusion forward process)
    let noise = backend.tensor_from_vec(vec![0.1f32; 16], &[4, 4]).unwrap();
    let noisy_image = ops.add(&image, &noise).unwrap();
    println!("Noisy image (after adding noise)");

    // Learned denoising step
    let denoiser = Linear::new(&backend, LinearConfig::new(16, 16)).unwrap();
    let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
    let flattened = ops.reshape(&noisy_image, &[1, 16]).unwrap();
    let denoised = denoiser.forward(flattened, &mut ctx).unwrap();
    println!("Denoised output shape: {:?}", ops.shape(&denoised));

    println!("\nKey concept:");
    println!("  1. Forward: gradually add noise to image");
    println!("  2. Reverse: learn to predict and remove noise");
    println!("  3. Start from pure noise, denoise step by step");
    println!("  4. Final result: generated image");
}
