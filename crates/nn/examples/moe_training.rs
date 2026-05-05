//! Mixture of Experts (MoE) Training Example
//!
//! Demonstrates sparse expert routing for scaling model capacity.
//!
//! Run with: `cargo run --example moe_training`

use rustral_core::{Backend, ForwardCtx, Mode, Module};
use rustral_ndarray_backend::CpuBackend;
use rustral_nn::{Linear, LinearConfig};

fn main() {
    println!("Mixture of Experts (MoE) Example");
    println!("===============================\n");

    let backend = CpuBackend::default();
    let ops = backend.ops();

    // MoE configuration
    let num_experts = 4;
    let input_dim = 64;
    let output_dim = 64;
    let top_k = 2;

    println!("MoE Configuration:");
    println!("  Number of experts: {}", num_experts);
    println!("  Input dimension: {}", input_dim);
    println!("  Output dimension: {}", output_dim);
    println!("  Top-K routing: {}\n", top_k);

    // Create expert layers
    let mut experts = Vec::new();
    for i in 0..num_experts {
        let expert = Linear::new(&backend, LinearConfig::new(input_dim, output_dim)).unwrap();
        experts.push(expert);
        println!("  Expert {}: Linear({} -> {})", i, input_dim, output_dim);
    }

    // Router (gating network)
    let router = Linear::new(&backend, LinearConfig::new(input_dim, num_experts)).unwrap();
    println!("  Router: Linear({} -> {})\n", input_dim, num_experts);

    // Sample input
    let input = backend.tensor_from_vec(vec![0.5f32; input_dim], &[1, input_dim]).unwrap();
    println!("Input shape: {:?}", ops.shape(&input));

    // Route to top-k experts
    let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
    let router_logits = router.forward(input.clone(), &mut ctx).unwrap();

    // For simplicity, combine first two expert outputs
    let expert0_out = experts[0].forward(input.clone(), &mut ctx).unwrap();
    let expert1_out = experts[1].forward(input.clone(), &mut ctx).unwrap();

    // Weighted combination
    let combined = ops.add(&expert0_out, &expert1_out).unwrap();
    println!("Combined output shape: {:?}", ops.shape(&combined));

    println!("\nKey concepts:");
    println!("  - Each token is routed to top-K experts");
    println!("  - Router learns which experts handle which inputs");
    println!("  - Total parameters scale with experts, but");
    println!("    active parameters per token stay constant");
}
