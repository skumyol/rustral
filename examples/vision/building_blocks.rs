//! Building Blocks: Understanding MNR's Core Abstractions
//!
//! Demonstrates the fundamental abstractions:
//! - Backend: Where computation happens (CPU/GPU)
//! - Tensor: Multi-dimensional arrays of numbers
//! - Module: The trait all layers implement
//! - ForwardCtx: Explicit context for computation
//!
//! Run with: `cargo run --example building_blocks`

use mnr_core::{Backend, ForwardCtx, Mode, Module};
use mnr_ndarray_backend::CpuBackend;
use mnr_nn::{Linear, LinearConfig};

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  MNR Building Blocks: Core Abstractions Explained        ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // ═══════════════════════════════════════════════════════════
    // PART 1: THE BACKEND
    // ═══════════════════════════════════════════════════════════
    println!("▶ PART 1: The Backend");
    println!("  The Backend is where all math happens. Think of it as the");
    println!("  'engine' that executes your neural network.\n");

    // Create a CPU backend
    let backend = CpuBackend::default();
    println!("  ✓ Created CpuBackend");

    // Get the operations interface
    let ops = backend.ops();
    println!("  ✓ Got TensorOps interface\n");

    // ═══════════════════════════════════════════════════════════
    // PART 2: TENSORS
    // ═══════════════════════════════════════════════════════════
    println!("▶ PART 2: Tensors");
    println!("  Tensors are multi-dimensional arrays of numbers. They are");
    println!("  the ONLY data type that flows through neural networks.\n");

    // Creating tensors
    println!("  Creating a 2D tensor (matrix):");
    let data = vec![
        1.0, 2.0, 3.0,  // Row 1
        4.0, 5.0, 6.0,  // Row 2
    ];
    let tensor_2d = backend.tensor_from_vec(data.clone(), &[2, 3]).unwrap();
    println!("    Data: {:?}", data);
    println!("    Shape: {:?}", ops.shape(&tensor_2d));
    println!("    Total elements: 6\n");

    // Tensor operations
    println!("  Tensor operations:");
    let zeros = ops.zeros(&[2, 2]).unwrap();
    println!("    Zeros [2,2]: created");

    // Create ones manually
    let ones = backend.tensor_from_vec(vec![1.0, 1.0, 1.0, 1.0], &[2, 2]).unwrap();
    println!("    Ones [2,2]: created manually\n");

    // Element-wise operations
    println!("  Element-wise operations:");
    let a = backend.tensor_from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let b = backend.tensor_from_vec(vec![10.0, 20.0, 30.0], &[3]).unwrap();
    let sum = ops.add(&a, &b).unwrap();
    println!("    [1, 2, 3] + [10, 20, 30] = {:?}", sum.values());

    // Multiplication
    let product = ops.mul(&a, &b).unwrap();
    println!("    [1, 2, 3] * [10, 20, 30] = {:?}\n", product.values());

    // Matrix multiplication
    println!("  Matrix multiplication:");
    let matrix_a = backend.tensor_from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let matrix_b = backend.tensor_from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
    let mat_product = ops.matmul(&matrix_a, &matrix_b).unwrap();
    println!("    Result shape: {:?}", ops.shape(&mat_product));
    println!("    Values: {:?}\n", mat_product.values());

    // Activation functions
    println!("  Activation functions:");
    let input = backend.tensor_from_vec(vec![-1.0, 0.0, 1.0, 2.0], &[4]).unwrap();
    let relu_out = ops.relu(&input).unwrap();
    println!("    ReLU([-1, 0, 1, 2]) = {:?}", relu_out.values());

    let sigmoid_out = ops.sigmoid(&input).unwrap();
    println!("    Sigmoid([-1, 0, 1, 2]) = {:?}", sigmoid_out.values());

    let tanh_out = ops.tanh(&input).unwrap();
    println!("    Tanh([-1, 0, 1, 2]) = {:?}\n", tanh_out.values());

    // ═══════════════════════════════════════════════════════════
    // PART 3: MODULES
    // ═══════════════════════════════════════════════════════════
    println!("▶ PART 3: Modules");
    println!("  A Module is anything that transforms input to output.");
    println!("  Linear, Conv2d, and even your entire model are Modules.\n");

    // Create a linear layer
    let linear = Linear::new(&backend, LinearConfig::new(3, 2))
        .expect("Failed to create linear layer");
    println!("  ✓ Created Linear layer (3 -> 2)");

    // Create input
    let input = backend.tensor_from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap();
    println!("  Input shape: [1, 3]\n");

    // ═══════════════════════════════════════════════════════════
    // PART 4: FORWARD CONTEXT
    // ═══════════════════════════════════════════════════════════
    println!("▶ PART 4: Forward Context");
    println!("  The ForwardCtx tells the system whether we're training");
    println!("  (needing gradients) or just doing inference.\n");

    // Inference mode
    let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
    println!("  Created ForwardCtx with Mode::Inference");

    let output = linear.forward(input.clone(), &mut ctx).unwrap();
    println!("  Output shape: {:?}\n", ops.shape(&output));

    // Training mode
    let mut train_ctx = ForwardCtx::new(&backend, Mode::Train);
    println!("  Created ForwardCtx with Mode::Train");

    let train_output = linear.forward(input, &mut train_ctx).unwrap();
    println!("  Output shape: {:?}\n", ops.shape(&train_output));

    // ═══════════════════════════════════════════════════════════
    // SUMMARY
    // ═══════════════════════════════════════════════════════════
    println!("▶ Summary");
    println!("  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐");
    println!("  │   Backend   │──▶│   Tensor    │──▶│   Module    │");
    println!("  │  (CPU/GPU)  │   │  (data)     │   │  (layer)    │");
    println!("  └─────────────┘   └─────────────┘   └─────────────┘");
    println!("         │                                    │");
    println!("         └────────────────────────────────────┘");
    println!("                    ForwardCtx");
    println!("                    (train/inference)\n");

    println!("  Key takeaways:");
    println!("    • Backend creates tensors and provides operations");
    println!("    • Tensors are the only data type in neural networks");
    println!("    • Modules transform tensors (layers, models)");
    println!("    • ForwardCtx controls gradient tracking");
}
