// WGSL Compute Shaders for MNR GPU Backend
// These shaders provide GPU-accelerated tensor operations

// ============================================================================
// Element-wise Operations
// ============================================================================

@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// Element-wise addition: output[i] = a[i] + b[i]
@compute @workgroup_size(256)
fn add(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output)) {
        return;
    }
    output[idx] = input_a[idx] + input_b[idx];
}

// Element-wise multiplication: output[i] = a[i] * b[i]
@compute @workgroup_size(256)
fn mul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output)) {
        return;
    }
    output[idx] = input_a[idx] * input_b[idx];
}

// Element-wise subtraction: output[i] = a[i] - b[i]
@compute @workgroup_size(256)
fn sub(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output)) {
        return;
    }
    output[idx] = input_a[idx] - input_b[idx];
}

// Element-wise division: output[i] = a[i] / b[i]
@compute @workgroup_size(256)
fn div(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output)) {
        return;
    }
    output[idx] = input_a[idx] / input_b[idx];
}

// Element-wise maximum: output[i] = max(a[i], b[i])
@compute @workgroup_size(256)
fn maximum(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output)) {
        return;
    }
    output[idx] = max(input_a[idx], input_b[idx]);
}

// ============================================================================
// Unary Operations
// ============================================================================

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_unary: array<f32>;

// ReLU: output[i] = max(0, input[i])
@compute @workgroup_size(256)
fn relu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output_unary)) {
        return;
    }
    let x = input[idx];
    output_unary[idx] = select(0.0, x, x > 0.0);
}

// Sigmoid: output[i] = 1 / (1 + exp(-x))
@compute @workgroup_size(256)
fn sigmoid(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output_unary)) {
        return;
    }
    let x = input[idx];
    output_unary[idx] = 1.0 / (1.0 + exp(-x));
}

// Tanh: output[i] = tanh(x)
@compute @workgroup_size(256)
fn tanh_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output_unary)) {
        return;
    }
    output_unary[idx] = tanh(input[idx]);
}

// Exponential: output[i] = exp(x)
@compute @workgroup_size(256)
fn exp_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output_unary)) {
        return;
    }
    output_unary[idx] = exp(input[idx]);
}

// Natural logarithm: output[i] = log(x)
@compute @workgroup_size(256)
fn log_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output_unary)) {
        return;
    }
    output_unary[idx] = log(input[idx]);
}

// Square root: output[i] = sqrt(x)
@compute @workgroup_size(256)
fn sqrt_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output_unary)) {
        return;
    }
    output_unary[idx] = sqrt(input[idx]);
}

// Negation: output[i] = -x
@compute @workgroup_size(256)
fn neg(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output_unary)) {
        return;
    }
    output_unary[idx] = -input[idx];
}

// ============================================================================
// Scalar Operations
// ============================================================================

@group(0) @binding(0) var<storage, read> input_scalar_op: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_scalar: array<f32>;
@group(0) @binding(2) var<uniform> scalar: f32;

// Add scalar: output[i] = input[i] + scalar
@compute @workgroup_size(256)
fn add_scalar(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output_scalar)) {
        return;
    }
    output_scalar[idx] = input_scalar_op[idx] + scalar;
}

// Multiply scalar: output[i] = input[i] * scalar
@compute @workgroup_size(256)
fn mul_scalar(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output_scalar)) {
        return;
    }
    output_scalar[idx] = input_scalar_op[idx] * scalar;
}

// Greater-than scalar: output[i] = 1.0 if input[i] > scalar else 0.0
@compute @workgroup_size(256)
fn gt_scalar(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output_scalar)) {
        return;
    }
    let x = input_scalar_op[idx];
    output_scalar[idx] = select(0.0, 1.0, x > scalar);
}

// ============================================================================
// Matrix Multiplication (Tiled)
// ============================================================================

struct MatMulParams {
    m: u32,  // rows in A, rows in C
    n: u32,  // cols in B, cols in C
    k: u32,  // cols in A, rows in B
}

@group(0) @binding(0) var<storage, read> matrix_a: array<f32>;
@group(0) @binding(1) var<storage, read> matrix_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> matrix_c: array<f32>;
@group(0) @binding(3) var<uniform> matmul_params: MatMulParams;

// Tiled matrix multiplication
// Each workgroup computes a TILE_SIZE x TILE_SIZE tile of the output
const TILE_SIZE: u32 = 16u;

var<workgroup> tile_a: array<f32, 256>;  // TILE_SIZE * TILE_SIZE
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn matmul_tiled(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let row = global_id.y;
    let col = global_id.x;
    let m = matmul_params.m;
    let n = matmul_params.n;
    let k = matmul_params.k;
    
    let local_row = local_id.y;
    let local_col = local_id.x;
    let local_idx = local_row * TILE_SIZE + local_col;
    
    var sum: f32 = 0.0;
    
    // Loop over tiles
    var num_tiles = (k + TILE_SIZE - 1u) / TILE_SIZE;
    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        // Load tile from A
        let a_col = t * TILE_SIZE + local_col;
        if (row < m && a_col < k) {
            tile_a[local_idx] = matrix_a[row * k + a_col];
        } else {
            tile_a[local_idx] = 0.0;
        }
        
        // Load tile from B
        let b_row = t * TILE_SIZE + local_row;
        if (b_row < k && col < n) {
            tile_b[local_idx] = matrix_b[b_row * n + col];
        } else {
            tile_b[local_idx] = 0.0;
        }
        
        workgroupBarrier();
        
        // Compute partial dot product
        for (var i: u32 = 0u; i < TILE_SIZE; i = i + 1u) {
            sum = sum + tile_a[local_row * TILE_SIZE + i] * tile_b[i * TILE_SIZE + local_col];
        }
        
        workgroupBarrier();
    }
    
    // Write result
    if (row < m && col < n) {
        matrix_c[row * n + col] = sum;
    }
}

// ============================================================================
// Reduction Operations
// ============================================================================

@group(0) @binding(0) var<storage, read> input_reduce: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_reduce: array<f32>;
@group(0) @binding(2) var<uniform> reduce_size: u32;

// Sum reduction (parallel)
@compute @workgroup_size(256)
fn sum_reduce(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let n = arrayLength(&input_reduce);
    
    if (idx >= n) {
        return;
    }
    
    // Simple version: each thread adds one element
    // In a full implementation, use parallel reduction
    // For now, CPU fallback handles full reduction
    output_reduce[idx] = input_reduce[idx];
}

// ============================================================================
// Copy/Fill Operations
// ============================================================================

@group(0) @binding(0) var<storage, read> input_fill: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_fill: array<f32>;
@group(0) @binding(2) var<uniform> fill_value: f32;

// Fill with constant value
@compute @workgroup_size(256)
fn fill(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output_fill)) {
        return;
    }
    output_fill[idx] = fill_value;
}

// Copy: output[i] = input[i]
@compute @workgroup_size(256)
fn copy(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output_fill)) {
        return;
    }
    output_fill[idx] = input_fill[idx];
}

// ============================================================================
// Softmax Operations
// ============================================================================

struct SoftmaxParams {
    batch_size: u32,
    num_classes: u32,
}

@group(0) @binding(0) var<storage, read> input_softmax: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_softmax: array<f32>;
@group(0) @binding(2) var<uniform> softmax_params: SoftmaxParams;

// Softmax: output[i,j] = exp(input[i,j] - max_row[i]) / sum(exp(input[i,k] - max_row[i]))
@compute @workgroup_size(256)
fn softmax(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_size = softmax_params.batch_size;
    let num_classes = softmax_params.num_classes;
    let n = arrayLength(&input_softmax);
    
    let idx = global_id.x;
    if (idx >= n) {
        return;
    }
    
    let batch_idx = idx / num_classes;
    let class_idx = idx % num_classes;
    
    // Compute max for numerical stability
    // Each thread works on one element, full max would require a separate pass
    // For now, simplified implementation assuming input is already shifted
    let x = input_softmax[idx];
    let exp_x = exp(x);
    output_softmax[idx] = exp_x;
}

// Softmax normalization pass: divide by row sum
@group(0) @binding(0) var<storage, read> softmax_unnorm: array<f32>;
@group(0) @binding(1) var<storage, read_write> softmax_output: array<f32>;
@group(0) @binding(2) var<uniform> softmax_norm_params: SoftmaxParams;

@compute @workgroup_size(256)
fn softmax_normalize(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_size = softmax_norm_params.batch_size;
    let num_classes = softmax_norm_params.num_classes;
    let n = arrayLength(&softmax_unnorm);
    
    let idx = global_id.x;
    if (idx >= n) {
        return;
    }
    
    let batch_idx = idx / num_classes;
    
    // Compute sum for this row (simplified - each thread computes sum independently)
    var row_sum: f32 = 0.0;
    let row_start = batch_idx * num_classes;
    for (var i: u32 = 0u; i < num_classes; i = i + 1u) {
        row_sum = row_sum + softmax_unnorm[row_start + i];
    }
    
    // Normalize
    if (row_sum > 0.0) {
        softmax_output[idx] = softmax_unnorm[idx] / row_sum;
    } else {
        softmax_output[idx] = softmax_unnorm[idx];
    }
}

// LogSoftmax: output[i,j] = log(softmax(input[i,j]))
@group(0) @binding(0) var<storage, read> input_logsoftmax: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_logsoftmax: array<f32>;
@group(0) @binding(2) var<uniform> logsoftmax_params: SoftmaxParams;

@compute @workgroup_size(256)
fn logsoftmax(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_size = logsoftmax_params.batch_size;
    let num_classes = logsoftmax_params.num_classes;
    let n = arrayLength(&input_logsoftmax);
    
    let idx = global_id.x;
    if (idx >= n) {
        return;
    }
    
    let batch_idx = idx / num_classes;
    
    // Find max for numerical stability
    var max_val: f32 = -3.402823466e+38; // -inf
    let row_start = batch_idx * num_classes;
    for (var i: u32 = 0u; i < num_classes; i = i + 1u) {
        let val = input_logsoftmax[row_start + i];
        if (val > max_val) {
            max_val = val;
        }
    }
    
    // Compute log-sum-exp: log(sum(exp(x - max))) + max
    var sum_exp: f32 = 0.0;
    for (var i: u32 = 0u; i < num_classes; i = i + 1u) {
        sum_exp = sum_exp + exp(input_logsoftmax[row_start + i] - max_val);
    }
    
    // logsoftmax(x) = x - log(sum_exp) - max
    output_logsoftmax[idx] = input_logsoftmax[idx] - log(sum_exp) - max_val;
}

// ============================================================================
// Transpose Operations
// ============================================================================

struct TransposeParams {
    rows: u32,
    cols: u32,
}

@group(0) @binding(0) var<storage, read> input_transpose: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_transpose: array<f32>;
@group(0) @binding(2) var<uniform> transpose_params: TransposeParams;

// Transpose: output[j,i] = input[i,j]
@compute @workgroup_size(16, 16, 1)
fn transpose(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let rows = transpose_params.rows;
    let cols = transpose_params.cols;
    
    let row = global_id.y;
    let col = global_id.x;
    
    if (row >= rows || col >= cols) {
        return;
    }
    
    // input[row, col] -> output[col, row]
    let input_idx = row * cols + col;
    let output_idx = col * rows + row;
    
    output_transpose[output_idx] = input_transpose[input_idx];
}

// Transpose with shared memory (optimized)
var<workgroup> tile_transpose: array<f32, 256>; // 16x16

@compute @workgroup_size(16, 16, 1)
fn transpose_tiled(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let rows = transpose_params.rows;
    let cols = transpose_params.cols;
    
    let row = global_id.y;
    let col = global_id.x;
    let local_row = local_id.y;
    let local_col = local_id.x;
    
    // Load into shared memory with coalesced read
    if (row < rows && col < cols) {
        let input_idx = row * cols + col;
        tile_transpose[local_row * 16u + local_col] = input_transpose[input_idx];
    }
    
    workgroupBarrier();
    
    // Write with bank-conflict-free access (transposed indices)
    // output[col, row] comes from shared memory at [local_col, local_row]
    let transposed_row = col;
    let transposed_col = row;
    
    if (transposed_row < cols && transposed_col < rows) {
        let output_idx = transposed_row * rows + transposed_col;
        output_transpose[output_idx] = tile_transpose[local_col * 16u + local_row];
    }
}

// ============================================================================
// Gather/Scatter Operations
// ============================================================================

struct GatherParams {
    num_indices: u32,
    index_dim: u32,
    input_dim0: u32,
    input_dim1: u32,
}

@group(0) @binding(0) var<storage, read> input_gather: array<f32>;
@group(0) @binding(1) var<storage, read> indices_gather: array<u32>;
@group(0) @binding(2) var<storage, read_write> output_gather: array<f32>;
@group(0) @binding(3) var<uniform> gather_params: GatherParams;

// Gather rows: output[i] = input[indices[i]]
// Assumes 2D input, gathers along first dimension
@compute @workgroup_size(256)
fn gather_rows(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let num_indices = gather_params.num_indices;
    let input_dim1 = gather_params.input_dim1;
    let n = arrayLength(&output_gather);
    
    let idx = global_id.x;
    if (idx >= n) {
        return;
    }
    
    let index_pos = idx / input_dim1;
    let col = idx % input_dim1;
    
    if (index_pos < num_indices) {
        let row_idx = indices_gather[index_pos];
        // Bounds check
        if (row_idx < gather_params.input_dim0) {
            let input_idx = row_idx * input_dim1 + col;
            if (input_idx < arrayLength(&input_gather)) {
                output_gather[idx] = input_gather[input_idx];
            }
        }
    }
}

// Scatter: output[indices[i]] = input[i]
@group(0) @binding(0) var<storage, read> input_scatter: array<f32>;
@group(0) @binding(1) var<storage, read> indices_scatter: array<u32>;
@group(0) @binding(2) var<storage, read_write> output_scatter: array<f32>;
@group(0) @binding(3) var<uniform> scatter_params: GatherParams;

@compute @workgroup_size(256)
fn scatter_rows(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let num_indices = scatter_params.num_indices;
    let input_dim1 = scatter_params.input_dim1;
    let n = arrayLength(&input_scatter);
    
    let idx = global_id.x;
    if (idx >= n) {
        return;
    }
    
    let index_pos = idx / input_dim1;
    let col = idx % input_dim1;
    
    if (index_pos < num_indices) {
        let row_idx = indices_scatter[index_pos];
        // Bounds check
        if (row_idx < scatter_params.input_dim0) {
            let output_idx = row_idx * input_dim1 + col;
            if (output_idx < arrayLength(&output_scatter)) {
                output_scatter[output_idx] = input_scatter[idx];
            }
        }
    }
}

// ============================================================================
// Indexing Operations
// ============================================================================

struct IndexParams {
    dim: u32,
    dim_size: u32,
}

@group(0) @binding(0) var<storage, read> input_index: array<f32>;
@group(0) @binding(1) var<storage, read> index_buffer: array<u32>;
@group(0) @binding(2) var<storage, read_write> output_index: array<f32>;
@group(0) @binding(3) var<uniform> index_params: IndexParams;

// Advanced gather with per-element indices
@compute @workgroup_size(256)
fn gather_advanced(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let n = arrayLength(&output_index);
    
    if (idx >= n) {
        return;
    }
    
    // Each output element is gathered from input at a different index
    let gather_idx = index_buffer[idx];
    if (gather_idx < arrayLength(&input_index)) {
        output_index[idx] = input_index[gather_idx];
    }
}

// ============================================================================
// Broadcast Operations
// ============================================================================

struct BroadcastParams {
    input_shape0: u32,
    input_shape1: u32,
    output_shape0: u32,
    output_shape1: u32,
}

@group(0) @binding(0) var<storage, read> input_broadcast: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_broadcast: array<f32>;
@group(0) @binding(2) var<uniform> broadcast_params: BroadcastParams;

// Broadcast: expand smaller tensor to match larger
@compute @workgroup_size(256)
fn broadcast_to(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let n = arrayLength(&output_broadcast);
    
    if (idx >= n) {
        return;
    }
    
    // Map output index to input index with broadcasting rules
    let out_row = idx / broadcast_params.output_shape1;
    let out_col = idx % broadcast_params.output_shape1;
    
    let in_row = out_row % broadcast_params.input_shape0;
    let in_col = out_col % broadcast_params.input_shape1;
    
    let input_idx = in_row * broadcast_params.input_shape1 + in_col;
    if (input_idx < arrayLength(&input_broadcast)) {
        output_broadcast[idx] = input_broadcast[input_idx];
    }
}
