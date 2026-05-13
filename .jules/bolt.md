## 2026-05-13 - [Normalization Optimization & Multi-dim Edge Cases]
**Learning:** Moving normalization layers (LayerNorm, BatchNorm) to be fully "on-device" and backend-agnostic provides massive speedups (up to 10x) by eliminating device-to-host synchronization. However, implementing these using core trait operations requires care:
1. Variance over multiple dimensions cannot be computed by nesting single-dimension variance calls. Use $Var(X) = E[X^2] - (E[X])^2$.
2. Backends often provide 1D fused kernels for LayerNorm. To support multi-dimensional `normalized_shape` (e.g., `[H, W]`), the input must be reshaped to `[prefix, product(norm_shape)]` before calling the backend.
**Action:** Always verify mathematical correctness for 4D tensors when using dimensionality-reduction ops, and use reshaping to bridge high-level module configurations with low-level fused kernel assumptions.

## 2026-05-13 - [MHA and Loss Optimization]
**Learning:** For competitive NLP performance (EMNLP systems level), standardizing core trait support for batched operations and efficient loss functions is mandatory.
1. `matmul_batched` avoids expensive sequential slicing on GPUs.
2. `cross_entropy_with_indices` eliminates O(V) memory allocation for one-hot tensors, making token-level classification (NER, LLM) significantly more efficient.
3. Multi-dimensional transposes (`transpose_axes`) are critical for parallel head processing in MHA.
**Action:** Always prefer trait-level batching over higher-level loops for performance-critical hot paths.
