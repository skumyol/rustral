# API Signatures

This document is an explicit, human-readable API inventory. It mirrors the public function signatures in the workspace and explains the purpose of each signature.

For roadmap/status (what to build next), see [`docs/master-plan.md`](master-plan.md).

## `rustral-core`

### Backend contract

```rust
pub trait Backend: Clone + Send + Sync + 'static {
    type Tensor: Clone + Send + Sync + 'static;
    type Device: Clone + Send + Sync + std::fmt::Debug + 'static;

    fn device(&self) -> Self::Device;
    fn ops(&self) -> &dyn TensorOps<Self>;

    fn fusion_ops(&self) -> Option<&dyn FusionOps<Self>> { None }
    fn attention_ops(&self) -> Option<&dyn AttentionOps<Self>> { None }
    fn quantization_ops(&self) -> Option<&dyn QuantizationOps<Self>> { None }

    fn capabilities(&self) -> BackendCapabilities { BackendCapabilities::default() }

    fn as_any(&self) -> &dyn std::any::Any;

    fn normal_parameter(&self, name: &str, shape: &[usize], seed: u64, scale: f32) -> Result<Parameter<Self>>
    where Self: Sized;
    fn parameter_from_vec(&self, name: &str, values: Vec<f32>, shape: &[usize]) -> Result<Parameter<Self>>
    where Self: Sized;
}
```

- `device`: returns the device handle where tensors and parameters live.
- `ops`: returns the backend operation table used by shared modules.
- `fusion_ops`: optional fused kernels (`LinearReLU` / `LinearGELU` use [`FusionOptimizer::apply_*`](../crates/core/src/fusion.rs) / `FusionHelper`). Default: `None`.
- `attention_ops`: optional optimized attention (e.g. Flash-style paths). Default: `None`; layers fall back to generic attention.
- `quantization_ops`: optional INT8 / quantization helpers. Default: `None`.
- `capabilities`: hardware and layout hints; see **BackendCapabilities** below and [`ARCHITECTURE.md`](../ARCHITECTURE.md) (e.g. `clamp_batch_size` is consumed; many fields are advisory).
- `as_any`: downcast hook for rare backend-specific code paths; prefer trait polymorphism when possible.
- `normal_parameter` / `parameter_from_vec`: construct `Parameter` values for layers and checkpoint restore.

### `BackendCapabilities`, `OperationType`, and related enums

```rust
pub struct BackendCapabilities {
    pub supports_fp16: bool,
    pub supports_bf16: bool,
    pub tensor_cores: bool,
    pub optimal_batch_size: usize,
    pub optimal_chunk_size: usize,
    pub max_allocation_size: usize,
    pub prefers_contiguous: bool,
    pub supports_in_place: bool,
    pub supports_mixed_precision: bool,
    pub recommended_training_dtype: TrainingDtype,
    pub supports_fast_fp16_tensor_cores: bool,
    pub preferred_conv_layout: ConvLayout,
    pub supports_strided_layouts: bool,
    pub supports_packed_layouts: bool,
}

impl BackendCapabilities {
    pub fn clamp_batch_size(&self, batch: usize) -> usize;
    pub fn recommends_mixed_precision(&self) -> bool;
    pub fn recommended_dtype_for_operation(&self, operation: OperationType) -> TrainingDtype;
}

pub enum OperationType {
    Matmul,
    Convolution,
    Reduction,
    Elementwise,
}

pub enum TrainingDtype { F32, F16, Bf16 }

pub enum ConvLayout { NCHW, NHWC }
```

- `clamp_batch_size`: soft upper bound using `optimal_batch_size` (at least 1).
- `recommends_mixed_precision`: `true` when mixed precision is supported and `recommended_training_dtype` is not `F32`.
- `recommended_dtype_for_operation`: capability-driven dtype hint (e.g. FP16 matmul on tensor-core-style flags when set).
- Fields such as `supports_fast_fp16_tensor_cores`, `preferred_conv_layout`, and layout flags are primarily **hints** for callers and documentation; see `ARCHITECTURE.md` for adoption status.

```rust
pub trait TensorOps<B: Backend>: Send + Sync {
    fn shape(&self, x: &B::Tensor) -> Vec<usize>;
    fn tensor_from_vec(&self, values: Vec<f32>, shape: &[usize]) -> Result<B::Tensor>;
    fn zeros(&self, shape: &[usize]) -> Result<B::Tensor>;
    fn matmul(&self, a: &B::Tensor, b: &B::Tensor) -> Result<B::Tensor>;
    fn add(&self, a: &B::Tensor, b: &B::Tensor) -> Result<B::Tensor>;
    fn add_row_vector(&self, a: &B::Tensor, row: &B::Tensor) -> Result<B::Tensor>;
    fn relu(&self, x: &B::Tensor) -> Result<B::Tensor>;
    fn gelu(&self, x: &B::Tensor) -> Result<B::Tensor>;
    fn softmax(&self, x: &B::Tensor) -> Result<B::Tensor>;
    fn argmax(&self, x: &B::Tensor) -> Result<usize>;
    fn gather_rows(&self, table: &Parameter<B>, ids: &[usize]) -> Result<B::Tensor>;
    fn linear(&self, input: &B::Tensor, weight: &Parameter<B>, bias: Option<&Parameter<B>>) -> Result<B::Tensor>;
}
```

- `shape`: returns row-major dimensions.
- `tensor_from_vec`: creates a tensor from flat values and shape.
- `zeros`: allocates a zero-filled tensor.
- `matmul`: multiplies rank-2 tensors.
- `add`: performs element-wise addition.
- `add_row_vector`: broadcasts a row vector across matrix rows.
- `relu`: applies ReLU element-wise.
- `gelu`: GELU activation (tanh approximation); used in transformer FFNs (eager and `Tape::gelu`).
- `softmax`: normalizes tensor values.
- `argmax`: returns the flat index of the maximum value.
- `gather_rows`: performs embedding-style row lookup.
- `linear`: applies `input * weight^T + bias`.

The real `TensorOps` trait in [`backend.rs`](../crates/core/src/backend.rs) also includes axis-aware softmax/reductions, conv helpers, dropout, and more. Notable for determinism:

```rust
fn dropout(&self, x: &B::Tensor, p: f32, training: bool) -> Result<B::Tensor>;
fn dropout_with_seed(&self, x: &B::Tensor, p: f32, seed: u64, training: bool) -> Result<B::Tensor> {
    /* default: ignores seed, delegates to dropout */
}
```

Backends may override `dropout_with_seed` for reproducible dropout; `ndarray-backend` and `wgpu-backend` implement seeded paths where applicable.

### Forward context

```rust
impl RunId {
    pub fn fresh() -> Self;
    pub fn get(self) -> u64;
}
```

- `fresh`: allocates a fresh execution id.
- `get`: exposes the numeric id for logging and tracing.

```rust
impl<'a, B: Backend> ForwardCtx<'a, B> {
    pub fn new(backend: &'a B, mode: Mode) -> Self;
    pub fn with_profiler(self, profiler: Arc<Mutex<OperationProfiler>>) -> Self;
    pub fn set_profiler(&mut self, profiler: Option<Arc<Mutex<OperationProfiler>>>);
    pub fn profiler(&self) -> Option<&Arc<Mutex<OperationProfiler>>>;
    pub fn set_shape_policy(&mut self, policy: ShapePolicy);
    pub fn shape_policy(&self) -> ShapePolicy;
    pub fn backend(&self) -> &'a B;
    pub fn mode(&self) -> Mode;
    pub fn run_id(&self) -> RunId;
    pub fn is_training(&self) -> bool;
}
```

- `new`: creates an explicit forward context.
- `with_profiler` / `set_profiler` / `profiler`: optional per-run operation timing.
- `set_shape_policy` / `shape_policy`: static vs dynamic shape hints for optimizers and tooling.
- `backend`: borrows the backend for module execution.
- `mode`: returns train or inference mode.
- `run_id`: returns the id of the current forward run.
- `is_training`: convenience check for train mode.

### Module contracts

```rust
pub trait Module<B: Backend>: Send + Sync {
    type Input;
    type Output;

    fn forward(&self, input: Self::Input, ctx: &mut ForwardCtx<B>) -> Result<Self::Output>;
}
```

- `forward`: runs a module with explicit input and context.

```rust
pub trait Trainable<B: Backend> {
    fn parameters(&self) -> Vec<ParameterRef>;
}
```

- `parameters`: returns trainable parameter references.

```rust
pub trait StatefulModule<B: Backend>: Module<B> {
    type State: Clone + Send + Sync + 'static;

    fn initial_state(&self) -> Self::State;
}
```

- `initial_state`: returns the default recurrent/stateful module state.

### Parameters

```rust
impl ParameterId {
    pub fn fresh() -> Self;
    pub fn get(self) -> u64;
}
```

- `fresh`: allocates a new parameter id.
- `get`: exposes the numeric id.

```rust
impl<B: Backend> Parameter<B> {
    pub fn new(name: impl Into<Arc<str>>, tensor: B::Tensor) -> Self;
    pub fn id(&self) -> ParameterId;
    pub fn name(&self) -> &str;
    pub fn tensor(&self) -> &B::Tensor;
    pub fn into_tensor(self) -> B::Tensor;
}
```

- `new`: creates an explicitly owned parameter.
- `id`: returns the stable parameter id.
- `name`: returns the parameter name.
- `tensor`: borrows the underlying tensor.
- `into_tensor`: consumes the parameter and returns its tensor.

### Shapes

```rust
impl Shape {
    pub fn new(dims: impl Into<Vec<usize>>) -> Result<Self>;
    pub fn as_slice(&self) -> &[usize];
    pub fn rank(&self) -> usize;
    pub fn elem_count(&self) -> usize;
}
```

- `new`: validates and creates a non-empty shape.
- `as_slice`: borrows the dimensions.
- `rank`: returns the number of dimensions.
- `elem_count`: returns total element count.

```rust
pub trait ShapeExt {
    fn elem_count(&self) -> usize;
}
```

- `elem_count`: computes the product of dimensions for slices.

## `rustral-ndarray-backend`

### CPU tensor

```rust
impl CpuTensor {
    pub fn new(values: Vec<f32>, shape: &[usize]) -> Result<Self>;
    pub fn values(&self) -> &[f32];
    pub fn shape(&self) -> &[usize];
    pub fn into_values(self) -> Vec<f32>;
}
```

- `new`: validates and creates a dense row-major tensor.
- `values`: borrows flat tensor values.
- `shape`: borrows tensor shape.
- `into_values`: consumes the tensor and returns its values.

### CPU backend

```rust
impl CpuBackend {
    pub fn tensor_from_vec(&self, values: Vec<f32>, shape: &[usize]) -> Result<CpuTensor>;
    pub fn normal_parameter(&self, name: &str, shape: &[usize], seed: u64, scale: f32) -> Result<Parameter<Self>>;
}
```

- `tensor_from_vec`: creates a CPU tensor.
- `normal_parameter`: creates a deterministic random parameter.

The `CpuBackend` implementation of `Backend` and `CpuOps` implementation of `TensorOps` provide the concrete behavior for all core tensor operations listed above.

## `rustral-symbolic`

```rust
impl Vocabulary {
    pub fn with_specials(unk: impl Into<String>) -> Self;
    pub fn insert(&mut self, token: impl Into<String>) -> Result<usize>;
    pub fn freeze(&mut self);
    pub fn is_frozen(&self) -> bool;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn unk_id(&self) -> usize;
    pub fn id(&self, token: &str) -> Option<usize>;
    pub fn id_or_unk(&self, token: &str) -> usize;
    pub fn token(&self, id: usize) -> Result<&str>;
    pub fn tokens(&self) -> impl Iterator<Item = &str>;
}
```

- `with_specials`: creates a vocabulary with an unknown token.
- `insert`: inserts a token unless frozen.
- `freeze`: prevents new token insertion.
- `is_frozen`: checks frozen state.
- `len`: returns token count.
- `is_empty`: checks whether no tokens exist.
- `unk_id`: returns the unknown-token id.
- `id`: looks up a token id.
- `id_or_unk`: looks up a token id with unknown fallback.
- `token`: retrieves a token by id.
- `tokens`: iterates over tokens in id order.

## `rustral-nn`

### Linear

```rust
impl<B: Backend> Linear<B> {
    pub fn from_parameters(config: LinearConfig, weight: Parameter<B>, bias: Option<Parameter<B>>) -> Self;
    pub fn config(&self) -> &LinearConfig;
    pub fn weight(&self) -> &Parameter<B>;
    pub fn bias(&self) -> Option<&Parameter<B>>;
}
```

- `from_parameters`: constructs a dense layer from explicit parameters.
- `config`: borrows layer dimensions and bias setting.
- `weight`: borrows the weight parameter.
- `bias`: borrows optional bias.

The `Module` implementation exposes:

```rust
fn forward(&self, input: B::Tensor, ctx: &mut ForwardCtx<B>) -> Result<B::Tensor>;
```

- `forward`: applies the affine projection.

The `Trainable` implementation exposes:

```rust
fn parameters(&self) -> Vec<ParameterRef>;
```

- `parameters`: returns weight and optional bias references.

### Embedding

```rust
impl<B: Backend> Embedding<B> {
    pub fn new(config: EmbeddingConfig, table: Parameter<B>, vocab: Arc<Vocabulary>) -> Self;
    pub fn vocab(&self) -> &Vocabulary;
    pub fn config(&self) -> &EmbeddingConfig;
}
```

- `new`: constructs an embedding lookup module.
- `vocab`: borrows the vocabulary.
- `config`: borrows embedding configuration.

The `Module` implementation exposes:

```rust
fn forward(&self, ids: Vec<usize>, ctx: &mut ForwardCtx<B>) -> Result<B::Tensor>;
```

- `forward`: gathers embedding rows.

### Readout

```rust
impl<B: Backend> Readout<B> {
    pub fn new(labels: Arc<Vocabulary>, projection: Linear<B>) -> Self;
    pub fn logits(&self, hidden: B::Tensor, ctx: &mut ForwardCtx<B>) -> Result<B::Tensor>;
    pub fn probabilities(&self, hidden: B::Tensor, ctx: &mut ForwardCtx<B>) -> Result<B::Tensor>;
    pub fn predict(&self, hidden: B::Tensor, ctx: &mut ForwardCtx<B>) -> Result<LabelPrediction>;
}
```

- `new`: constructs a readout from labels and projection.
- `logits`: computes unnormalized label scores.
- `probabilities`: applies softmax to logits.
- `predict`: returns the highest-scoring label.

### Sequential composition

```rust
impl<A, Bm> Sequential2<A, Bm> {
    pub fn new(first: A, second: Bm) -> Self;
}
```

- `new`: composes two modules.

The `Module` implementation exposes:

```rust
fn forward(&self, input: I, ctx: &mut ForwardCtx<BE>) -> Result<O>;
```

- `forward`: runs the first module, then the second.

## `rustral-runtime`

### Learner and trainer

```rust
impl Default for TrainerConfig {
    fn default() -> Self;
}
```

- `default`: creates conservative training defaults.

```rust
pub trait Learner<D>: Send + Sync {
    type BatchUpdate: Send;

    fn loss_and_update(&self, datum: &D) -> anyhow::Result<(f32, Self::BatchUpdate)>;
    fn merge_updates(&self, updates: Vec<Self::BatchUpdate>) -> anyhow::Result<Self::BatchUpdate>;
    fn apply_update(&mut self, update: Self::BatchUpdate) -> anyhow::Result<()>;
}
```

- `loss_and_update`: computes loss and local update for one datum.
- `merge_updates`: reduces per-example updates into one batch update.
- `apply_update`: mutates learner state with a merged update.

```rust
impl ParallelTrainer {
    pub fn new(config: TrainerConfig) -> Self;
    pub fn train<D, L>(&self, learner: &mut L, data: &[D]) -> anyhow::Result<Vec<EpochStats>>
    where
        D: Send + Sync,
        L: Learner<D>;
}
```

- `new`: creates a trainer.
- `train`: runs epoch/batch training with optional parallel map phase.

### Inference pool

```rust
impl<I, O> InferencePool<I, O>
where
    I: Send + 'static,
    O: Send + 'static,
{
    pub fn new<F>(workers: usize, queue_bound: usize, handler: F) -> anyhow::Result<Self>
    where
        F: Fn(I) -> anyhow::Result<O> + Send + Sync + 'static;

    pub fn infer(&self, input: I) -> anyhow::Result<O>;
    pub fn worker_count(&self) -> usize;
}
```

- `new`: starts a bounded fixed-size inference worker pool.
- `infer`: submits one request and waits for the result.
- `worker_count`: returns the number of worker threads.
