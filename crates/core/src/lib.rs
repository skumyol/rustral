//! Core contracts for the Rustral.
//!
//! This crate deliberately keeps state explicit. There is no global parameter
//! collection and no implicit computation graph renewal.

/// Primary env var to enable optional parallel reductions on CPU (`ndarray-backend`).
///
/// When set to `"1"`, [`parallel_reductions_enabled`] is true. See also [`PAR_REDUCE_ENV`].
///
/// Parallel paths use fixed reduction order per chunk/output element; enabling this can still
/// change numeric results versus the fully serial path due to floating-point association.
///
/// Default: unset or not `"1"` (serial SIMD reductions where implemented).
pub const PARALLEL_REDUCTIONS_ENV: &str = "RUSTRAL_PARALLEL_REDUCTIONS";

/// Shorter alias for [`PARALLEL_REDUCTIONS_ENV`], accepted by `ndarray-backend` and this helper.
pub const PAR_REDUCE_ENV: &str = "RUSTRAL_PAR_REDUCE";

/// `true` if either [`PARALLEL_REDUCTIONS_ENV`] or [`PAR_REDUCE_ENV`] is set to `"1"`.
pub fn parallel_reductions_enabled() -> bool {
    std::env::var(PARALLEL_REDUCTIONS_ENV).as_deref() == Ok("1")
        || std::env::var(PAR_REDUCE_ENV).as_deref() == Ok("1")
}

mod backend;
mod context;
mod error;
mod fusion;
mod fusion_tests;
mod memory_profiler;
mod module;
mod numerics;
mod operation_profiler;
mod parameter;
mod shape;
mod shape_policy;
mod tensor_pool;
mod tolerance;

pub use backend::{
    AttentionOps, Backend, BackendCapabilities, ConvLayout, FusionOps, OperationType, QuantizationOps,
    TensorInPlaceOps, TensorOps, TensorView, TrainingDtype,
};
pub use context::{ForwardCtx, Mode, RunId};
pub use shape_policy::ShapePolicy;
pub use error::{CoreError, Result};
pub use fusion::{FusionOptimizer, FusionPattern, Op, OpType, PatternMatcher};
pub use fusion_tests::{
    generate_constant_data, generate_random_data, FusionTestConfig, FusionTestHarness, FusionTestResult,
    FusionTestSuiteResult,
};
pub use memory_profiler::{
    global_profiler, AllocationEvent, AllocationTracker, MemoryProfiler, MemorySnapshot, MemorySummary,
    OomRisk,
};
pub use module::{
    collect_named_parameter_ids, collect_named_parameters, Module, NamedParameters, Saveable, StatefulModule,
    Trainable,
};
pub use numerics::{
    DType, NumericsConfig, NumericsError, NumericsValidator, Tolerance, ValidationResult,
};
pub use operation_profiler::{
    DeviceType, MatmulDim, OperationGuard, OperationProfiler, OperationStats, ProfilerSnapshot, ProfilingHooks,
    ShapeBucket, SnapshotOp,
};
pub use parameter::{Parameter, ParameterGroup, ParameterId, ParameterRef};
pub use shape::{Shape, ShapeExt, TensorShape};
pub use tensor_pool::{PoolStats, PoolStrategy, PooledTensor, TensorPool};
pub use tolerance::{OpFamily, ToleranceConfig};
