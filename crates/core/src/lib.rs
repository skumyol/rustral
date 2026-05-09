//! Core contracts for the Rustral.
//!
//! This crate deliberately keeps state explicit. There is no global parameter
//! collection and no implicit computation graph renewal.

/// Environment variable to enable parallel reductions (non-deterministic order).
///
/// When set to "1", reduction operations may use parallel execution (e.g., rayon)
/// which can improve performance but may produce non-deterministic results due to
/// floating-point operation reordering.
///
/// Default: "0" (deterministic fixed-order reductions)
pub const PARALLEL_REDUCTIONS_ENV: &str = "RUSTRAL_PARALLEL_REDUCTIONS";

/// Check if parallel reductions are enabled via environment variable.
pub fn parallel_reductions_enabled() -> bool {
    std::env::var(PARALLEL_REDUCTIONS_ENV).as_deref() == Ok("1")
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
