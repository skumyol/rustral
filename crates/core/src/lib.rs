//! Core contracts for the Rustral.
//!
//! This crate deliberately keeps state explicit. There is no global parameter
//! collection and no implicit computation graph renewal.

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

pub use backend::{
    AttentionOps, Backend, BackendCapabilities, ConvLayout, FusionOps, QuantizationOps, TensorInPlaceOps,
    TensorOps, TensorView, TrainingDtype,
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
    DeviceType, MatmulDim, OperationGuard, OperationProfiler, OperationStats, ProfilingHooks, ShapeBucket,
};
pub use parameter::{Parameter, ParameterGroup, ParameterId, ParameterRef};
pub use shape::{Shape, ShapeExt, TensorShape};
pub use tensor_pool::{PoolStats, PoolStrategy, PooledTensor, TensorPool};
