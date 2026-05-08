//! Core contracts for the Rustral.
//!
//! This crate deliberately keeps state explicit. There is no global parameter
//! collection and no implicit computation graph renewal.

mod backend;
mod context;
mod error;
mod memory_profiler;
mod module;
mod operation_profiler;
mod parameter;
mod shape;
mod tensor_pool;

pub use backend::{Backend, BackendCapabilities, FusionOps, TensorInPlaceOps, TensorOps, TensorView};
pub use context::{ForwardCtx, Mode, RunId};
pub use error::{CoreError, Result};
pub use memory_profiler::{
    global_profiler, AllocationEvent, AllocationTracker, MemoryProfiler, MemorySnapshot, MemorySummary,
    OomRisk,
};
pub use module::{
    collect_named_parameter_ids, collect_named_parameters, Module, NamedParameters, Saveable, StatefulModule, Trainable,
};
pub use operation_profiler::{OperationGuard, OperationProfiler, OperationStats};
pub use parameter::{Parameter, ParameterGroup, ParameterId, ParameterRef};
pub use shape::{Shape, ShapeExt, TensorShape};
pub use tensor_pool::{PoolStats, PooledTensor, TensorPool};
