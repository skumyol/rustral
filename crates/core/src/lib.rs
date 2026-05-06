//! Core contracts for the Rustral.
//!
//! This crate deliberately keeps state explicit. There is no global parameter
//! collection and no implicit computation graph renewal.

mod backend;
mod context;
mod error;
mod memory_profiler;
mod module;
mod parameter;
mod shape;

pub use backend::{Backend, TensorInPlaceOps, TensorOps, TensorView};
pub use context::{ForwardCtx, Mode, RunId};
pub use error::{CoreError, Result};
pub use memory_profiler::{
    global_profiler, AllocationEvent, AllocationTracker, MemoryProfiler, MemorySnapshot, MemorySummary,
    OomRisk,
};
pub use module::{
    collect_named_parameter_ids, collect_named_parameters, Module, NamedParameters, Saveable, StatefulModule, Trainable,
};
pub use parameter::{Parameter, ParameterGroup, ParameterId, ParameterRef};
pub use shape::{Shape, ShapeExt, TensorShape};
