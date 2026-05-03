//! Core contracts for the Modular Neural Runtime.
//!
//! This crate deliberately keeps state explicit. There is no global parameter
//! collection and no implicit computation graph renewal.

mod backend;
mod context;
mod error;
mod module;
mod parameter;
mod shape;

pub use backend::{Backend, TensorInPlaceOps, TensorOps, TensorView};
pub use context::{ForwardCtx, Mode, RunId};
pub use error::{CoreError, Result};
pub use module::{Module, Saveable, StatefulModule, Trainable};
pub use parameter::{Parameter, ParameterGroup, ParameterId, ParameterRef};
pub use shape::{Shape, ShapeExt, TensorShape};
