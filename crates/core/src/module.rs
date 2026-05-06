use crate::{Backend, ForwardCtx, Parameter, ParameterRef, Result};
use std::collections::HashMap;

/// Stateless forward-computation contract.
///
/// Modules define explicit input and output types. A module may own parameters,
/// but it must not rely on global computation state.
pub trait Module<B: Backend>: Send + Sync {
    /// Input type accepted by the module.
    type Input;

    /// Output type produced by the module.
    type Output;

    /// Run the module for one explicit forward context.
    fn forward(&self, input: Self::Input, ctx: &mut ForwardCtx<B>) -> Result<Self::Output>;
}

/// Contract for modules that expose trainable parameters.
pub trait Trainable<B: Backend> {
    /// Return stable references to trainable parameters owned by this module.
    fn parameters(&self) -> Vec<ParameterRef>;
}

/// Extension for modules with recurrent or persistent state.
pub trait StatefulModule<B: Backend>: Module<B> {
    /// State type carried between calls.
    type State: Clone + Send + Sync + 'static;

    /// Return the default initial state using the given forward context.
    ///
    /// This receives the context so the module can create zero tensors
    /// via the backend, avoiding the need to store a backend reference.
    fn initial_state(&self, ctx: &mut ForwardCtx<B>) -> Result<Self::State>;
}

/// Contract for modules that can be serialized and deserialized.
///
/// This enables saving trained models to disk and loading them back.
/// Implementations should provide stable, hierarchical keys for parameters.
pub trait Saveable<B: Backend> {
    /// Export the module's state as a collection of named parameters.
    ///
    /// Keys should be hierarchical (e.g., "layer1.weight", "layer1.bias")
    /// to support nested modules like Sequential2.
    fn state_dict(&self) -> Vec<(String, ParameterRef)>;

    /// Load parameters from a state dictionary.
    ///
    /// # Arguments
    /// * `dict` - Map from parameter names to flat f32 values
    /// * `backend` - Backend for creating tensors
    ///
    /// # Errors
    /// Returns an error if a required key is missing or shapes don't match.
    fn load_state_dict(
        &mut self,
        dict: &std::collections::HashMap<String, Vec<f32>>,
        backend: &B,
    ) -> Result<()>;
}

/// Named parameter traversal for nested modules.
///
/// This closes the "parameter ownership loop": optimizers, checkpointing, and tooling
/// can walk an owned module tree without callers assembling flat parameter arrays.
///
/// Names must be **stable** and **hierarchical**, e.g.:
/// - `encoder.layers.0.self_attn.q_proj.weight`
/// - `mlp.0.weight`
#[allow(dead_code)]
pub trait NamedParameters<B: Backend> {
    /// Visit all parameters owned by this module (immutable).
    fn visit_parameters(&self, f: &mut dyn FnMut(&str, &Parameter<B>));

    /// Visit all parameters owned by this module (mutable).
    fn visit_parameters_mut(&mut self, f: &mut dyn FnMut(&str, &mut Parameter<B>));
}

/// Collect all parameters with their stable names as [`ParameterRef`]s.
pub fn collect_named_parameters<B: Backend, M: NamedParameters<B>>(model: &M) -> Vec<(String, ParameterRef)> {
    let mut out = Vec::new();
    model.visit_parameters(&mut |name, p| {
        out.push((name.to_string(), ParameterRef { id: p.id() }));
    });
    out
}

/// Collect a reverse map from parameter id to its stable name.
///
/// This is useful for trainer logs and checkpoint keys (id -> path).
pub fn collect_named_parameter_ids<B: Backend, M: NamedParameters<B>>(model: &M) -> HashMap<crate::ParameterId, String> {
    let mut out: HashMap<crate::ParameterId, String> = HashMap::new();
    model.visit_parameters(&mut |name, p| {
        out.insert(p.id(), name.to_string());
    });
    out
}
