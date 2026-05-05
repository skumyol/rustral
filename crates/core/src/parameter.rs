use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::Backend;

static NEXT_PARAMETER_ID: AtomicU64 = AtomicU64::new(1);

/// Stable identifier for a trainable parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ParameterId(u64);

impl ParameterId {
    /// Allocate a fresh parameter id.
    pub fn fresh() -> Self {
        Self(NEXT_PARAMETER_ID.fetch_add(1, Ordering::Relaxed))
    }

    /// Return the numeric value of this id.
    pub fn get(self) -> u64 {
        self.0
    }
}

/// Named trainable tensor owned by a module.
///
/// Parameters are explicit values, not entries in a global collection. This
/// makes ownership, serialization, and composition auditable.
#[derive(Clone)]
pub struct Parameter<B: Backend> {
    id: ParameterId,
    name: Arc<str>,
    tensor: B::Tensor,
}

impl<B: Backend> Parameter<B> {
    /// Create a new named parameter from an existing backend tensor.
    pub fn new(name: impl Into<Arc<str>>, tensor: B::Tensor) -> Self {
        Self { id: ParameterId::fresh(), name: name.into(), tensor }
    }

    /// Return this parameter's stable id.
    pub fn id(&self) -> ParameterId {
        self.id
    }

    /// Return this parameter's human-readable name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Borrow the underlying backend tensor.
    pub fn tensor(&self) -> &B::Tensor {
        &self.tensor
    }

    /// Consume the parameter and return the underlying backend tensor.
    pub fn into_tensor(self) -> B::Tensor {
        self.tensor
    }

    /// Replace the tensor while preserving id and name.
    pub fn with_tensor(self, tensor: B::Tensor) -> Self {
        Self { id: self.id, name: self.name, tensor }
    }
}

/// Lightweight reference to a parameter by id.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ParameterRef {
    /// Referenced parameter id.
    pub id: ParameterId,
}

/// Named group of parameter references for typed, role-based access.
///
/// Modules can expose a `ParameterGroup` so consumers can retrieve
/// parameters by semantic role (e.g. `"weight"`, `"bias"`) instead of
/// relying on positional indexing in a flat `Vec<ParameterRef>`.
#[derive(Clone, Debug, Default)]
pub struct ParameterGroup {
    name: Arc<str>,
    params: Vec<(Arc<str>, ParameterRef)>,
}

impl ParameterGroup {
    /// Create an empty parameter group with the given name.
    pub fn new(name: impl Into<Arc<str>>) -> Self {
        Self { name: name.into(), params: Vec::new() }
    }

    /// Insert a parameter under a named role.
    pub fn insert(&mut self, role: impl Into<Arc<str>>, param: ParameterRef) {
        self.params.push((role.into(), param));
    }

    /// Look up a parameter by its role name.
    pub fn get(&self, role: &str) -> Option<ParameterRef> {
        self.params.iter().find(|(r, _)| r.as_ref() == role).map(|(_, p)| *p)
    }

    /// Return all parameter references in this group.
    pub fn all(&self) -> Vec<ParameterRef> {
        self.params.iter().map(|(_, p)| *p).collect()
    }

    /// Return the number of parameters in this group.
    pub fn len(&self) -> usize {
        self.params.len()
    }

    /// Return true if this group contains no parameters.
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// Return the group name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Consume the group and return a flat vector of parameter references.
    ///
    /// This is useful for implementing [`Trainable::parameters`] on top of
    /// parameter groups.
    pub fn into_refs(self) -> Vec<ParameterRef> {
        self.all()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_id_fresh() {
        let id1 = ParameterId::fresh();
        let id2 = ParameterId::fresh();
        assert_ne!(id1.get(), id2.get());
    }

    #[test]
    fn test_parameter_ref_equality() {
        let id = ParameterId::fresh();
        let ref1 = ParameterRef { id };
        let ref2 = ParameterRef { id };
        assert_eq!(ref1, ref2);
    }

    #[test]
    fn test_parameter_group() {
        let mut group = ParameterGroup::new("encoder");
        assert_eq!(group.name(), "encoder");
        assert!(group.is_empty());

        let id = ParameterId::fresh();
        let pref = ParameterRef { id };
        group.insert("weight", pref);
        assert_eq!(group.len(), 1);
        assert_eq!(group.get("weight"), Some(pref));
        assert_eq!(group.get("bias"), None);

        let all = group.all();
        assert_eq!(all.len(), 1);

        let refs = group.into_refs();
        assert_eq!(refs.len(), 1);
    }

    #[test]
    fn test_parameter_group_default() {
        let group: ParameterGroup = Default::default();
        assert!(group.is_empty());
    }
}
