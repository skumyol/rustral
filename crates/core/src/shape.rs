use crate::{CoreError, Result};
use serde::{Deserialize, Serialize};

/// Validated tensor shape.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shape(pub Vec<usize>);

impl Shape {
    /// Create a new non-empty shape with all dimensions greater than zero.
    pub fn new(dims: impl Into<Vec<usize>>) -> Result<Self> {
        let dims = dims.into();
        if dims.is_empty() {
            return Err(CoreError::InvalidShape { shape: dims, reason: "rank must be at least one".into() });
        }
        if dims.iter().any(|&d| d == 0) {
            return Err(CoreError::InvalidShape { shape: dims, reason: "dimensions must be non-zero".into() });
        }
        Ok(Self(dims))
    }

    /// Borrow the dimensions as a slice.
    pub fn as_slice(&self) -> &[usize] {
        &self.0
    }

    /// Return the number of dimensions.
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// Return the product of all dimensions.
    pub fn elem_count(&self) -> usize {
        self.0.iter().product()
    }
}

/// Convenience extension for shape-like slices.
pub trait ShapeExt {
    /// Return the product of all dimensions.
    fn elem_count(&self) -> usize;
}

impl ShapeExt for [usize] {
    fn elem_count(&self) -> usize {
        self.iter().product()
    }
}

/// Trait for accessing tensor shape information.
///
/// This trait is implemented by backend tensors to allow shape inspection
/// during serialization and other operations without requiring full
/// [`TensorOps`] functionality.
pub trait TensorShape {
    /// Return the tensor shape as a slice of dimensions.
    fn shape(&self) -> &[usize];
}
