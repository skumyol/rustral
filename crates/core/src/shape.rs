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
        if dims.contains(&0) {
            return Err(CoreError::InvalidShape {
                shape: dims,
                reason: "dimensions must be non-zero".into(),
            });
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_new_valid() {
        let shape = Shape::new(vec![2, 3, 4]).unwrap();
        assert_eq!(shape.0, vec![2, 3, 4]);
    }

    #[test]
    fn test_shape_new_empty_fails() {
        let result = Shape::new(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_shape_new_zero_dim_fails() {
        let result = Shape::new(vec![2, 0, 4]);
        assert!(result.is_err());
    }

    #[test]
    fn test_shape_as_slice() {
        let shape = Shape::new(vec![2, 3]).unwrap();
        assert_eq!(shape.as_slice(), &[2, 3]);
    }

    #[test]
    fn test_shape_rank() {
        let shape = Shape::new(vec![2, 3, 4]).unwrap();
        assert_eq!(shape.rank(), 3);
    }

    #[test]
    fn test_shape_elem_count() {
        let shape = Shape::new(vec![2, 3, 4]).unwrap();
        assert_eq!(shape.elem_count(), 24);
    }

    #[test]
    fn test_shape_elem_count_single() {
        let shape = Shape::new(vec![5]).unwrap();
        assert_eq!(shape.elem_count(), 5);
    }

    #[test]
    fn test_shape_ext_empty() {
        let slice: &[usize] = &[];
        assert_eq!(slice.elem_count(), 1);
    }

    #[test]
    fn test_shape_ext() {
        assert_eq!([2, 3, 4].elem_count(), 24);
    }
}
