use thiserror::Error;

/// Standard result type for core runtime operations.
pub type Result<T> = std::result::Result<T, CoreError>;

/// Error type shared by core traits and reference backends.
#[derive(Debug, Error)]
pub enum CoreError {
    /// A tensor or parameter had an unexpected shape.
    #[error("shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// Expected dimensions or element counts.
        expected: Vec<usize>,
        /// Actual dimensions or element counts.
        actual: Vec<usize>,
    },

    /// A shape was structurally invalid.
    #[error("invalid shape {shape:?}: {reason}")]
    InvalidShape {
        /// Shape that failed validation.
        shape: Vec<usize>,
        /// Human-readable reason.
        reason: String,
    },

    /// A public function received an invalid argument.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// Backend-specific failure.
    #[error("backend error: {0}")]
    Backend(String),

    /// Save/load failure.
    #[error("serialization error: {0}")]
    Serialization(String),
}
