//! Hints for how tensor shapes behave during a run (static vs dynamic).
//!
//! Used by [`crate::ForwardCtx`] and optimization code to decide when aggressive reuse
//! (CUDA graphs, inference-oriented pooling) is safe.

/// Policy describing how shapes may vary across steps.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ShapePolicy {
    /// Fixed shapes for the whole run (typical inference with constant batch/sequence).
    Static,
    /// Shapes may change but stay within known bounds (bucketing-friendly).
    DynamicBounded,
    /// No assumptions (training with variable length or dynamic batch).
    #[default]
    DynamicUnbounded,
}

impl ShapePolicy {
    /// Whether CUDA graph capture is generally safe for this policy.
    pub fn supports_cuda_graph_capture(self) -> bool {
        matches!(self, Self::Static)
    }

    /// Whether an arena-style pool reset between steps is preferred.
    pub fn prefers_arena_pool(self) -> bool {
        matches!(self, Self::DynamicBounded)
    }
}
