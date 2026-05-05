use std::sync::atomic::{AtomicU64, Ordering};

use crate::Backend;

static NEXT_RUN_ID: AtomicU64 = AtomicU64::new(1);

/// Execution mode for a forward pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Training mode. Modules may enable stochastic behavior such as dropout.
    Train,
    /// Inference mode. Modules should use deterministic behavior.
    Inference,
}

/// Monotonic identifier assigned to each forward context.
///
/// This is useful for tracing, cache invalidation, metrics, and debugging
/// without tying correctness to object lifetime or global computation graphs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RunId(u64);

impl RunId {
    /// Allocate a fresh run id.
    pub fn fresh() -> Self {
        Self(NEXT_RUN_ID.fetch_add(1, Ordering::Relaxed))
    }

    /// Return the numeric value of this id.
    pub fn get(self) -> u64 {
        self.0
    }
}

/// Explicit context for one forward computation.
///
/// The context carries the backend, mode, and run id. It replaces DyNet-style
/// implicit graph renewal with visible execution state.
pub struct ForwardCtx<'a, B: Backend> {
    backend: &'a B,
    mode: Mode,
    run_id: RunId,
}

impl<'a, B: Backend> ForwardCtx<'a, B> {
    /// Create a new forward context for the given backend and mode.
    pub fn new(backend: &'a B, mode: Mode) -> Self {
        Self { backend, mode, run_id: RunId::fresh() }
    }

    /// Borrow the backend used for this computation.
    pub fn backend(&self) -> &'a B {
        self.backend
    }

    /// Return the current execution mode.
    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// Return the unique id for this forward computation.
    pub fn run_id(&self) -> RunId {
        self.run_id
    }

    /// Return true when the context is in training mode.
    pub fn is_training(&self) -> bool {
        matches!(self.mode, Mode::Train)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_id_fresh() {
        let id1 = RunId::fresh();
        let id2 = RunId::fresh();
        assert_ne!(id1.get(), id2.get());
    }

    #[test]
    fn test_mode_equality() {
        assert_eq!(Mode::Train, Mode::Train);
        assert_eq!(Mode::Inference, Mode::Inference);
        assert_ne!(Mode::Train, Mode::Inference);
    }
}
