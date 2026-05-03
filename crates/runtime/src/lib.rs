//! Runtime utilities for parallel training and inference orchestration.

mod inference;
mod trainer;

pub use inference::{InferencePool, InferenceRequest, InferenceResponse};
pub use trainer::{EpochStats, Learner, ParallelTrainer, TrainerConfig};
