//! Runtime utilities for parallel training and inference orchestration.

mod inference;
#[cfg(feature = "training")]
pub mod serious_training;
mod trainer;

pub use inference::{InferencePool, InferenceRequest, InferenceResponse};
#[cfg(feature = "training")]
pub use serious_training::{SeriousTrainingConfig, SeriousTrainingOutcome, train_synthetic_classification};
pub use trainer::{EpochStats, Learner, ParallelTrainer, TrainerConfig};
