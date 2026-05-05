//! Runtime utilities for parallel training and inference orchestration.

mod inference;
#[cfg(feature = "training")]
pub mod serious_training;
#[cfg(feature = "training")]
pub mod tape_trainer;
mod trainer;

pub use inference::{InferencePool, InferenceRequest, InferenceResponse};
#[cfg(feature = "training")]
pub use serious_training::{train_synthetic_classification, SeriousTrainingConfig, SeriousTrainingOutcome};
#[cfg(feature = "training")]
pub use tape_trainer::{TapeTrainer, TapeTrainerConfig};
pub use trainer::{EpochStats, Learner, ParallelTrainer, TrainerConfig};
