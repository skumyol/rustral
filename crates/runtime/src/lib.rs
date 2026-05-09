//! Runtime utilities for parallel training and inference orchestration.

mod inference;
#[cfg(feature = "training")]
pub mod model_io;
#[cfg(feature = "training")]
pub mod serious_training;
#[cfg(feature = "training")]
pub mod tape_trainer;
mod trainer;
#[cfg(all(feature = "training", feature = "tui"))]
pub mod tui_hook;

pub use inference::{InferencePool, InferenceRequest, InferenceResponse};
#[cfg(feature = "training")]
pub use model_io::{load_model, save_model};
#[cfg(feature = "training")]
pub use serious_training::{train_synthetic_classification, SeriousTrainingConfig, SeriousTrainingOutcome};
#[cfg(feature = "training")]
pub use tape_trainer::{SupervisedTapeModel, TapeTrainer, TapeTrainerConfig, TrainingReport};
pub use trainer::{EpochStats, Learner, ParallelTrainer, TrainerConfig};
