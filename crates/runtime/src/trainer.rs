use std::time::{Duration, Instant};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Configuration for the parallel training loop.
#[derive(Clone, Debug)]
pub struct TrainerConfig {
    /// Number of full passes over the dataset.
    pub epochs: usize,

    /// Number of examples per update batch.
    pub batch_size: usize,

    /// Requested logical parallelism.
    ///
    /// The current rayon-backed implementation uses the process-level rayon
    /// pool. A future implementation can use this value to construct a scoped
    /// pool for stricter control.
    pub parallelism: usize,
}

impl Default for TrainerConfig {
    /// Create a conservative default training configuration.
    fn default() -> Self {
        let parallelism = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        Self { epochs: 1, batch_size: 32, parallelism }
    }
}

/// Summary statistics for one completed epoch.
#[derive(Clone, Debug)]
pub struct EpochStats {
    /// Zero-based epoch index.
    pub epoch: usize,

    /// Number of examples seen in the epoch.
    pub examples: usize,

    /// Mean loss across examples.
    pub mean_loss: f32,

    /// Wall-clock elapsed time for the epoch.
    pub elapsed: Duration,
}

/// Minimal training contract.
///
/// The runtime controls batching and parallel map/reduce. The learner controls
/// how a single example is evaluated and how a reduced batch update is applied.
pub trait Learner<D>: Send + Sync {
    /// Batch-level update representation produced by the learner.
    type BatchUpdate: Send;

    /// Evaluate one datum and produce its scalar loss plus local update.
    fn loss_and_update(&self, datum: &D) -> anyhow::Result<(f32, Self::BatchUpdate)>;

    /// Merge per-example updates into one batch update.
    fn merge_updates(&self, updates: Vec<Self::BatchUpdate>) -> anyhow::Result<Self::BatchUpdate>;

    /// Mutate the learner by applying a merged update.
    fn apply_update(&mut self, update: Self::BatchUpdate) -> anyhow::Result<()>;
}

/// Parallel map/reduce trainer.
pub struct ParallelTrainer {
    config: TrainerConfig,
}

impl ParallelTrainer {
    /// Create a trainer with explicit configuration.
    pub fn new(config: TrainerConfig) -> Self {
        Self { config }
    }

    /// Train a learner on a slice of data and return epoch statistics.
    pub fn train<D, L>(&self, learner: &mut L, data: &[D]) -> anyhow::Result<Vec<EpochStats>>
    where
        D: Send + Sync,
        L: Learner<D>,
    {
        if self.config.batch_size == 0 {
            anyhow::bail!("batch_size must be non-zero");
        }
        let mut stats = Vec::with_capacity(self.config.epochs);
        for epoch in 0..self.config.epochs {
            let start = Instant::now();
            let mut losses = Vec::new();

            for batch in data.chunks(self.config.batch_size) {
                #[cfg(feature = "parallel")]
                let results: Vec<_> = batch.par_iter().map(|d| learner.loss_and_update(d)).collect();

                #[cfg(not(feature = "parallel"))]
                let results: Vec<_> = batch.iter().map(|d| learner.loss_and_update(d)).collect();

                let mut updates = Vec::with_capacity(results.len());
                for item in results {
                    let (loss, update) = item?;
                    losses.push(loss);
                    updates.push(update);
                }
                let merged = learner.merge_updates(updates)?;
                learner.apply_update(merged)?;
            }

            let mean_loss = if losses.is_empty() { 0.0 } else { losses.iter().sum::<f32>() / losses.len() as f32 };
            stats.push(EpochStats { epoch, examples: data.len(), mean_loss, elapsed: start.elapsed() });
        }
        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug)]
    struct DummyUpdate(f32);

    struct DummyLearner;

    impl Learner<f32> for DummyLearner {
        type BatchUpdate = DummyUpdate;

        fn loss_and_update(&self, _datum: &f32) -> anyhow::Result<(f32, Self::BatchUpdate)> {
            Ok((1.0, DummyUpdate(0.5)))
        }

        fn merge_updates(&self, updates: Vec<Self::BatchUpdate>) -> anyhow::Result<Self::BatchUpdate> {
            let sum = updates.iter().map(|u| u.0).sum::<f32>();
            Ok(DummyUpdate(sum))
        }

        fn apply_update(&mut self, _update: Self::BatchUpdate) -> anyhow::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn test_trainer_config_default() {
        let config = TrainerConfig::default();
        assert_eq!(config.epochs, 1);
        assert_eq!(config.batch_size, 32);
        assert!(config.parallelism > 0);
    }

    #[test]
    fn test_parallel_trainer_new() {
        let config = TrainerConfig { epochs: 1, batch_size: 2, parallelism: 1 };
        let trainer = ParallelTrainer::new(config);
        let mut learner = DummyLearner;
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let stats = trainer.train(&mut learner, &data).unwrap();
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].epoch, 0);
        assert_eq!(stats[0].examples, 4);
    }

    #[test]
    fn test_parallel_trainer_zero_batch_size_fails() {
        let config = TrainerConfig { epochs: 1, batch_size: 0, parallelism: 1 };
        let trainer = ParallelTrainer::new(config);
        let mut learner = DummyLearner;
        let result = trainer.train(&mut learner, &[1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_epoch_stats_debug() {
        let stats = EpochStats { epoch: 0, examples: 10, mean_loss: 0.5, elapsed: Duration::from_secs(1) };
        let debug = format!("{:?}", stats);
        assert!(debug.contains("0.5"));
    }
}
