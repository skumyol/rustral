//! North-star high-level training API.
//!
//! This wraps [`crate::tape_trainer::TapeTrainer`] with a builder-style surface:
//!
//! ```rust,ignore
//! use rustral_optim::Adam;
//! use rustral_runtime::Trainer;
//!
//! let report = Trainer::classification(Adam::new(1e-3))
//!     .epochs(3)
//!     .batch_size(64)
//!     .fit(&backend, &mut model, &train)?;
//! ```

use std::marker::PhantomData;

use rustral_core::Backend;
use rustral_optim::Optimizer;

use crate::tape_trainer::{SupervisedTapeModel, TapeTrainer, TapeTrainerConfig, TrainingReport};

pub struct Classification;
pub struct Regression;

/// High-level trainer facade with a builder-style API.
pub struct Trainer<O, Task> {
    cfg: TapeTrainerConfig,
    optimizer: O,
    _task: PhantomData<Task>,
}

impl<O> Trainer<O, Classification> {
    pub fn classification(optimizer: O) -> Self {
        Self { cfg: TapeTrainerConfig::default(), optimizer, _task: PhantomData }
    }
}

impl<O> Trainer<O, Regression> {
    pub fn regression(optimizer: O) -> Self {
        Self { cfg: TapeTrainerConfig::default(), optimizer, _task: PhantomData }
    }
}

impl<O, Task> Trainer<O, Task> {
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.cfg.epochs = epochs;
        self
    }

    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.cfg.batch_size = batch_size;
        self
    }

    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.cfg.shuffle = shuffle;
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.cfg.seed = seed;
        self
    }
}

impl<O> Trainer<O, Regression> {
    pub fn fit<B, M, X, Y>(self, backend: &B, model: &mut M, train: &[(X, Y)]) -> anyhow::Result<TrainingReport>
    where
        B: Backend,
        B::Tensor: Clone,
        O: Optimizer<B>,
        M: SupervisedTapeModel<B, X, Y>,
        X: Clone,
        Y: Clone,
    {
        let mut t = TapeTrainer::new(self.cfg, self.optimizer);
        t.fit(backend, model, train)
    }
}

impl<O> Trainer<O, Classification> {
    pub fn fit<B, M, X, Y>(self, backend: &B, model: &mut M, train: &[(X, Y)]) -> anyhow::Result<TrainingReport>
    where
        B: Backend,
        B::Tensor: Clone,
        O: Optimizer<B>,
        M: SupervisedTapeModel<B, X, Y>,
        X: Clone,
        Y: Copy + Into<usize>,
    {
        let mut t = TapeTrainer::new(self.cfg, self.optimizer);
        t.fit_classification(backend, model, train)
    }
}

