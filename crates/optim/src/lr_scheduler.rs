//! Learning Rate Schedulers for Rustral
//!
//! Provides various learning rate scheduling strategies for stable and
//! efficient training of neural networks.
//!
//! # Available Schedulers
//!
//! | Scheduler | Use Case |
//! |-----------|----------|
//! | `ConstantLR` | Baseline, testing |
//! | `LinearWarmup` | Prevent early training instability |
//! | `CosineAnnealingLR` | Smooth decay to minimum |
//! | `StepDecay` | Milestone-based reduction |
//! | `ExponentialLR` | Continuous decay |
//! | `PlateauLR` | Reduce when loss plateaus |
//! | `OneCycleLR` | Super-convergence training |
//! | `PolynomialLR` | Polynomial decay |
//! | `WarmupCosine` | Linear warmup + cosine decay (most popular) |
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use rustral_optim::lr_scheduler::{WarmupCosine, LRScheduler};
//!
//! let mut scheduler = WarmupCosine::new(1e-4, 1e-6, 1000, 100000);
//!
//! for step in 0..100000 {
//!     let lr = scheduler.get_lr(step);
//!     optimizer.set_lr(lr);
//!     // ... train step ...
//! }
//! ```

/// Trait for all learning rate schedulers.
pub trait LRScheduler {
    /// Get the learning rate for a given step.
    fn get_lr(&self, step: u64) -> f64;

    /// Get the current learning rate (uses internal step counter).
    fn current_lr(&self) -> f64;

    /// Step the internal counter and return new LR.
    fn step(&mut self) -> f64;

    /// Reset to initial state.
    fn reset(&mut self);

    /// Get the scheduler name.
    fn name(&self) -> &str;
}

/// Constant learning rate (no scheduling).
pub struct ConstantLR {
    lr: f64,
}

impl ConstantLR {
    pub fn new(lr: f64) -> Self {
        Self { lr }
    }
}

impl LRScheduler for ConstantLR {
    fn get_lr(&self, _step: u64) -> f64 {
        self.lr
    }

    fn current_lr(&self) -> f64 {
        self.lr
    }

    fn step(&mut self) -> f64 {
        self.lr
    }

    fn reset(&mut self) {}

    fn name(&self) -> &str {
        "constant"
    }
}

/// Linear warmup scheduler.
///
/// LR increases linearly from `initial_lr` to `peak_lr` over `warmup_steps`.
pub struct LinearWarmup {
    initial_lr: f64,
    peak_lr: f64,
    warmup_steps: u64,
    current_step: u64,
}

impl LinearWarmup {
    pub fn new(initial_lr: f64, peak_lr: f64, warmup_steps: u64) -> Self {
        Self { initial_lr, peak_lr, warmup_steps, current_step: 0 }
    }
}

impl LRScheduler for LinearWarmup {
    fn get_lr(&self, step: u64) -> f64 {
        if step >= self.warmup_steps {
            self.peak_lr
        } else {
            let progress = step as f64 / self.warmup_steps as f64;
            self.initial_lr + (self.peak_lr - self.initial_lr) * progress
        }
    }

    fn current_lr(&self) -> f64 {
        self.get_lr(self.current_step)
    }

    fn step(&mut self) -> f64 {
        let lr = self.get_lr(self.current_step);
        self.current_step += 1;
        lr
    }

    fn reset(&mut self) {
        self.current_step = 0;
    }

    fn name(&self) -> &str {
        "linear_warmup"
    }
}

/// Cosine annealing scheduler.
///
/// LR decays from `initial_lr` to `min_lr` following a cosine curve.
pub struct CosineAnnealingLR {
    initial_lr: f64,
    min_lr: f64,
    total_steps: u64,
    current_step: u64,
}

impl CosineAnnealingLR {
    pub fn new(initial_lr: f64, min_lr: f64, total_steps: u64) -> Self {
        Self { initial_lr, min_lr, total_steps, current_step: 0 }
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn get_lr(&self, step: u64) -> f64 {
        if step >= self.total_steps {
            self.min_lr
        } else {
            let progress = step as f64 / self.total_steps as f64;
            let cosine = (1.0 + (std::f64::consts::PI * progress).cos()) / 2.0;
            self.min_lr + (self.initial_lr - self.min_lr) * cosine
        }
    }

    fn current_lr(&self) -> f64 {
        self.get_lr(self.current_step)
    }

    fn step(&mut self) -> f64 {
        let lr = self.get_lr(self.current_step);
        self.current_step += 1;
        lr
    }

    fn reset(&mut self) {
        self.current_step = 0;
    }

    fn name(&self) -> &str {
        "cosine_annealing"
    }
}

/// Warmup + Cosine annealing (most popular for transformers).
///
/// Combines linear warmup with cosine decay.
pub struct WarmupCosine {
    initial_lr: f64,
    peak_lr: f64,
    min_lr: f64,
    warmup_steps: u64,
    total_steps: u64,
    current_step: u64,
}

impl WarmupCosine {
    pub fn new(peak_lr: f64, min_lr: f64, warmup_steps: u64, total_steps: u64) -> Self {
        Self { initial_lr: 0.0, peak_lr, min_lr, warmup_steps, total_steps, current_step: 0 }
    }

    /// Create with custom initial LR (for fine-tuning where you don't start from 0).
    pub fn with_initial_lr(mut self, initial_lr: f64) -> Self {
        self.initial_lr = initial_lr;
        self
    }
}

impl LRScheduler for WarmupCosine {
    fn get_lr(&self, step: u64) -> f64 {
        if step < self.warmup_steps {
            // Linear warmup
            let progress = step as f64 / self.warmup_steps as f64;
            self.initial_lr + (self.peak_lr - self.initial_lr) * progress
        } else {
            // Cosine decay
            let decay_steps = self.total_steps - self.warmup_steps;
            let decay_progress = (step - self.warmup_steps) as f64 / decay_steps as f64;
            let cosine = (1.0 + (std::f64::consts::PI * decay_progress).cos()) / 2.0;
            self.min_lr + (self.peak_lr - self.min_lr) * cosine
        }
    }

    fn current_lr(&self) -> f64 {
        self.get_lr(self.current_step)
    }

    fn step(&mut self) -> f64 {
        let lr = self.get_lr(self.current_step);
        self.current_step += 1;
        lr
    }

    fn reset(&mut self) {
        self.current_step = 0;
    }

    fn name(&self) -> &str {
        "warmup_cosine"
    }
}

/// Step decay scheduler.
///
/// Reduce LR by `decay_factor` every `step_size` steps.
pub struct StepDecay {
    initial_lr: f64,
    decay_factor: f64,
    step_size: u64,
    current_step: u64,
}

impl StepDecay {
    pub fn new(initial_lr: f64, decay_factor: f64, step_size: u64) -> Self {
        assert!(decay_factor > 0.0 && decay_factor < 1.0, "Decay factor must be in (0, 1)");
        Self { initial_lr, decay_factor, step_size, current_step: 0 }
    }
}

impl LRScheduler for StepDecay {
    fn get_lr(&self, step: u64) -> f64 {
        let num_decays = step / self.step_size;
        self.initial_lr * self.decay_factor.powi(num_decays as i32)
    }

    fn current_lr(&self) -> f64 {
        self.get_lr(self.current_step)
    }

    fn step(&mut self) -> f64 {
        let lr = self.get_lr(self.current_step);
        self.current_step += 1;
        lr
    }

    fn reset(&mut self) {
        self.current_step = 0;
    }

    fn name(&self) -> &str {
        "step_decay"
    }
}

/// Exponential decay scheduler.
///
/// Continuous exponential decay: `lr = initial_lr * exp(-step * decay_rate)`.
pub struct ExponentialLR {
    initial_lr: f64,
    decay_rate: f64,
    current_step: u64,
}

impl ExponentialLR {
    pub fn new(initial_lr: f64, decay_rate: f64) -> Self {
        assert!(decay_rate > 0.0, "Decay rate must be positive");
        Self { initial_lr, decay_rate, current_step: 0 }
    }
}

impl LRScheduler for ExponentialLR {
    fn get_lr(&self, step: u64) -> f64 {
        self.initial_lr * (-(step as f64) * self.decay_rate).exp()
    }

    fn current_lr(&self) -> f64 {
        self.get_lr(self.current_step)
    }

    fn step(&mut self) -> f64 {
        let lr = self.get_lr(self.current_step);
        self.current_step += 1;
        lr
    }

    fn reset(&mut self) {
        self.current_step = 0;
    }

    fn name(&self) -> &str {
        "exponential"
    }
}

/// Reduce on plateau scheduler.
///
/// Reduces LR when loss stops improving.
pub struct PlateauLR {
    initial_lr: f64,
    min_lr: f64,
    decay_factor: f64,
    patience: u64,
    cooldown: u64,
    best_loss: f64,
    bad_steps: u64,
    cooldown_steps: u64,
    current_step: u64,
    num_decays: u64,
}

impl PlateauLR {
    pub fn new(initial_lr: f64, min_lr: f64, decay_factor: f64, patience: u64) -> Self {
        Self {
            initial_lr,
            min_lr,
            decay_factor,
            patience,
            cooldown: 0,
            best_loss: f64::INFINITY,
            bad_steps: 0,
            cooldown_steps: 0,
            current_step: 0,
            num_decays: 0,
        }
    }

    pub fn with_cooldown(mut self, cooldown: u64) -> Self {
        self.cooldown = cooldown;
        self
    }

    /// Report loss and get updated LR.
    pub fn report_loss(&mut self, loss: f64) -> f64 {
        if self.cooldown_steps > 0 {
            self.cooldown_steps -= 1;
        } else if loss < self.best_loss {
            self.best_loss = loss;
            self.bad_steps = 0;
        } else {
            self.bad_steps += 1;
        }

        if self.bad_steps >= self.patience {
            self.num_decays += 1;
            self.bad_steps = 0;
            self.cooldown_steps = self.cooldown;
        }

        self.current_lr()
    }
}

impl LRScheduler for PlateauLR {
    fn get_lr(&self, _step: u64) -> f64 {
        let lr = self.initial_lr * self.decay_factor.powi(self.num_decays as i32);
        lr.max(self.min_lr)
    }

    fn current_lr(&self) -> f64 {
        self.get_lr(self.current_step)
    }

    fn step(&mut self) -> f64 {
        let lr = self.get_lr(self.current_step);
        self.current_step += 1;
        lr
    }

    fn reset(&mut self) {
        self.current_step = 0;
        self.num_decays = 0;
        self.bad_steps = 0;
        self.best_loss = f64::INFINITY;
        self.cooldown_steps = 0;
    }

    fn name(&self) -> &str {
        "plateau"
    }
}

/// One-cycle learning rate scheduler (Smith 2017).
///
/// LR increases then decreases following a single cycle, with momentum adjustment.
pub struct OneCycleLR {
    max_lr: f64,
    min_lr: f64,
    total_steps: u64,
    current_step: u64,
}

impl OneCycleLR {
    pub fn new(max_lr: f64, total_steps: u64) -> Self {
        Self { max_lr, min_lr: max_lr / 10.0, total_steps, current_step: 0 }
    }

    pub fn with_min_lr(mut self, min_lr: f64) -> Self {
        self.min_lr = min_lr;
        self
    }
}

impl LRScheduler for OneCycleLR {
    fn get_lr(&self, step: u64) -> f64 {
        if step >= self.total_steps {
            return self.min_lr;
        }

        let progress = step as f64 / self.total_steps as f64;

        if progress < 0.5 {
            // Warmup phase (first half)
            let warmup_progress = progress * 2.0;
            self.min_lr + (self.max_lr - self.min_lr) * warmup_progress
        } else {
            // Decay phase (second half)
            let decay_progress = (progress - 0.5) * 2.0;
            let cosine = (1.0 + (std::f64::consts::PI * decay_progress).cos()) / 2.0;
            self.min_lr + (self.max_lr - self.min_lr) * cosine
        }
    }

    fn current_lr(&self) -> f64 {
        self.get_lr(self.current_step)
    }

    fn step(&mut self) -> f64 {
        let lr = self.get_lr(self.current_step);
        self.current_step += 1;
        lr
    }

    fn reset(&mut self) {
        self.current_step = 0;
    }

    fn name(&self) -> &str {
        "one_cycle"
    }
}

/// Polynomial decay scheduler.
///
/// LR decays as `(1 - step/total_steps)^power`.
pub struct PolynomialLR {
    initial_lr: f64,
    min_lr: f64,
    total_steps: u64,
    power: f64,
    current_step: u64,
}

impl PolynomialLR {
    pub fn new(initial_lr: f64, min_lr: f64, total_steps: u64, power: f64) -> Self {
        assert!(power > 0.0, "Power must be positive");
        Self { initial_lr, min_lr, total_steps, power, current_step: 0 }
    }
}

impl LRScheduler for PolynomialLR {
    fn get_lr(&self, step: u64) -> f64 {
        if step >= self.total_steps {
            return self.min_lr;
        }
        let progress = step as f64 / self.total_steps as f64;
        let decay = (1.0 - progress).powf(self.power);
        self.min_lr + (self.initial_lr - self.min_lr) * decay
    }

    fn current_lr(&self) -> f64 {
        self.get_lr(self.current_step)
    }

    fn step(&mut self) -> f64 {
        let lr = self.get_lr(self.current_step);
        self.current_step += 1;
        lr
    }

    fn reset(&mut self) {
        self.current_step = 0;
    }

    fn name(&self) -> &str {
        "polynomial"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_lr() {
        let mut scheduler = ConstantLR::new(0.001);
        assert_eq!(scheduler.step(), 0.001);
        assert_eq!(scheduler.step(), 0.001);
        assert_eq!(scheduler.get_lr(1000), 0.001);
    }

    #[test]
    fn test_linear_warmup() {
        let scheduler = LinearWarmup::new(0.0, 0.001, 100);

        assert_eq!(scheduler.get_lr(0), 0.0);
        assert_eq!(scheduler.get_lr(50), 0.0005);
        assert_eq!(scheduler.get_lr(100), 0.001);
        assert_eq!(scheduler.get_lr(200), 0.001); // Stays at peak
    }

    #[test]
    fn test_cosine_annealing() {
        let scheduler = CosineAnnealingLR::new(0.001, 1e-6, 1000);

        assert_eq!(scheduler.get_lr(0), 0.001);
        assert!((scheduler.get_lr(500) - 0.0005005).abs() < 0.001);
        assert_eq!(scheduler.get_lr(1000), 1e-6);
    }

    #[test]
    fn test_warmup_cosine() {
        let scheduler = WarmupCosine::new(0.001, 1e-6, 100, 1000);

        // Warmup phase
        assert_eq!(scheduler.get_lr(0), 0.0);
        assert_eq!(scheduler.get_lr(50), 0.0005);
        assert_eq!(scheduler.get_lr(100), 0.001);

        // Cosine phase
        assert!(scheduler.get_lr(500) < 0.001);
        assert!(scheduler.get_lr(500) > 1e-6);
        assert_eq!(scheduler.get_lr(1000), 1e-6);
    }

    #[test]
    fn test_step_decay() {
        let scheduler = StepDecay::new(0.001, 0.5, 100);

        assert_eq!(scheduler.get_lr(0), 0.001);
        assert_eq!(scheduler.get_lr(99), 0.001);
        assert_eq!(scheduler.get_lr(100), 0.0005);
        assert_eq!(scheduler.get_lr(200), 0.00025);
    }

    #[test]
    fn test_exponential_decay() {
        let scheduler = ExponentialLR::new(0.001, 0.001);

        assert_eq!(scheduler.get_lr(0), 0.001);
        assert!(scheduler.get_lr(1000) < 0.001);
        assert!(scheduler.get_lr(1000) > 0.0);
    }

    #[test]
    fn test_plateau() {
        let mut scheduler = PlateauLR::new(0.001, 1e-6, 0.5, 3);

        // Initial LR
        assert_eq!(scheduler.current_lr(), 0.001);

        // Report improving loss
        scheduler.report_loss(1.0);
        scheduler.report_loss(0.9);
        scheduler.report_loss(0.8);
        assert_eq!(scheduler.current_lr(), 0.001); // No decay yet

        // Report worse loss 3 times (patience=3)
        scheduler.report_loss(0.85);
        scheduler.report_loss(0.9);
        scheduler.report_loss(0.95);
        assert_eq!(scheduler.current_lr(), 0.0005); // Decayed once
    }

    #[test]
    fn test_one_cycle() {
        let scheduler = OneCycleLR::new(0.01, 1000);

        assert!((scheduler.get_lr(0) - 0.001).abs() < 1e-9); // min_lr = max/10
        assert!((scheduler.get_lr(500) - 0.01).abs() < 1e-9); // Peak at midpoint
        assert!((scheduler.get_lr(1000) - 0.001).abs() < 1e-9); // Back to min
    }

    #[test]
    fn test_polynomial() {
        let scheduler = PolynomialLR::new(0.001, 1e-6, 1000, 2.0);

        assert_eq!(scheduler.get_lr(0), 0.001);
        assert_eq!(scheduler.get_lr(1000), 1e-6);

        // At 50% progress, should be 25% of the way (power=2)
        let mid = scheduler.get_lr(500);
        assert!(mid > 1e-6 && mid < 0.001);
    }

    #[test]
    fn test_scheduler_step_counter() {
        let mut scheduler = WarmupCosine::new(0.001, 1e-6, 100, 1000);

        for _ in 0..50 {
            scheduler.step();
        }
        assert_eq!(scheduler.current_step, 50);

        scheduler.reset();
        assert_eq!(scheduler.current_step, 0);
        assert_eq!(scheduler.current_lr(), 0.0); // At step 0, warmup starts from 0
    }

    #[test]
    fn test_scheduler_names() {
        assert_eq!(ConstantLR::new(0.001).name(), "constant");
        assert_eq!(LinearWarmup::new(0.0, 0.001, 100).name(), "linear_warmup");
        assert_eq!(CosineAnnealingLR::new(0.001, 1e-6, 1000).name(), "cosine_annealing");
        assert_eq!(StepDecay::new(0.001, 0.5, 100).name(), "step_decay");
        assert_eq!(ExponentialLR::new(0.001, 0.001).name(), "exponential");
        assert_eq!(PlateauLR::new(0.001, 1e-6, 0.5, 3).name(), "plateau");
        assert_eq!(OneCycleLR::new(0.01, 1000).name(), "one_cycle");
        assert_eq!(PolynomialLR::new(0.001, 1e-6, 1000, 2.0).name(), "polynomial");
        assert_eq!(WarmupCosine::new(0.001, 1e-6, 100, 1000).name(), "warmup_cosine");
    }

    #[test]
    fn test_one_cycle_with_min_lr() {
        let scheduler = OneCycleLR::new(0.01, 1000).with_min_lr(0.002);
        assert_eq!(scheduler.min_lr, 0.002);
    }

    #[test]
    fn test_constant_lr_current_and_reset() {
        let mut scheduler = ConstantLR::new(0.001);
        assert_eq!(scheduler.current_lr(), 0.001);
        scheduler.step();
        scheduler.reset();
        assert_eq!(scheduler.current_lr(), 0.001);
    }

    #[test]
    fn test_exponential_lr_current() {
        let mut scheduler = ExponentialLR::new(0.001, 0.001);
        assert_eq!(scheduler.current_lr(), 0.001);
        scheduler.step();
        assert!(scheduler.current_lr() < 0.001);
    }

    #[test]
    fn test_polynomial_lr_step_and_reset() {
        let mut scheduler = PolynomialLR::new(0.001, 1e-6, 100, 2.0);
        let lr1 = scheduler.step();
        let lr2 = scheduler.step();
        assert!(lr2 <= lr1);
        scheduler.reset();
        assert_eq!(scheduler.current_step, 0);
    }
}
