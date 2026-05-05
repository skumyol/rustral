//! Search strategies for kernel configuration tuning.

use crate::kernel_config::{ConfigSpace, KernelConfig, WorkgroupConfig};
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::HashMap;

/// Search strategy trait.
pub trait SearchStrategy {
    /// Get the next configuration to try.
    fn next_config(&mut self, space: &ConfigSpace) -> Option<KernelConfig>;

    /// Report results for a configuration.
    fn report_result(&mut self, config: &KernelConfig, time_us: f64);

    /// Check if search is complete.
    fn is_complete(&self) -> bool;

    /// Get the best configuration found so far.
    fn best_config(&self) -> Option<&KernelConfig>;

    /// Get the best time found so far.
    fn best_time(&self) -> f64;
}

/// Exhaustive grid search.
pub struct GridSearch {
    configs: Vec<KernelConfig>,
    current: usize,
    best: Option<KernelConfig>,
    best_time: f64,
}

impl GridSearch {
    /// Create a new grid search over a configuration space.
    pub fn new(space: &ConfigSpace) -> Self {
        let configs = enumerate_configs(space);
        Self {
            configs,
            current: 0,
            best: None,
            best_time: f64::MAX,
        }
    }

    /// Get all configurations that will be tried.
    pub fn all_configs(&self) -> &[KernelConfig] {
        &self.configs
    }
}

impl SearchStrategy for GridSearch {
    fn next_config(&mut self, _space: &ConfigSpace) -> Option<KernelConfig> {
        if self.current < self.configs.len() {
            let config = self.configs[self.current].clone();
            self.current += 1;
            Some(config)
        } else {
            None
        }
    }

    fn report_result(&mut self, config: &KernelConfig, time_us: f64) {
        if time_us < self.best_time {
            self.best_time = time_us;
            self.best = Some(config.clone());
        }
    }

    fn is_complete(&self) -> bool {
        self.current >= self.configs.len()
    }

    fn best_config(&self) -> Option<&KernelConfig> {
        self.best.as_ref()
    }

    fn best_time(&self) -> f64 {
        self.best_time
    }
}

/// Random search with optional early stopping.
pub struct RandomSearch {
    space: ConfigSpace,
    max_iterations: usize,
    current: usize,
    best: Option<KernelConfig>,
    best_time: f64,
    rng: rand::rngs::StdRng,
    early_stop_threshold: f64,
    no_improvement_count: usize,
    max_no_improvement: usize,
}

impl RandomSearch {
    /// Create a new random search.
    pub fn new(space: ConfigSpace, max_iterations: usize, seed: u64) -> Self {
        Self {
            space,
            max_iterations,
            current: 0,
            best: None,
            best_time: f64::MAX,
            rng: rand::SeedableRng::seed_from_u64(seed),
            early_stop_threshold: 0.95, // Stop if within 5% of best
            no_improvement_count: 0,
            max_no_improvement: 20,
        }
    }

    /// Set early stopping parameters.
    pub fn with_early_stop(mut self, threshold: f64, max_no_improvement: usize) -> Self {
        self.early_stop_threshold = threshold;
        self.max_no_improvement = max_no_improvement;
        self
    }

    /// Sample a random configuration from the space.
    fn sample_config(&mut self) -> KernelConfig {
        let wg = *self.space.workgroup_sizes.choose(&mut self.rng)
            .unwrap_or(&WorkgroupConfig::default());

        let alg = self.space.algorithms.choose(&mut self.rng)
            .cloned()
            .unwrap_or_else(|| crate::kernel_config::AlgorithmConfig::Elementwise);

        let mem = self.space.memory_configs.choose(&mut self.rng)
            .cloned()
            .unwrap_or_default();

        let mut params = HashMap::new();
        for (key, values) in &self.space.param_ranges {
            if let Some(&val) = values.choose(&mut self.rng) {
                params.insert(key.clone(), val);
            }
        }

        KernelConfig {
            workgroup: wg,
            algorithm: alg,
            memory: mem,
            params,
        }
    }
}

impl SearchStrategy for RandomSearch {
    fn next_config(&mut self, _space: &ConfigSpace) -> Option<KernelConfig> {
        if self.current >= self.max_iterations {
            return None;
        }

        if self.no_improvement_count >= self.max_no_improvement {
            return None; // Early stop
        }

        self.current += 1;
        Some(self.sample_config())
    }

    fn report_result(&mut self, config: &KernelConfig, time_us: f64) {
        let improved = if time_us < self.best_time {
            self.best_time = time_us;
            self.best = Some(config.clone());
            self.no_improvement_count = 0;
            true
        } else {
            self.no_improvement_count += 1;
            false
        };

        // Check early stopping
        if !improved && self.best_time < f64::MAX {
            let ratio = self.best_time / time_us;
            if ratio >= self.early_stop_threshold {
                self.no_improvement_count += 1;
            }
        }
    }

    fn is_complete(&self) -> bool {
        self.current >= self.max_iterations ||
        self.no_improvement_count >= self.max_no_improvement
    }

    fn best_config(&self) -> Option<&KernelConfig> {
        self.best.as_ref()
    }

    fn best_time(&self) -> f64 {
        self.best_time
    }
}

/// Evolutionary search with mutation and crossover.
pub struct EvolutionarySearch {
    population_size: usize,
    elite_size: usize,
    mutation_rate: f64,
    generations: usize,
    current_gen: usize,
    population: Vec<(KernelConfig, f64)>,
    space: ConfigSpace,
    rng: rand::rngs::StdRng,
}

impl EvolutionarySearch {
    /// Create a new evolutionary search.
    pub fn new(space: ConfigSpace, population_size: usize, generations: usize, seed: u64) -> Self {
        let mut rng = rand::SeedableRng::seed_from_u64(seed);
        let mut population = Vec::with_capacity(population_size);

        // Initialize with random configs
        for _ in 0..population_size {
            let config = sample_random(&space, &mut rng);
            population.push((config, f64::MAX));
        }

        Self {
            population_size,
            elite_size: population_size / 4,
            mutation_rate: 0.1,
            generations,
            current_gen: 0,
            population,
            space,
            rng,
        }
    }

    /// Get configs to evaluate for current generation.
    pub fn current_configs(&self) -> &[KernelConfig] {
        // This would return references, but we need owned
        // For simplicity, return empty and handle differently
        &[]
    }

    /// Mutate a configuration.
    fn mutate(&mut self, config: &mut KernelConfig) {
        if self.rng.gen::<f64>() < self.mutation_rate {
            // Mutate workgroup size
            if let Some(&wg) = self.space.workgroup_sizes.choose(&mut self.rng) {
                config.workgroup = wg;
            }
        }

        if self.rng.gen::<f64>() < self.mutation_rate {
            // Mutate algorithm
            if let Some(alg) = self.space.algorithms.choose(&mut self.rng) {
                config.algorithm = alg.clone();
            }
        }

        // Mutate params
        for (key, values) in &self.space.param_ranges {
            if self.rng.gen::<f64>() < self.mutation_rate {
                if let Some(&val) = values.choose(&mut self.rng) {
                    config.params.insert(key.clone(), val);
                }
            }
        }
    }

    /// Crossover two configurations.
    fn crossover(&mut self, a: &KernelConfig, b: &KernelConfig) -> KernelConfig {
        KernelConfig {
            workgroup: if self.rng.gen::<bool>() { a.workgroup } else { b.workgroup },
            algorithm: if self.rng.gen::<bool>() { a.algorithm.clone() } else { b.algorithm.clone() },
            memory: if self.rng.gen::<bool>() { a.memory } else { b.memory },
            params: a.params.clone(), // Simplified: take from a
        }
    }

    /// Evolve to next generation.
    fn evolve(&mut self) {
        // Sort by fitness (lower time is better)
        self.population.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Keep elite
        let mut new_population = self.population[..self.elite_size].to_vec();

        // Generate offspring
        while new_population.len() < self.population_size {
            // Tournament selection - clone to avoid borrow issues
            let parent1 = self.tournament_select().0.clone();
            let parent2 = self.tournament_select().0.clone();

            let mut child = self.crossover(&parent1, &parent2);
            self.mutate(&mut child);

            new_population.push((child, f64::MAX));
        }

        self.population = new_population;
        self.current_gen += 1;
    }

    fn tournament_select(&mut self) -> &(KernelConfig, f64) {
        // Simple random selection from top 50%
        let idx = self.rng.gen_range(0..self.population_size / 2);
        &self.population[idx]
    }
}

/// Bayesian optimization (simplified version).
pub struct BayesianSearch {
    space: ConfigSpace,
    max_iterations: usize,
    current: usize,
    observations: Vec<(KernelConfig, f64)>,
    best: Option<KernelConfig>,
    best_time: f64,
    exploration_factor: f64,
}

impl BayesianSearch {
    /// Create a new Bayesian search.
    pub fn new(space: ConfigSpace, max_iterations: usize) -> Self {
        Self {
            space,
            max_iterations,
            current: 0,
            observations: Vec::new(),
            best: None,
            best_time: f64::MAX,
            exploration_factor: 0.1,
        }
    }

    /// Compute expected improvement for a configuration.
    fn expected_improvement(&self, _config: &KernelConfig) -> f64 {
        // Simplified EI computation
        // In practice, would use Gaussian Process surrogate model
        1.0 / (self.observations.len() as f64 + 1.0)
    }

    /// Find configuration with highest acquisition function value.
    fn acquisition_max(&self) -> KernelConfig {
        // Simplified: use a few random samples and pick best EI
        // In practice, would optimize the acquisition function
        let mut rng = rand::thread_rng();

        let mut best_config = None;
        let mut best_ei = 0.0;

        for _ in 0..100 {
            let config = sample_random(&self.space, &mut rng);
            let ei = self.expected_improvement(&config);
            if ei > best_ei {
                best_ei = ei;
                best_config = Some(config);
            }
        }

        best_config.unwrap_or_else(KernelConfig::default)
    }
}

impl SearchStrategy for BayesianSearch {
    fn next_config(&mut self, _space: &ConfigSpace) -> Option<KernelConfig> {
        if self.current >= self.max_iterations {
            return None;
        }

        self.current += 1;

        // Initial random sampling
        if self.observations.len() < 5 {
            let mut rng = rand::thread_rng();
            return Some(sample_random(&self.space, &mut rng));
        }

        // Acquisition function optimization
        Some(self.acquisition_max())
    }

    fn report_result(&mut self, config: &KernelConfig, time_us: f64) {
        self.observations.push((config.clone(), time_us));

        if time_us < self.best_time {
            self.best_time = time_us;
            self.best = Some(config.clone());
        }
    }

    fn is_complete(&self) -> bool {
        self.current >= self.max_iterations
    }

    fn best_config(&self) -> Option<&KernelConfig> {
        self.best.as_ref()
    }

    fn best_time(&self) -> f64 {
        self.best_time
    }
}

/// Helper function to enumerate all configurations in a space.
fn enumerate_configs(space: &ConfigSpace) -> Vec<KernelConfig> {
    let mut configs = Vec::new();

    for wg in &space.workgroup_sizes {
        for alg in &space.algorithms {
            for mem in &space.memory_configs {
                // Generate all parameter combinations
                let param_combos = generate_param_combos(&space.param_ranges);
                for params in param_combos {
                    configs.push(KernelConfig {
                        workgroup: *wg,
                        algorithm: alg.clone(),
                        memory: *mem,
                        params,
                    });
                }
            }
        }
    }

    configs
}

/// Generate all combinations of parameters.
fn generate_param_combos(ranges: &HashMap<String, Vec<i32>>) -> Vec<HashMap<String, i32>> {
    let keys: Vec<_> = ranges.keys().cloned().collect();

    if keys.is_empty() {
        return vec![HashMap::new()];
    }

    let mut result = vec![HashMap::new()];

    for key in keys {
        let values = ranges.get(&key).unwrap();
        let mut new_result = Vec::new();

        for base in &result {
            for &val in values {
                let mut new_map = base.clone();
                new_map.insert(key.clone(), val);
                new_result.push(new_map);
            }
        }

        result = new_result;
    }

    result
}

/// Sample a random configuration from the space.
fn sample_random<R: Rng>(space: &ConfigSpace, rng: &mut R) -> KernelConfig {
    let wg = *space.workgroup_sizes.choose(rng)
        .unwrap_or(&WorkgroupConfig::default());

    let alg = space.algorithms.choose(rng)
        .cloned()
        .unwrap_or_else(|| crate::kernel_config::AlgorithmConfig::Elementwise);

    let mem = space.memory_configs.choose(rng)
        .cloned()
        .unwrap_or_default();

    let mut params = HashMap::new();
    for (key, values) in &space.param_ranges {
        if let Some(&val) = values.choose(rng) {
            params.insert(key.clone(), val);
        }
    }

    KernelConfig {
        workgroup: wg,
        algorithm: alg,
        memory: mem,
        params,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_config::{AlgorithmConfig, MatmulAlgorithm};

    #[test]
    fn test_grid_search() {
        let space = ConfigSpace::matmul_reduced();
        let mut search = GridSearch::new(&space);

        assert!(!search.is_complete());

        let mut count = 0;
        while let Some(_config) = search.next_config(&space) {
            count += 1;
            search.report_result(&_config, 100.0 - count as f64);
        }

        assert!(search.is_complete());
        assert!(count > 0);
        assert!(search.best_config().is_some());
    }

    #[test]
    fn test_random_search() {
        let space = ConfigSpace::matmul_reduced();
        let mut search = RandomSearch::new(space.clone(), 20, 42);

        while let Some(config) = search.next_config(&space) {
            let time = rand::random::<f64>() * 100.0;
            search.report_result(&config, time);
        }

        assert!(search.best_time() < f64::MAX);
    }

    #[test]
    fn test_enumerate_configs() {
        let space = ConfigSpace::matmul_reduced();
        let configs = enumerate_configs(&space);

        assert!(!configs.is_empty());
        assert!(configs.len() <= space.size());
    }

    #[test]
    fn test_grid_search_all_configs() {
        let space = ConfigSpace::matmul_reduced();
        let search = GridSearch::new(&space);

        let all = search.all_configs();
        assert!(!all.is_empty());
        assert_eq!(all.len(), enumerate_configs(&space).len());
    }

    #[test]
    fn test_random_search_with_early_stop() {
        let space = ConfigSpace::matmul_reduced();
        let mut search = RandomSearch::new(space, 100, 42)
            .with_early_stop(0.99, 5);

        assert_eq!(search.max_iterations, 100);

        let mut count = 0;
        while let Some(_config) = search.next_config(&ConfigSpace::matmul_reduced()) {
            count += 1;
            if count > 10 {
                break;
            }
        }
        assert!(count > 0);
    }

    #[test]
    fn test_random_search_sample_config() {
        let space = ConfigSpace::matmul_reduced();
        let mut search = RandomSearch::new(space.clone(), 10, 42);

        let config = search.sample_config();
        assert!(space.workgroup_sizes.contains(&config.workgroup));
    }

    #[test]
    fn test_evolutionary_search() {
        let space = ConfigSpace::matmul_reduced();
        let mut search = EvolutionarySearch::new(space.clone(), 10, 5, 42);

        // Mutate a config
        let mut config = KernelConfig::default();
        search.mutate(&mut config);

        // Crossover
        let a = KernelConfig::default();
        let b = KernelConfig::default();
        let child = search.crossover(&a, &b);
        assert!(child.workgroup.total_threads() > 0);

        // Evolve
        search.evolve();
        assert_eq!(search.current_gen, 1);

        // Tournament select
        let selected = search.tournament_select();
        assert!(selected.1 >= 0.0);
    }

    #[test]
    fn test_evolutionary_search_current_configs() {
        let space = ConfigSpace::matmul_reduced();
        let search = EvolutionarySearch::new(space, 10, 5, 42);
        let configs = search.current_configs();
        assert!(configs.is_empty());
    }

    #[test]
    fn test_bayesian_search() {
        let space = ConfigSpace::matmul_reduced();
        let mut search = BayesianSearch::new(space, 10);

        // First few should be random
        let mut count = 0;
        while let Some(_config) = search.next_config(&ConfigSpace::matmul_reduced()) {
            count += 1;
            if count >= 3 {
                break;
            }
        }
        assert_eq!(count, 3);

        // Report a result
        let config = KernelConfig::default();
        search.report_result(&config, 100.0);
        assert!(search.best_config().is_some());
        assert_eq!(search.best_time(), 100.0);

        // Expected improvement
        let ei = search.expected_improvement(&config);
        assert!(ei > 0.0);

        // Acquisition max
        let _best = search.acquisition_max();
    }

    #[test]
    fn test_generate_param_combos_empty() {
        let ranges: std::collections::HashMap<String, Vec<i32>> = std::collections::HashMap::new();
        let combos = generate_param_combos(&ranges);
        assert_eq!(combos.len(), 1);
        assert!(combos[0].is_empty());
    }

    #[test]
    fn test_generate_param_combos() {
        let mut ranges = std::collections::HashMap::new();
        ranges.insert("a".to_string(), vec![1, 2]);
        ranges.insert("b".to_string(), vec![10, 20]);

        let combos = generate_param_combos(&ranges);
        assert_eq!(combos.len(), 4);
    }

    #[test]
    fn test_sample_random() {
        let space = ConfigSpace::matmul_reduced();
        let mut rng = rand::thread_rng();
        let config = sample_random(&space, &mut rng);
        assert!(space.workgroup_sizes.contains(&config.workgroup));
    }
}
