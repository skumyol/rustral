//! Continuous Batching for Inference Serving (vLLM-style)
//!
//! Maximizes GPU utilization by:
//! 1. PagedAttention: Block-based KV cache management
//! 2. Continuous batching: Add/remove requests dynamically
//! 3. Preemption: Pause low-priority requests
//! 4. Iteration-level scheduling
//!
//! # Request Lifecycle
//! ```text
//! [Prompt Processing] → [Generation] → [Completion]
//!       ↑                     ↑
//!   High compute          Low compute
//!   (can batch more)      (cache-bound)
//! ```
//!
//! # Example
//!```rust,ignore
//! use rustral_nn::continuous_batching::{Scheduler, SchedulingPolicy};
//!
//! let mut scheduler = Scheduler::new(model, SchedulingPolicy::Fcfs);
//! scheduler.add_request(prompt1, 100)?;
//! scheduler.add_request(prompt2, 50)?;
//!
//! while !scheduler.is_empty() {
//!     let batch = scheduler.schedule(max_tokens_per_step)?;
//!     let outputs = model.generate(batch)?;
//!     scheduler.update(outputs)?;
//!}
//!```

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use rustral_core::{Backend, CoreError, Result};

use super::kv_cache::PagedCache;

/// Request priority
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum RequestPriority {
    /// Critical (cannot be preempted)
    Critical = 3,
    /// High priority
    High = 2,
    /// Normal
    Normal = 1,
    /// Low (first to preempt)
    Low = 0,
}

/// Request state
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RequestState {
    /// Waiting to be scheduled
    Waiting,
    /// Prompt tokens being processed
    Prefill,
    /// Generating new tokens
    Generating,
    /// Paused (preempted)
    Paused,
    /// Completed
    Completed,
}

/// Single inference request
pub struct Request {
    /// Unique ID
    pub id: u64,
    /// Prompt tokens
    pub prompt: Vec<u32>,
    /// Generated tokens so far
    pub generated: Vec<u32>,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Current state
    state: RequestState,
    /// Priority
    priority: RequestPriority,
    /// Arrival time
    arrival_time: Instant,
    /// Last generation time
    last_gen_time: Instant,
    /// KV cache blocks assigned
    blocks: Vec<usize>,
    /// Whether completed
    is_completed: bool,
}

impl Request {
    pub fn new(id: u64, prompt: Vec<u32>, max_tokens: usize, priority: RequestPriority) -> Self {
        let now = Instant::now();
        Self {
            id,
            prompt,
            generated: Vec::new(),
            max_tokens,
            state: RequestState::Waiting,
            priority,
            arrival_time: now,
            last_gen_time: now,
            blocks: Vec::new(),
            is_completed: false,
        }
    }

    /// Total sequence length (prompt + generated)
    pub fn seq_len(&self) -> usize {
        self.prompt.len() + self.generated.len()
    }

    /// Get next token position
    pub fn next_pos(&self) -> usize {
        self.seq_len()
    }

    /// Check if request should be completed
    pub fn should_complete(&self) -> bool {
        if self.generated.len() >= self.max_tokens {
            return true;
        }
        // Check for stop token (e.g., EOS)
        if let Some(&last) = self.generated.last() {
            if last == 2 {
                // Assuming 2 is EOS
                return true;
            }
        }
        false
    }

    /// Mark as completed
    pub fn complete(&mut self) {
        self.is_completed = true;
        self.state = RequestState::Completed;
    }

    /// Add generated token
    pub fn add_token(&mut self, token: u32) {
        self.generated.push(token);
        self.last_gen_time = Instant::now();
    }
}

/// Scheduling policy
#[derive(Clone, Copy, Debug)]
pub enum SchedulingPolicy {
    /// First-come-first-served
    Fcfs,
    /// Shortest remaining time first
    Srtf,
    /// Priority-based with preemption
    Priority,
}

/// Batch of requests for one iteration
pub struct Batch {
    /// Request IDs
    pub request_ids: Vec<u64>,
    /// Input tensors (token IDs)
    pub input_ids: Vec<Vec<u32>>,
    /// Sequence lengths
    pub seq_lens: Vec<usize>,
    /// Block tables for PagedAttention
    pub block_tables: Vec<Vec<usize>>,
}

/// Request scheduler
pub struct Scheduler<B: Backend> {
    /// Pending requests
    pending: VecDeque<Request>,
    /// Running requests
    running: HashMap<u64, Request>,
    /// Paused requests (preempted)
    paused: Vec<Request>,
    /// Completed requests
    completed: Vec<Request>,
    /// Paged cache manager
    cache: PagedCache<B>,
    /// Scheduling policy
    policy: SchedulingPolicy,
    /// Next request ID
    next_id: u64,
    /// Block size
    block_size: usize,
    /// Maximum batch size
    max_batch_size: usize,
    /// Maximum sequence length
    max_seq_len: usize,
    /// Preemption threshold
    preemption_threshold: f32,
}

impl<B: Backend> Scheduler<B>
where
    B::Tensor: Clone,
{
    pub fn new(
        cache: PagedCache<B>,
        policy: SchedulingPolicy,
        max_batch_size: usize,
        max_seq_len: usize,
    ) -> Self {
        Self {
            pending: VecDeque::new(),
            running: HashMap::new(),
            paused: Vec::new(),
            completed: Vec::new(),
            cache,
            policy,
            next_id: 0,
            block_size: 16,
            max_batch_size,
            max_seq_len,
            preemption_threshold: 0.5,
        }
    }

    /// Add new request to queue
    pub fn add_request(&mut self, prompt: Vec<u32>, max_tokens: usize) -> u64 {
        let id = self.next_id;
        self.next_id += 1;

        let request = Request::new(id, prompt, max_tokens, RequestPriority::Normal);
        self.pending.push_back(request);

        id
    }

    /// Add high priority request
    pub fn add_priority_request(
        &mut self,
        prompt: Vec<u32>,
        max_tokens: usize,
        priority: RequestPriority,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;

        let request = Request::new(id, prompt, max_tokens, priority);
        self.pending.push_back(request);

        id
    }

    /// Schedule requests for next iteration
    pub fn schedule(&mut self, _max_new_tokens: usize) -> Option<Batch> {
        // 1. Move completed from running to completed
        let completed_ids: Vec<u64> =
            self.running.values().filter(|r| r.should_complete()).map(|r| r.id).collect();

        for id in completed_ids {
            if let Some(mut req) = self.running.remove(&id) {
                req.complete();
                self.completed.push(req);
            }
        }

        // 2. Try to resume paused requests
        self.try_resume_paused();

        // 3. Admit new requests from pending
        self.admit_pending();

        // 4. Check if we need to preempt
        self.maybe_preempt();

        // 5. Construct batch
        if self.running.is_empty() {
            return None;
        }

        let request_ids: Vec<u64> = self.running.keys().copied().collect();
        let mut input_ids = Vec::new();
        let mut seq_lens = Vec::new();
        let mut block_tables = Vec::new();

        for id in &request_ids {
            if let Some(req) = self.running.get(id) {
                // For prefill: full prompt
                // For generation: just last token
                let input = if req.generated.is_empty() {
                    req.prompt.clone()
                } else {
                    vec![*req.generated.last().unwrap_or(&0)]
                };

                input_ids.push(input);
                seq_lens.push(req.seq_len());
                block_tables.push(req.blocks.clone());
            }
        }

        Some(Batch { request_ids, input_ids, seq_lens, block_tables })
    }

    /// Update requests with generated tokens
    pub fn update(&mut self, outputs: HashMap<u64, u32>) -> Result<()> {
        for (id, token) in outputs {
            if let Some(req) = self.running.get_mut(&id) {
                req.add_token(token);

                // Allocate new cache blocks if needed
                let seq_len = req.seq_len();
                let blocks_needed = (seq_len + self.block_size - 1) / self.block_size;
                let current_blocks = req.blocks.len();

                if blocks_needed > current_blocks {
                    let new_blocks = self.cache.allocate(id as usize, blocks_needed - current_blocks);
                    if let Some(blocks) = new_blocks {
                        req.blocks.extend(blocks);
                    } else {
                        // Out of memory - should preempt
                        return Err(CoreError::Other("KV cache out of memory".to_string()));
                    }
                }
            }
        }

        Ok(())
    }

    /// Admit pending requests
    fn admit_pending(&mut self) {
        while self.running.len() < self.max_batch_size {
            if let Some(mut req) = self.pending.pop_front() {
                // Allocate blocks for prompt
                let prompt_len = req.prompt.len();
                let blocks_needed = (prompt_len + self.block_size - 1) / self.block_size;

                if let Some(blocks) = self.cache.allocate(req.id as usize, blocks_needed) {
                    req.blocks = blocks;
                    req.state = RequestState::Prefill;
                    self.running.insert(req.id, req);
                } else {
                    // Not enough memory, put back
                    self.pending.push_front(req);
                    break;
                }
            } else {
                break;
            }
        }
    }

    /// Try to resume paused requests
    fn try_resume_paused(&mut self) {
        let mut to_resume = Vec::new();

        for (idx, _req) in self.paused.iter().enumerate() {
            if self.running.len() < self.max_batch_size {
                to_resume.push(idx);
            } else {
                break;
            }
        }

        for idx in to_resume.into_iter().rev() {
            let req = self.paused.remove(idx);
            self.running.insert(req.id, req);
        }
    }

    /// Preempt low-priority requests if needed
    fn maybe_preempt(&mut self) {
        // Check memory pressure
        let memory_pressure = self.running.len() as f32 / self.max_batch_size as f32;

        if memory_pressure > self.preemption_threshold {
            // Find lowest priority request to preempt
            let to_preempt: Option<u64> = self
                .running
                .values()
                .filter(|r| r.priority == RequestPriority::Low)
                .min_by_key(|r| r.priority as u8)
                .map(|r| r.id);

            if let Some(id) = to_preempt {
                if let Some(mut req) = self.running.remove(&id) {
                    // Free cache blocks (but keep on CPU for resume)
                    self.cache.free(req.id as usize);
                    req.blocks.clear();
                    req.state = RequestState::Paused;
                    self.paused.push(req);
                }
            }
        }
    }

    /// Check if all requests completed
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty() && self.running.is_empty() && self.paused.is_empty()
    }

    /// Get number of pending requests
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Get number of running requests
    pub fn running_count(&self) -> usize {
        self.running.len()
    }

    /// Get number of completed requests
    pub fn completed_count(&self) -> usize {
        self.completed.len()
    }

    /// Get statistics
    pub fn stats(&self) -> SchedulerStats {
        let total_tokens_generated: usize = self.completed.iter().map(|r| r.generated.len()).sum();

        let avg_latency_ms = if !self.completed.is_empty() {
            self.completed
                .iter()
                .map(|r| r.last_gen_time.duration_since(r.arrival_time).as_millis() as f32)
                .sum::<f32>()
                / self.completed.len() as f32
        } else {
            0.0
        };

        SchedulerStats {
            pending: self.pending.len(),
            running: self.running.len(),
            paused: self.paused.len(),
            completed: self.completed.len(),
            total_tokens_generated,
            avg_latency_ms,
            throughput_tok_per_sec: if avg_latency_ms > 0.0 {
                total_tokens_generated as f32 / (avg_latency_ms / 1000.0)
            } else {
                0.0
            },
        }
    }
}

/// Scheduler statistics
#[derive(Debug, Clone)]
pub struct SchedulerStats {
    pub pending: usize,
    pub running: usize,
    pub paused: usize,
    pub completed: usize,
    pub total_tokens_generated: usize,
    pub avg_latency_ms: f32,
    pub throughput_tok_per_sec: f32,
}

/// Token samplers for generation
pub struct Sampler;

impl Sampler {
    /// Greedy sampling (argmax)
    pub fn greedy(logits: &[f32]) -> u32 {
        let (idx, _) =
            logits.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap_or((0, &0.0));
        idx as u32
    }

    /// Temperature sampling
    pub fn temperature(logits: &[f32], temperature: f32) -> u32 {
        if temperature == 0.0 {
            return Self::greedy(logits);
        }

        // Apply temperature
        let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();

        // Softmax
        let max_logit = scaled.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_sum: f32 = scaled.iter().map(|&x| (x - max_logit).exp()).sum();
        let probs: Vec<f32> = scaled.iter().map(|&x| (x - max_logit).exp() / exp_sum).collect();

        // Sample
        use std::f32;
        let r: f32 = 0.5; // In real impl, use random
        let mut cumsum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if cumsum >= r {
                return i as u32;
            }
        }
        (probs.len() - 1) as u32
    }
}

/// Request completion callback
type CompletionCallback = Box<dyn Fn(u64, Vec<u32>)>;

/// Serving engine with HTTP-like interface
pub struct ServingEngine<B: Backend> {
    scheduler: Scheduler<B>,
    /// Callbacks for completed requests
    callbacks: HashMap<u64, CompletionCallback>,
}

impl<B: Backend> ServingEngine<B>
where
    B::Tensor: Clone,
{
    pub fn new(scheduler: Scheduler<B>) -> Self {
        Self { scheduler, callbacks: HashMap::new() }
    }

    /// Submit request and get future
    pub fn submit(&mut self, prompt: Vec<u32>, max_tokens: usize) -> u64 {
        self.scheduler.add_request(prompt, max_tokens)
    }

    /// Run one scheduling iteration.
    ///
    /// **Placeholder decode:** emits dummy token `1` per request. Real Llama (or GPT) serving should
    /// run model forwards here using a **per-request** KV cache such as [`crate::LlamaDecodeCache`]
    /// (prefill + per-token steps). Reference HTTP wiring: `rustral-llama-server` in
    /// `crates/inference-server` (`/v1/generate`, `/v1/generate/stream`).
    pub fn step(&mut self) -> Result<()> {
        if let Some(batch) = self.scheduler.schedule(1) {
            // In real impl, would run model forward here
            // For now, simulate generation
            let outputs: HashMap<u64, u32> = batch
                .request_ids
                .iter()
                .map(|&id| (id, 1u32)) // Dummy token
                .collect();

            self.scheduler.update(outputs)?;
        }

        // Check completions
        for req in &self.scheduler.completed {
            if let Some(callback) = self.callbacks.remove(&req.id) {
                callback(req.id, req.generated.clone());
            }
        }

        Ok(())
    }

    /// Run until all requests complete
    pub fn run_to_completion(&mut self) -> Result<()> {
        while !self.scheduler.is_empty() {
            self.step()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustral_ndarray_backend::CpuBackend;

    #[test]
    fn test_request_creation() {
        let req = Request::new(0, vec![1, 2, 3], 100, RequestPriority::Normal);
        assert_eq!(req.id, 0);
        assert_eq!(req.prompt, vec![1, 2, 3]);
        assert_eq!(req.seq_len(), 3);
        assert_eq!(req.state, RequestState::Waiting);
    }

    #[test]
    fn test_request_add_token() {
        let mut req = Request::new(0, vec![1, 2, 3], 100, RequestPriority::Normal);
        assert_eq!(req.seq_len(), 3);

        req.add_token(4);
        assert_eq!(req.generated, vec![4]);
        assert_eq!(req.seq_len(), 4);
    }

    #[test]
    fn test_request_should_complete() {
        let mut req = Request::new(0, vec![1, 2], 5, RequestPriority::Normal);
        assert!(!req.should_complete());

        // Generate 5 tokens
        for i in 0..5 {
            req.add_token(i as u32);
        }
        assert!(req.should_complete());
    }

    #[test]
    fn test_greedy_sampler() {
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let token = Sampler::greedy(&logits);
        assert_eq!(token, 3); // Index of 0.9
    }

    #[test]
    fn test_scheduler_stats() {
        let backend = CpuBackend::default();
        let cache = PagedCache::new(16, 100, &backend).unwrap();
        let scheduler = Scheduler::new(cache, SchedulingPolicy::Fcfs, 8, 8192);

        let stats = scheduler.stats();
        assert_eq!(stats.pending, 0);
        assert_eq!(stats.running, 0);
        assert_eq!(stats.completed, 0);
    }

    #[test]
    fn test_scheduler_add_request() {
        let backend = CpuBackend::default();
        let cache = PagedCache::new(16, 100, &backend).unwrap();
        let mut scheduler = Scheduler::new(cache, SchedulingPolicy::Fcfs, 8, 8192);

        let _id1 = scheduler.add_request(vec![1, 2, 3], 100);
        let _id2 = scheduler.add_request(vec![4, 5], 50);

        assert_eq!(scheduler.pending_count(), 2);
        assert!(!scheduler.is_empty());
    }

    #[test]
    fn test_request_priority() {
        let high = RequestPriority::Critical;
        let low = RequestPriority::Low;
        assert!(high > low);
    }

    #[test]
    fn test_request_next_pos() {
        let req = Request::new(0, vec![1, 2, 3], 100, RequestPriority::Normal);
        assert_eq!(req.next_pos(), 3);
    }

    #[test]
    fn test_request_should_complete_eos() {
        let mut req = Request::new(0, vec![1, 2], 5, RequestPriority::Normal);
        req.add_token(2); // Assuming 2 is EOS
        assert!(req.should_complete());
    }

    #[test]
    fn test_request_complete() {
        let mut req = Request::new(0, vec![1, 2, 3], 100, RequestPriority::Normal);
        req.complete();
        assert!(req.is_completed);
        assert_eq!(req.state, RequestState::Completed);
    }

    #[test]
    fn test_scheduler_add_priority_request() {
        let backend = CpuBackend::default();
        let cache = PagedCache::new(16, 100, &backend).unwrap();
        let mut scheduler = Scheduler::new(cache, SchedulingPolicy::Fcfs, 8, 8192);

        let id = scheduler.add_priority_request(vec![1, 2, 3], 100, RequestPriority::High);
        assert_eq!(id, 0);
        assert_eq!(scheduler.pending_count(), 1);
    }

    #[test]
    fn test_scheduler_schedule_and_update() {
        let backend = CpuBackend::default();
        let cache = PagedCache::new(16, 100, &backend).unwrap();
        let mut scheduler = Scheduler::new(cache, SchedulingPolicy::Fcfs, 8, 8192);

        let id = scheduler.add_request(vec![1, 2, 3], 5);
        // Schedule should admit the request and return a batch
        let batch = scheduler.schedule(1);
        assert!(batch.is_some());
        let batch = batch.unwrap();
        assert!(batch.request_ids.contains(&id));

        // Update with generated tokens
        let mut outputs = std::collections::HashMap::new();
        outputs.insert(id, 10u32);
        scheduler.update(outputs).unwrap();

        assert_eq!(scheduler.running_count(), 1);
    }

    #[test]
    fn test_scheduler_completion_and_counts() {
        let backend = CpuBackend::default();
        let cache = PagedCache::new(16, 100, &backend).unwrap();
        let mut scheduler = Scheduler::new(cache, SchedulingPolicy::Fcfs, 8, 8192);

        let id = scheduler.add_request(vec![1, 2], 1);
        let batch = scheduler.schedule(1).unwrap();
        assert_eq!(batch.request_ids.len(), 1);

        let mut outputs = std::collections::HashMap::new();
        outputs.insert(id, 1u32);
        scheduler.update(outputs).unwrap();

        // After max_tokens=1 reached, schedule should move to completed
        let _ = scheduler.schedule(1);
        assert_eq!(scheduler.completed_count(), 1);
        assert_eq!(scheduler.running_count(), 0);
        assert_eq!(scheduler.pending_count(), 0);
    }

    #[test]
    fn test_scheduler_stats_with_completed() {
        let backend = CpuBackend::default();
        let cache = PagedCache::new(16, 100, &backend).unwrap();
        let mut scheduler = Scheduler::new(cache, SchedulingPolicy::Fcfs, 8, 8192);

        let id = scheduler.add_request(vec![1, 2], 1);
        let _ = scheduler.schedule(1);
        let mut outputs = std::collections::HashMap::new();
        outputs.insert(id, 1u32);
        scheduler.update(outputs).unwrap();
        let _ = scheduler.schedule(1);

        let stats = scheduler.stats();
        assert_eq!(stats.completed, 1);
        assert_eq!(stats.total_tokens_generated, 1);
    }

    #[test]
    fn test_sampler_temperature_zero() {
        let logits = vec![0.1f32, 0.5, 0.3, 0.9, 0.2];
        let token = Sampler::temperature(&logits, 0.0);
        assert_eq!(token, 3); // same as greedy
    }

    #[test]
    fn test_sampler_temperature_nonzero() {
        let logits = vec![0.1f32, 0.5, 0.3, 0.9, 0.2];
        let token = Sampler::temperature(&logits, 1.0);
        // With fixed r=0.5, it should return some token
        assert!(token < 5);
    }

    #[test]
    fn test_serving_engine() {
        let backend = CpuBackend::default();
        let cache = PagedCache::new(16, 100, &backend).unwrap();
        let scheduler = Scheduler::new(cache, SchedulingPolicy::Fcfs, 8, 8192);
        let mut engine = ServingEngine::new(scheduler);

        let id = engine.submit(vec![1, 2, 3], 2);
        assert_eq!(id, 0);

        // Run a few steps
        for _ in 0..5 {
            engine.step().unwrap();
        }

        // Run to completion
        engine.run_to_completion().unwrap();
    }
}
