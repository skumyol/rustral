use std::sync::Arc;
use std::thread::{self, JoinHandle};

use crossbeam_channel::{bounded, Receiver, Sender};

/// Request sent to an inference worker.
pub struct InferenceRequest<I, O> {
    /// Inference input payload.
    pub input: I,

    /// One-shot reply channel for the inference result.
    pub reply: Sender<anyhow::Result<O>>,
}

/// Response wrapper for inference outputs.
pub struct InferenceResponse<O> {
    /// Model output payload.
    pub output: O,
}

/// Fixed-size inference worker pool with bounded queueing.
pub struct InferencePool<I, O> {
    sender: Sender<InferenceRequest<I, O>>,
    workers: Vec<JoinHandle<()>>,
}

impl<I, O> InferencePool<I, O>
where
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
{
    /// Start an inference pool.
    ///
    /// `handler` is shared by all workers. It must be thread-safe and must not
    /// rely on mutable global state unless that state is explicitly synchronized.
    pub fn new<F>(workers: usize, queue_bound: usize, handler: F) -> anyhow::Result<Self>
    where
        F: Fn(I) -> anyhow::Result<O> + Send + Sync + 'static,
    {
        if workers == 0 {
            anyhow::bail!("workers must be non-zero");
        }
        if queue_bound == 0 {
            anyhow::bail!("queue_bound must be non-zero");
        }

        let (sender, receiver): (Sender<InferenceRequest<I, O>>, Receiver<InferenceRequest<I, O>>) = bounded(queue_bound);
        let handler = Arc::new(handler);
        let mut handles = Vec::with_capacity(workers);

        for _ in 0..workers {
            let rx = receiver.clone();
            let h = Arc::clone(&handler);
            handles.push(thread::spawn(move || {
                while let Ok(req) = rx.recv() {
                    let _ = req.reply.send(h(req.input));
                }
            }));
        }

        Ok(Self { sender, workers: handles })
    }

    /// Submit one inference request and block until the worker replies.
    pub fn infer(&self, input: I) -> anyhow::Result<O> {
        let (reply_tx, reply_rx) = bounded(1);
        self.sender.send(InferenceRequest { input, reply: reply_tx })?;
        reply_rx.recv()?
    }

    /// Return the number of worker threads.
    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }
}
