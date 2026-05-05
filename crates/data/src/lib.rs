//! Data loading and dataset abstractions for the Modular Neural Runtime.
//!
//! Provides traits and implementations for handling datasets of various sizes,
//! from small in-memory datasets to large streaming datasets.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

/// Trait for datasets that can provide indexed access to elements.
///
/// This is the core abstraction for data sources. Implementations can be
/// in-memory, memory-mapped, or streaming.
pub trait Dataset<D>: Send + Sync {
    /// Return the number of elements in the dataset.
    fn len(&self) -> usize;

    /// Return true if the dataset is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get an element by index.
    ///
    /// Returns `None` if the index is out of bounds.
    fn get(&self, index: usize) -> Option<D>;

    /// Get multiple elements by their indices.
    ///
    /// Default implementation calls `get` for each index.
    fn get_batch(&self, indices: &[usize]) -> Vec<D> {
        indices.iter().filter_map(|&i| self.get(i)).collect()
    }
}

/// A dataset stored entirely in memory.
pub struct InMemoryDataset<D: Clone + Send + Sync + 'static> {
    data: Vec<D>,
}

impl<D: Clone + Send + Sync + 'static> InMemoryDataset<D> {
    /// Create a new in-memory dataset from a vector.
    pub fn new(data: Vec<D>) -> Self {
        Self { data }
    }

    /// Consume the dataset and return the underlying data.
    pub fn into_inner(self) -> Vec<D> {
        self.data
    }
}

impl<D: Clone + Send + Sync + 'static> Dataset<D> for InMemoryDataset<D> {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn get(&self, index: usize) -> Option<D> {
        self.data.get(index).cloned()
    }

    fn get_batch(&self, indices: &[usize]) -> Vec<D> {
        indices.iter().filter_map(|&i| self.data.get(i).cloned()).collect()
    }
}

/// A streaming dataset that reads data from a file line by line.
///
/// Useful for very large text datasets that don't fit in memory.
/// The dataset reads and parses data on-demand.
pub struct StreamingDataset<D, F> {
    /// Path to the data file.
    path: std::path::PathBuf,
    /// Total number of elements (pre-computed or estimated).
    len: usize,
    /// Parser function to convert a line to data element.
    parser: F,
    /// Phantom data to track D.
    _phantom: std::marker::PhantomData<D>,
}

impl<D, F> StreamingDataset<D, F>
where
    F: Fn(&str) -> Option<D> + Clone + Send + Sync + 'static,
{
    /// Create a new streaming dataset from a file.
    ///
    /// # Arguments
    /// * `path` - Path to the data file
    /// * `parser` - Function to parse each line into a data element
    ///
    /// # Returns
    /// The dataset with length computed by counting lines.
    pub fn from_file<P: AsRef<Path>>(path: P, parser: F) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();

        // Count lines for length
        let file = File::open(&path)?;
        let reader = io::BufReader::new(file);
        let len = reader.lines().count();

        Ok(Self { path, len, parser, _phantom: std::marker::PhantomData })
    }

    /// Create with known length (faster, skips line counting).
    pub fn from_file_with_len<P: AsRef<Path>>(path: P, len: usize, parser: F) -> Self {
        Self { path: path.as_ref().to_path_buf(), len, parser, _phantom: std::marker::PhantomData }
    }
}

impl<D, F> Dataset<D> for StreamingDataset<D, F>
where
    D: Clone + Send + Sync + 'static,
    F: Fn(&str) -> Option<D> + Clone + Send + Sync + 'static,
{
    fn len(&self) -> usize {
        self.len
    }

    fn get(&self, index: usize) -> Option<D> {
        if index >= self.len {
            return None;
        }

        let file = File::open(&self.path).ok()?;
        let reader = io::BufReader::new(file);

        reader.lines().nth(index).and_then(|line| line.ok().and_then(|l| (self.parser)(&l)))
    }
}

/// A memory-mapped dataset for fast random access to large binary files.
///
/// Uses memory mapping for efficient access to files larger than RAM.
/// Each element must have a fixed size.
pub struct MmapDataset {
    /// Memory-mapped file.
    #[allow(dead_code)]
    mmap: memmap2::Mmap,
    /// Element size in bytes.
    element_size: usize,
    /// Total number of elements.
    len: usize,
}

impl MmapDataset {
    /// Create a memory-mapped dataset from a binary file.
    ///
    /// # Arguments
    /// * `path` - Path to the binary file
    /// * `element_size` - Size of each element in bytes
    pub fn from_file<P: AsRef<Path>>(path: P, element_size: usize) -> io::Result<Self> {
        let file = File::open(path)?;
        let len = file.metadata()?.len() as usize / element_size;

        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        Ok(Self { mmap, element_size, len })
    }

    /// Get raw bytes for an element.
    pub fn get_bytes(&self, index: usize) -> Option<&[u8]> {
        if index >= self.len {
            return None;
        }

        let start = index * self.element_size;
        let end = start + self.element_size;

        self.mmap.get(start..end)
    }

    /// Get element as f32 slice (for tensor data).
    pub fn get_f32_slice(&self, index: usize) -> Option<&[f32]> {
        let bytes = self.get_bytes(index)?;

        // Safety: we're assuming the bytes are valid f32 values
        // This is safe because we know the element size and alignment
        if bytes.len() % 4 != 0 {
            return None;
        }

        let len = bytes.len() / 4;
        Some(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, len) })
    }
}

impl Dataset<Vec<f32>> for MmapDataset {
    fn len(&self) -> usize {
        self.len
    }

    fn get(&self, index: usize) -> Option<Vec<f32>> {
        self.get_f32_slice(index).map(|s| s.to_vec())
    }
}

/// Configuration for a DataLoader.
pub struct DataLoaderConfig {
    /// Number of samples per batch.
    pub batch_size: usize,
    /// Whether to shuffle data each epoch.
    pub shuffle: bool,
    /// Random seed for shuffling (deterministic if set).
    pub seed: Option<u64>,
    /// Number of worker threads for loading (0 = single-threaded).
    pub num_workers: usize,
}

impl Default for DataLoaderConfig {
    fn default() -> Self {
        Self { batch_size: 1, shuffle: false, seed: None, num_workers: 0 }
    }
}

/// DataLoader that batches and optionally shuffles data from a dataset.
pub struct DataLoader<D: Clone + Send + Sync + 'static> {
    dataset: Box<dyn Dataset<D>>,
    config: DataLoaderConfig,
    indices: Vec<usize>,
    position: usize,
    rng: StdRng,
}

impl<D: Clone + Send + Sync + 'static> DataLoader<D> {
    /// Create a new DataLoader from a dataset.
    pub fn new(dataset: Box<dyn Dataset<D>>, config: DataLoaderConfig) -> Self {
        let len = dataset.len();
        let mut indices: Vec<usize> = (0..len).collect();

        let mut rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        if config.shuffle {
            indices.shuffle(&mut rng);
        }

        Self { dataset, config, indices, position: 0, rng }
    }

    /// Get the next batch of data.
    ///
    /// Returns `None` when all data has been consumed (one epoch completed).
    pub fn next_batch(&mut self) -> Option<Vec<D>> {
        if self.position >= self.indices.len() {
            return None;
        }

        let end = (self.position + self.config.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.position..end];
        let batch = self.dataset.get_batch(batch_indices);
        self.position = end;

        if batch.is_empty() {
            None
        } else {
            Some(batch)
        }
    }

    /// Reset to the beginning of the dataset.
    ///
    /// If shuffling is enabled, re-shuffles the data.
    pub fn reset(&mut self) {
        self.position = 0;

        if self.config.shuffle {
            self.indices.shuffle(&mut self.rng);
        }
    }

    /// Get the total number of batches per epoch.
    pub fn num_batches(&self) -> usize {
        (self.indices.len() + self.config.batch_size - 1) / self.config.batch_size
    }

    /// Get the dataset length.
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Check if the current epoch is complete.
    pub fn is_epoch_done(&self) -> bool {
        self.position >= self.indices.len()
    }
}

impl<D: Clone + Send + Sync + 'static> Iterator for DataLoader<D> {
    type Item = Vec<D>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_batch()
    }
}

/// Helper function to create a simple DataLoader without configuration.
pub fn simple_loader<D: Clone + Send + Sync + 'static>(dataset: Vec<D>, batch_size: usize) -> DataLoader<D> {
    DataLoader::new(
        Box::new(InMemoryDataset::new(dataset)),
        DataLoaderConfig { batch_size, shuffle: false, seed: None, num_workers: 0 },
    )
}

/// Helper function to create a shuffled DataLoader.
pub fn shuffled_loader<D: Clone + Send + Sync + 'static>(
    dataset: Vec<D>,
    batch_size: usize,
    seed: u64,
) -> DataLoader<D> {
    DataLoader::new(
        Box::new(InMemoryDataset::new(dataset)),
        DataLoaderConfig { batch_size, shuffle: true, seed: Some(seed), num_workers: 0 },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_in_memory_dataset() {
        let data = vec![1, 2, 3, 4, 5];
        let dataset = InMemoryDataset::new(data);

        assert_eq!(dataset.len(), 5);
        assert_eq!(dataset.get(0), Some(1));
        assert_eq!(dataset.get(4), Some(5));
        assert_eq!(dataset.get(5), None);
    }

    #[test]
    fn test_data_loader_batching() {
        let data: Vec<i32> = (0..10).collect();
        let dataset = InMemoryDataset::new(data);
        let mut loader = DataLoader::new(
            Box::new(dataset),
            DataLoaderConfig { batch_size: 3, shuffle: false, seed: None, num_workers: 0 },
        );

        // First batch: [0, 1, 2]
        let batch1 = loader.next_batch().unwrap();
        assert_eq!(batch1, vec![0, 1, 2]);

        // Second batch: [3, 4, 5]
        let batch2 = loader.next_batch().unwrap();
        assert_eq!(batch2, vec![3, 4, 5]);

        // Third batch: [6, 7, 8]
        let batch3 = loader.next_batch().unwrap();
        assert_eq!(batch3, vec![6, 7, 8]);

        // Fourth batch: [9] (remainder)
        let batch4 = loader.next_batch().unwrap();
        assert_eq!(batch4, vec![9]);

        // No more batches
        assert!(loader.next_batch().is_none());
    }

    #[test]
    fn test_data_loader_shuffle() {
        let data: Vec<i32> = (0..100).collect();

        // Create two loaders with same seed
        let mut loader1 = DataLoader::new(
            Box::new(InMemoryDataset::new(data.clone())),
            DataLoaderConfig { batch_size: 10, shuffle: true, seed: Some(42), num_workers: 0 },
        );

        let mut loader2 = DataLoader::new(
            Box::new(InMemoryDataset::new(data)),
            DataLoaderConfig { batch_size: 10, shuffle: true, seed: Some(42), num_workers: 0 },
        );

        // Both should produce the same shuffled order
        let batch1: Vec<_> = loader1.next_batch().unwrap();
        let batch2: Vec<_> = loader2.next_batch().unwrap();
        assert_eq!(batch1, batch2);

        // Verify it's actually shuffled (not in original order)
        let original_first_10: Vec<i32> = (0..10).collect();
        assert_ne!(batch1, original_first_10);
    }

    #[test]
    fn test_data_loader_reset() {
        let data: Vec<i32> = (0..6).collect();
        let dataset = InMemoryDataset::new(data);
        let mut loader = DataLoader::new(
            Box::new(dataset),
            DataLoaderConfig { batch_size: 3, shuffle: false, seed: None, num_workers: 0 },
        );

        // Consume first epoch
        let _ = loader.next_batch();
        let _ = loader.next_batch();
        assert!(loader.next_batch().is_none());

        // Reset and consume again
        loader.reset();
        let batch1 = loader.next_batch().unwrap();
        assert_eq!(batch1, vec![0, 1, 2]);
    }

    #[test]
    fn test_simple_loader() {
        let data = vec![1, 2, 3, 4];
        let mut loader = simple_loader(data, 2);

        assert_eq!(loader.next_batch(), Some(vec![1, 2]));
        assert_eq!(loader.next_batch(), Some(vec![3, 4]));
        assert_eq!(loader.next_batch(), None);
    }

    #[test]
    fn test_iterator_interface() {
        let data: Vec<i32> = (0..6).collect();
        let dataset = InMemoryDataset::new(data);
        let loader = DataLoader::new(
            Box::new(dataset),
            DataLoaderConfig { batch_size: 3, shuffle: false, seed: None, num_workers: 0 },
        );

        let batches: Vec<_> = loader.collect();
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0], vec![0, 1, 2]);
        assert_eq!(batches[1], vec![3, 4, 5]);
    }

    #[test]
    fn test_dataset_is_empty() {
        let empty = InMemoryDataset::new(Vec::<i32>::new());
        assert!(empty.is_empty());
        let non_empty = InMemoryDataset::new(vec![1]);
        assert!(!non_empty.is_empty());
    }

    #[test]
    fn test_in_memory_dataset_into_inner() {
        let data = vec![1, 2, 3];
        let dataset = InMemoryDataset::new(data.clone());
        assert_eq!(dataset.into_inner(), data);
    }

    #[test]
    fn test_in_memory_dataset_get_batch() {
        let data: Vec<i32> = (0..10).collect();
        let dataset = InMemoryDataset::new(data);
        let batch = dataset.get_batch(&[0, 2, 4]);
        assert_eq!(batch, vec![0, 2, 4]);
    }

    #[test]
    fn test_data_loader_num_batches() {
        let data: Vec<i32> = (0..10).collect();
        let loader = DataLoader::new(
            Box::new(InMemoryDataset::new(data)),
            DataLoaderConfig { batch_size: 3, shuffle: false, seed: None, num_workers: 0 },
        );
        assert_eq!(loader.num_batches(), 4);
    }

    #[test]
    fn test_data_loader_len_and_epoch_done() {
        let mut loader = simple_loader(vec![1, 2, 3, 4], 2);
        assert_eq!(loader.len(), 4);
        assert!(!loader.is_epoch_done());
        let _ = loader.next_batch();
        let _ = loader.next_batch();
        assert!(loader.is_epoch_done());
    }

    #[test]
    fn test_shuffled_loader() {
        let mut loader = shuffled_loader(vec![1, 2, 3, 4], 2, 42);
        let batch = loader.next_batch().unwrap();
        assert_eq!(batch.len(), 2);
    }

    #[test]
    fn test_dataloader_default_config() {
        let config: DataLoaderConfig = Default::default();
        assert_eq!(config.batch_size, 1);
        assert!(!config.shuffle);
        assert_eq!(config.seed, None);
        assert_eq!(config.num_workers, 0);
    }
}
