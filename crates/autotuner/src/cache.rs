//! Cache for kernel tuning results.

use crate::kernel_config::KernelConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// A cached tuning entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// The kernel configuration.
    pub config: KernelConfig,
    /// Execution time in microseconds.
    pub time_us: f64,
    /// Hardware/GPU identifier.
    pub device_id: String,
    /// Timestamp when tuned.
    pub timestamp: u64,
    /// MNR version when tuned.
    pub version: String,
    /// Number of samples averaged.
    pub num_samples: usize,
}

impl CacheEntry {
    /// Create a new cache entry.
    pub fn new(config: KernelConfig, time_us: f64, device_id: String) -> Self {
        Self {
            config,
            time_us,
            device_id,
            timestamp: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            num_samples: 1,
        }
    }

    /// Check if this entry is still valid (not too old).
    pub fn is_fresh(&self, max_age_days: u64) -> bool {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let age_secs = now.saturating_sub(self.timestamp);
        let max_age_secs = max_age_days * 24 * 60 * 60;

        age_secs < max_age_secs
    }

    /// Merge with another entry (average times).
    pub fn merge(&mut self, other: &CacheEntry) {
        let total_samples = self.num_samples + other.num_samples;
        self.time_us = (self.time_us * self.num_samples as f64
            + other.time_us * other.num_samples as f64)
            / total_samples as f64;
        self.num_samples = total_samples;
    }
}

/// Persistent cache for tuning configurations.
pub struct ConfigCache {
    /// In-memory cache: "kernel_key:input_sig" -> entry
    entries: HashMap<String, CacheEntry>,
    /// Cache file path.
    cache_path: PathBuf,
    /// Maximum age in days for cache entries.
    max_age_days: u64,
    /// Device identifier for this machine.
    device_id: String,
}

/// Create a cache key from kernel key and input signature.
fn make_cache_key(kernel_key: &str, input_sig: &[usize]) -> String {
    format!("{}:{:?}", kernel_key, input_sig)
}

impl ConfigCache {
    /// Create a new cache with default location.
    pub fn new() -> Self {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from(".cache"))
            .join("mnr")
            .join("autotuner");

        let _ = fs::create_dir_all(&cache_dir);

        let cache_path = cache_dir.join("kernel_configs.json");

        let device_id = Self::detect_device_id();

        let mut cache = Self {
            entries: HashMap::new(),
            cache_path,
            max_age_days: 30,
            device_id,
        };

        // Load existing cache
        let _ = cache.load();

        cache
    }

    /// Create a new cache with custom path.
    pub fn with_path<P: AsRef<Path>>(path: P) -> Self {
        let device_id = Self::detect_device_id();

        let mut cache = Self {
            entries: HashMap::new(),
            cache_path: path.as_ref().to_path_buf(),
            max_age_days: 30,
            device_id,
        };

        let _ = cache.load();
        cache
    }

    /// Set maximum cache entry age.
    pub fn with_max_age(mut self, days: u64) -> Self {
        self.max_age_days = days;
        self
    }

    /// Get the best cached configuration for a kernel.
    pub fn get(&self, kernel_key: &str, input_sig: &[usize]) -> Option<&CacheEntry> {
        let key = make_cache_key(kernel_key, input_sig);

        self.entries.get(&key).filter(|e| {
            e.is_fresh(self.max_age_days) && e.device_id == self.device_id
        })
    }

    /// Insert or update a cache entry.
    pub fn insert(&mut self, kernel_key: &str, input_sig: &[usize], entry: CacheEntry) {
        let key = make_cache_key(kernel_key, input_sig);

        if let Some(existing) = self.entries.get_mut(&key) {
            existing.merge(&entry);
        } else {
            self.entries.insert(key, entry);
        }
    }

    /// Save cache to disk.
    pub fn save(&self) -> Result<(), String> {
        let json = serde_json::to_string_pretty(&self.entries)
            .map_err(|e| format!("Failed to serialize cache: {}", e))?;

        fs::write(&self.cache_path, json)
            .map_err(|e| format!("Failed to write cache: {}", e))?;

        Ok(())
    }

    /// Load cache from disk.
    pub fn load(&mut self) -> Result<(), String> {
        if !self.cache_path.exists() {
            return Ok(());
        }

        let json = fs::read_to_string(&self.cache_path)
            .map_err(|e| format!("Failed to read cache: {}", e))?;

        self.entries = serde_json::from_str(&json)
            .map_err(|e| format!("Failed to parse cache: {}", e))?;

        // Filter out stale entries
        let device_id = self.device_id.clone();
        let max_age = self.max_age_days;
        self.entries.retain(|_, e| {
            e.is_fresh(max_age) && e.device_id == device_id
        });

        Ok(())
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Get number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        let total = self.entries.len();
        let by_device: HashMap<String, usize> = self.entries
            .values()
            .fold(HashMap::new(), |mut acc, e| {
                *acc.entry(e.device_id.clone()).or_insert(0) += 1;
                acc
            });

        CacheStats {
            total_entries: total,
            by_device,
            cache_path: self.cache_path.clone(),
        }
    }

    /// Detect GPU/device identifier.
    fn detect_device_id() -> String {
        // Try to get GPU info from wgpu or system
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = fs::read_to_string("/proc/driver/nvidia/gpus/0/information") {
                for line in content.lines() {
                    if line.starts_with("Model:") {
                        return line.split(':').nth(1).unwrap_or("unknown").trim().to_string();
                    }
                }
            }
        }

        // Fallback: use hostname + architecture
        let hostname = std::env::var("HOSTNAME")
            .or_else(|_| std::env::var("COMPUTERNAME"))
            .unwrap_or_else(|_| "unknown".to_string());

        format!("{}-{}", hostname, std::env::consts::ARCH)
    }
}

impl Default for ConfigCache {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ConfigCache {
    fn drop(&mut self) {
        // Auto-save on drop
        let _ = self.save();
    }
}

/// Cache statistics.
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub total_entries: usize,
    pub by_device: HashMap<String, usize>,
    pub cache_path: PathBuf,
}

impl CacheStats {
    /// Print cache statistics.
    pub fn print(&self) {
        println!("Config Cache Statistics:");
        println!("  Total entries: {}", self.total_entries);
        println!("  Cache path: {}", self.cache_path.display());
        println!("  By device:");
        for (device, count) in &self.by_device {
            println!("    {}: {}", device, count);
        }
    }
}

/// Simplified tuning cache interface.
pub struct TuningCache {
    cache: ConfigCache,
}

impl TuningCache {
    /// Create a new tuning cache.
    pub fn new() -> Self {
        Self {
            cache: ConfigCache::new(),
        }
    }

    /// Get a configuration if cached.
    pub fn get_config(&self, kernel: &str, shapes: &[usize]) -> Option<KernelConfig> {
        self.cache.get(kernel, shapes).map(|e| e.config.clone())
    }

    /// Store a configuration.
    pub fn store_config(&mut self, kernel: &str, shapes: &[usize], config: KernelConfig, time_us: f64) {
        let entry = CacheEntry::new(config, time_us, self.cache.device_id.clone());
        self.cache.insert(kernel, shapes, entry);
    }

    /// Get cache hit rate stats.
    pub fn hit_rate(&self) -> f64 {
        // Would track hits/misses in practice
        if self.cache.is_empty() {
            0.0
        } else {
            1.0 // Assume all cached for now
        }
    }
}

impl Default for TuningCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_config::KernelConfig;

    #[test]
    fn test_cache_entry() {
        let config = KernelConfig::default_matmul();
        let entry = CacheEntry::new(config.clone(), 100.0, "test-gpu".to_string());

        assert!(entry.is_fresh(30));
        assert_eq!(entry.time_us, 100.0);
        assert_eq!(entry.device_id, "test-gpu");
    }

    #[test]
    fn test_cache_entry_merge() {
        let config = KernelConfig::default_matmul();
        let mut entry1 = CacheEntry::new(config.clone(), 100.0, "test".to_string());
        entry1.num_samples = 2;

        let entry2 = CacheEntry::new(config.clone(), 200.0, "test".to_string());

        entry1.merge(&entry2);

        // (100*2 + 200*1) / 3 = 133.33...
        assert!((entry1.time_us - 133.33).abs() < 1.0);
        assert_eq!(entry1.num_samples, 3);
    }

    #[test]
    fn test_config_cache() {
        let temp_dir = std::env::temp_dir();
        let cache_path = temp_dir.join("test_mnr_cache.json");

        let mut cache = ConfigCache::with_path(&cache_path);
        cache.clear();

        let config = KernelConfig::default_matmul();
        let entry = CacheEntry::new(config, 50.0, cache.device_id.clone());

        cache.insert("matmul", &[1024, 1024, 1024], entry);
        assert_eq!(cache.len(), 1);

        // Save and reload
        cache.save().unwrap();

        let mut cache2 = ConfigCache::with_path(&cache_path);
        cache2.load().unwrap();
        assert_eq!(cache2.len(), 1);

        // Cleanup
        let _ = fs::remove_file(&cache_path);
    }

    #[test]
    fn test_tuning_cache() {
        let mut cache = TuningCache::new();

        let config = KernelConfig::default_matmul();
        cache.store_config("matmul", &[1024, 1024, 1024], config.clone(), 100.0);

        let retrieved = cache.get_config("matmul", &[1024, 1024, 1024]);
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_cache_entry_fresh_and_stale() {
        let config = KernelConfig::default_matmul();
        let entry = CacheEntry::new(config, 100.0, "test-device".to_string());
        assert!(entry.is_fresh(30));

        // Very old entry should not be fresh
        let mut stale = entry.clone();
        stale.timestamp = 0; // Unix epoch
        assert!(!stale.is_fresh(1));
    }

    #[test]
    fn test_config_cache_with_path_and_max_age() {
        let temp_path = std::env::temp_dir().join("test_mnr_cache_unique.json");
        let cache = ConfigCache::with_path(&temp_path).with_max_age(7);
        assert_eq!(cache.max_age_days, 7);
    }

    #[test]
    fn test_config_cache_get_missing_and_stale() {
        let temp_path = std::env::temp_dir().join("test_mnr_cache_get.json");
        let mut cache = ConfigCache::with_path(&temp_path);
        cache.clear();

        // Missing entry
        assert!(cache.get("missing", &[1, 2, 3]).is_none());

        // Insert and retrieve
        let config = KernelConfig::default_matmul();
        let entry = CacheEntry::new(config, 100.0, cache.device_id.clone());
        cache.insert("matmul", &[64, 64, 64], entry.clone());
        assert!(cache.get("matmul", &[64, 64, 64]).is_some());

        // Stale entry with different device
        let mut stale_entry = entry.clone();
        stale_entry.device_id = "other-device".to_string();
        stale_entry.timestamp = 0;
        cache.insert("old", &[1], stale_entry);
        assert!(cache.get("old", &[1]).is_none());
    }

    #[test]
    fn test_config_cache_insert_merge() {
        let temp_path = std::env::temp_dir().join("test_mnr_cache_merge.json");
        let mut cache = ConfigCache::with_path(&temp_path);
        cache.clear();

        let config = KernelConfig::default_matmul();
        let device = cache.device_id.clone();
        let entry1 = CacheEntry::new(config.clone(), 100.0, device.clone());
        let entry2 = CacheEntry::new(config.clone(), 200.0, device.clone());

        cache.insert("matmul", &[128, 128], entry1);
        cache.insert("matmul", &[128, 128], entry2);

        let merged = cache.get("matmul", &[128, 128]).unwrap();
        assert_eq!(merged.num_samples, 2);
    }

    #[test]
    fn test_config_cache_clear_and_len() {
        let temp_path = std::env::temp_dir().join("test_mnr_cache_len.json");
        let mut cache = ConfigCache::with_path(&temp_path);
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);

        let config = KernelConfig::default_matmul();
        let entry = CacheEntry::new(config, 50.0, "test".to_string());
        cache.insert("matmul", &[256], entry);
        assert!(!cache.is_empty());
        assert_eq!(cache.len(), 1);

        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_config_cache_stats() {
        let temp_path = std::env::temp_dir().join("test_mnr_cache_stats.json");
        let mut cache = ConfigCache::with_path(&temp_path);
        cache.clear();

        let config = KernelConfig::default_matmul();
        let entry = CacheEntry::new(config, 50.0, cache.device_id.clone());
        cache.insert("matmul", &[256], entry);

        let stats = cache.stats();
        assert_eq!(stats.total_entries, 1);
        assert!(!stats.by_device.is_empty());

        // Verify print doesn't panic
        stats.print();
    }

    #[test]
    fn test_config_cache_detect_device_id() {
        let device_id = ConfigCache::detect_device_id();
        assert!(!device_id.is_empty());
    }

    #[test]
    fn test_tuning_cache_hit_rate() {
        let mut cache = TuningCache::new();
        // Store a config so cache is non-empty
        let config = KernelConfig::default_matmul();
        cache.store_config("matmul", &[1024], config, 100.0);
        // hit_rate returns 1.0 when cache is non-empty
        assert_eq!(cache.hit_rate(), 1.0);
    }

    #[test]
    fn test_tuning_cache_default() {
        let cache: TuningCache = Default::default();
        // hit_rate is either 0.0 (empty) or 1.0 (has entries loaded from disk)
        assert!(cache.hit_rate() == 0.0 || cache.hit_rate() == 1.0);
    }
}
