//! MNIST Handwritten Digit Classification Example
//!
//! This example implements a LeNet-5 style CNN for MNIST classification.
//! Architecture: Conv2d -> ReLU -> MaxPool -> Conv2d -> ReLU -> MaxPool -> Linear -> ReLU -> Linear -> 10 outputs
//!
//! # Running this example
//!
//! Download MNIST data files and place them in `data/mnist/`:
//! - train-images-idx3-ubyte.gz
//! - train-labels-idx1-ubyte.gz
//! - t10k-images-idx3-ubyte.gz
//! - t10k-labels-idx1-ubyte.gz
//!
//! Then run: `cargo run --bin mnist`

use mnr_core::{Backend, ForwardCtx, Mode, Module};
use mnr_data::{DataLoader, DataLoaderConfig, Dataset};
use mnr_ndarray_backend::CpuBackend;
use mnr_nn::{Conv2d, Conv2dConfig, CrossEntropyLoss, LinearBuilder, max_pool2d};
use std::fs::File;
use std::io::{Read, BufReader, Result as IoResult};

/// MNIST image: 28x28 pixels, flattened to 784 values
#[derive(Clone)]
pub struct MnistSample {
    pub image: Vec<f32>,  // 784 elements, normalized to [0, 1]
    pub label: usize,     // 0-9
}

/// MNIST dataset loader
pub struct MnistDataset {
    samples: Vec<MnistSample>,
}

impl MnistDataset {
    /// Load MNIST dataset from ubyte files
    pub fn load(images_path: &str, labels_path: &str) -> IoResult<Self> {
        let images = Self::load_images(images_path)?;
        let labels = Self::load_labels(labels_path)?;

        let samples: Vec<MnistSample> = images.into_iter()
            .zip(labels.into_iter())
            .map(|(image, label)| MnistSample { image, label })
            .collect();

        println!("Loaded {} MNIST samples", samples.len());
        Ok(Self { samples })
    }

    fn load_images(path: &str) -> IoResult<Vec<Vec<f32>>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read magic number and header
        let mut header = [0u8; 16];
        reader.read_exact(&mut header)?;

        let magic = u32::from_be_bytes([header[0], header[1], header[2], header[3]]);
        if magic != 2051 {
            panic!("Invalid MNIST image file magic number: {}", magic);
        }

        let num_images = u32::from_be_bytes([header[4], header[5], header[6], header[7]]) as usize;
        let num_rows = u32::from_be_bytes([header[8], header[9], header[10], header[11]]) as usize;
        let num_cols = u32::from_be_bytes([header[12], header[13], header[14], header[15]]) as usize;

        println!("Loading {} images of size {}x{}", num_images, num_rows, num_cols);

        let mut images = Vec::with_capacity(num_images);
        let mut pixel_buffer = vec![0u8; num_rows * num_cols];

        for _ in 0..num_images {
            reader.read_exact(&mut pixel_buffer)?;
            let pixels: Vec<f32> = pixel_buffer.iter()
                .map(|&p| p as f32 / 255.0)  // Normalize to [0, 1]
                .collect();
            images.push(pixels);
        }

        Ok(images)
    }

    fn load_labels(path: &str) -> IoResult<Vec<usize>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read magic number and count
        let mut header = [0u8; 8];
        reader.read_exact(&mut header)?;

        let magic = u32::from_be_bytes([header[0], header[1], header[2], header[3]]);
        if magic != 2049 {
            panic!("Invalid MNIST label file magic number: {}", magic);
        }

        let num_labels = u32::from_be_bytes([header[4], header[5], header[6], header[7]]) as usize;
        println!("Loading {} labels", num_labels);

        let mut labels = Vec::with_capacity(num_labels);
        let mut label_buffer = vec![0u8; num_labels];
        reader.read_exact(&mut label_buffer)?;

        for &label in &label_buffer {
            labels.push(label as usize);
        }

        Ok(labels)
    }
}

impl Dataset<MnistSample> for MnistDataset {
    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, index: usize) -> Option<MnistSample> {
        self.samples.get(index).cloned()
    }
}

/// LeNet-5 style model for MNIST
pub struct MnistModel {
    conv1: Conv2d<CpuBackend>,
    conv2: Conv2d<CpuBackend>,
    fc1: mnr_nn::Linear<CpuBackend>,
    fc2: mnr_nn::Linear<CpuBackend>,
}

impl MnistModel {
    pub fn new(backend: &CpuBackend) -> Self {
        // Conv2d(1 -> 6, 5x5) -> ReLU -> MaxPool(2x2)
        let conv1_config = Conv2dConfig::new(6, 5, 5)
            .with_stride(1, 1)
            .with_bias();
        let filter1 = backend.normal_parameter("conv1/filter", &[6, 1, 5, 5], 42, 0.1).unwrap();
        let bias1 = backend.normal_parameter("conv1/bias", &[6], 43, 0.1).unwrap();
        let conv1 = Conv2d::from_parameters(conv1_config, filter1, Some(bias1));

        // Conv2d(6 -> 16, 5x5) -> ReLU -> MaxPool(2x2)
        let conv2_config = Conv2dConfig::new(16, 5, 5)
            .with_stride(1, 1)
            .with_bias();
        let filter2 = backend.normal_parameter("conv2/filter", &[16, 6, 5, 5], 44, 0.1).unwrap();
        let bias2 = backend.normal_parameter("conv2/bias", &[16], 45, 0.1).unwrap();
        let conv2 = Conv2d::from_parameters(conv2_config, filter2, Some(bias2));

        // After two conv layers and two 2x2 max pools:
        // Input: 28x28 -> Conv1: 24x24 -> Pool1: 12x12 -> Conv2: 8x8 -> Pool2: 4x4
        // Flattened: 16 channels * 4 * 4 = 256

        // FC1: 256 -> 120
        let fc1 = LinearBuilder::new(256, 120)
            .with_bias(true)
            .build(backend)
            .expect("Failed to create fc1");

        // FC2: 120 -> 10
        let fc2 = LinearBuilder::new(120, 10)
            .with_bias(true)
            .build(backend)
            .expect("Failed to create fc2");

        Self { conv1, conv2, fc1, fc2 }
    }

    /// Forward pass through the network
    pub fn forward(&self, image: &[f32], ctx: &mut ForwardCtx<CpuBackend>) -> Vec<f32> {
        let ops = ctx.backend().ops();

        // Reshape to [1, 1, 28, 28] - batch, channels, height, width
        let input = ops.tensor_from_vec(image.to_vec(), &[1, 1, 28, 28]).unwrap();

        // Conv1: [1, 1, 28, 28] -> [1, 6, 24, 24]
        let x = self.conv1.forward(input, ctx).unwrap();
        let x = ops.relu(&x).unwrap();
        let x = max_pool2d(&x, 2, 2, 2, 2, true, ctx.backend().ops()).unwrap(); // [1, 6, 12, 12]

        // Conv2: [1, 6, 12, 12] -> [1, 16, 8, 8]
        let x = self.conv2.forward(x, ctx).unwrap();
        let x = ops.relu(&x).unwrap();
        let x = max_pool2d(&x, 2, 2, 2, 2, true, ctx.backend().ops()).unwrap(); // [1, 16, 4, 4]

        // Flatten: [16, 4, 4] or [1, 16, 4, 4] -> [1, 256]
        let shape = ops.shape(&x);
        let flattened_size = if shape.len() == 4 {
            shape[1] * shape[2] * shape[3]
        } else {
            shape[0] * shape[1] * shape[2]
        };
        let flattened = ops.reshape(&x, &[1, flattened_size]).unwrap();

        // FC1: [1, 256] -> [1, 120]
        let x = self.fc1.forward(flattened, ctx).unwrap();
        let x = ops.relu(&x).unwrap();

        // FC2: [1, 120] -> [1, 10]
        let logits = self.fc2.forward(x, ctx).unwrap();

        // Extract output values
        let output_shape = ops.shape(&logits);
        let num_classes = output_shape[1];
        let mut result = Vec::with_capacity(num_classes);
        for i in 0..num_classes {
            result.push(ops.tensor_element(&logits, i).unwrap());
        }

        result
    }

    /// Count trainable parameters
    pub fn num_parameters(&self) -> usize {
        let mut total = 0;
        // Conv1: 6 filters * 1 channel * 5 * 5 = 150 weights + 6 biases
        total += 6 * 1 * 5 * 5 + 6;
        // Conv2: 16 filters * 6 channels * 5 * 5 = 2400 weights + 16 biases
        total += 16 * 6 * 5 * 5 + 16;
        // FC1: 256 * 120 = 30720 weights + 120 biases
        total += 256 * 120 + 120;
        // FC2: 120 * 10 = 1200 weights + 10 biases
        total += 120 * 10 + 10;
        total
    }
}

fn main() {
    println!("MNIST Digit Classification Example");
    println!("==================================\n");

    // Initialize backend
    let backend = CpuBackend::default();

    // Check for MNIST data files
    let data_dir = "data/mnist";
    let train_images_path = format!("{}/train-images-idx3-ubyte", data_dir);
    let train_labels_path = format!("{}/train-labels-idx1-ubyte", data_dir);
    let test_images_path = format!("{}/t10k-images-idx3-ubyte", data_dir);
    let test_labels_path = format!("{}/t10k-labels-idx1-ubyte", data_dir);

    // Try to load MNIST dataset
    let train_dataset = match MnistDataset::load(&train_images_path, &train_labels_path) {
        Ok(ds) => ds,
        Err(e) => {
            println!("Could not load MNIST training data: {}", e);
            println!("\nTo run this example, download MNIST from:");
            println!("  https://yann.lecun.com/exdb/mnist/");
            println!("\nPlace the extracted files in: {}", data_dir);
            println!("  (unzip the .gz files first: gunzip *.gz)");
            println!("\nRunning demo mode with synthetic data instead.\n");
            return run_demo(&backend);
        }
    };

    let _test_dataset = match MnistDataset::load(&test_images_path, &test_labels_path) {
        Ok(ds) => ds,
        Err(e) => {
            println!("Could not load MNIST test data: {}", e);
            return;
        }
    };

    // Create model
    let model = MnistModel::new(&backend);
    println!("Model created with {} parameters\n", model.num_parameters());

    // Create data loaders
    let mut train_loader = DataLoader::new(
        Box::new(train_dataset),
        DataLoaderConfig {
            batch_size: 32,
            shuffle: true,
            seed: Some(42),
            num_workers: 0,
        },
    );

    // Evaluate on a few training samples
    println!("Evaluating on training samples:");
    let mut correct = 0;
    let mut total = 0;

    for (batch_idx, batch) in train_loader.by_ref().take(10).enumerate() {
        for sample in batch {
            let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
            let logits = model.forward(&sample.image, &mut ctx);

            // Find predicted class (argmax)
            let predicted = logits.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();

            if predicted == sample.label {
                correct += 1;
            }
            total += 1;

            if batch_idx == 0 && total <= 5 {
                println!("  Sample {}: Predicted={}, True={}", total, predicted, sample.label);
            }
        }
    }

    println!("\nAccuracy on first 10 batches: {}/{} ({:.1}%)",
        correct, total, 100.0 * correct as f32 / total as f32);

    // Note about full training
    println!("\nNote: Full training with backprop requires completing the autodiff integration.");
    println!("The forward pass is fully functional; gradients can be computed for individual ops.");
}

/// Run a demo with synthetic data when MNIST files aren't available
fn run_demo(backend: &CpuBackend) {
    println!("Demo Mode with Synthetic Data");
    println!("-----------------------------\n");

    // Create a small synthetic dataset
    let mut synthetic_data = Vec::new();
    for label in 0..10 {
        // Create 10 samples per class with some pattern
        for _i in 0..10 {
            let mut image = vec![0.0f32; 784];
            // Add some pattern based on label
            for row in 0..28 {
                for col in 0..28 {
                    let idx = row * 28 + col;
                    // Simple pattern: diagonal lines
                    image[idx] = if (row + col + label * 3) % 7 == 0 { 1.0 } else { 0.0 };
                }
            }
            synthetic_data.push(MnistSample { image, label });
        }
    }

    // Create model
    let model = MnistModel::new(backend);
    println!("Model created with {} parameters\n", model.num_parameters());

    // Evaluate on synthetic data
    let mut correct = 0;
    let loss_fn = CrossEntropyLoss::new();

    for (idx, sample) in synthetic_data.iter().take(20).enumerate() {
        let mut ctx = ForwardCtx::new(backend, Mode::Inference);
        let logits = model.forward(&sample.image, &mut ctx);

        // Find predicted class
        let predicted = logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        // Compute loss (requires tensor format)
        let logits_tensor = backend.ops()
            .tensor_from_vec(logits.clone(), &[1, 10])
            .unwrap();
        let loss = loss_fn.forward_indices(&logits_tensor, &[sample.label], &mut ctx).unwrap();
        let loss_val = backend.ops().tensor_element(&loss, 0).unwrap();

        if predicted == sample.label {
            correct += 1;
        }

        println!("  Sample {}: Predicted={}, True={}, Loss={:.4}",
            idx, predicted, sample.label, loss_val);
    }

    println!("\nDemo accuracy: {}/20 ({:.1}%)", correct, 100.0 * correct as f32 / 20.0);
}
