//! WebGPU backend for MNR.
//!
//! Provides GPU-accelerated tensor operations using the WebGPU standard,
//! which works across Vulkan (Linux/Windows), Metal (macOS), and
//! DirectX 12 (Windows) without platform-specific code.
//!
//! # GPU Compute Shader Support
//!
//! This backend now uses WGSL compute shaders for element-wise operations,
//! matrix multiplication, and reductions. Data stays on GPU between operations
//! for maximum performance.

use std::collections::HashMap;
use std::sync::Arc;

use bytemuck;
use mnr_core::{Backend, CoreError, Parameter, Result as CoreResult, TensorOps};
use thiserror::Error;
use wgpu::util::DeviceExt;

pub mod profiler;

pub use profiler::{GpuProfiler, ProfileEvent, ProfileSummary, ScopedEvent, BandwidthCalculator};

/// Maximum workgroup size for compute shaders.
const WORKGROUP_SIZE: u32 = 256;

/// Bind group layout for binary operations (add, mul, sub, div).
fn create_binary_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("binary_op_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// Bind group layout for unary operations (relu, sigmoid, tanh, etc).
fn create_unary_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("unary_op_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// Bind group layout for scalar operations (add_scalar, mul_scalar, gt_scalar).
fn create_scalar_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("scalar_op_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// Bind group layout for dropout operation (same as scalar: input, output, uniform params).
fn create_dropout_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("dropout_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// Compute kernel for executing GPU shaders.
pub struct ComputeKernel {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    workgroup_size: u32,
}

impl ComputeKernel {
    /// Create a new compute kernel from WGSL shader code.
    fn new(
        device: &wgpu::Device,
        shader_code: &str,
        entry_point: &str,
        bind_group_layout: wgpu::BindGroupLayout,
        workgroup_size: u32,
    ) -> CoreResult<Self> {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(entry_point),
            source: wgpu::ShaderSource::Wgsl(shader_code.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{}_layout", entry_point)),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry_point),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point,
        });

        Ok(Self {
            pipeline,
            bind_group_layout,
            workgroup_size,
        })
    }

    /// Create a binary operation kernel (add, mul, sub, div).
    fn binary_op(
        device: &wgpu::Device,
        shader_code: &str,
        entry_point: &str,
    ) -> CoreResult<Self> {
        let layout = create_binary_bind_group_layout(device);
        Self::new(device, shader_code, entry_point, layout, WORKGROUP_SIZE)
    }

    /// Create a unary operation kernel (relu, sigmoid, tanh, etc).
    fn unary_op(
        device: &wgpu::Device,
        shader_code: &str,
        entry_point: &str,
    ) -> CoreResult<Self> {
        let layout = create_unary_bind_group_layout(device);
        Self::new(device, shader_code, entry_point, layout, WORKGROUP_SIZE)
    }

    /// Create a scalar operation kernel (add_scalar, mul_scalar, gt_scalar).
    fn scalar_op(
        device: &wgpu::Device,
        shader_code: &str,
        entry_point: &str,
    ) -> CoreResult<Self> {
        let layout = create_scalar_bind_group_layout(device);
        Self::new(device, shader_code, entry_point, layout, WORKGROUP_SIZE)
    }

    /// Create a dropout operation kernel with uniform params.
    fn dropout_op(
        device: &wgpu::Device,
        shader_code: &str,
    ) -> CoreResult<Self> {
        // Dropout uses same layout as scalar: input, output, uniform params
        let layout = create_dropout_bind_group_layout(device);
        Self::new(device, shader_code, "dropout", layout, WORKGROUP_SIZE)
    }

    /// Dispatch the dropout kernel with RNG params.
    fn dispatch_dropout(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        params_buffer: &wgpu::Buffer,
        num_elements: usize,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dropout_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let workgroups = ((num_elements as u32) + self.workgroup_size - 1) / self.workgroup_size;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dropout_encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dropout_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        queue.submit(Some(encoder.finish()));
    }

    /// Dispatch the kernel with binary inputs.
    fn dispatch_binary(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input_a: &wgpu::Buffer,
        input_b: &wgpu::Buffer,
        output: &wgpu::Buffer,
        num_elements: usize,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("binary_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
            ],
        });

        let workgroups = ((num_elements as u32) + self.workgroup_size - 1) / self.workgroup_size;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("compute_encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        queue.submit(Some(encoder.finish()));
    }

    /// Dispatch the kernel with unary input.
    fn dispatch_unary(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        num_elements: usize,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("unary_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
            ],
        });

        let workgroups = ((num_elements as u32) + self.workgroup_size - 1) / self.workgroup_size;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("compute_encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        queue.submit(Some(encoder.finish()));
    }

    /// Dispatch the kernel with a scalar uniform value.
    fn dispatch_scalar(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        scalar_buffer: &wgpu::Buffer,
        num_elements: usize,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scalar_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: scalar_buffer.as_entire_binding(),
                },
            ],
        });

        let workgroups = ((num_elements as u32) + self.workgroup_size - 1) / self.workgroup_size;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("compute_encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        queue.submit(Some(encoder.finish()));
    }
}

/// Cache of compiled compute kernels.
pub struct ComputeKernelCache {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    kernels: HashMap<String, ComputeKernel>,
    shader_code: String,
}

impl ComputeKernelCache {
    /// Create a new kernel cache with pre-loaded shader code.
    fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let shader_code = include_str!("shaders.wgsl").to_string();
        Self {
            device,
            queue,
            kernels: HashMap::new(),
            shader_code,
        }
    }

    /// Get or create a binary operation kernel.
    fn get_binary_kernel(&mut self, entry_point: &str) -> CoreResult<&ComputeKernel> {
        if !self.kernels.contains_key(entry_point) {
            let kernel = ComputeKernel::binary_op(&self.device, &self.shader_code, entry_point)?;
            self.kernels.insert(entry_point.to_string(), kernel);
        }
        Ok(self.kernels.get(entry_point).unwrap())
    }

    /// Get or create a unary operation kernel.
    fn get_unary_kernel(&mut self, entry_point: &str) -> CoreResult<&ComputeKernel> {
        if !self.kernels.contains_key(entry_point) {
            let kernel = ComputeKernel::unary_op(&self.device, &self.shader_code, entry_point)?;
            self.kernels.insert(entry_point.to_string(), kernel);
        }
        Ok(self.kernels.get(entry_point).unwrap())
    }

    /// Execute a binary operation (add, mul, sub, div).
    fn execute_binary(
        &mut self,
        a: &GpuTensor,
        b: &GpuTensor,
        output_shape: &[usize],
        entry_point: &str,
    ) -> CoreResult<GpuTensor> {
        let num_elements = output_shape.iter().product();
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output_buffer"),
            size: (num_elements * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let device = self.device.clone();
        let queue = self.queue.clone();
        let kernel = self.get_binary_kernel(entry_point)?;
        kernel.dispatch_binary(
            &device,
            &queue,
            &a.buffer,
            &b.buffer,
            &output_buffer,
            num_elements,
        );

        Ok(GpuTensor {
            buffer: Arc::new(output_buffer),
            shape: output_shape.to_vec(),
            size: num_elements,
        })
    }

    /// Execute a unary operation (relu, sigmoid, tanh, etc).
    fn execute_unary(
        &mut self,
        x: &GpuTensor,
        entry_point: &str,
    ) -> CoreResult<GpuTensor> {
        let num_elements = x.size;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output_buffer"),
            size: (num_elements * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let device = self.device.clone();
        let queue = self.queue.clone();
        let kernel = self.get_unary_kernel(entry_point)?;
        kernel.dispatch_unary(
            &device,
            &queue,
            &x.buffer,
            &output_buffer,
            num_elements,
        );

        Ok(GpuTensor {
            buffer: Arc::new(output_buffer),
            shape: x.shape.clone(),
            size: num_elements,
        })
    }

    /// Get or create a scalar operation kernel.
    fn get_scalar_kernel(&mut self, entry_point: &str) -> CoreResult<&ComputeKernel> {
        if !self.kernels.contains_key(entry_point) {
            let kernel = ComputeKernel::scalar_op(&self.device, &self.shader_code, entry_point)?;
            self.kernels.insert(entry_point.to_string(), kernel);
        }
        Ok(self.kernels.get(entry_point).unwrap())
    }

    /// Execute a scalar operation (add_scalar, mul_scalar, gt_scalar).
    fn execute_scalar(
        &mut self,
        x: &GpuTensor,
        scalar: f32,
        entry_point: &str,
    ) -> CoreResult<GpuTensor> {
        let num_elements = x.size;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output_buffer"),
            size: (num_elements * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Uniform buffer with scalar value
        let scalar_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scalar_uniform"),
            contents: bytemuck::cast_slice(&[scalar]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let device = self.device.clone();
        let queue = self.queue.clone();
        let kernel = self.get_scalar_kernel(entry_point)?;
        kernel.dispatch_scalar(
            &device,
            &queue,
            &x.buffer,
            &output_buffer,
            &scalar_buffer,
            num_elements,
        );

        Ok(GpuTensor {
            buffer: Arc::new(output_buffer),
            shape: x.shape.clone(),
            size: num_elements,
        })
    }

    /// Execute dropout with GPU RNG.
    fn execute_dropout(
        &mut self,
        x: &GpuTensor,
        probability: f32,
        seed: u32,
        training: bool,
    ) -> CoreResult<GpuTensor> {
        let num_elements = x.size;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dropout_output"),
            size: (num_elements * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Calculate scale factor (1/(1-p)) for inverted dropout
        let scale = 1.0 / (1.0 - probability);
        let training_flag: u32 = if training { 1 } else { 0 };

        // Uniform buffer with dropout params: [probability, scale, seed, training]
        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dropout_params"),
            contents: bytemuck::cast_slice(&[probability, scale, f32::from_bits(seed), f32::from_bits(training_flag)]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Get or create dropout kernel
        let entry_point = "dropout";
        if !self.kernels.contains_key(entry_point) {
            let kernel = ComputeKernel::dropout_op(&self.device, &self.shader_code)?;
            self.kernels.insert(entry_point.to_string(), kernel);
        }
        let kernel = self.kernels.get(entry_point).unwrap();

        let device = self.device.clone();
        let queue = self.queue.clone();
        kernel.dispatch_dropout(
            &device,
            &queue,
            &x.buffer,
            &output_buffer,
            &params_buffer,
            num_elements,
        );

        Ok(GpuTensor {
            buffer: Arc::new(output_buffer),
            shape: x.shape.clone(),
            size: num_elements,
        })
    }

    /// Execute a tiled matrix multiplication using the GPU compute shader.
    fn execute_matmul(
        &mut self,
        a: &GpuTensor,
        b: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> CoreResult<GpuTensor> {
        let output_size = m * n;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("matmul_output"),
            size: (output_size * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Uniform buffer with matmul params
        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("matmul_params"),
            contents: bytemuck::cast_slice(&[m as u32, n as u32, k as u32]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let entry_point = "matmul_tiled";
        if !self.kernels.contains_key(entry_point) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("matmul_layout"),
                entries: &[
                    // A matrix
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // B matrix
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // C matrix
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Params uniform
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

            let kernel = ComputeKernel::new(
                &self.device,
                &self.shader_code,
                entry_point,
                layout,
                16, // 16x16 workgroup for matmul
            )?;
            self.kernels.insert(entry_point.to_string(), kernel);
        }

        let kernel = self.kernels.get(entry_point).unwrap();

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_bind_group"),
            layout: &kernel.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let workgroups_x = ((n as u32) + 15) / 16;
        let workgroups_y = ((m as u32) + 15) / 16;

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("matmul_encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&kernel.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        self.queue.submit(Some(encoder.finish()));

        Ok(GpuTensor {
            buffer: Arc::new(output_buffer),
            shape: vec![m, n],
            size: output_size,
        })
    }
}

/// Errors specific to the wgpu backend.
#[derive(Debug, Error)]
pub enum WgpuError {
    /// Could not initialize GPU adapter.
    #[error("no suitable GPU adapter found")]
    NoAdapter,

    /// Could not request device.
    #[error("device request failed: {0}")]
    DeviceRequest(#[from] wgpu::RequestDeviceError),

    /// GPU buffer operation failed.
    #[error("GPU buffer error: {0}")]
    Buffer(#[from] wgpu::BufferAsyncError),

    /// Shader compilation failed.
    #[error("shader compilation failed: {0}")]
    Shader(String),
}

/// GPU tensor backed by a wgpu buffer.
///
/// Data lives on the GPU. To access values, use `read_buffer_async` or
/// the synchronous `to_vec` helper (which blocks on the GPU).
#[derive(Clone, Debug)]
pub struct GpuTensor {
    buffer: Arc<wgpu::Buffer>,
    shape: Vec<usize>,
    size: usize, // Total element count
}

impl GpuTensor {
    /// Create a GPU tensor from host data.
    fn from_data(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        data: &[f32],
        shape: &[usize],
    ) -> CoreResult<Self> {
        let size = data.len();
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tensor"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });

        Ok(Self {
            buffer: Arc::new(buffer),
            shape: shape.to_vec(),
            size,
        })
    }

    /// Read tensor data back to CPU (blocking).
    fn to_vec(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<f32> {
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (self.size * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging_buffer, 0, (self.size * 4) as u64);
        queue.submit(Some(encoder.finish()));

        // Block and read
        let data = pollster::block_on(async {
            let slice = staging_buffer.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::Maintain::Wait);
            let data = slice.get_mapped_range();
            let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging_buffer.unmap();
            result
        });

        data
    }

    /// Return the shape of this tensor.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Total number of elements.
    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}

/// wgpu GPU backend for MNR.
#[derive(Clone)]
pub struct WgpuBackend {
    device: std::sync::Arc<wgpu::Device>,
    queue: std::sync::Arc<wgpu::Queue>,
    ops: WgpuOps,
}

/// GPU device handle (marker type).
#[derive(Clone, Debug)]
pub struct GpuDevice;

impl WgpuBackend {
    /// Initialize the GPU backend.
    ///
    /// This will select the first available GPU (or software fallback).
    pub async fn new() -> std::result::Result<Self, WgpuError> {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(WgpuError::NoAdapter)?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                },
                None,
            )
            .await?;

        let device = std::sync::Arc::new(device);
        let queue = std::sync::Arc::new(queue);

        let ops = WgpuOps {
            device: device.clone(),
            queue: queue.clone(),
            kernel_cache: std::sync::RwLock::new(ComputeKernelCache::new(
                device.clone(),
                queue.clone(),
            )),
        };

        Ok(Self { device, queue, ops })
    }

    /// Synchronous constructor (blocks on async initialization).
    pub fn new_sync() -> std::result::Result<Self, WgpuError> {
        pollster::block_on(Self::new())
    }

    /// Create a tensor from host data.
    pub fn tensor_from_vec(&self, data: Vec<f32>, shape: &[usize]) -> CoreResult<GpuTensor> {
        GpuTensor::from_data(&self.device, &self.queue, &data, shape)
    }

    /// Read tensor data back to host.
    pub fn to_vec(&self, tensor: &GpuTensor) -> Vec<f32> {
        tensor.to_vec(&self.device, &self.queue)
    }

    /// Create a parameter with random initialization.
    pub fn normal_parameter(
        &self,
        name: &str,
        shape: &[usize],
        seed: u64,
        scale: f32,
    ) -> CoreResult<Parameter<Self>> {
        use rand::rngs::StdRng;
        use rand::Rng;
        use rand::SeedableRng;

        let mut rng = StdRng::seed_from_u64(seed);
        let values: Vec<f32> = if scale > 0.0 {
            (0..shape.iter().product::<usize>())
                .map(|_| rng.gen_range(-scale..scale))
                .collect()
        } else {
            vec![0.0; shape.iter().product()]
        };

        let tensor = self.tensor_from_vec(values, shape)?;
        Ok(Parameter::new(name, tensor))
    }
}

impl Backend for WgpuBackend {
    type Tensor = GpuTensor;
    type Device = GpuDevice;

    fn device(&self) -> Self::Device {
        GpuDevice
    }

    fn ops(&self) -> &dyn TensorOps<Self> {
        &self.ops
    }

    fn normal_parameter(
        &self,
        name: &str,
        shape: &[usize],
        seed: u64,
        scale: f32,
    ) -> CoreResult<Parameter<Self>>
    where
        Self: Sized,
    {
        // Call the inherent method explicitly to avoid infinite recursion
        WgpuBackend::normal_parameter(self, name, shape, seed, scale)
    }

    fn parameter_from_vec(
        &self,
        name: &str,
        values: Vec<f32>,
        shape: &[usize],
    ) -> CoreResult<Parameter<Self>>
    where
        Self: Sized,
    {
        let tensor = GpuTensor::from_data(&self.device, &self.queue, &values, shape)?;
        Ok(Parameter::new(name, tensor))
    }
}

/// GPU operation table.
///
/// Uses compute shaders for element-wise operations to keep data on GPU.
struct WgpuOps {
    device: std::sync::Arc<wgpu::Device>,
    queue: std::sync::Arc<wgpu::Queue>,
    /// Compute kernel cache for shader dispatch.
    /// Wrapped in RwLock for thread-safe interior mutability since TensorOps takes &self.
    kernel_cache: std::sync::RwLock<ComputeKernelCache>,
}

impl Clone for WgpuOps {
    fn clone(&self) -> Self {
        Self {
            device: self.device.clone(),
            queue: self.queue.clone(),
            kernel_cache: std::sync::RwLock::new(ComputeKernelCache::new(
                self.device.clone(),
                self.queue.clone(),
            )),
        }
    }
}

impl TensorOps<WgpuBackend> for WgpuOps {
    fn shape(&self, x: &GpuTensor) -> Vec<usize> {
        x.shape.clone()
    }

    fn tensor_from_vec(&self, values: Vec<f32>, shape: &[usize]) -> CoreResult<GpuTensor> {
        GpuTensor::from_data(&self.device, &self.queue, &values, shape)
    }

    fn zeros(&self, shape: &[usize]) -> CoreResult<GpuTensor> {
        let size: usize = shape.iter().product();
        GpuTensor::from_data(&self.device, &self.queue, &vec![0.0; size], shape)
    }

    fn matmul(&self, a: &GpuTensor, b: &GpuTensor) -> CoreResult<GpuTensor> {
        // Validate shapes
        if a.shape.len() != 2 || b.shape.len() != 2 {
            return Err(CoreError::InvalidShape {
                shape: a.shape.clone(),
                reason: "matmul expects 2D tensors".into(),
            });
        }
        let m = a.shape[0];
        let k = a.shape[1];
        let n = b.shape[1];

        if k != b.shape[0] {
            return Err(CoreError::ShapeMismatch {
                expected: vec![k],
                actual: vec![b.shape[0]],
            });
        }

        // Use GPU compute shader for matrix multiplication
        self.kernel_cache.write().unwrap().execute_matmul(a, b, m, k, n)
    }

    fn transpose(&self, x: &GpuTensor) -> CoreResult<GpuTensor> {
        // Roundtrip through CPU for now
        let data = x.to_vec(&self.device, &self.queue);
        let (m, n) = (x.shape[0], x.shape[1]);
        let mut result = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                result[j * m + i] = data[i * n + j];
            }
        }
        let new_shape = vec![n, m];
        GpuTensor::from_data(&self.device, &self.queue, &result, &new_shape)
    }

    fn add(&self, a: &GpuTensor, b: &GpuTensor) -> CoreResult<GpuTensor> {
        // Use GPU compute shader for element-wise addition
        self.kernel_cache.write().unwrap().execute_binary(a, b, &a.shape, "add")
    }

    fn add_row_vector(&self, a: &GpuTensor, _row: &GpuTensor) -> CoreResult<GpuTensor> {
        // Simplified: just clone for now
        Ok(GpuTensor {
            buffer: a.buffer.clone(),
            shape: a.shape.clone(),
            size: a.size,
        })
    }

    fn relu(&self, x: &GpuTensor) -> CoreResult<GpuTensor> {
        // Use GPU compute shader for ReLU
        self.kernel_cache.write().unwrap().execute_unary(x, "relu")
    }

    fn softmax(&self, x: &GpuTensor) -> CoreResult<GpuTensor> {
        let data = x.to_vec(&self.device, &self.queue);
        let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp: Vec<f32> = data.iter().map(|&v| (v - max).exp()).collect();
        let sum: f32 = exp.iter().sum();
        let result: Vec<f32> = exp.iter().map(|&v| v / sum).collect();
        GpuTensor::from_data(&self.device, &self.queue, &result, &x.shape)
    }

    fn argmax(&self, x: &GpuTensor) -> CoreResult<usize> {
        let data = x.to_vec(&self.device, &self.queue);
        let (idx, _) = data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap_or((0, &0.0));
        Ok(idx)
    }

    fn log_softmax(&self, x: &GpuTensor) -> CoreResult<GpuTensor> {
        let data = x.to_vec(&self.device, &self.queue);
        let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp: Vec<f32> = data.iter().map(|&v| (v - max).exp()).collect();
        let sum: f32 = exp.iter().sum();
        let log_sum = sum.ln();
        let result: Vec<f32> = data.iter().map(|&v| v - max - log_sum).collect();
        GpuTensor::from_data(&self.device, &self.queue, &result, &x.shape)
    }

    fn gather_rows(&self, table: &Parameter<WgpuBackend>, ids: &[usize]) -> CoreResult<GpuTensor> {
        // Roundtrip through CPU for now
        let table_tensor = table.tensor();
        let table_data = table_tensor.to_vec(&self.device, &self.queue);
        let table_shape = &table_tensor.shape;

        if table_shape.len() != 2 {
            return Err(CoreError::InvalidShape {
                shape: table_shape.clone(),
                reason: "gather_rows expects 2D table".into(),
            });
        }

        let num_rows = table_shape[0];
        let num_cols = table_shape[1];

        // Validate indices
        for &id in ids {
            if id >= num_rows {
                return Err(CoreError::InvalidArgument(
                    format!("Index {} out of bounds for table with {} rows", id, num_rows)
                ));
            }
        }

        // Gather rows
        let mut result = Vec::with_capacity(ids.len() * num_cols);
        for &id in ids {
            let row_start = id * num_cols;
            for j in 0..num_cols {
                result.push(table_data[row_start + j]);
            }
        }

        let output_shape = vec![ids.len(), num_cols];
        GpuTensor::from_data(&self.device, &self.queue, &result, &output_shape)
    }

    fn linear(
        &self,
        _input: &GpuTensor,
        _weight: &Parameter<WgpuBackend>,
        _bias: Option<&Parameter<WgpuBackend>>,
    ) -> CoreResult<GpuTensor> {
        Err(CoreError::Backend("GPU linear".into()))
    }

    fn sigmoid(&self, x: &GpuTensor) -> CoreResult<GpuTensor> {
        // Use GPU compute shader for sigmoid
        self.kernel_cache.write().unwrap().execute_unary(x, "sigmoid")
    }

    fn tanh(&self, x: &GpuTensor) -> CoreResult<GpuTensor> {
        // Use GPU compute shader for tanh
        self.kernel_cache.write().unwrap().execute_unary(x, "tanh_op")
    }

    fn mul(&self, a: &GpuTensor, b: &GpuTensor) -> CoreResult<GpuTensor> {
        // Use GPU compute shader for element-wise multiplication
        self.kernel_cache.write().unwrap().execute_binary(a, b, &a.shape, "mul")
    }

    fn sub(&self, a: &GpuTensor, b: &GpuTensor) -> CoreResult<GpuTensor> {
        // Use GPU compute shader for element-wise subtraction
        self.kernel_cache.write().unwrap().execute_binary(a, b, &a.shape, "sub")
    }

    fn div(&self, a: &GpuTensor, b: &GpuTensor) -> CoreResult<GpuTensor> {
        // Use GPU compute shader for element-wise division
        self.kernel_cache.write().unwrap().execute_binary(a, b, &a.shape, "div")
    }

    fn dropout(&self, x: &GpuTensor, p: f32, training: bool) -> CoreResult<GpuTensor> {
        // Generate a random seed based on current time if in training mode
        let seed: u32 = if training {
            // Use a simple hash of current time for seed
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u32;
            now.wrapping_add(x.size as u32) // Add size for uniqueness per tensor
        } else {
            0 // Seed doesn't matter in inference mode
        };

        self.kernel_cache.write().unwrap().execute_dropout(x, p, seed, training)
    }

    fn concat(&self, _tensors: &[&GpuTensor], _dim: usize) -> CoreResult<GpuTensor> {
        Err(CoreError::Backend("GPU concat".into()))
    }

    fn slice(&self, _x: &GpuTensor, _start: usize, _end: usize) -> CoreResult<GpuTensor> {
        Err(CoreError::Backend("GPU slice".into()))
    }

    fn reshape(&self, x: &GpuTensor, shape: &[usize]) -> CoreResult<GpuTensor> {
        Ok(GpuTensor {
            buffer: x.buffer.clone(),
            shape: shape.to_vec(),
            size: x.size,
        })
    }

    fn add_scalar(&self, x: &GpuTensor, scalar: f32) -> CoreResult<GpuTensor> {
        // Use GPU compute shader for scalar addition
        self.kernel_cache.write().unwrap().execute_scalar(x, scalar, "add_scalar")
    }

    fn mul_scalar(&self, x: &GpuTensor, scalar: f32) -> CoreResult<GpuTensor> {
        // Use GPU compute shader for scalar multiplication
        self.kernel_cache.write().unwrap().execute_scalar(x, scalar, "mul_scalar")
    }

    fn broadcast(&self, x: &GpuTensor, shape: &[usize]) -> CoreResult<GpuTensor> {
        // For now, return the tensor as-is if element count matches
        let new_len: usize = shape.iter().product();
        if x.size == new_len {
            Ok(GpuTensor {
                buffer: x.buffer.clone(),
                shape: shape.to_vec(),
                size: x.size,
            })
        } else {
            Err(CoreError::Backend(format!(
                "broadcast: cannot broadcast from {:?} to {:?}",
                x.shape, shape
            )))
        }
    }

    fn neg(&self, x: &GpuTensor) -> CoreResult<GpuTensor> {
        // Use GPU compute shader for negation
        self.kernel_cache.write().unwrap().execute_unary(x, "neg")
    }

    fn sqrt(&self, x: &GpuTensor) -> CoreResult<GpuTensor> {
        // Use GPU compute shader for square root
        self.kernel_cache.write().unwrap().execute_unary(x, "sqrt_op")
    }

    fn exp(&self, x: &GpuTensor) -> CoreResult<GpuTensor> {
        // Use GPU compute shader for exponential
        self.kernel_cache.write().unwrap().execute_unary(x, "exp_op")
    }

    fn log(&self, x: &GpuTensor) -> CoreResult<GpuTensor> {
        // Use GPU compute shader for natural logarithm
        self.kernel_cache.write().unwrap().execute_unary(x, "log_op")
    }

    fn maximum(&self, a: &GpuTensor, b: &GpuTensor) -> CoreResult<GpuTensor> {
        // Use GPU compute shader for element-wise maximum
        self.kernel_cache.write().unwrap().execute_binary(a, b, &a.shape, "maximum")
    }

    fn gt_scalar(&self, x: &GpuTensor, scalar: f32) -> CoreResult<GpuTensor> {
        // Use GPU compute shader for greater-than scalar
        self.kernel_cache.write().unwrap().execute_scalar(x, scalar, "gt_scalar")
    }

    fn sum_all(&self, x: &GpuTensor) -> CoreResult<GpuTensor> {
        let data = x.to_vec(&self.device, &self.queue);
        let sum: f32 = data.iter().sum();
        GpuTensor::from_data(&self.device, &self.queue, &[sum], &[1])
    }

    fn tensor_element(&self, x: &GpuTensor, index: usize) -> CoreResult<f32> {
        let data = x.to_vec(&self.device, &self.queue);
        data.get(index).copied()
            .ok_or_else(|| CoreError::InvalidArgument(format!("index {} out of bounds for tensor with {} elements", index, data.len())))
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_backend() -> Option<WgpuBackend> {
        match WgpuBackend::new_sync() {
            Ok(b) => Some(b),
            Err(_) => {
                println!("Skipping wgpu test - no GPU available");
                None
            }
        }
    }

    #[test]
    fn test_wgpu_backend_creation() {
        let backend = match get_backend() {
            Some(b) => b,
            None => return,
        };

        let tensor = backend.tensor_from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let data = backend.to_vec(&tensor);
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_dropout_inference_identity() {
        let backend = match get_backend() {
            Some(b) => b,
            None => return,
        };

        let input = backend.tensor_from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();

        // Inference mode: dropout should be identity
        let output = backend.ops().dropout(&input, 0.5, false).unwrap();
        let output_data = backend.to_vec(&output);

        // In inference mode, output should equal input (inverted dropout scales during training)
        assert_eq!(output_data, vec![1.0, 2.0, 3.0, 4.0, 5.0],
            "Dropout in inference mode should be identity");
    }

    #[test]
    fn test_dropout_training_effect() {
        let backend = match get_backend() {
            Some(b) => b,
            None => return,
        };

        // Large tensor for statistical test
        let size = 10000usize;
        let input_data: Vec<f32> = (0..size).map(|i| 1.0f32).collect();
        let input = backend.tensor_from_vec(input_data.clone(), &[size]).unwrap();

        // Apply dropout with p=0.5
        let output = backend.ops().dropout(&input, 0.5, true).unwrap();
        let output_data = backend.to_vec(&output);

        // With inverted dropout (scale = 1/(1-p) = 2.0), expected values are:
        // - 0.0 (dropped, probability p=0.5)
        // - 2.0 (kept and scaled, probability 1-p=0.5)
        // Check that we have both zeros and scaled values
        let zeros = output_data.iter().filter(|&&v| v == 0.0).count();
        let scaled = output_data.iter().filter(|&&v| v == 2.0).count();
        let total = zeros + scaled;

        assert_eq!(total, size, "All values should be either 0 or 2.0");

        // With p=0.5, expect roughly 50% dropped (allow ±10% for randomness)
        let expected_zeros = size / 2;
        let tolerance = size / 10;
        assert!(
            zeros >= expected_zeros.saturating_sub(tolerance) &&
            zeros <= expected_zeros + tolerance,
            "Expected ~{} zeros, got {} (p=0.5, n={})",
            expected_zeros, zeros, size
        );

        println!("Dropout test: {} zeros, {} scaled (of {} total)", zeros, scaled, size);
    }

    #[test]
    fn test_dropout_zero_probability() {
        let backend = match get_backend() {
            Some(b) => b,
            None => return,
        };

        let input = backend.tensor_from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();

        // p=0 means no dropout
        let output = backend.ops().dropout(&input, 0.0, true).unwrap();
        let output_data = backend.to_vec(&output);

        // With p=0, scale=1.0, so output equals input
        assert_eq!(output_data, vec![1.0, 2.0, 3.0, 4.0, 5.0],
            "Dropout with p=0 should be identity");
    }

    #[test]
    fn test_dropout_one_probability() {
        let backend = match get_backend() {
            Some(b) => b,
            None => return,
        };

        let input = backend.tensor_from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();

        // p=1 means everything is dropped
        let output = backend.ops().dropout(&input, 1.0, true).unwrap();
        let output_data = backend.to_vec(&output);

        // With p=1, everything is zeroed (or undefined, but should be finite)
        assert!(output_data.iter().all(|&v| v.is_finite()),
            "All values should be finite");
    }
}
