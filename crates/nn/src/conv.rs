//! Convolutional layers for neural networks.

use mnr_core::{Backend, CoreError, ForwardCtx, Module, Parameter, ParameterRef, Result, ShapeExt, Trainable};
use serde::{Deserialize, Serialize};

/// Configuration for a 2D convolutional layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Conv2dConfig {
    /// Number of output channels.
    pub out_channels: usize,
    /// Filter height.
    pub kernel_h: usize,
    /// Filter width.
    pub kernel_w: usize,
    /// Stride between rows.
    pub stride_h: usize,
    /// Stride between columns.
    pub stride_w: usize,
    /// Whether to use bias.
    pub bias: bool,
    /// Whether to disable padding (valid convolution).
    pub no_padding: bool,
}

impl Conv2dConfig {
    /// Create a new Conv2D configuration.
    pub fn new(out_channels: usize, kernel_h: usize, kernel_w: usize) -> Self {
        Self {
            out_channels,
            kernel_h,
            kernel_w,
            stride_h: 1,
            stride_w: 1,
            bias: false,
            no_padding: true,
        }
    }

    /// Set stride.
    pub fn with_stride(mut self, h: usize, w: usize) -> Self {
        self.stride_h = h;
        self.stride_w = w;
        self
    }

    /// Enable bias.
    pub fn with_bias(mut self) -> Self {
        self.bias = true;
        self
    }

    /// Enable padding.
    pub fn with_padding(mut self) -> Self {
        self.no_padding = false;
        self
    }
}

/// 2D convolutional layer.
pub struct Conv2d<B: Backend> {
    config: Conv2dConfig,
    filter: Parameter<B>,
    bias: Option<Parameter<B>>,
}

impl<B: Backend> Conv2d<B> {
    /// Create a Conv2D layer from explicit parameters.
    pub fn from_parameters(config: Conv2dConfig, filter: Parameter<B>, bias: Option<Parameter<B>>) -> Self {
        Self { config, filter, bias }
    }

    /// Calculate output dimensions for given input.
    pub fn output_size(&self, input_h: usize, input_w: usize) -> (usize, usize) {
        let (out_h, out_w) = if self.config.no_padding {
            (
                (input_h - self.config.kernel_h) / self.config.stride_h + 1,
                (input_w - self.config.kernel_w) / self.config.stride_w + 1,
            )
        } else {
            (
                (input_h + self.config.stride_h - 1) / self.config.stride_h,
                (input_w + self.config.stride_w - 1) / self.config.stride_w,
            )
        };
        (out_h, out_w)
    }

    /// Access the configuration.
    pub fn config(&self) -> &Conv2dConfig {
        &self.config
    }
}

impl<B: Backend> Module<B> for Conv2d<B> {
    type Input = B::Tensor;
    type Output = B::Tensor;

    fn forward(&self, input: Self::Input, ctx: &mut ForwardCtx<B>) -> Result<Self::Output> {
        let ops = ctx.backend().ops();
        let input_shape = ops.shape(&input);

        // Expected input: [batch, in_channels, height, width] or [in_channels, height, width]
        let (batch, in_channels, in_h, in_w) = match input_shape.len() {
            3 => (1, input_shape[0], input_shape[1], input_shape[2]),
            4 => (input_shape[0], input_shape[1], input_shape[2], input_shape[3]),
            _ => return Err(CoreError::InvalidShape {
                shape: input_shape,
                reason: "Conv2d expects 3D [C,H,W] or 4D [N,C,H,W] input".into(),
            }),
        };

        let (out_h, out_w) = self.output_size(in_h, in_w);
        let out_channels = self.config.out_channels;

        // Get filter tensor and reshape for computation
        // Filter shape: [out_channels, in_channels, kernel_h, kernel_w]
        let filter = self.filter.tensor();
        let filter_shape = ops.shape(filter);
        if filter_shape.len() != 4 {
            return Err(CoreError::InvalidShape {
                shape: filter_shape,
                reason: "Conv2d filter must be 4D [out_c, in_c, kH, kW]".into(),
            });
        }

        let kernel_h = self.config.kernel_h;
        let kernel_w = self.config.kernel_w;
        let stride_h = self.config.stride_h;
        let stride_w = self.config.stride_w;
        let padding = !self.config.no_padding;

        // Extract input values - this is O(n) scalar extraction
        // For production use, backends should implement Conv2d via specialized kernels
        let input_values: Vec<f32> = (0..input_shape.elem_count())
            .filter_map(|i| ops.tensor_element(&input, i).ok())
            .collect();

        // Extract filter values
        let filter_values: Vec<f32> = (0..filter_shape.elem_count())
            .filter_map(|i| ops.tensor_element(filter, i).ok())
            .collect();

        // Perform convolution
        let mut output_values = vec![0.0f32; batch * out_channels * out_h * out_w];

        for n in 0..batch {
            for oc in 0..out_channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0f32;

                        for ic in 0..in_channels {
                            for kh in 0..kernel_h {
                                for kw in 0..kernel_w {
                                    let (ih, iw) = if padding {
                                        let ih_start = (oh * stride_h).saturating_sub(kernel_h / 2);
                                        let iw_start = (ow * stride_w).saturating_sub(kernel_w / 2);
                                        (ih_start + kh, iw_start + kw)
                                    } else {
                                        (oh * stride_h + kh, ow * stride_w + kw)
                                    };

                                    if ih < in_h && iw < in_w {
                                        // Input index
                                        let input_idx = n * in_channels * in_h * in_w
                                            + ic * in_h * in_w
                                            + ih * in_w
                                            + iw;
                                        // Filter index
                                        let filter_idx = oc * in_channels * kernel_h * kernel_w
                                            + ic * kernel_h * kernel_w
                                            + kh * kernel_w
                                            + kw;

                                        if input_idx < input_values.len() && filter_idx < filter_values.len() {
                                            sum += input_values[input_idx] * filter_values[filter_idx];
                                        }
                                    }
                                }
                            }
                        }

                        // Add bias if present
                        if let Some(ref bias) = self.bias {
                            sum += ops.tensor_element(bias.tensor(), oc).unwrap_or(0.0);
                        }

                        let output_idx = n * out_channels * out_h * out_w
                            + oc * out_h * out_w
                            + oh * out_w
                            + ow;
                        output_values[output_idx] = sum;
                    }
                }
            }
        }

        let output_shape = if batch == 1 {
            vec![out_channels, out_h, out_w]
        } else {
            vec![batch, out_channels, out_h, out_w]
        };

        ops.tensor_from_vec(output_values, &output_shape)
    }
}

impl<B: Backend> Trainable<B> for Conv2d<B> {
    fn parameters(&self) -> Vec<ParameterRef> {
        let mut params = vec![ParameterRef { id: self.filter.id() }];
        if let Some(ref b) = self.bias {
            params.push(ParameterRef { id: b.id() });
        }
        params
    }
}

/// Max pooling 2D operation.
pub fn max_pool2d<B: Backend>(
    input: &B::Tensor,
    window_h: usize,
    window_w: usize,
    stride_h: usize,
    stride_w: usize,
    no_padding: bool,
    ops: &dyn mnr_core::TensorOps<B>,
) -> Result<B::Tensor> {
    let input_shape = ops.shape(input);

    // Expected input: [channels, height, width] or [batch, channels, height, width]
    let (batch, channels, in_h, in_w) = match input_shape.len() {
        3 => (1, input_shape[0], input_shape[1], input_shape[2]),
        4 => (input_shape[0], input_shape[1], input_shape[2], input_shape[3]),
        _ => return Err(CoreError::InvalidShape {
            shape: input_shape.clone(),
            reason: "max_pool2d expects 3D [C,H,W] or 4D [N,C,H,W] input".into(),
        }),
    };

    // Calculate output dimensions
    let out_h = if no_padding {
        (in_h - window_h) / stride_h + 1
    } else {
        (in_h + stride_h - 1) / stride_h
    };
    let out_w = if no_padding {
        (in_w - window_w) / stride_w + 1
    } else {
        (in_w + stride_w - 1) / stride_w
    };

    // Extract input values
    let input_values: Vec<f32> = (0..input_shape.elem_count())
        .filter_map(|i| ops.tensor_element(input, i).ok())
        .collect();

    let mut output_values = vec![f32::NEG_INFINITY; batch * channels * out_h * out_w];

    for n in 0..batch {
        for c in 0..channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let h_start = oh * stride_h;
                    let w_start = ow * stride_w;

                    for kh in 0..window_h {
                        for kw in 0..window_w {
                            let ih = h_start + kh;
                            let iw = w_start + kw;

                            if ih < in_h && iw < in_w {
                                let input_idx = n * channels * in_h * in_w
                                    + c * in_h * in_w
                                    + ih * in_w
                                    + iw;
                                let output_idx = n * channels * out_h * out_w
                                    + c * out_h * out_w
                                    + oh * out_w
                                    + ow;

                                if input_idx < input_values.len() && output_idx < output_values.len() {
                                    output_values[output_idx] = output_values[output_idx].max(input_values[input_idx]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let output_shape = if batch == 1 {
        vec![channels, out_h, out_w]
    } else {
        vec![batch, channels, out_h, out_w]
    };

    ops.tensor_from_vec(output_values, &output_shape)
}

/// Global max pooling over spatial dimensions.
pub fn global_max_pool2d<B: Backend>(input: &B::Tensor, ops: &dyn mnr_core::TensorOps<B>) -> Result<B::Tensor> {
    let shape = ops.shape(input);
    if shape.len() != 3 && shape.len() != 4 {
        return Err(CoreError::InvalidShape {
            shape: shape.clone(),
            reason: "global_max_pool2d expects 3D [C,H,W] or 4D [N,C,H,W] tensor".into(),
        });
    }

    let (h, w) = if shape.len() == 3 {
        (shape[1], shape[2])
    } else {
        (shape[2], shape[3])
    };

    max_pool2d(input, h, w, 1, 1, true, ops)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_config() {
        let config = Conv2dConfig::new(64, 3, 3)
            .with_stride(2, 2)
            .with_bias()
            .with_padding();

        assert_eq!(config.out_channels, 64);
        assert_eq!(config.kernel_h, 3);
        assert_eq!(config.kernel_w, 3);
        assert_eq!(config.stride_h, 2);
        assert_eq!(config.stride_w, 2);
        assert!(config.bias);
        assert!(!config.no_padding);
    }

    #[test]
    fn test_output_size_no_padding() {
        let config = Conv2dConfig::new(32, 3, 3);
        let mock_backend = mnr_ndarray_backend::CpuBackend::default();
        let filter = mock_backend.normal_parameter("W", &[32, 3, 3, 3], 42, 0.1).unwrap();
        let conv = Conv2d::from_parameters(config.clone(), filter, None);

        let (h, w) = conv.output_size(28, 28);
        assert_eq!(h, 26);  // (28 - 3) / 1 + 1
        assert_eq!(w, 26);
    }

    #[test]
    fn test_output_size_with_stride() {
        let config = Conv2dConfig::new(32, 3, 3).with_stride(2, 2);
        let mock_backend = mnr_ndarray_backend::CpuBackend::default();
        let filter = mock_backend.normal_parameter("W", &[32, 3, 3, 3], 42, 0.1).unwrap();
        let conv = Conv2d::from_parameters(config.clone(), filter, None);

        let (h, w) = conv.output_size(28, 28);
        assert_eq!(h, 13);  // (28 - 3) / 2 + 1
        assert_eq!(w, 13);
    }

    #[test]
    fn test_conv2d_forward_3d() {
        let backend = mnr_ndarray_backend::CpuBackend::default();
        let mut ctx = mnr_core::ForwardCtx::new(&backend, mnr_core::Mode::Inference);

        let config = Conv2dConfig::new(2, 2, 2).with_stride(1, 1);
        // 2 output channels, 1 input channel, 2x2 kernel
        let filter = backend.tensor_from_vec(
            vec![1.0, 0.0, 0.0, 1.0,  // output channel 0 (identity-like)
                 0.0, 1.0, 1.0, 0.0], // output channel 1 (swapped)
            &[2, 1, 2, 2]
        ).unwrap();
        let filter_param = mnr_core::Parameter::new("W", filter);

        let conv = Conv2d::from_parameters(config, filter_param, None);

        // Input: 1 channel, 3x3 image
        let input = backend.tensor_from_vec(
            vec![1.0, 2.0, 3.0,
                 4.0, 5.0, 6.0,
                 7.0, 8.0, 9.0],
            &[1, 3, 3]
        ).unwrap();

        let output = conv.forward(input, &mut ctx).unwrap();
        let shape = backend.ops().shape(&output);
        assert_eq!(shape, &[2, 2, 2]); // 2 channels, 2x2 output
    }

    #[test]
    fn test_max_pool2d() {
        let backend = mnr_ndarray_backend::CpuBackend::default();

        // Input: 1 channel, 4x4
        let input = backend.tensor_from_vec(
            vec![1.0, 2.0, 3.0, 4.0,
                 5.0, 6.0, 7.0, 8.0,
                 9.0, 10.0, 11.0, 12.0,
                 13.0, 14.0, 15.0, 16.0],
            &[1, 4, 4]
        ).unwrap();

        let output = max_pool2d(&input, 2, 2, 2, 2, true, backend.ops()).unwrap();
        let shape = backend.ops().shape(&output);
        assert_eq!(shape, &[1, 2, 2]);

        let values: Vec<f32> = (0..4)
            .filter_map(|i| backend.ops().tensor_element(&output, i).ok())
            .collect();
        assert_eq!(values, vec![6.0, 8.0, 14.0, 16.0]);
    }

    #[test]
    fn test_global_max_pool2d() {
        let backend = mnr_ndarray_backend::CpuBackend::default();

        // Input: 2 channels, 2x2
        let input = backend.tensor_from_vec(
            vec![1.0, 3.0,
                 2.0, 4.0,
                 5.0, 7.0,
                 6.0, 8.0],
            &[2, 2, 2]
        ).unwrap();

        let output = global_max_pool2d(&input, backend.ops()).unwrap();
        let shape = backend.ops().shape(&output);
        assert_eq!(shape, &[2, 1, 1]);

        let values: Vec<f32> = (0..2)
            .filter_map(|i| backend.ops().tensor_element(&output, i).ok())
            .collect();
        assert_eq!(values, vec![4.0, 8.0]);
    }
}
