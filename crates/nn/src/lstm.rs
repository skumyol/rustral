//! LSTM and RNN modules for sequential processing.
//!
//! This module provides LSTM cell implementations following the legacy
//! DyNet wrapper architecture, but with explicit state management.

use rustral_core::{Backend, ForwardCtx, Module, Parameter, ParameterRef, Result, StatefulModule, Trainable};
use serde::{Deserialize, Serialize};

/// Configuration for an LSTM cell.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LstmConfig {
    /// Input dimension.
    pub input_dim: usize,
    /// Hidden/output dimension.
    pub hidden_dim: usize,
}

impl LstmConfig {
    /// Create a new LSTM configuration.
    /// When only one dimension is given, input and hidden dims are set equal.
    pub fn new(dim: usize) -> Self {
        Self { input_dim: dim, hidden_dim: dim }
    }

    /// Create with explicit input and hidden dimensions.
    pub fn new_with_dims(input_dim: usize, hidden_dim: usize) -> Self {
        Self { input_dim, hidden_dim }
    }
}

/// LSTM cell state containing cell state and hidden state.
#[derive(Clone, Debug)]
pub struct LstmState<B: Backend> {
    /// Cell state (c).
    pub cell: B::Tensor,
    /// Hidden state (h).
    pub hidden: B::Tensor,
}

/// Vanilla LSTM cell implementation.
///
/// Uses the standard LSTM formulation with input, forget, and output gates.
pub struct LstmCell<B: Backend> {
    config: LstmConfig,
    /// Weights for input transformation (combined gates).
    wx: Parameter<B>,
    /// Weights for hidden transformation (combined gates).
    wh: Parameter<B>,
    /// Bias (combined gates).
    b: Parameter<B>,
}

impl<B: Backend> LstmCell<B> {
    /// Create an LSTM cell with randomly initialized parameters.
    pub fn new(backend: &B, config: LstmConfig) -> Result<Self> {
        let four_h = config.hidden_dim * 4;
        let wx = backend.normal_parameter("wx", &[four_h, config.input_dim], 42, 0.1)?;
        let wh = backend.normal_parameter("wh", &[four_h, config.hidden_dim], 43, 0.1)?;
        let b = backend.normal_parameter("b", &[four_h], 44, 0.1)?;
        Ok(Self { config, wx, wh, b })
    }

    /// Create an LSTM cell from explicit parameters.
    pub fn from_parameters(config: LstmConfig, wx: Parameter<B>, wh: Parameter<B>, b: Parameter<B>) -> Self {
        Self { config, wx, wh, b }
    }

    /// Return default initial state (zeros).
    pub fn default_state(&self, backend: &B) -> Result<LstmState<B>> {
        let ops = backend.ops();
        let hidden_dim = self.config.hidden_dim;
        Ok(LstmState { cell: ops.zeros(&[hidden_dim])?, hidden: ops.zeros(&[hidden_dim])? })
    }

    /// Access the cell configuration.
    pub fn config(&self) -> &LstmConfig {
        &self.config
    }
}

impl<B: Backend> LstmCell<B> {
    /// Split combined gates tensor into individual gates.
    fn split_gates(
        &self,
        gates: &B::Tensor,
        ops: &dyn rustral_core::TensorOps<B>,
    ) -> Result<(B::Tensor, B::Tensor, B::Tensor, B::Tensor)> {
        let hidden_dim = self.config.hidden_dim;
        let shape = ops.shape(gates);
        // If gates is 2D [1, 4*hidden_dim], reshape to [4*hidden_dim] for slicing
        let gates_1d =
            if shape.len() == 2 && shape[0] == 1 { ops.reshape(gates, &[shape[1]])? } else { gates.clone() };
        let i = ops.slice(&gates_1d, 0, hidden_dim)?;
        let f = ops.slice(&gates_1d, hidden_dim, hidden_dim * 2)?;
        let o = ops.slice(&gates_1d, hidden_dim * 2, hidden_dim * 3)?;
        let g = ops.slice(&gates_1d, hidden_dim * 3, hidden_dim * 4)?;
        Ok((i, f, o, g))
    }
}

impl<B: Backend> Module<B> for LstmCell<B> {
    type Input = (LstmState<B>, B::Tensor);
    type Output = LstmState<B>;

    /// Single step forward pass implementing standard LSTM.
    ///
    /// Input: (previous_state, input_tensor)
    /// Output: new_state (with updated hidden state)
    fn forward(&self, (prev_state, x): Self::Input, ctx: &mut ForwardCtx<B>) -> Result<Self::Output> {
        let ops = ctx.backend().ops();
        let h = &prev_state.hidden;
        let c = &prev_state.cell;
        let _hidden_dim = self.config.hidden_dim;

        // Concatenate input and hidden for single matrix multiply
        // xh = concat([x, h], dim=0) -> [input_dim + hidden_dim]
        let _xh = ops.concat(&[&x, h], 0)?;

        // Compute all gates at once: Wx has shape [4*hidden_dim, input_dim], Wh has shape [4*hidden_dim, hidden_dim]
        // We need to do this in two parts: Wx @ x and Wh @ h
        let gates_x = ops.linear(&x, &self.wx, None::<&Parameter<B>>)?;
        let gates_h = ops.linear(h, &self.wh, None::<&Parameter<B>>)?;
        let b_tensor = self.b.tensor();

        // gates = gates_x + gates_h + b (element-wise)
        let gates_sum = ops.add(&gates_x, &gates_h)?;
        let gates = ops.add_row_vector(&gates_sum, b_tensor)?;

        // Split into individual gates
        let (i_raw, f_raw, o_raw, g_raw) = self.split_gates(&gates, ops)?;

        // Apply activations
        let i = ops.sigmoid(&i_raw)?; // input gate
        let f = ops.sigmoid(&f_raw)?; // forget gate
        let o = ops.sigmoid(&o_raw)?; // output gate
        let g = ops.tanh(&g_raw)?; // cell candidate

        // c_new = f * c + i * g (element-wise)
        let f_c = ops.mul(&f, c)?;
        let i_g = ops.mul(&i, &g)?;
        let c_new = ops.add(&f_c, &i_g)?;

        // h_new = o * tanh(c_new)
        let tanh_c = ops.tanh(&c_new)?;
        let h_new = ops.mul(&o, &tanh_c)?;

        Ok(LstmState { cell: c_new, hidden: h_new })
    }
}

impl<B: Backend> StatefulModule<B> for LstmCell<B> {
    type State = LstmState<B>;

    fn initial_state(&self, ctx: &mut ForwardCtx<B>) -> Result<Self::State> {
        self.default_state(ctx.backend())
    }
}

impl<B: Backend> Trainable<B> for LstmCell<B> {
    fn parameters(&self) -> Vec<ParameterRef> {
        vec![
            ParameterRef { id: self.wx.id() },
            ParameterRef { id: self.wh.id() },
            ParameterRef { id: self.b.id() },
        ]
    }
}

/// Stacked LSTM: multiple LSTM layers applied sequentially.
pub struct StackedLstm<B: Backend> {
    layers: Vec<LstmCell<B>>,
}

impl<B: Backend> StackedLstm<B> {
    /// Create a stacked LSTM from layers.
    pub fn new(layers: Vec<LstmCell<B>>) -> Self {
        Self { layers }
    }

    /// Process a sequence of inputs through all layers.
    ///
    /// Returns the final state and output sequence from the last layer.
    pub fn forward_sequence(
        &self,
        initial_states: Vec<LstmState<B>>,
        inputs: Vec<B::Tensor>,
        ctx: &mut ForwardCtx<B>,
    ) -> Result<(Vec<LstmState<B>>, Vec<B::Tensor>)> {
        let mut states = initial_states;
        let mut outputs = inputs;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let mut new_outputs = Vec::with_capacity(outputs.len());
            let mut state =
                states.get(layer_idx).cloned().unwrap_or_else(|| layer.initial_state(ctx).unwrap());

            for input in outputs {
                state = layer.forward((state, input), ctx)?;
                new_outputs.push(state.hidden.clone());
            }

            states.push(state);
            outputs = new_outputs;
        }

        Ok((states, outputs))
    }
}

impl<B: Backend> Trainable<B> for StackedLstm<B> {
    fn parameters(&self) -> Vec<ParameterRef> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}

/// Configuration for a GRU cell.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GruConfig {
    /// Input dimension.
    pub input_dim: usize,
    /// Hidden/output dimension.
    pub hidden_dim: usize,
}

/// GRU cell state (only hidden state, no cell state like LSTM).
#[derive(Clone, Debug)]
pub struct GruState<B: Backend> {
    /// Hidden state (h).
    pub hidden: B::Tensor,
}

/// GRU cell implementation.
///
/// Uses the standard GRU formulation with update and reset gates.
pub struct GruCell<B: Backend> {
    config: GruConfig,
    /// Weights for input transformation.
    wx: Parameter<B>,
    /// Weights for hidden transformation.
    wh: Parameter<B>,
    /// Bias.
    b: Parameter<B>,
}

impl<B: Backend> GruCell<B> {
    /// Create a GRU cell with randomly initialized parameters.
    pub fn new(backend: &B, config: GruConfig) -> Result<Self> {
        let three_h = config.hidden_dim * 3;
        let wx = backend.normal_parameter("wx", &[three_h, config.input_dim], 42, 0.1)?;
        let wh = backend.normal_parameter("wh", &[three_h, config.hidden_dim], 43, 0.1)?;
        let b = backend.normal_parameter("b", &[three_h], 44, 0.1)?;
        Ok(Self { config, wx, wh, b })
    }

    /// Create a GRU cell from explicit parameters.
    pub fn from_parameters(config: GruConfig, wx: Parameter<B>, wh: Parameter<B>, b: Parameter<B>) -> Self {
        Self { config, wx, wh, b }
    }

    /// Return default initial state (zeros).
    pub fn default_state(&self, backend: &B) -> Result<GruState<B>> {
        let ops = backend.ops();
        let hidden_dim = self.config.hidden_dim;
        Ok(GruState { hidden: ops.zeros(&[hidden_dim])? })
    }

    /// Access the cell configuration.
    pub fn config(&self) -> &GruConfig {
        &self.config
    }

    /// Split combined gates tensor into update, reset, and new gates.
    fn split_gates(
        &self,
        gates: &B::Tensor,
        ops: &dyn rustral_core::TensorOps<B>,
    ) -> Result<(B::Tensor, B::Tensor, B::Tensor)> {
        let hidden_dim = self.config.hidden_dim;
        let shape = ops.shape(gates);
        // If gates is 2D [1, 3*hidden_dim], reshape to [3*hidden_dim] for slicing
        let gates_1d =
            if shape.len() == 2 && shape[0] == 1 { ops.reshape(gates, &[shape[1]])? } else { gates.clone() };
        let z = ops.slice(&gates_1d, 0, hidden_dim)?; // update gate
        let r = ops.slice(&gates_1d, hidden_dim, hidden_dim * 2)?; // reset gate
        let n = ops.slice(&gates_1d, hidden_dim * 2, hidden_dim * 3)?; // new gate
        Ok((z, r, n))
    }
}

impl<B: Backend> Module<B> for GruCell<B> {
    type Input = (GruState<B>, B::Tensor);
    type Output = GruState<B>;

    /// Single step forward pass implementing standard GRU.
    fn forward(&self, (prev_state, x): Self::Input, ctx: &mut ForwardCtx<B>) -> Result<Self::Output> {
        let ops = ctx.backend().ops();
        let h = &prev_state.hidden;
        let _hidden_dim = self.config.hidden_dim;

        // Compute gates: Wx has shape [3*hidden_dim, input_dim], Wh has shape [3*hidden_dim, hidden_dim]
        let gates_x = ops.linear(&x, &self.wx, None::<&Parameter<B>>)?;
        let gates_h = ops.linear(h, &self.wh, None::<&Parameter<B>>)?;
        let b_tensor = self.b.tensor();

        // gates = gates_x + gates_h + b (element-wise)
        let gates_sum = ops.add(&gates_x, &gates_h)?;
        let gates = ops.add_row_vector(&gates_sum, b_tensor)?;

        // Split into individual gates
        let (z_raw, r_raw, n_raw) = self.split_gates(&gates, ops)?;

        // Apply activations
        let z = ops.sigmoid(&z_raw)?; // update gate
        let r = ops.sigmoid(&r_raw)?; // reset gate
        let n = ops.tanh(&n_raw)?; // new gate

        // h_new = z * n + (1 - z) * h
        let one_minus_z = ops.add_scalar(&ops.neg(&z)?, 1.0)?;
        let _r_h = ops.mul(&r, h)?; // For reset gate computation (future use)
        let z_n = ops.mul(&z, &n)?;
        let one_minus_z_h = ops.mul(&one_minus_z, h)?;
        let h_new = ops.add(&z_n, &one_minus_z_h)?;

        Ok(GruState { hidden: h_new })
    }
}

impl<B: Backend> StatefulModule<B> for GruCell<B> {
    type State = GruState<B>;

    fn initial_state(&self, ctx: &mut ForwardCtx<B>) -> Result<Self::State> {
        self.default_state(ctx.backend())
    }
}

impl<B: Backend> Trainable<B> for GruCell<B> {
    fn parameters(&self) -> Vec<ParameterRef> {
        vec![
            ParameterRef { id: self.wx.id() },
            ParameterRef { id: self.wh.id() },
            ParameterRef { id: self.b.id() },
        ]
    }
}

/// Bidirectional RNN wrapper.
///
/// Processes input in both forward and backward directions,
/// concatenating the outputs.
pub struct BidirectionalRnn<F, Bw> {
    forward: F,
    backward: Bw,
}

impl<F, Bw> BidirectionalRnn<F, Bw> {
    /// Create a bidirectional RNN from forward and backward cells.
    pub fn new(forward: F, backward: Bw) -> Self {
        Self { forward, backward }
    }
}

/// Trait for RNN cells that can be used in a bidirectional wrapper.
pub trait RnnCell<B: Backend>:
    Module<B, Input = (<Self as RnnCell<B>>::State, B::Tensor), Output = <Self as RnnCell<B>>::State>
    + StatefulModule<B>
{
    type State: Clone;
    fn default_state(&self, backend: &B) -> Result<<Self as RnnCell<B>>::State>;
    fn hidden<'a>(&self, state: &'a <Self as RnnCell<B>>::State) -> &'a B::Tensor;

    /// Process a sequence of inputs, returning all outputs and the final state.
    ///
    /// This is a convenience method that iterates over the input sequence,
    /// applying the cell at each step and collecting the outputs.
    fn forward_sequence(
        &self,
        initial_state: <Self as RnnCell<B>>::State,
        inputs: Vec<B::Tensor>,
        ctx: &mut ForwardCtx<B>,
    ) -> Result<(<Self as RnnCell<B>>::State, Vec<<Self as RnnCell<B>>::State>)> {
        let mut state = initial_state;
        let mut outputs = Vec::with_capacity(inputs.len());
        for input in inputs {
            state = self.forward((state, input), ctx)?;
            outputs.push(state.clone());
        }
        Ok((state, outputs))
    }
}

impl<B: Backend> RnnCell<B> for LstmCell<B> {
    type State = LstmState<B>;

    fn default_state(&self, backend: &B) -> Result<<Self as RnnCell<B>>::State> {
        LstmCell::default_state(self, backend)
    }

    fn hidden<'a>(&self, state: &'a <Self as RnnCell<B>>::State) -> &'a B::Tensor {
        &state.hidden
    }
}

impl<B: Backend> RnnCell<B> for GruCell<B> {
    type State = GruState<B>;

    fn default_state(&self, backend: &B) -> Result<<Self as RnnCell<B>>::State> {
        GruCell::default_state(self, backend)
    }

    fn hidden<'a>(&self, state: &'a <Self as RnnCell<B>>::State) -> &'a B::Tensor {
        &state.hidden
    }
}

/// Output of bidirectional RNN: concatenated forward and backward hidden states.
#[derive(Clone, Debug)]
pub struct BidirectionalOutput<B: Backend> {
    /// Forward direction hidden state.
    pub forward: B::Tensor,
    /// Backward direction hidden state.
    pub backward: B::Tensor,
    /// Concatenated hidden state (forward || backward).
    pub concatenated: B::Tensor,
}

impl<F, Bw> BidirectionalRnn<F, Bw> {
    /// Process a sequence through both directions and return outputs.
    pub fn forward_sequence<B: Backend>(
        &self,
        inputs: Vec<B::Tensor>,
        ctx: &mut ForwardCtx<B>,
    ) -> Result<Vec<BidirectionalOutput<B>>>
    where
        F: RnnCell<B>,
        Bw: RnnCell<B, State = <F as RnnCell<B>>::State>,
    {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let backend = ctx.backend();

        // Forward direction
        let forward_init = self.forward.default_state(backend)?;
        let (_, forward_states) = self.forward.forward_sequence(forward_init, inputs.clone(), ctx)?;

        // Backward direction (reverse input sequence)
        let backward_init = self.backward.default_state(backend)?;
        let mut reversed_inputs = inputs.clone();
        reversed_inputs.reverse();
        let (_, backward_states) = self.backward.forward_sequence(backward_init, reversed_inputs, ctx)?;

        // Combine outputs (reverse backward outputs to align with forward)
        let mut outputs = Vec::with_capacity(inputs.len());
        let ops = backend.ops();

        for i in 0..inputs.len() {
            let forward_state = &forward_states[i];
            let backward_state = &backward_states[inputs.len() - 1 - i];

            let forward_hidden = self.forward.hidden(forward_state).clone();
            let backward_hidden = self.backward.hidden(backward_state).clone();

            // Concatenate forward and backward hidden states
            let concatenated = ops.concat(&[&forward_hidden, &backward_hidden], 0)?;

            outputs.push(BidirectionalOutput {
                forward: forward_hidden,
                backward: backward_hidden,
                concatenated,
            });
        }

        Ok(outputs)
    }

    /// Get final hidden states from both directions.
    pub fn final_hidden<B: Backend>(
        &self,
        inputs: Vec<B::Tensor>,
        ctx: &mut ForwardCtx<B>,
    ) -> Result<(<F as RnnCell<B>>::State, <Bw as RnnCell<B>>::State)>
    where
        F: RnnCell<B>,
        Bw: RnnCell<B, State = <F as RnnCell<B>>::State>,
    {
        if inputs.is_empty() {
            let backend = ctx.backend();
            let init_f = self.forward.default_state(backend)?;
            let init_b = self.backward.default_state(backend)?;
            return Ok((init_f, init_b));
        }

        // Forward direction
        let backend = ctx.backend();
        let forward_init = self.forward.default_state(backend)?;
        let (forward_final, _) = self.forward.forward_sequence(forward_init, inputs.clone(), ctx)?;

        // Backward direction (reverse input)
        let backward_init = self.backward.default_state(backend)?;
        let mut reversed_inputs = inputs;
        reversed_inputs.reverse();
        let (backward_final, _) = self.backward.forward_sequence(backward_init, reversed_inputs, ctx)?;

        Ok((forward_final, backward_final))
    }
}

impl<B: Backend, F, Bw> Trainable<B> for BidirectionalRnn<F, Bw>
where
    F: RnnCell<B> + Trainable<B>,
    Bw: RnnCell<B, State = <F as RnnCell<B>>::State> + Trainable<B>,
{
    fn parameters(&self) -> Vec<rustral_core::ParameterRef> {
        let mut params = self.forward.parameters();
        params.extend(self.backward.parameters());
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustral_core::{ForwardCtx, Mode, Parameter};
    use rustral_ndarray_backend::CpuBackend;

    fn create_mock_lstm_cell(input_dim: usize, hidden_dim: usize) -> LstmCell<CpuBackend> {
        let backend = CpuBackend::default();

        // Wx: [4*hidden_dim, input_dim]
        let wx_values: Vec<f32> = (0..4 * hidden_dim * input_dim).map(|i| (i as f32) * 0.001).collect();
        let wx =
            Parameter::new("wx", backend.tensor_from_vec(wx_values, &[4 * hidden_dim, input_dim]).unwrap());

        // Wh: [4*hidden_dim, hidden_dim]
        let wh_values: Vec<f32> = (0..4 * hidden_dim * hidden_dim).map(|i| (i as f32) * 0.001).collect();
        let wh =
            Parameter::new("wh", backend.tensor_from_vec(wh_values, &[4 * hidden_dim, hidden_dim]).unwrap());

        // b: [4*hidden_dim]
        let b_values: Vec<f32> = (0..4 * hidden_dim).map(|_i| 0.0).collect();
        let b = Parameter::new("b", backend.tensor_from_vec(b_values, &[4 * hidden_dim]).unwrap());

        let config = LstmConfig { input_dim, hidden_dim };
        LstmCell::from_parameters(config, wx, wh, b)
    }

    fn create_mock_gru_cell(input_dim: usize, hidden_dim: usize) -> GruCell<CpuBackend> {
        let backend = CpuBackend::default();

        // Wx: [3*hidden_dim, input_dim]
        let wx_values: Vec<f32> = (0..3 * hidden_dim * input_dim).map(|i| (i as f32) * 0.001).collect();
        let wx =
            Parameter::new("wx", backend.tensor_from_vec(wx_values, &[3 * hidden_dim, input_dim]).unwrap());

        // Wh: [3*hidden_dim, hidden_dim]
        let wh_values: Vec<f32> = (0..3 * hidden_dim * hidden_dim).map(|i| (i as f32) * 0.001).collect();
        let wh =
            Parameter::new("wh", backend.tensor_from_vec(wh_values, &[3 * hidden_dim, hidden_dim]).unwrap());

        // b: [3*hidden_dim]
        let b_values: Vec<f32> = (0..3 * hidden_dim).map(|_i| 0.0).collect();
        let b = Parameter::new("b", backend.tensor_from_vec(b_values, &[3 * hidden_dim]).unwrap());

        let config = GruConfig { input_dim, hidden_dim };
        GruCell::from_parameters(config, wx, wh, b)
    }

    #[test]
    fn test_lstm_default_state() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let cell = create_mock_lstm_cell(5, 4);
        let state = cell.initial_state(&mut ctx).unwrap();

        assert_eq!(state.cell.shape(), &[4]);
        assert_eq!(state.hidden.shape(), &[4]);
        assert!(state.cell.values().iter().all(|&v| v == 0.0));
        assert!(state.hidden.values().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_lstm_forward_preserves_shapes() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let cell = create_mock_lstm_cell(3, 2);

        let state = cell.initial_state(&mut ctx).unwrap();
        let input = backend.tensor_from_vec(vec![0.1, 0.2, 0.3], &[3]).unwrap();

        let new_state = cell.forward((state, input), &mut ctx).unwrap();

        assert_eq!(new_state.cell.shape(), &[2]);
        assert_eq!(new_state.hidden.shape(), &[2]);
    }

    #[test]
    fn test_lstm_single_step() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let cell = create_mock_lstm_cell(4, 3);

        let state = cell.initial_state(&mut ctx).unwrap();
        let input = backend.tensor_from_vec(vec![0.1; 4], &[4]).unwrap();

        let new_state = cell.forward((state, input), &mut ctx).unwrap();

        // Hidden state should be non-zero after first step
        let hidden_sum: f32 = new_state.hidden.values().iter().sum();
        assert_ne!(hidden_sum, 0.0, "LSTM hidden state should change after forward pass");
    }

    #[test]
    fn test_lstm_multiple_steps() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let cell = create_mock_lstm_cell(2, 2);

        let mut state = cell.initial_state(&mut ctx).unwrap();
        let inputs = vec![
            backend.tensor_from_vec(vec![0.1, 0.2], &[2]).unwrap(),
            backend.tensor_from_vec(vec![0.3, 0.4], &[2]).unwrap(),
            backend.tensor_from_vec(vec![0.5, 0.6], &[2]).unwrap(),
        ];

        for input in inputs {
            state = cell.forward((state, input), &mut ctx).unwrap();
        }

        // After multiple steps, state should be evolved
        assert_eq!(state.hidden.shape(), &[2]);
        assert_eq!(state.cell.shape(), &[2]);
    }

    #[test]
    fn test_lstm_stacked_sequence() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);

        // Create 2-layer stacked LSTM
        let layer1 = create_mock_lstm_cell(3, 4);
        let layer2 = create_mock_lstm_cell(4, 4); // input is hidden from layer1
        let stacked = StackedLstm::new(vec![layer1, layer2]);

        let inputs: Vec<_> =
            (0..5).map(|i| backend.tensor_from_vec(vec![(i as f32) * 0.1; 3], &[3]).unwrap()).collect();

        let (states, outputs) = stacked.forward_sequence(vec![], inputs, &mut ctx).unwrap();

        // Should have 2 layer states (one per layer)
        assert_eq!(states.len(), 2);
        // Should have 5 outputs (one per input timestep)
        assert_eq!(outputs.len(), 5);
        // Each output should be hidden_dim (4)
        assert_eq!(outputs[0].shape(), &[4]);
    }

    #[test]
    fn test_lstm_trainable() {
        let cell = create_mock_lstm_cell(3, 2);
        let params = cell.parameters();
        assert_eq!(params.len(), 3); // Wx, Wh, b
    }

    #[test]
    fn test_gru_default_state() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let cell = create_mock_gru_cell(5, 4);
        let state = cell.initial_state(&mut ctx).unwrap();

        assert_eq!(state.hidden.shape(), &[4]);
        assert!(state.hidden.values().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_gru_forward_shape() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let cell = create_mock_gru_cell(3, 2);

        let state = cell.initial_state(&mut ctx).unwrap();
        let input = backend.tensor_from_vec(vec![0.1, 0.2, 0.3], &[3]).unwrap();

        let new_state = cell.forward((state, input), &mut ctx).unwrap();
        assert_eq!(new_state.hidden.shape(), &[2]);
    }

    #[test]
    fn test_gru_single_step() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let cell = create_mock_gru_cell(4, 3);

        let state = cell.initial_state(&mut ctx).unwrap();
        let input = backend.tensor_from_vec(vec![0.1; 4], &[4]).unwrap();

        let new_state = cell.forward((state, input), &mut ctx).unwrap();

        let hidden_sum: f32 = new_state.hidden.values().iter().sum();
        assert_ne!(hidden_sum, 0.0, "GRU hidden state should change after forward pass");
    }

    #[test]
    fn test_gru_trainable() {
        let cell = create_mock_gru_cell(3, 2);
        let params = cell.parameters();
        assert_eq!(params.len(), 3); // Wx, Wh, b
    }

    #[test]
    fn test_bidirectional_lstm_sequence() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);

        let forward_cell = create_mock_lstm_cell(3, 4);
        let backward_cell = create_mock_lstm_cell(3, 4);
        let bilstm = BidirectionalRnn::new(forward_cell, backward_cell);

        let inputs: Vec<_> =
            (0..3).map(|i| backend.tensor_from_vec(vec![(i as f32) * 0.1; 3], &[3]).unwrap()).collect();

        let outputs = bilstm.forward_sequence(inputs, &mut ctx).unwrap();

        // Should have 3 outputs (one per input timestep)
        assert_eq!(outputs.len(), 3);
        // Each concatenated output should be 8 (forward 4 + backward 4)
        assert_eq!(outputs[0].concatenated.shape(), &[8]);
        assert_eq!(outputs[1].concatenated.shape(), &[8]);
        assert_eq!(outputs[2].concatenated.shape(), &[8]);
    }

    #[test]
    fn test_bidirectional_lstm_final_hidden() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);

        let forward_cell = create_mock_lstm_cell(3, 4);
        let backward_cell = create_mock_lstm_cell(3, 4);
        let bilstm = BidirectionalRnn::new(forward_cell, backward_cell);

        let inputs: Vec<_> =
            (0..3).map(|i| backend.tensor_from_vec(vec![(i as f32) * 0.1; 3], &[3]).unwrap()).collect();

        let (forward_final, backward_final) = bilstm.final_hidden(inputs, &mut ctx).unwrap();

        assert_eq!(forward_final.hidden.shape(), &[4]);
        assert_eq!(forward_final.cell.shape(), &[4]);
        assert_eq!(backward_final.hidden.shape(), &[4]);
        assert_eq!(backward_final.cell.shape(), &[4]);
    }

    #[test]
    fn test_bidirectional_empty_sequence() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);

        let forward_cell = create_mock_lstm_cell(3, 4);
        let backward_cell = create_mock_lstm_cell(3, 4);
        let bilstm = BidirectionalRnn::new(forward_cell, backward_cell);

        let outputs = bilstm.forward_sequence::<CpuBackend>(vec![], &mut ctx).unwrap();
        assert_eq!(outputs.len(), 0);
    }

    #[test]
    fn test_bidirectional_trainable() {
        let forward_cell = create_mock_lstm_cell(3, 4);
        let backward_cell = create_mock_lstm_cell(3, 4);
        let bilstm = BidirectionalRnn::new(forward_cell, backward_cell);

        let params = bilstm.parameters();
        assert_eq!(params.len(), 6); // 3 forward + 3 backward
    }

    #[test]
    fn test_lstm_config_constructors() {
        let config1 = LstmConfig::new(8);
        assert_eq!(config1.input_dim, 8);
        assert_eq!(config1.hidden_dim, 8);

        let config2 = LstmConfig::new_with_dims(5, 3);
        assert_eq!(config2.input_dim, 5);
        assert_eq!(config2.hidden_dim, 3);
    }

    #[test]
    fn test_lstm_cell_new() {
        let backend = CpuBackend::default();
        let config = LstmConfig::new(4);
        let cell = LstmCell::new(&backend, config).unwrap();

        // Check config accessor
        assert_eq!(cell.config().input_dim, 4);
        assert_eq!(cell.config().hidden_dim, 4);

        // Check default state via inherent method
        let state = cell.default_state(&backend).unwrap();
        assert_eq!(state.cell.shape(), &[4]);
        assert_eq!(state.hidden.shape(), &[4]);
    }

    #[test]
    fn test_stacked_lstm_parameters() {
        let backend = CpuBackend::default();
        let cell1 = LstmCell::new(&backend, LstmConfig::new(3)).unwrap();
        let cell2 = LstmCell::new(&backend, LstmConfig::new(3)).unwrap();
        let stacked = StackedLstm::new(vec![cell1, cell2]);
        assert_eq!(stacked.parameters().len(), 6); // 3 per cell
    }

    #[test]
    fn test_gru_config_and_state() {
        let backend = CpuBackend::default();
        let cell = create_mock_gru_cell(3, 2);

        // config accessor
        assert_eq!(cell.config().input_dim, 3);
        assert_eq!(cell.config().hidden_dim, 2);

        // default state via inherent method
        let state = cell.default_state(&backend).unwrap();
        assert_eq!(state.hidden.shape(), &[2]);
    }

    #[test]
    fn test_gru_cell_trait_methods() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let cell = create_mock_gru_cell(3, 2);

        // Test RnnCell trait methods for GruCell
        let state = RnnCell::default_state(&cell, &backend).unwrap();
        assert_eq!(state.hidden.shape(), &[2]);

        let h = RnnCell::hidden(&cell, &state);
        assert_eq!(h.shape(), &[2]);

        // Forward via trait
        let input = backend.tensor_from_vec(vec![0.1f32; 3], &[3]).unwrap();
        let new_state = RnnCell::forward_sequence(&cell, state, vec![input], &mut ctx).unwrap();
        assert_eq!(new_state.0.hidden.shape(), &[2]);
        assert_eq!(new_state.1.len(), 1);
    }

    #[test]
    fn test_bidirectional_empty_final_hidden() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);

        let forward_cell = create_mock_lstm_cell(3, 4);
        let backward_cell = create_mock_lstm_cell(3, 4);
        let bilstm = BidirectionalRnn::new(forward_cell, backward_cell);

        let (forward_final, backward_final) = bilstm.final_hidden(vec![], &mut ctx).unwrap();
        assert_eq!(forward_final.hidden.shape(), &[4]);
        assert_eq!(forward_final.cell.shape(), &[4]);
        assert_eq!(backward_final.hidden.shape(), &[4]);
        assert_eq!(backward_final.cell.shape(), &[4]);
    }
}
