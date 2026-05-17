//! Candle-direct NLP workload runner.
use candle_core::{DType, Device, Tensor};
use candle_nn::{LayerNorm, Linear, Module, VarBuilder, VarMap};
use rustral_bench::{samples_to_json, time_runs, Sample};

const BACKEND: &str = "candle-cpu";

struct CandleTransformerLayer {
    self_attn: (Linear, Linear, Linear, Linear),
    ln1: LayerNorm,
    ln2: LayerNorm,
    linear1: Linear,
    linear2: Linear,
    d_model: usize,
}
impl CandleTransformerLayer {
    fn new(vs: VarBuilder, d_model: usize, ff_dim: usize) -> candle_core::Result<Self> {
        let q = candle_nn::linear(d_model, d_model, vs.pp("q"))?;
        let k = candle_nn::linear(d_model, d_model, vs.pp("k"))?;
        let v = candle_nn::linear(d_model, d_model, vs.pp("v"))?;
        let out = candle_nn::linear(d_model, d_model, vs.pp("out"))?;
        let ln1 = candle_nn::layer_norm(d_model, 1e-5, vs.pp("ln1"))?;
        let ln2 = candle_nn::layer_norm(d_model, 1e-5, vs.pp("ln2"))?;
        let linear1 = candle_nn::linear(d_model, ff_dim, vs.pp("l1"))?;
        let linear2 = candle_nn::linear(ff_dim, d_model, vs.pp("l2"))?;
        Ok(Self { self_attn: (q, k, v, out), ln1, ln2, linear1, linear2, d_model })
    }
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let q = self.self_attn.0.forward(x)?;
        let k = self.self_attn.1.forward(x)?;
        let v = self.self_attn.2.forward(x)?;
        let kt = k.transpose(1, 2)?;
        let scores = q.matmul(&kt)?;
        let scaled = (scores * (1.0 / (self.d_model as f64).sqrt()))?;
        let probs = candle_nn::ops::softmax(&scaled, 2)?;
        let attn = probs.matmul(&v)?;
        let x = (x + self.self_attn.3.forward(&attn)?)?;
        let x = self.ln1.forward(&x)?;
        let h = self.linear1.forward(&x)?.gelu()?;
        let x = (x + self.linear2.forward(&h)?)?;
        let x = self.ln2.forward(&x)?;
        Ok(x)
    }
}
fn main() {
    let device = Device::Cpu;
    let mut samples: Vec<Sample> = Vec::new();
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let d_model = 128;
    let ff_dim = 512;
    let layer1 = CandleTransformerLayer::new(vs.pp("l1"), d_model, ff_dim).unwrap();
    let layer2 = CandleTransformerLayer::new(vs.pp("l2"), d_model, ff_dim).unwrap();
    let embedding = candle_nn::embedding(1000, d_model, vs.pp("emb")).unwrap();
    let runs = time_runs(
        || {
            let input = Tensor::new(&[[1u32, 2, 3, 4, 5]], &device).unwrap();
            let x = embedding.forward(&input).unwrap();
            let x = layer1.forward(&x).unwrap();
            let _y = layer2.forward(&x).unwrap();
        },
        1,
        5,
    );
    samples.push(Sample::cpu_f32("nlp.full_pipeline", BACKEND, vec![("d_model".into(), "128".into())], runs));
    println!("{}", samples_to_json("candle-nlp", &samples));
}
