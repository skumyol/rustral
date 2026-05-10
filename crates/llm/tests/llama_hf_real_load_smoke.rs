//! End-to-end smoke: download **hf-internal-testing/tiny-random-LlamaForCausalLM** from the Hub, merge
//! safetensors, load into [`LlamaCausalLm`].
//!
//! **Opt-in** (no network in default CI):
//! ```text
//! RUSTRAL_TEST_HF_NETWORK=1 cargo test -p rustral-llm --test llama_hf_real_load_smoke
//! ```

use rustral_io::{load_meta_state_dict_from_paths, MetaStateDict};
use rustral_llm::{HfLlamaConfig, LlamaCausalLm};

#[test]
fn hf_tiny_random_llama_meta_load_smoke() {
    if std::env::var("RUSTRAL_TEST_HF_NETWORK").ok().as_deref() != Some("1") {
        eprintln!("skipping HF network test (set RUSTRAL_TEST_HF_NETWORK=1 to enable)");
        return;
    }

    let snap =
        rustral_hf::snapshot_model("hf-internal-testing/tiny-random-LlamaForCausalLM").expect("Hub snapshot");
    let cfg_path = snap.files.config_json.as_deref().expect("config.json from Hub model");

    let cfg = HfLlamaConfig::from_json_file(cfg_path).expect("parse HF Llama config");

    assert!(!snap.files.safetensors_files.is_empty(), "expected at least one safetensors shard");
    let meta: MetaStateDict =
        load_meta_state_dict_from_paths(&snap.files.safetensors_files).expect("merge safetensors meta");

    let (model, report) = LlamaCausalLm::from_hf_meta(&cfg, &meta, 0).expect("build Llama + load weights");

    assert!(
        !report.loaded_rustral_keys.is_empty(),
        "expected some Rustral keys loaded from checkpoint, got {:?}",
        report.loaded_rustral_keys
    );
    assert!(
        report.loaded_rustral_keys.iter().any(|k| k == "token_embedding.embed"),
        "expected embed_tokens→token_embedding.embed in {:?}",
        report.loaded_rustral_keys
    );
    assert!(
        report.skipped_parameters.is_empty(),
        "expected full Llama tensor map; still skipped {:?}",
        report.skipped_parameters
    );

    let logits_run = model.generate_greedy(vec![0usize], 1);
    assert!(logits_run.is_ok(), "generate after load failed: {:?}", logits_run.err());
}
