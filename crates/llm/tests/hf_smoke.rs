#[cfg(feature = "hf-tokenizers")]
mod hf_smoke {
    use rustral_llm::gpt2::{Gpt2Decoder, HfGpt2Config};
    use rustral_llm::TokenizerHandle;

    #[test]
    fn hf_tiny_random_gpt2_snapshot_smoke() {
        if std::env::var("RUSTRAL_TEST_HF_NETWORK").ok().as_deref() != Some("1") {
            eprintln!("skipping HF network smoke test (set RUSTRAL_TEST_HF_NETWORK=1 to enable)");
            return;
        }

        let snap = rustral_hf::snapshot_model("hf-internal-testing/tiny-random-gpt2")
            .expect("snapshot tiny-random-gpt2");
        let cfg_path = snap.files.config_json.as_deref().expect("config.json");
        let tok_path = snap.files.tokenizer_json.as_deref().expect("tokenizer.json");

        let cfg = HfGpt2Config::from_json_file(cfg_path).expect("parse config");
        let tok = TokenizerHandle::from_file(tok_path).expect("load tokenizer");

        let prompt_ids_u32 = tok.encode("Rust is").expect("encode");
        let prompt_ids: Vec<usize> = prompt_ids_u32.iter().map(|&x| x as usize).collect();

        let model = Gpt2Decoder::new_random(&cfg, 0).expect("init random model");
        let out = model.generate_greedy(prompt_ids, 2).expect("generate");
        assert!(out.len() >= 2);
    }
}

