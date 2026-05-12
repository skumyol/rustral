#[cfg(feature = "hf-tokenizers")]
mod hf_tokenizers_tests {
    use rustral_llm::TokenizerHandle;

    #[test]
    fn encode_decode_roundtrip_wordlevel() {
        // Build a tiny word-level tokenizer, write to json, reload via our wrapper.
        let mut vocab: ahash::AHashMap<String, u32> = ahash::AHashMap::new();
        vocab.insert("[UNK]".to_string(), 0);
        vocab.insert("Rust".to_string(), 1);
        vocab.insert("is".to_string(), 2);
        vocab.insert("fast".to_string(), 3);

        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .build()
            .expect("build wordlevel");

        let mut tok = tokenizers::Tokenizer::new(model);
        tok.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace::default()));

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("tokenizer.json");
        tok.save(&path, false).expect("save tokenizer");

        let handle = TokenizerHandle::from_file(&path).expect("load tokenizer");
        let ids = handle.encode("Rust is fast").expect("encode");
        assert_eq!(ids, vec![1, 2, 3]);

        let text = handle.decode(&ids).expect("decode");
        assert_eq!(text, "Rust is fast");
    }
}
