use std::time::Instant;

use rustral_llm::{gpt2::HfGpt2Config, gpt2::Gpt2Decoder, LlmError};

#[cfg(feature = "hf-tokenizers")]
use rustral_llm::TokenizerHandle;

fn main() -> Result<(), LlmError> {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        eprintln!("rustral-llm (experimental)");
        eprintln!();
        eprintln!("Commands:");
        eprintln!("  generate --model <hf_id> --prompt <text> --max-new-tokens <n> [--seed <u64>]");
        eprintln!();
        eprintln!("Notes:");
        eprintln!("  - tokenizer support requires building with: --features hf-tokenizers");
        eprintln!("  - prints JSON metrics (hub_snapshot_ms, model_init_ms, first_token_ms, tokens_per_sec, …) then generated text");
        return Ok(());
    }

    if args.len() >= 2 && args[1] == "generate" {
        return cmd_generate(&args[2..]);
    }

    let start = Instant::now();
    eprintln!("rustral-llm (experimental) ready in {:?}. Use --help.", start.elapsed());
    Ok(())
}

fn cmd_generate(args: &[String]) -> Result<(), LlmError> {
    let mut model_id: Option<String> = None;
    let mut prompt: Option<String> = None;
    let mut max_new_tokens: usize = 32;
    let mut seed: u64 = 0;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model_id = args.get(i).cloned();
            }
            "--prompt" => {
                i += 1;
                prompt = args.get(i).cloned();
            }
            "--max-new-tokens" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or_else(|| LlmError::InvalidArg("--max-new-tokens requires a value".to_string()))?;
                max_new_tokens = v
                    .parse::<usize>()
                    .map_err(|_| LlmError::InvalidArg(format!("invalid --max-new-tokens: {v}")))?;
            }
            "--seed" => {
                i += 1;
                let v = args.get(i).ok_or_else(|| LlmError::InvalidArg("--seed requires a value".to_string()))?;
                seed = v.parse::<u64>().map_err(|_| LlmError::InvalidArg(format!("invalid --seed: {v}")))?;
            }
            other => return Err(LlmError::InvalidArg(format!("unknown arg: {other}"))),
        }
        i += 1;
    }

    let model_id = model_id.ok_or_else(|| LlmError::InvalidArg("--model is required".to_string()))?;
    let prompt = prompt.ok_or_else(|| LlmError::InvalidArg("--prompt is required".to_string()))?;

    let t_snap = Instant::now();
    let snap = rustral_hf::snapshot_model(&model_id).map_err(|e| anyhow::anyhow!("{e}"))?;
    let hub_snapshot_ms = t_snap.elapsed().as_secs_f64() * 1000.0;

    let cfg_path = snap.files.config_json.as_deref().ok_or_else(|| LlmError::MissingFile("config.json".to_string()))?;
    let cfg = HfGpt2Config::from_json_file(cfg_path)?;

    let t_model = Instant::now();
    let model = Gpt2Decoder::new_random(&cfg, seed)?;
    let model_init_ms = t_model.elapsed().as_secs_f64() * 1000.0;

    #[cfg(not(feature = "hf-tokenizers"))]
    {
        let _model = model;
        let _cfg = cfg;
        let _prompt = prompt;
        let _max_new_tokens = max_new_tokens;
        let metrics = serde_json::json!({
            "model_id": model_id,
            "hub_snapshot_ms": hub_snapshot_ms,
            "model_init_ms": model_init_ms,
            "error": "tokenizer feature disabled; rebuild with --features hf-tokenizers",
        });
        eprintln!("{}", serde_json::to_string_pretty(&metrics).map_err(|e| LlmError::Anyhow(e.into()))?);
        return Err(LlmError::InvalidArg(
            "tokenizer support is disabled; rebuild with `cargo run -p rustral-llm --features hf-tokenizers -- generate ...`"
                .to_string(),
        ));
    }

    #[cfg(feature = "hf-tokenizers")]
    {
        let tok_path = snap
            .files
            .tokenizer_json
            .as_deref()
            .ok_or_else(|| LlmError::MissingFile("tokenizer.json".to_string()))?;

        let t_tok = Instant::now();
        let tok = TokenizerHandle::from_file(tok_path)?;
        let tokenizer_load_ms = t_tok.elapsed().as_secs_f64() * 1000.0;

        let prompt_ids_u32 = tok.encode(&prompt)?;
        let prompt_ids: Vec<usize> = prompt_ids_u32.iter().map(|&x| x as usize).collect();

        let (out_ids, decode_timing) = model.generate_greedy_timed(prompt_ids.clone(), max_new_tokens)?;

        let tokens_per_sec = if max_new_tokens > 0 && decode_timing.decode_wall_ms > 0.0 {
            Some(max_new_tokens as f64 / (decode_timing.decode_wall_ms / 1000.0))
        } else {
            None
        };

        let out_ids_u32: Vec<u32> = out_ids.iter().map(|&x| x as u32).collect();
        let out_text = tok.decode(&out_ids_u32)?;

        let metrics = serde_json::json!({
            "model_id": model_id,
            "backend": "ndarray",
            "dtype": "f32",
            "seed": seed,
            "hub_snapshot_ms": hub_snapshot_ms,
            "model_init_ms": model_init_ms,
            "tokenizer_load_ms": tokenizer_load_ms,
            "prompt_tokens": prompt_ids.len(),
            "max_new_tokens": max_new_tokens,
            "first_token_ms": decode_timing.first_token_ms,
            "decode_wall_ms": decode_timing.decode_wall_ms,
            "tokens_per_sec": tokens_per_sec,
        });
        println!("{}", serde_json::to_string_pretty(&metrics).map_err(|e| LlmError::Anyhow(e.into()))?);
        println!();
        println!("{out_text}");
        Ok(())
    }
}

