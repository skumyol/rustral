//! Lightweight in-tree tokenizer adapters for the SST-2 / WikiText-2 NLP examples.
//!
//! This module is deliberately small and dependency-free. Both included tokenizers are
//! word-level (whitespace split, with a configurable lowercasing pass and `<unk>`/`<pad>`
//! handling). They are sufficient for honest small-baseline experiments on SST-2 and
//! WikiText-2, where word-level tokenization is the conventional reporting choice.
//!
//! BPE / wordpiece tokenization (HuggingFace `tokenizers`) is intentionally deferred —
//! adding the full HF tokenizer dependency tree just for a baseline would inflate compile
//! times and freeze a specific HF crate version that may diverge from the published
//! tokenizer artifacts. When a real third consumer needs BPE, this module is the place to
//! grow it (or to graduate into a `rustral-tokenizers` crate).

use std::collections::HashMap;

/// Padding / unknown / boundary token strings used by both tokenizers.
pub const PAD_TOKEN: &str = "<pad>";
pub const UNK_TOKEN: &str = "<unk>";
pub const BOS_TOKEN: &str = "<bos>";
pub const EOS_TOKEN: &str = "<eos>";

/// A vocabulary mapping `token -> id` with deterministic ordering.
#[derive(Clone, Debug)]
pub struct Vocab {
    /// `tokens[i]` is the token whose id is `i`.
    pub tokens: Vec<String>,
    /// Inverse lookup.
    pub id_of: HashMap<String, usize>,
    pub pad_id: usize,
    pub unk_id: usize,
    pub bos_id: usize,
    pub eos_id: usize,
}

impl Vocab {
    /// Initialise a vocabulary that just has the four reserved tokens.
    pub fn empty() -> Self {
        let mut v = Self {
            tokens: Vec::new(),
            id_of: HashMap::new(),
            pad_id: 0,
            unk_id: 0,
            bos_id: 0,
            eos_id: 0,
        };
        v.pad_id = v.intern(PAD_TOKEN);
        v.unk_id = v.intern(UNK_TOKEN);
        v.bos_id = v.intern(BOS_TOKEN);
        v.eos_id = v.intern(EOS_TOKEN);
        v
    }

    /// Insert a token if not present, returning its id.
    pub fn intern(&mut self, tok: &str) -> usize {
        if let Some(&id) = self.id_of.get(tok) {
            return id;
        }
        let id = self.tokens.len();
        self.tokens.push(tok.to_string());
        self.id_of.insert(tok.to_string(), id);
        id
    }

    /// Lookup `tok`, falling back to `unk_id`.
    pub fn lookup(&self, tok: &str) -> usize {
        *self.id_of.get(tok).unwrap_or(&self.unk_id)
    }

    pub fn size(&self) -> usize {
        self.tokens.len()
    }
}

/// Configuration for the [`WordLevelTokenizer`].
#[derive(Clone, Debug)]
pub struct WordLevelConfig {
    /// If true, lowercase tokens before vocab lookup.
    pub lowercase: bool,
    /// Cap on vocab size; tokens past this rank are mapped to `<unk>`.
    pub max_vocab: Option<usize>,
    /// Drop tokens with frequency strictly below this threshold (1 = keep singletons).
    pub min_freq: usize,
}

impl Default for WordLevelConfig {
    fn default() -> Self {
        Self { lowercase: true, max_vocab: None, min_freq: 1 }
    }
}

/// Whitespace-split word-level tokenizer.
///
/// Suitable for SST-2 sentiment classification (cheap baseline) and for WikiText-2
/// language modelling (where WikiText-2 is canonically reported word-level).
#[derive(Clone, Debug)]
pub struct WordLevelTokenizer {
    pub config: WordLevelConfig,
    pub vocab: Vocab,
}

impl WordLevelTokenizer {
    /// Build a tokenizer by counting tokens across `corpus_lines` (each line is a sentence
    /// or a document). Reserved tokens are always present.
    pub fn fit_from_iter<I, S>(config: WordLevelConfig, corpus_lines: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for line in corpus_lines {
            for tok in Self::raw_tokens(line.as_ref(), config.lowercase) {
                *counts.entry(tok).or_insert(0) += 1;
            }
        }

        // Sort by descending frequency, stable on token string for reproducibility.
        let mut ranked: Vec<(String, usize)> = counts
            .into_iter()
            .filter(|(_, c)| *c >= config.min_freq)
            .collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        let mut vocab = Vocab::empty();
        let cap = config.max_vocab.unwrap_or(usize::MAX);
        for (tok, _) in ranked.into_iter().take(cap.saturating_sub(vocab.size())) {
            vocab.intern(&tok);
        }
        Self { config, vocab }
    }

    fn raw_tokens(s: &str, lowercase: bool) -> impl Iterator<Item = String> + '_ {
        s.split_whitespace().map(move |t| if lowercase { t.to_lowercase() } else { t.to_string() })
    }

    /// Encode a string to token ids, lowercased if configured.
    pub fn encode(&self, s: &str) -> Vec<usize> {
        Self::raw_tokens(s, self.config.lowercase)
            .map(|t| self.vocab.lookup(&t))
            .collect()
    }

    /// Encode `s`, prepend `<bos>`, append `<eos>`.
    pub fn encode_with_specials(&self, s: &str) -> Vec<usize> {
        let mut out = Vec::new();
        out.push(self.vocab.bos_id);
        out.extend(self.encode(s));
        out.push(self.vocab.eos_id);
        out
    }

    /// Decode ids to a space-joined string. Reserved ids are emitted as their token
    /// strings; unknown ids fall back to `<unk>`.
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .map(|&i| self.vocab.tokens.get(i).cloned().unwrap_or_else(|| UNK_TOKEN.to_string()))
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_and_roundtrip() {
        let lines = ["the cat sat", "the dog ran", "the cat ran"];
        let tok = WordLevelTokenizer::fit_from_iter(WordLevelConfig::default(), lines.iter().copied());
        // Reserved tokens always present.
        assert!(tok.vocab.size() >= 4);
        assert_eq!(tok.vocab.tokens[tok.vocab.pad_id], PAD_TOKEN);
        assert_eq!(tok.vocab.tokens[tok.vocab.unk_id], UNK_TOKEN);

        let ids = tok.encode("the cat");
        assert_eq!(ids.len(), 2);
        assert!(ids[0] != tok.vocab.unk_id);
        assert!(ids[1] != tok.vocab.unk_id);

        // Out-of-vocab maps to <unk>.
        let oov = tok.encode("zebra");
        assert_eq!(oov, vec![tok.vocab.unk_id]);

        // Specials.
        let with_specials = tok.encode_with_specials("the cat");
        assert_eq!(with_specials.first().copied(), Some(tok.vocab.bos_id));
        assert_eq!(with_specials.last().copied(), Some(tok.vocab.eos_id));

        // Decode is loss-free for in-vocab tokens.
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, "the cat");
    }

    #[test]
    fn min_freq_filter_drops_rare_tokens() {
        let lines = ["a a a b c"];
        let cfg = WordLevelConfig { min_freq: 2, ..Default::default() };
        let tok = WordLevelTokenizer::fit_from_iter(cfg, lines.iter().copied());
        // Only "a" survives the min_freq=2 cut.
        assert!(tok.vocab.id_of.contains_key("a"));
        assert!(!tok.vocab.id_of.contains_key("b"));
        assert!(!tok.vocab.id_of.contains_key("c"));
    }

    #[test]
    fn max_vocab_caps_total_size() {
        let lines = ["x x x x y y y z z w"];
        let cfg = WordLevelConfig { max_vocab: Some(6), ..Default::default() };
        let tok = WordLevelTokenizer::fit_from_iter(cfg, lines.iter().copied());
        // 4 reserved + 2 most-frequent.
        assert_eq!(tok.vocab.size(), 6);
        assert!(tok.vocab.id_of.contains_key("x"));
        assert!(tok.vocab.id_of.contains_key("y"));
    }
}
