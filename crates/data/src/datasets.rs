//! Built-in real-corpus datasets used by the NLP examples.
//!
//! All dataset URLs are pinned by SHA-256 in this module and verified by `fetch::fetch_url`.
//! Pinning prevents silent corpus drift; if a mirror changes the file we fail loudly.

#[cfg(feature = "fetch")]
pub mod sst2;
#[cfg(feature = "fetch")]
pub mod wikitext2;
