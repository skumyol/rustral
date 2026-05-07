//! Placeholder binary for the (mostly empty) `rustral-examples` crate.
//!
//! The real, working examples live under `crates/runtime/examples/` so they can use the
//! runtime workspace's `training` feature directly. Run them with:
//!
//! ```bash
//! cargo run -p rustral-runtime --features training --example sst2_classifier
//! cargo run -p rustral-runtime --features training --example wikitext2_lm
//! cargo run -p rustral-runtime --features training --example tape_train_demo
//! ```
//!
//! This binary just prints the same pointer at runtime so a curious user who runs
//! `cargo run --manifest-path examples/Cargo.toml` is not left wondering.

fn main() {
    println!(
        "rustral-examples placeholder.\n\n\
         Working examples live under crates/runtime/examples/. Try:\n\
           cargo run -p rustral-runtime --features training --example sst2_classifier\n\
           cargo run -p rustral-runtime --features training --example wikitext2_lm\n\
           cargo run -p rustral-runtime --features training --example tape_train_demo\n"
    );
}
