//! `Lang` — re-exported from the in-house `whisper-cpp` crate.
//!
//! `Lang` is structurally a whisper.cpp concept (the set of language
//! codes whisper.cpp's vocabulary supports), so it lives in
//! `crates/whisper-cpp/src/lang.rs` alongside the FFI surface that
//! produces it. Asry re-exports here so existing call sites
//! (`asry::Lang`) continue to compile unchanged. The serde
//! impls — lowercase ISO-639-1 wire format, case-insensitive
//! deserialise, validation against `[a-zA-Z]{1,8}` — are gated
//! behind asry's `serde` feature, which chains to
//! `whisper-cpp/serde`.
//!
//! Round-trip + canonicalisation invariants are documented on the
//! re-exported type itself.

pub use whispercpp::Lang;
