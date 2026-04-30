//! 8-step alignment algorithm modules. See spec §6.3.2.
//!
//! The pipeline stages live in separate files so each step has its
//! own unit-test surface; `Aligner::align` glues them in Task 14.

pub(crate) mod silence_mask;
pub(crate) mod tokenize;
