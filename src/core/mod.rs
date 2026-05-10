//! Sans-I/O core state machine.

mod buffer;
mod command;
mod cut;
mod dispatch;
mod event;
pub mod oov;
mod transcriber;

pub use command::{
  AlignmentResult, AsrParams, AsrParamsOverride, AsrResult, Command, SamplingStrategy,
};
pub use event::Event;
pub use oov::{
  OovDecision, OovEvent, OovKind, ResolvedOov, default_oov_decisions, fail_closed_all_decisions,
  wildcard_all_decisions,
};
pub use transcriber::{LanguagePolicy, Transcriber, TranscriberOptions};
