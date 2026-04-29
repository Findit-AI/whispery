//! Sans-I/O core state machine.

mod command;
mod cut;
mod event;

pub use command::{
    AlignmentResult, AsrParams, AsrParamsOverride, AsrResult, Command, SamplingStrategy,
};
pub use event::Event;

// `cut` is crate-private; nothing in it crosses the public surface.
