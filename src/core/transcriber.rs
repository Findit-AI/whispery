//! Transcriber — the public Sans-I/O surface.
//!
//! `Transcriber` is `Send + !Sync` (every public mutating method
//! takes `&mut self`). Multi-threaded drivers must wrap in
//! `Mutex<Transcriber>` themselves.
//!
//! See spec §5.1.

use core::time::Duration;

use mediatime::{Timebase, Timestamp};

use crate::core::buffer::SampleBuffer;
use crate::core::command::{AsrParams, Command};
use crate::core::cut::Cut;
use crate::core::dispatch::Dispatch;
use crate::core::event::Event;
use crate::types::Lang;

/// Language-detection / locking strategy.
#[derive(Clone, Debug)]
pub enum LanguagePolicy {
    /// Each chunk independently auto-detects.
    Auto,
    /// Caller supplies the language; whisper is given a hard hint
    /// and never auto-detects.
    Lock {
        /// Locked language.
        hint: Lang,
    },
    /// Auto-detect on the first `n` chunks that emit non-empty text,
    /// then lock the most-frequent detected language for the rest of
    /// the session. WhisperX-equivalent default; `n = 1` matches
    /// WhisperX exactly.
    AutoLockAfter(usize),
}

impl Default for LanguagePolicy {
    fn default() -> Self {
        Self::AutoLockAfter(1)
    }
}

/// Configuration for the core state machine.
#[derive(Clone, Debug)]
pub struct TranscriberConfig {
    /// Maximum duration of a merged chunk. Default 30 s.
    pub chunk_size: Duration,
    /// Max samples kept in the internal buffer before push returns
    /// Backpressure. Default 60 s × 16 kHz = 960 000.
    pub buffer_cap_samples: usize,
    /// Maximum forward-gap that is silently zero-filled. Default
    /// 200 ms × 16 kHz = 3200.
    pub gap_tolerance_samples: u64,
    /// Whether to emit `RunAlignment` after each ASR completion.
    pub word_alignment: bool,
    /// Maximum chunks in flight. Default `worker_count + 2`; without
    /// runner context, the core defaults to 6.
    pub max_in_flight: usize,
    /// Default ASR params injected into every `RunAsr` command.
    pub asr_params: AsrParams,
    /// Language detection / locking strategy.
    pub language_policy: LanguagePolicy,
}

impl Default for TranscriberConfig {
    fn default() -> Self {
        Self {
            chunk_size: Duration::from_secs(30),
            buffer_cap_samples: 60 * 16_000,
            gap_tolerance_samples: 200 * 16, // 200 ms at 16 kHz
            word_alignment: false,
            max_in_flight: 6,
            asr_params: AsrParams::default(),
            language_policy: LanguagePolicy::default(),
        }
    }
}

/// The Sans-I/O state machine. See spec §5.1.
///
/// `Transcriber` is `Send` (movable across threads) but `!Sync`
/// (every public mutating method takes `&mut self`). A consumer that
/// wants to drive it from multiple threads must wrap it in
/// `Mutex<Transcriber>` themselves; whispery does not provide
/// internal synchronisation.
pub struct Transcriber {
    #[allow(dead_code)] // consumed in Tasks 19/20
    config: TranscriberConfig,
    buffer: SampleBuffer,
    cut: Cut,
    dispatch: Dispatch,
    #[allow(dead_code)] // consumed in Tasks 19/20
    next_chunk_id: u64,
    #[allow(dead_code)] // consumed in Tasks 19/20
    eof_signaled: bool,
}

impl Transcriber {
    /// Construct from config.
    pub fn new(config: TranscriberConfig) -> Self {
        let buffer = SampleBuffer::new(config.buffer_cap_samples, config.gap_tolerance_samples);
        let cut = Cut::new(config.chunk_size);
        let dispatch = Dispatch::new(
            config.asr_params.clone(),
            config.word_alignment,
            config.max_in_flight,
        );
        Self {
            config,
            buffer,
            cut,
            dispatch,
            next_chunk_id: 0,
            eof_signaled: false,
        }
    }

    /// Pop the front command, consulting `unpoll_command`'s parked
    /// slot first.
    pub fn poll_command(&mut self) -> Option<Command> {
        self.dispatch.poll_command()
    }

    /// Pop the front event.
    pub fn poll_event(&mut self) -> Option<Event> {
        self.dispatch.poll_event()
    }

    /// Re-park the front of the command queue. **Visibility:
    /// `pub(crate)`** — the runner module is the only legitimate
    /// caller. Out-of-tree consumers driving the state machine
    /// themselves do not need this affordance.
    pub(crate) fn unpoll_command(&mut self, cmd: Command) {
        self.dispatch.unpoll_command(cmd);
    }

    /// True iff every queue is empty: no buffered samples, no
    /// pending command/event, no in_flight chunks, no cut_pending
    /// entries. Pre-restart in-flight chunks (those still working
    /// through whisper or alignment) keep `is_idle()` false until
    /// they emit; `restart_at` does not synthetically clear them.
    pub fn is_idle(&self) -> bool {
        self.dispatch.is_idle() && self.buffer.buffered_samples() == 0
    }

    /// Live buffer length in samples.
    pub fn buffered_samples(&self) -> usize {
        self.buffer.buffered_samples()
    }

    /// Output timebase recorded from the first `push_samples` call.
    pub fn output_timebase(&self) -> Option<Timebase> {
        self.buffer.output_timebase()
    }

    /// Authoritative output-timebase PTS the buffer expects for the
    /// next contiguous `push_samples` call. Returns `None` before
    /// the first push.
    pub fn next_expected_starts_at(&self) -> Option<Timestamp> {
        self.buffer.next_expected_starts_at()
    }

    /// Non-mutating predicate: would the next push of `samples_len`
    /// audio samples plus `vad_count` VAD segments fit under the
    /// configured caps?
    pub fn would_accept(&self, samples_len: usize, _vad_count: usize) -> bool {
        self.buffered_samples() + samples_len <= self.config.buffer_cap_samples
    }
}
