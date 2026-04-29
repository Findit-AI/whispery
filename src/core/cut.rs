//! Cut state machine — incremental WhisperX `merge_chunks`.
//!
//! All internal arithmetic is in 16 kHz analysis sample-index space
//! (`SampleRange`); conversion to the output timebase happens at
//! emission time. See spec §5.3.

use alloc::vec::Vec;
use core::time::Duration;

use crate::types::VadSegment;

/// Half-open range in 16 kHz analysis sample indices, stream-relative
/// (i.e., absolute since stream start, not relative to the live
/// buffer). Crate-private; only `TimeRange` (in the output timebase)
/// crosses the public surface.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct SampleRange {
    /// First sample of the range (inclusive).
    pub start: u64,
    /// One past the last sample of the range (exclusive).
    pub end: u64,
}

impl SampleRange {
    /// Construct from start and end. Panics if `end < start`.
    pub(crate) const fn new(start: u64, end: u64) -> Self {
        if end < start {
            panic!("SampleRange::new requires end >= start");
        }
        Self { start, end }
    }

    /// Length in samples.
    pub(crate) const fn len(&self) -> u64 {
        self.end - self.start
    }
}

/// Provenance tag on a `SubRange` inside a `MergedChunk.subs` list.
/// Lets downstream code distinguish a real silero VAD segment from a
/// hard-split fragment of an over-long segment.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum SubOrigin {
    /// Came directly from a `VadSegment` as pushed.
    Vad {
        /// Monotonic counter assigned by `Cut` on push.
        vad_seq: u32,
    },
    /// Result of hard-splitting a `VadSegment` longer than
    /// `chunk_size`. The full original VAD segment can be
    /// reconstructed by joining all `SubRange`s sharing this
    /// `vad_seq`.
    HardSplit {
        /// Original VAD segment's sequence number.
        vad_seq: u32,
        /// Zero-based index of this fragment.
        part: u8,
        /// Total number of fragments the original segment was split
        /// into.
        total_parts: u8,
    },
}

/// One sub-range inside a merged chunk, with provenance.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct SubRange {
    /// Sample-index range.
    pub range: SampleRange,
    /// Origin tag.
    pub origin: SubOrigin,
}

/// Output of the cut state machine.
#[derive(Clone, Debug)]
pub(crate) struct MergedChunk {
    /// Bounds of the merged chunk in 16 kHz sample-index space.
    pub range: SampleRange,
    /// Sub-VAD-segments composing the chunk, with origin tags.
    pub subs: Vec<SubRange>,
}

/// Internal state of the cut machine.
pub(crate) struct Cut {
    /// `chunk_size` expressed in 16 kHz samples (Duration ×
    /// SAMPLE_RATE_HZ at construction).
    chunk_size_samples: u64,
    /// Monotonic VAD-sequence counter.
    next_vad_seq: u32,
    /// Currently accumulating chunk's start (sample index, inclusive).
    /// `None` between chunks.
    current_start: Option<u64>,
    /// Currently accumulating chunk's end (sample index, exclusive).
    /// Maintained equal to `current_start` immediately after step 3.
    current_end: u64,
    /// Sub-ranges accumulated for the current chunk.
    current_subs: Vec<SubRange>,
}

impl Cut {
    /// Construct with the given chunk-size duration. The duration is
    /// converted to 16 kHz samples once.
    pub(crate) fn new(chunk_size: Duration) -> Self {
        let secs = chunk_size.as_secs_f64();
        let samples = (secs * crate::time::SAMPLE_RATE_HZ as f64).round() as u64;
        Self {
            chunk_size_samples: samples,
            next_vad_seq: 0,
            current_start: None,
            current_end: 0,
            current_subs: Vec::new(),
        }
    }

    /// Currently-configured chunk size in 16 kHz samples. Exposed
    /// for tests.
    pub(crate) fn chunk_size_samples(&self) -> u64 {
        self.chunk_size_samples
    }

    // push_segment and flush land in the next task.
    #[allow(dead_code)]
    fn _placeholder_for_subsequent_tasks(_: VadSegment) {}
}
