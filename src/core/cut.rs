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

    /// Push a VAD segment through the cut state machine. Returns
    /// `Some(MergedChunk)` if this push closed an accumulating
    /// chunk; `None` otherwise.
    pub(crate) fn push_segment(&mut self, seg: VadSegment) -> Vec<MergedChunk> {
        let len = seg.sample_count();
        let vad_seq = self.next_vad_seq;
        self.next_vad_seq += 1;

        let mut emitted = Vec::new();
        if len > self.chunk_size_samples {
            // Pre-split overlong segment into n equal-ish parts.
            // n = ceil(len / chunk_size_samples).
            let n_full = len.div_ceil(self.chunk_size_samples);
            // SubOrigin::HardSplit.total_parts is u8 (255 max), so a
            // single VAD segment longer than 255 × chunk_size is
            // outside the design envelope. For default chunk_size=30s
            // that's 127 minutes of continuous speech in one segment;
            // pathological. Hard-fail rather than silently drop data.
            assert!(
                n_full <= u8::MAX as u64,
                "VadSegment of {} samples exceeds 255×chunk_size ({}); refuse to drop data",
                len,
                255 * self.chunk_size_samples,
            );
            let n = n_full as u8;
            for i in 0..n {
                let part_start = seg.start_sample() + (i as u64 * len) / n as u64;
                let part_end = if i == n - 1 {
                    seg.end_sample()
                } else {
                    seg.start_sample() + ((i + 1) as u64 * len) / n as u64
                };
                let sub = SubRange {
                    range: SampleRange::new(part_start, part_end),
                    origin: SubOrigin::HardSplit { vad_seq, part: i, total_parts: n },
                };
                if let Some(chunk) = self.feed_sub(sub) {
                    emitted.push(chunk);
                }
            }
        } else {
            let sub = SubRange {
                range: SampleRange::new(seg.start_sample(), seg.end_sample()),
                origin: SubOrigin::Vad { vad_seq },
            };
            if let Some(chunk) = self.feed_sub(sub) {
                emitted.push(chunk);
            }
        }
        emitted
    }

    /// Flush the accumulating chunk on EOF. Returns the partial
    /// chunk if any was being accumulated.
    pub(crate) fn flush(&mut self) -> Option<MergedChunk> {
        let start = self.current_start.take()?;
        let subs = core::mem::take(&mut self.current_subs);
        Some(MergedChunk {
            range: SampleRange::new(start, self.current_end),
            subs,
        })
    }

    /// Feed one sub-range through the merge logic.
    fn feed_sub(&mut self, sub: SubRange) -> Option<MergedChunk> {
        // Step 3: initialise current_start AND current_end if absent.
        if self.current_start.is_none() {
            self.current_start = Some(sub.range.start);
            self.current_end = sub.range.start;
        }
        let cs = self.current_start.expect("just initialised");

        let mut emitted = None;

        // Step 4: emit when adding `sub` would exceed chunk_size, AND
        // we have at least one segment already in this chunk.
        if sub.range.end.saturating_sub(cs) > self.chunk_size_samples
            && self.current_end > cs
        {
            let subs = core::mem::take(&mut self.current_subs);
            emitted = Some(MergedChunk {
                range: SampleRange::new(cs, self.current_end),
                subs,
            });
            self.current_start = Some(sub.range.start);
            self.current_end = sub.range.start;
        }

        // Step 5: extend the current chunk with sub.
        self.current_end = sub.range.end;
        self.current_subs.push(sub);

        emitted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cut(chunk_size_secs: u64) -> Cut {
        Cut::new(Duration::from_secs(chunk_size_secs))
    }

    #[test]
    fn empty_flush_returns_none() {
        let mut c = cut(30);
        assert!(c.flush().is_none());
    }

    #[test]
    fn single_segment_under_chunk_does_not_flush_until_eof() {
        let mut c = cut(30);
        let emitted = c.push_segment(VadSegment::new(0, 16_000));
        assert!(emitted.is_empty(), "no chunk yet, segment is short");
        let final_chunk = c.flush().unwrap();
        assert_eq!(final_chunk.range, SampleRange::new(0, 16_000));
        assert_eq!(final_chunk.subs.len(), 1);
        assert!(matches!(final_chunk.subs[0].origin, SubOrigin::Vad { vad_seq: 0 }));
    }

    #[test]
    fn segments_summing_under_chunk_merge_into_one() {
        let mut c = cut(30);
        // chunk_size = 30s = 480_000 samples
        c.push_segment(VadSegment::new(0, 100_000));
        c.push_segment(VadSegment::new(120_000, 200_000));
        c.push_segment(VadSegment::new(220_000, 300_000));
        let final_chunk = c.flush().unwrap();
        assert_eq!(final_chunk.range, SampleRange::new(0, 300_000));
        assert_eq!(final_chunk.subs.len(), 3);
    }

    #[test]
    fn segments_exceeding_chunk_flush_at_boundary() {
        let mut c = cut(30);
        // Three 200_000-sample segments, each within chunk_size, but
        // their union (start 0 → end 600_000+) exceeds 480_000.
        let r1 = c.push_segment(VadSegment::new(0, 200_000));
        let r2 = c.push_segment(VadSegment::new(210_000, 400_000));
        // Adding the 3rd: 600_000 - 0 = 600_000 > 480_000 → flush.
        let r3 = c.push_segment(VadSegment::new(410_000, 600_000));
        assert!(r1.is_empty());
        assert!(r2.is_empty());
        assert_eq!(r3.len(), 1);
        assert_eq!(r3[0].range, SampleRange::new(0, 400_000));
        assert_eq!(r3[0].subs.len(), 2);

        // The third segment is now accumulating in a fresh chunk.
        let final_chunk = c.flush().unwrap();
        assert_eq!(final_chunk.range, SampleRange::new(410_000, 600_000));
        assert_eq!(final_chunk.subs.len(), 1);
    }

    #[test]
    fn over_long_single_segment_hard_splits_with_per_index_formula() {
        let mut c = Cut::new(Duration::from_millis(625)); // 10_000 samples @ 16 kHz
        // len = 29_000, chunk_size = 10_000 → n = 3.
        // Per-index: start = [0, 29000/3 = 9666, 2*29000/3 = 19333]
        //            end   = [9666, 19333, 29000]
        // Each part length: 9666, 9667, 9667 — all ≤ 10_000.
        let emitted = c.push_segment(VadSegment::new(0, 29_000));
        assert_eq!(emitted.len(), 2, "first two of three parts emit a chunk each");
        assert_eq!(emitted[0].range, SampleRange::new(0, 9_666));
        assert_eq!(emitted[1].range, SampleRange::new(9_666, 19_333));

        // Third part is left accumulating.
        let final_chunk = c.flush().unwrap();
        assert_eq!(final_chunk.range, SampleRange::new(19_333, 29_000));

        // Verify origin tags.
        assert_eq!(emitted[0].subs.len(), 1);
        match emitted[0].subs[0].origin {
            SubOrigin::HardSplit { vad_seq: 0, part: 0, total_parts: 3 } => {}
            o => panic!("unexpected origin {:?}", o),
        }
        match emitted[1].subs[0].origin {
            SubOrigin::HardSplit { vad_seq: 0, part: 1, total_parts: 3 } => {}
            o => panic!("unexpected origin {:?}", o),
        }
        match final_chunk.subs[0].origin {
            SubOrigin::HardSplit { vad_seq: 0, part: 2, total_parts: 3 } => {}
            o => panic!("unexpected origin {:?}", o),
        }
    }

    #[test]
    fn hard_split_strict_bound_holds_on_pathological_lengths() {
        // The audit's failure case: len=29, chunk=10, n=3 must produce
        // parts ≤ 10 — never 9, 9, 11.
        // We need a chunk_size_samples of exactly 10 — build Cut directly.
        let mut c = Cut {
            chunk_size_samples: 10,
            next_vad_seq: 0,
            current_start: None,
            current_end: 0,
            current_subs: Vec::new(),
        };
        let emitted = c.push_segment(VadSegment::new(0, 29));
        // n=3, parts: [0,9), [9,19), [19,29) → each length 10. None
        // exceeds chunk_size_samples=10. Two emit, third stays.
        assert_eq!(emitted.len(), 2);
        assert!(emitted[0].range.len() <= 10);
        assert!(emitted[1].range.len() <= 10);
        let last = c.flush().unwrap();
        assert!(last.range.len() <= 10);
    }
}
