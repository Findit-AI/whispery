//! Cut state machine — incremental WhisperX `merge_chunks`.
//!
//! All internal arithmetic is in 16 kHz analysis sample-index space
//! (`SampleRange`); conversion to the output timebase happens at
//! emission time.

use alloc::vec::Vec;
use core::time::Duration;

use crate::{core::AsrParamsOverride, types::VadSegment};

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
  ///
  /// `part` and `total_parts` are `u32`. An earlier `u8` shape
  /// asserted `n_full <= 255`; with smaller `chunk_size`
  /// settings, realistic long-form audio (lectures, podcasts)
  /// can need more than 255 hard-split parts — the assertion
  /// would turn valid input into a process panic. Widening to
  /// `u32` removes the artificial ceiling; with default
  /// `chunk_size = 30 s` the bound is >2 hours per VAD segment,
  /// well past anything seen in practice.
  HardSplit {
    /// Original VAD segment's sequence number.
    vad_seq: u32,
    /// Zero-based index of this fragment.
    part: u32,
    /// Total number of fragments the original segment was split
    /// into.
    total_parts: u32,
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
  /// Per-packet `AsrParamsOverride` snapshot taken at the moment
  /// this chunk's accumulation *started* — i.e., when the first
  /// `SubRange` was fed to the cut machine. NOT the override at
  /// emission time.
  ///
  /// Reason: a chunk can accumulate across multiple
  /// `process_packet` calls when a VAD segment doesn't close the
  /// chunk in a single packet (silence-flush threshold or
  /// `chunk_size` wasn't crossed). With "override at emit time",
  /// audio pushed under packet A's override could be transcribed
  /// under packet B's override (or none) just because B was the
  /// packet whose silence happened to close the segment. Codex
  /// round-30 flagged this. Binding to start-time is the
  /// "first override wins" semantic users intend when they
  /// stamp an override with audio.
  pub override_at_start: Option<AsrParamsOverride>,
}

/// Internal state of the cut machine.
pub(crate) struct Cut {
  /// `chunk_size` expressed in 16 kHz samples (Duration ×
  /// SAMPLE_RATE_HZ at construction).
  chunk_size_samples: u64,
  /// If `Some`, flush the current chunk whenever a new sub-range
  /// arrives after a silence gap (`sub.start - current_end`)
  /// larger than this threshold. `None` keeps the WhisperX-style
  /// continuous batching where small silences are merged into a
  /// chunk for whisper context.
  silence_flush_samples: Option<u64>,
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
  /// Per-packet `AsrParamsOverride` snapshot captured the instant
  /// the currently-accumulating chunk *started* (when
  /// `current_start` transitioned `None → Some`). Carries the
  /// override forward so it can be attached to the
  /// `MergedChunk.override_at_start` when the chunk emits, even
  /// if accumulation spans multiple `process_packet` calls.
  /// `None` between chunks AND when no override was active at
  /// start; `current_start.is_some()` disambiguates.
  current_override_at_start: Option<AsrParamsOverride>,
}

impl Cut {
  /// Construct with the given chunk-size duration and optional
  /// silence-flush threshold. The durations are converted to
  /// 16 kHz samples once.
  pub(crate) fn new(chunk_size: Duration, silence_flush_gap: Option<Duration>) -> Self {
    let secs = chunk_size.as_secs_f64();
    // `.round()` is not available in `no_std`; add 0.5 then truncate,
    // which is equivalent for non-negative values.
    let samples = (secs * crate::time::SAMPLE_RATE_HZ as f64 + 0.5) as u64;
    let silence_flush_samples = silence_flush_gap
      .map(|d| (d.as_secs_f64() * crate::time::SAMPLE_RATE_HZ as f64 + 0.5) as u64);
    Self {
      chunk_size_samples: samples,
      silence_flush_samples,
      next_vad_seq: 0,
      current_start: None,
      current_end: 0,
      current_subs: Vec::new(),
      current_override_at_start: None,
    }
  }

  /// Currently-configured chunk size in 16 kHz samples. Exposed
  /// for tests.
  pub(crate) fn chunk_size_samples(&self) -> u64 {
    self.chunk_size_samples
  }

  /// Highest sample index ever pushed (inclusive of last segment's
  /// end_sample). `None` before any push. Used by `Transcriber`
  /// to enforce strict-monotonic VAD segment ordering.
  pub(crate) fn last_pushed_end(&self) -> Option<u64> {
    if self.next_vad_seq == 0 {
      None
    } else {
      Some(self.current_end)
    }
  }

  /// Start sample of the chunk currently accumulating in the cut
  /// state machine, if any. `None` between chunks. Used by trim's
  /// low-water computation: samples back to this index are still
  /// referenced by the unextracted partial chunk and must not be
  /// dropped before that chunk emits via push_segment or flush.
  pub(crate) fn pending_start(&self) -> Option<u64> {
    self.current_start
  }

  /// Whether a hypothetical sub-range starting at `sample_index`
  /// would force the current chunk to flush — either because it
  /// would push the chunk past `chunk_size_samples`, or because
  /// the silence gap from `current_end` to `sample_index` exceeds
  /// the configured `flush_on_silence_gap` threshold. Returns
  /// `false` if no chunk is accumulating yet.
  ///
  /// `handle_no_speech_through` calls this to pre-flush whenever a
  /// hypothetical future segment at `sample_index` would have
  /// triggered a flush — the caller has already declared that no
  /// such segment is coming, so the partial chunk can yield now
  /// instead of sitting until EOF or chunk_size.
  pub(crate) fn would_flush_at(&self, sample_index: u64) -> bool {
    let Some(start) = self.current_start else {
      return false;
    };
    if self.current_end <= start {
      // No real subs in the current chunk; nothing to flush.
      return false;
    }
    // Chunk-size flush.
    if sample_index.saturating_sub(start) > self.chunk_size_samples {
      return true;
    }
    // Silence-gap flush.
    if let Some(threshold) = self.silence_flush_samples {
      if sample_index.saturating_sub(self.current_end) > threshold {
        return true;
      }
    }
    false
  }

  /// Push a VAD segment through the cut state machine. Returns
  /// `Some(MergedChunk)` if this push closed an accumulating
  /// chunk; `None` otherwise.
  ///
  /// `current_override` is the per-packet `AsrParamsOverride`
  /// stamped on the dispatch for the duration of this
  /// `process_packet` call. It's snapshotted onto a chunk's
  /// state when accumulation starts and travels with the
  /// emitted `MergedChunk.override_at_start`. Pass `None` when
  /// no override is in effect.
  pub(crate) fn push_segment(
    &mut self,
    seg: VadSegment,
    current_override: Option<&AsrParamsOverride>,
  ) -> Vec<MergedChunk> {
    let len = seg.sample_count();
    let vad_seq = self.next_vad_seq;
    self.next_vad_seq += 1;

    let mut emitted = Vec::new();
    if len > self.chunk_size_samples {
      // Pre-split overlong segment into n equal-ish parts.
      // n = ceil(len / chunk_size_samples).
      let n_full = len.div_ceil(self.chunk_size_samples);
      // The previous u8 ceiling (255 parts) made realistic
      // long-form audio panic at small chunk_size settings.
      // SubOrigin::HardSplit's `part` / `total_parts` are now
      // u32 — only truly absurd input (>4 G parts) would
      // overflow, and that's well past any realistic upper bound
      // on `len / chunk_size_samples`.
      assert!(
        n_full <= u32::MAX as u64,
        "VadSegment of {} samples exceeds u32::MAX × chunk_size_samples ({}); pathological input",
        len,
        self.chunk_size_samples,
      );
      let n = n_full as u32;
      for i in 0..n {
        let part_start = seg.start_sample() + (i as u64 * len) / n as u64;
        let part_end = if i == n - 1 {
          seg.end_sample()
        } else {
          seg.start_sample() + ((i + 1) as u64 * len) / n as u64
        };
        let sub = SubRange {
          range: SampleRange::new(part_start, part_end),
          origin: SubOrigin::HardSplit {
            vad_seq,
            part: i,
            total_parts: n,
          },
        };
        if let Some(chunk) = self.feed_sub(sub, current_override) {
          emitted.push(chunk);
        }
      }
    } else {
      let sub = SubRange {
        range: SampleRange::new(seg.start_sample(), seg.end_sample()),
        origin: SubOrigin::Vad { vad_seq },
      };
      if let Some(chunk) = self.feed_sub(sub, current_override) {
        emitted.push(chunk);
      }
    }
    emitted
  }

  /// Flush the accumulating chunk on EOF. Returns the partial
  /// chunk if any was being accumulated. The chunk's
  /// `override_at_start` is the snapshot captured when this
  /// chunk's accumulation began — NOT the override at flush time.
  pub(crate) fn flush(&mut self) -> Option<MergedChunk> {
    let start = self.current_start.take()?;
    let subs = core::mem::take(&mut self.current_subs);
    let override_at_start = self.current_override_at_start.take();
    Some(MergedChunk {
      range: SampleRange::new(start, self.current_end),
      subs,
      override_at_start,
    })
  }

  /// Feed one sub-range through the merge logic.
  ///
  /// `current_override` is snapshotted when this sub starts a new
  /// chunk (i.e., transitions `current_start` from `None` to
  /// `Some`), or after an emit resets the accumulator and this
  /// sub becomes the seed of the next chunk. Carrying
  /// `Option<&...>` avoids cloning when we don't need to.
  fn feed_sub(
    &mut self,
    sub: SubRange,
    current_override: Option<&AsrParamsOverride>,
  ) -> Option<MergedChunk> {
    let mut emitted = None;

    // Step 0: silence-flush. If a chunk is accumulating and the
    // gap between the new sub's start and the current chunk's
    // end exceeds the configured threshold, flush the current
    // chunk before adding the new sub. This gives utterance-
    // boundary chunking when callers want it (TranscriberOptions::
    // flush_on_silence_gap = Some(threshold)). When the threshold
    // is None (default), small silences stay merged into one
    // chunk for better whisper context — original WhisperX
    // semantics.
    if let (Some(threshold), Some(cs)) = (self.silence_flush_samples, self.current_start) {
      let gap = sub.range.start.saturating_sub(self.current_end);
      if gap > threshold && self.current_end > cs {
        let subs = core::mem::take(&mut self.current_subs);
        let override_at_start = self.current_override_at_start.take();
        emitted = Some(MergedChunk {
          range: SampleRange::new(cs, self.current_end),
          subs,
          override_at_start,
        });
        self.current_start = None;
      }
    }

    // Step 3: initialise current_start AND current_end if absent.
    // This is also where we capture the override snapshot for
    // the new accumulation. Two cases get here: (a) very first
    // sub of a chunk (was None forever); (b) we just emitted
    // above (silence-flush) and this sub seeds the next chunk.
    // Either way, the override active right now is what binds.
    if self.current_start.is_none() {
      self.current_start = Some(sub.range.start);
      self.current_end = sub.range.start;
      self.current_override_at_start = current_override.cloned();
    }
    let cs = self.current_start.expect("just initialised");

    // Step 4: emit when adding `sub` would exceed chunk_size, AND
    // we have at least one segment already in this chunk. Skipped
    // if step 0 already emitted a chunk (a hard-split sub never
    // alone exceeds chunk_size, so we won't double-emit).
    if emitted.is_none()
      && sub.range.end.saturating_sub(cs) > self.chunk_size_samples
      && self.current_end > cs
    {
      let subs = core::mem::take(&mut self.current_subs);
      let prev_override = self.current_override_at_start.take();
      emitted = Some(MergedChunk {
        range: SampleRange::new(cs, self.current_end),
        subs,
        override_at_start: prev_override,
      });
      self.current_start = Some(sub.range.start);
      self.current_end = sub.range.start;
      // The next chunk is seeded by this sub right now, so its
      // override-at-start is the same `current_override` we were
      // passed.
      self.current_override_at_start = current_override.cloned();
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
    Cut::new(Duration::from_secs(chunk_size_secs), None)
  }

  #[test]
  fn empty_flush_returns_none() {
    let mut c = cut(30);
    assert!(c.flush().is_none());
  }

  #[test]
  fn single_segment_under_chunk_does_not_flush_until_eof() {
    let mut c = cut(30);
    let emitted = c.push_segment(VadSegment::new(0, 16_000), None);
    assert!(emitted.is_empty(), "no chunk yet, segment is short");
    let final_chunk = c.flush().unwrap();
    assert_eq!(final_chunk.range, SampleRange::new(0, 16_000));
    assert_eq!(final_chunk.subs.len(), 1);
    assert!(matches!(
      final_chunk.subs[0].origin,
      SubOrigin::Vad { vad_seq: 0 }
    ));
  }

  #[test]
  fn segments_summing_under_chunk_merge_into_one() {
    let mut c = cut(30);
    // chunk_size = 30s = 480_000 samples
    c.push_segment(VadSegment::new(0, 100_000), None);
    c.push_segment(VadSegment::new(120_000, 200_000), None);
    c.push_segment(VadSegment::new(220_000, 300_000), None);
    let final_chunk = c.flush().unwrap();
    assert_eq!(final_chunk.range, SampleRange::new(0, 300_000));
    assert_eq!(final_chunk.subs.len(), 3);
  }

  #[test]
  fn segments_exceeding_chunk_flush_at_boundary() {
    let mut c = cut(30);
    // Three 200_000-sample segments, each within chunk_size, but
    // their union (start 0 → end 600_000+) exceeds 480_000.
    let r1 = c.push_segment(VadSegment::new(0, 200_000), None);
    let r2 = c.push_segment(VadSegment::new(210_000, 400_000), None);
    // Adding the 3rd: 600_000 - 0 = 600_000 > 480_000 → flush.
    let r3 = c.push_segment(VadSegment::new(410_000, 600_000), None);
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
    let mut c = Cut::new(Duration::from_millis(625), None); // 10_000 samples (no silence-flush) @ 16 kHz
    // len = 29_000, chunk_size = 10_000 → n = 3.
    // Per-index: start = [0, 29000/3 = 9666, 2*29000/3 = 19333]
    //            end   = [9666, 19333, 29000]
    // Each part length: 9666, 9667, 9667 — all ≤ 10_000.
    let emitted = c.push_segment(VadSegment::new(0, 29_000), None);
    assert_eq!(
      emitted.len(),
      2,
      "first two of three parts emit a chunk each"
    );
    assert_eq!(emitted[0].range, SampleRange::new(0, 9_666));
    assert_eq!(emitted[1].range, SampleRange::new(9_666, 19_333));

    // Third part is left accumulating.
    let final_chunk = c.flush().unwrap();
    assert_eq!(final_chunk.range, SampleRange::new(19_333, 29_000));

    // Verify origin tags.
    assert_eq!(emitted[0].subs.len(), 1);
    match emitted[0].subs[0].origin {
      SubOrigin::HardSplit {
        vad_seq: 0,
        part: 0,
        total_parts: 3,
      } => {}
      o => panic!("unexpected origin {:?}", o),
    }
    match emitted[1].subs[0].origin {
      SubOrigin::HardSplit {
        vad_seq: 0,
        part: 1,
        total_parts: 3,
      } => {}
      o => panic!("unexpected origin {:?}", o),
    }
    match final_chunk.subs[0].origin {
      SubOrigin::HardSplit {
        vad_seq: 0,
        part: 2,
        total_parts: 3,
      } => {}
      o => panic!("unexpected origin {:?}", o),
    }
  }

  /// A single VAD segment longer than 255 × chunk_size used to
  /// panic in the old u8-bounded code. With chunk_size=625ms
  /// (10_000 samples), a ~3-minute segment (300 parts) is
  /// realistic for lectures / podcasts and must split
  /// successfully rather than aborting the process.
  #[test]
  fn hard_split_supports_more_than_255_parts() {
    let mut c = Cut::new(Duration::from_millis(625), None); // 10_000 samples
    let parts_wanted: u64 = 300;
    let len = parts_wanted * 10_000;
    let emitted = c.push_segment(VadSegment::new(0, len), None);
    // n_full = len.div_ceil(10_000) = 300 → 299 chunks emit, the
    // last accumulates and only emerges from flush().
    assert_eq!(emitted.len(), (parts_wanted - 1) as usize);

    // Verify total_parts on every emitted chunk.
    for sub_chunk in &emitted {
      for sub in &sub_chunk.subs {
        match sub.origin {
          SubOrigin::HardSplit { total_parts, .. } => {
            assert_eq!(total_parts as u64, parts_wanted);
          }
          other => panic!("expected HardSplit, got {:?}", other),
        }
      }
    }

    let last = c.flush().unwrap();
    match last.subs[0].origin {
      SubOrigin::HardSplit {
        part, total_parts, ..
      } => {
        assert_eq!(part as u64, parts_wanted - 1);
        assert_eq!(total_parts as u64, parts_wanted);
      }
      other => panic!("expected HardSplit, got {:?}", other),
    }
  }

  /// Silence-flush threshold (`Some(threshold)`) flushes the
  /// current chunk when the gap to the new sub exceeds it. The
  /// segments individually stay under chunk_size, so without the
  /// threshold they would merge into one chunk (WhisperX-style).
  #[test]
  fn silence_flush_threshold_separates_chunks_at_gap() {
    // chunk_size = 30 s, silence threshold = 1 s (16_000 samples).
    let mut c = Cut::new(Duration::from_secs(30), Some(Duration::from_secs(1)));
    // First segment ends at 16_000.
    let r1 = c.push_segment(VadSegment::new(0, 16_000), None);
    assert!(r1.is_empty());
    // Second segment starts at 48_000 — gap of 32_000 (2 s) > 1 s.
    // Should flush chunk 0 and start chunk 1.
    let r2 = c.push_segment(VadSegment::new(48_000, 64_000), None);
    assert_eq!(r2.len(), 1);
    assert_eq!(r2[0].range, SampleRange::new(0, 16_000));
    let final_chunk = c.flush().unwrap();
    assert_eq!(final_chunk.range, SampleRange::new(48_000, 64_000));
  }

  /// Silence-flush threshold (`Some(threshold)`) does NOT flush
  /// when the gap is under the threshold — small silences stay
  /// merged for whisper context.
  #[test]
  fn silence_flush_threshold_keeps_short_gap_merged() {
    // chunk_size = 30 s, silence threshold = 2 s.
    let mut c = Cut::new(Duration::from_secs(30), Some(Duration::from_secs(2)));
    let r1 = c.push_segment(VadSegment::new(0, 16_000), None);
    // Gap of 16_000 samples (1 s) — under threshold.
    let r2 = c.push_segment(VadSegment::new(32_000, 48_000), None);
    assert!(r1.is_empty());
    assert!(r2.is_empty());
    let final_chunk = c.flush().unwrap();
    assert_eq!(
      final_chunk.range,
      SampleRange::new(0, 48_000),
      "small gap kept the two segments in one chunk"
    );
    assert_eq!(final_chunk.subs.len(), 2);
  }

  /// `None` threshold (default) preserves original WhisperX
  /// behavior — large silences don't trigger flush, only
  /// chunk_size does.
  #[test]
  fn silence_flush_none_preserves_whisperx_batching() {
    let mut c = Cut::new(Duration::from_secs(30), None);
    let r1 = c.push_segment(VadSegment::new(0, 16_000), None);
    // 5 s gap — would trip a silence-flush threshold, but None.
    let r2 = c.push_segment(VadSegment::new(96_000, 112_000), None);
    assert!(r1.is_empty());
    assert!(r2.is_empty());
    let final_chunk = c.flush().unwrap();
    // Both segments in one chunk because chunk_size = 30 s
    // (480_000 samples) wasn't exceeded.
    assert_eq!(final_chunk.range, SampleRange::new(0, 112_000));
    assert_eq!(final_chunk.subs.len(), 2);
  }

  #[test]
  fn hard_split_strict_bound_holds_on_pathological_lengths() {
    // The audit's failure case: len=29, chunk=10, n=3 must produce
    // parts ≤ 10 — never 9, 9, 11.
    // We need a chunk_size_samples of exactly 10 — build Cut directly.
    let mut c = Cut {
      chunk_size_samples: 10,
      silence_flush_samples: None,
      next_vad_seq: 0,
      current_start: None,
      current_end: 0,
      current_subs: Vec::new(),
      current_override_at_start: None,
    };
    let emitted = c.push_segment(VadSegment::new(0, 29), None);
    // n=3, parts: [0,9), [9,19), [19,29) → each length 10. None
    // exceeds chunk_size_samples=10. Two emit, third stays.
    assert_eq!(emitted.len(), 2);
    assert!(emitted[0].range.len() <= 10);
    assert!(emitted[1].range.len() <= 10);
    let last = c.flush().unwrap();
    assert!(last.range.len() <= 10);
  }
}
