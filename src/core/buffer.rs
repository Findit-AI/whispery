//! `SampleBuffer` — bounded f32 buffer with output-timebase PTS
//! arithmetic anchored at the first push.
//!
//! Invariants: `base_pts_out_anchor` is immutable after the first
//! push (so trim doesn't accumulate drift on non-integer-ratio
//! output timebases); the regression check runs in output-PTS
//! space (so contiguous caller pushes on NTSC-like timebases
//! don't produce spurious `PtsRegression`); trim's low-water is
//! computed from `cut_pending` only, not `in_flight`, because
//! in-flight chunks already hold their audio in their own
//! `Arc<[f32]>` (decoupled from the live buffer).

use alloc::vec::Vec;

use mediatime::{Timebase, Timestamp};

use crate::{time::ANALYSIS_TIMEBASE, types::TranscriberError};

/// Live audio buffer.
pub(crate) struct SampleBuffer {
  /// Output timebase recorded from the first push.
  output_tb: Option<Timebase>,
  /// PTS (in `output_tb`) of stream-zero. **Immutable** after the
  /// first push.
  base_pts_out_anchor: i64,
  /// Total samples ever appended (monotonic; reset only by
  /// `handle_restart`).
  absolute_sample_offset: u64,
  /// Samples dropped by trim (monotonic).
  buffer_drop_offset: u64,
  /// Live samples in the range
  /// `[buffer_drop_offset, absolute_sample_offset)`.
  samples: Vec<f32>,
  /// Cap on `samples.len()` before `append` returns Backpressure.
  cap: usize,
  /// Maximum forward-gap that is silently zero-filled, in 16 kHz
  /// samples.
  gap_tolerance_samples: u64,
}

impl SampleBuffer {
  /// Construct an empty buffer with the given caps.
  pub(crate) fn new(cap: usize, gap_tolerance_samples: u64) -> Self {
    Self {
      output_tb: None,
      base_pts_out_anchor: 0,
      absolute_sample_offset: 0,
      buffer_drop_offset: 0,
      samples: Vec::new(),
      cap,
      gap_tolerance_samples,
    }
  }

  /// Output timebase (None until first push).
  pub(crate) fn output_timebase(&self) -> Option<Timebase> {
    self.output_tb
  }

  /// PTS-anchor at stream-zero, in the current output timebase.
  /// Mutates only on `handle_restart`; chunks extracted within a
  /// single between-restart epoch share this value. The
  /// alignment dispatch snapshots it onto each chunk record at
  /// extract time so post-restart word-mapping for surviving
  /// pre-restart chunks uses the original epoch's anchor
  /// rather than whatever the buffer is currently anchored at.
  #[cfg(feature = "alignment")]
  pub(crate) fn base_pts_out_anchor(&self) -> i64 {
    self.base_pts_out_anchor
  }

  /// Append a packet of samples whose first sample's PTS is
  /// `starts_at` in the output timebase. Returns `Backpressure`
  /// when the buffer would exceed its cap; `PtsRegression` /
  /// `GapExceedsTolerance` / `InconsistentTimebase` per their
  /// usual contracts.
  pub(crate) fn append(
    &mut self,
    starts_at: Timestamp,
    packet: &[f32],
    extra_queued_samples: usize,
  ) -> Result<(), TranscriberError> {
    // Do NOT commit `output_tb` / `base_pts_out_anchor` until
    // every error path has been cleared. Pre-fix code wrote the
    // anchor on first push *before* the capacity check, so a
    // first-push Backpressure left a "ghost" timebase that later
    // retries (with a corrected timebase or smaller packet) would
    // race against, tripping InconsistentTimebase / PtsRegression.
    // Compute against the *effective* anchor (the one we'd commit
    // if every check passes) without writing to `self` until then.
    let (effective_tb, effective_anchor, would_be_first_push) = match self.output_tb {
      Some(expected_tb) => {
        if starts_at.timebase() != expected_tb {
          return Err(TranscriberError::InconsistentTimebase {
            expected: expected_tb,
            got: starts_at.timebase(),
          });
        }
        (expected_tb, self.base_pts_out_anchor, false)
      }
      None => (starts_at.timebase(), starts_at.pts(), true),
    };

    // Compute expected next-PTS in output-tb space, then the
    // delta against caller's starts_at. The regression check
    // stays in output-PTS space so contiguous pushes on
    // non-integer-ratio output timebases don't trip spurious
    // regressions through round-trip truncation.
    let expected_pts_out = effective_anchor
      + Timebase::rescale_pts(
        self.absolute_sample_offset as i64,
        ANALYSIS_TIMEBASE,
        effective_tb,
      );
    let delta_pts_out = starts_at.pts() - expected_pts_out;

    let delta_samples: u64 = if delta_pts_out < 0 {
      return Err(TranscriberError::PtsRegression {
        kind: crate::types::PushKind::Samples,
        advance: delta_pts_out,
      });
    } else if delta_pts_out == 0 {
      0
    } else {
      // Convert the gap back to 16 kHz samples for the
      // zero-fill width / tolerance check.
      let g = Timebase::rescale_pts(delta_pts_out, effective_tb, ANALYSIS_TIMEBASE);
      if (g as u64) > self.gap_tolerance_samples {
        return Err(TranscriberError::GapExceedsTolerance {
          gap_samples: g as u64,
          tolerance_samples: self.gap_tolerance_samples,
        });
      }
      g as u64
    };

    // Check capacity BEFORE mutating. The doc on
    // TranscriberError::Backpressure says "buffered samples
    // *would* exceed the cap" — earlier code mutated first then
    // reported, which left the caller in an un-retryable
    // position (samples committed, retry trips PtsRegression).
    // With the pre-mutation check, Backpressure is a true atomic
    // rejection: the input is dropped on the floor and the
    // caller can retry the same packet later.
    //
    // Include `extra_queued_samples` (audio already held in
    // cut_pending Arcs from the pre-extraction design). Without
    // this term, a slow runner could let cut_pending grow
    // unboundedly because trim emptied the live buffer.
    // Codex round-37 round-26 [high]: overflow-safe capacity
    // arithmetic. Pre-fix `samples.len() + delta_samples as
    // usize + packet.len() + extra_queued_samples` was unchecked
    // — a public `gap_tolerance_samples` near `u64::MAX` plus a
    // large `delta_samples` could wrap to a small `usize` and
    // bypass the `> self.cap` guard, letting the subsequent
    // zero-fill `extend` attempt a multi-GB allocation. Cast +
    // sum via `usize::try_from` and `checked_add`; treat any
    // overflow as backpressure (the input would not fit by any
    // measure).
    let delta_usize = match usize::try_from(delta_samples) {
      Ok(v) => v,
      Err(_) => {
        return Err(TranscriberError::Backpressure {
          buffered: usize::MAX,
          cap: self.cap,
        });
      }
    };
    let total_with_queued = self
      .samples
      .len()
      .checked_add(delta_usize)
      .and_then(|v| v.checked_add(packet.len()))
      .and_then(|v| v.checked_add(extra_queued_samples));
    let total_with_queued = match total_with_queued {
      Some(v) => v,
      None => {
        return Err(TranscriberError::Backpressure {
          buffered: usize::MAX,
          cap: self.cap,
        });
      }
    };
    if total_with_queued > self.cap {
      return Err(TranscriberError::Backpressure {
        buffered: total_with_queued,
        cap: self.cap,
      });
    }

    // Codex round-37 + round-38: empty-packet must never mutate
    // stream state. Three cases cover all empty-packet inputs:
    //
    // 1. Empty packet at `delta_pts_out > 0` (a "heartbeat" at a
    //    slightly future PTS): used to commit the anchor (if
    //    first push), zero-fill, and advance
    //    `absolute_sample_offset`. The next real packet at the
    //    originally-expected PTS then trips `PtsRegression`. Now
    //    a true no-op.
    // 2. Empty FIRST push (round-38 finding 4): used to commit
    //    the anchor at the heartbeat's PTS. Real first-audio at
    //    a different PTS would then trip `PtsRegression` /
    //    `InconsistentTimebase`. Now a true no-op — the anchor
    //    is reserved for the first non-empty push.
    // 3. Empty subsequent push at `delta == 0`: already a no-op
    //    (delta_samples = 0, packet.len() = 0, no first-push
    //    commit). Preserved for callers using empty packets as
    //    explicit heartbeats.
    if packet.is_empty() && (delta_samples > 0 || would_be_first_push) {
      return Ok(());
    }

    // All checks passed. Commit the anchor on first push, then
    // zero-fill any tolerated gap and append the packet.
    if would_be_first_push {
      self.output_tb = Some(effective_tb);
      self.base_pts_out_anchor = effective_anchor;
    }
    if delta_samples > 0 {
      self
        .samples
        .extend(core::iter::repeat_n(0.0_f32, delta_samples as usize));
      self.absolute_sample_offset += delta_samples;
    }
    self.samples.extend_from_slice(packet);
    self.absolute_sample_offset += packet.len() as u64;

    Ok(())
  }

  /// Total samples ever appended (after handle_restart, this restarts
  /// from 0). Crate-private; the cut state machine consumes this.
  pub(crate) fn absolute_sample_offset(&self) -> u64 {
    self.absolute_sample_offset
  }

  /// Length of the live buffer.
  pub(crate) fn buffered_samples(&self) -> usize {
    self.samples.len()
  }

  /// Output-timebase PTS the buffer expects for the next contiguous
  /// push. None before the first push.
  pub(crate) fn next_expected_starts_at(&self) -> Option<Timestamp> {
    let tb = self.output_tb?;
    let pts = self.base_pts_out_anchor
      + Timebase::rescale_pts(self.absolute_sample_offset as i64, ANALYSIS_TIMEBASE, tb);
    Some(Timestamp::new(pts, tb))
  }

  /// Extract a chunk's samples as a fresh `Arc<[f32]>` without
  /// mutating the buffer. The range is in stream-relative 16 kHz
  /// indices (i.e., absolute, not relative to the live buffer).
  pub(crate) fn extract(&self, range: crate::core::cut::SampleRange) -> alloc::sync::Arc<[f32]> {
    let lo = (range.start - self.buffer_drop_offset) as usize;
    let hi = (range.end - self.buffer_drop_offset) as usize;
    let slice = &self.samples[lo..hi];
    slice.into()
  }

  /// Convert a 16 kHz `SampleRange` (stream-relative) to a
  /// `mediatime::TimeRange` in the output timebase. Always
  /// rescales from the immutable anchor; the round-trip error is
  /// at most ±1 PTS regardless of trim history.
  pub(crate) fn samples_to_output_range(
    &self,
    range: crate::core::cut::SampleRange,
  ) -> mediatime::TimeRange {
    let tb = self
      .output_tb
      .expect("samples_to_output_range called before any push");
    let start_out =
      self.base_pts_out_anchor + Timebase::rescale_pts(range.start as i64, ANALYSIS_TIMEBASE, tb);
    let end_out =
      self.base_pts_out_anchor + Timebase::rescale_pts(range.end as i64, ANALYSIS_TIMEBASE, tb);
    mediatime::TimeRange::new(start_out, end_out, tb)
  }

  /// Build a `samples_to_output_range` closure from an explicit
  /// `(timebase, base_pts_out_anchor)` snapshot rather than from
  /// the buffer's current state. The dispatch layer captures the
  /// pair onto each chunk record at extract time (see
  /// `dispatch::ChunkRecord::output_tb`) and feeds it back here
  /// at alignment-dispatch time, so word ranges stay anchored in
  /// the chunk's own PTS epoch even after a `handle_restart` shifts
  /// the live buffer onto a new one.
  ///
  /// Conversion math is identical to
  /// [`samples_to_output_range`](Self::samples_to_output_range)
  /// (drift-free): `out_pts = base_pts_out_anchor + rescale(sample, ANALYSIS_TIMEBASE, tb)`.
  #[cfg(feature = "alignment")]
  pub(crate) fn samples_to_output_range_fn_at(
    tb: Timebase,
    base_pts_out_anchor: i64,
  ) -> alloc::sync::Arc<dyn Fn(u64, u64) -> mediatime::TimeRange + Send + Sync> {
    alloc::sync::Arc::new(
      move |start_sample: u64, end_sample: u64| -> mediatime::TimeRange {
        let s_pts =
          base_pts_out_anchor + Timebase::rescale_pts(start_sample as i64, ANALYSIS_TIMEBASE, tb);
        let e_pts =
          base_pts_out_anchor + Timebase::rescale_pts(end_sample as i64, ANALYSIS_TIMEBASE, tb);
        mediatime::TimeRange::new(s_pts, e_pts, tb)
      },
    )
  }

  /// Drop samples up to (but not including) `low_water_samples`.
  /// `base_pts_out_anchor` is *not* touched; `buffer_drop_offset`
  /// advances. Used by the dispatch state machine after chunks
  /// past `low_water_samples` are no longer reachable from
  /// `cut_pending`.
  pub(crate) fn trim_to(&mut self, low_water_samples: u64) {
    if low_water_samples <= self.buffer_drop_offset {
      return;
    }
    let drop_count = (low_water_samples - self.buffer_drop_offset) as usize;
    let drop_count = drop_count.min(self.samples.len());
    self.samples.drain(..drop_count);
    self.buffer_drop_offset += drop_count as u64;
  }

  /// Reset the buffer's anchor for `handle_restart`. Clears the live
  /// `Vec<f32>`, sets `base_pts_out_anchor` to `starts_at.pts()`,
  /// and zeroes both offsets so the next push starts a fresh
  /// contiguous segment with `delta_pts_out == 0` exactly.
  /// Pre-restart in-flight chunks are unaffected — they hold their
  /// audio in their own `Arc<[f32]>`s.
  pub(crate) fn handle_restart(&mut self, starts_at: Timestamp) {
    self.output_tb = Some(starts_at.timebase());
    self.base_pts_out_anchor = starts_at.pts();
    self.absolute_sample_offset = 0;
    self.buffer_drop_offset = 0;
    self.samples.clear();
  }

  /// Buffer drop offset (in 16 kHz samples). Used by the dispatch
  /// state machine when computing trim's low-water against
  /// `cut_pending` ranges.
  pub(crate) fn buffer_drop_offset(&self) -> u64 {
    self.buffer_drop_offset
  }
}

/// Construct a default `SampleBuffer` (60 s × 16 kHz cap, 200 ms
/// gap tolerance). Used by tests and as the default in
/// `TranscriberOptions`.
pub(crate) fn default_buffer() -> SampleBuffer {
  SampleBuffer::new(60 * 16_000, 200 * 16) // 200 ms × 16 samples/ms = 3200
}

#[cfg(test)]
mod tests {
  use super::*;
  use core::num::NonZeroU32;

  fn tb_48k() -> Timebase {
    Timebase::new(1, NonZeroU32::new(48_000).unwrap())
  }

  fn ts_at_48k(pts: i64) -> Timestamp {
    Timestamp::new(pts, tb_48k())
  }

  #[test]
  fn first_push_records_anchor_and_timebase() {
    let mut b = SampleBuffer::new(1_000_000, 3200);
    b.append(ts_at_48k(48_000), &[0.0; 100], 0).unwrap();
    assert_eq!(b.output_timebase(), Some(tb_48k()));
    assert_eq!(b.absolute_sample_offset(), 100);
    // Next expected: 48_000 + rescale(100, 1/16k, 1/48k) = 48_000 + 300 = 48_300
    assert_eq!(b.next_expected_starts_at().unwrap().pts(), 48_300);
  }

  #[test]
  fn contiguous_push_succeeds() {
    let mut b = SampleBuffer::new(1_000_000, 3200);
    b.append(ts_at_48k(0), &[0.0; 1000], 0).unwrap();
    let next = b.next_expected_starts_at().unwrap();
    b.append(next, &[0.0; 500], 0).unwrap();
    assert_eq!(b.absolute_sample_offset(), 1500);
  }

  #[test]
  fn pts_regression_returns_error() {
    let mut b = SampleBuffer::new(1_000_000, 3200);
    b.append(ts_at_48k(48_000), &[0.0; 100], 0).unwrap();
    let result = b.append(ts_at_48k(47_000), &[0.0; 100], 0);
    assert!(matches!(
      result,
      Err(TranscriberError::PtsRegression {
        kind: crate::types::PushKind::Samples,
        ..
      })
    ));
  }

  #[test]
  fn forward_gap_within_tolerance_zero_fills() {
    let mut b = SampleBuffer::new(1_000_000, 3200);
    b.append(ts_at_48k(0), &[1.0; 100], 0).unwrap();
    // Skip 300 PTS at 1/48000 = 100 16 kHz samples (within tolerance).
    b.append(ts_at_48k(600), &[2.0; 100], 0).unwrap();
    // First 100 samples = 1.0; next 100 = zero-fill; next 100 = 2.0.
    assert_eq!(b.absolute_sample_offset(), 300);
  }

  #[test]
  fn forward_gap_above_tolerance_errors() {
    // gap_tolerance_samples is in 16 kHz.
    let mut b = SampleBuffer::new(1_000_000, 100);
    b.append(ts_at_48k(0), &[0.0; 100], 0).unwrap();
    // 1300 PTS at 1/48000 = 1300 * 16 / 48 ≈ 433 samples > 100.
    let r = b.append(ts_at_48k(1300), &[0.0; 100], 0);
    assert!(matches!(
      r,
      Err(TranscriberError::GapExceedsTolerance { .. })
    ));
  }

  #[test]
  fn backpressure_at_cap() {
    let mut b = SampleBuffer::new(150, 3200);
    let r = b.append(ts_at_48k(0), &[0.0; 200], 0);
    assert!(
      matches!(r, Err(TranscriberError::Backpressure { buffered, cap }) if buffered == 200 && cap == 150)
    );
    // Backpressure must NOT mutate state. The buffer should be
    // empty and absolute_sample_offset should still be 0 — the
    // caller can retry the same packet later (e.g., after the
    // runner drains chunks and the cap has been raised, or with
    // a smaller packet).
    assert_eq!(
      b.buffered_samples(),
      0,
      "Backpressure must not commit samples"
    );
    assert_eq!(
      b.absolute_sample_offset(),
      0,
      "Backpressure must not advance offset"
    );
  }

  /// The rejected packet from a Backpressure can be retried
  /// after the buffer drains. Without the pre-mutation check,
  /// the state advanced, retrying the same packet would have
  /// tripped PtsRegression.
  #[test]
  fn backpressure_allows_retry_with_smaller_packet() {
    let mut b = SampleBuffer::new(150, 3200);
    // First push at cap is rejected with no state advance.
    let r = b.append(ts_at_48k(0), &[0.0; 200], 0);
    assert!(matches!(r, Err(TranscriberError::Backpressure { .. })));
    assert_eq!(b.buffered_samples(), 0);
    assert_eq!(b.absolute_sample_offset(), 0);
    // Same anchor PTS still works on a smaller packet.
    b.append(ts_at_48k(0), &[1.0; 100], 0).unwrap();
    assert_eq!(b.buffered_samples(), 100);
    assert_eq!(b.absolute_sample_offset(), 100);
  }

  /// A Backpressure on the FIRST push must be fully atomic —
  /// the rejected packet must not commit the stream's timebase
  /// or anchor. Without the fix, a retry (after the cap is
  /// raised, or with a smaller packet, or with a corrected
  /// timebase) would race against an already-fixed anchor and
  /// trip InconsistentTimebase / PtsRegression /
  /// GapExceedsTolerance even though the rejected input was
  /// supposed to be uncommitted.
  #[test]
  fn first_push_backpressure_does_not_commit_timebase() {
    let mut b = SampleBuffer::new(150, 3200);
    // First push fails with Backpressure (200 > 150).
    let r = b.append(ts_at_48k(48_000), &[0.0; 200], 0);
    assert!(matches!(r, Err(TranscriberError::Backpressure { .. })));
    // Timebase and anchor must remain uncommitted.
    assert_eq!(
      b.output_timebase(),
      None,
      "Backpressure on first push must not commit output timebase"
    );
    assert!(
      b.next_expected_starts_at().is_none(),
      "Backpressure on first push must not commit anchor PTS"
    );
  }

  /// After a first-push Backpressure, the buffer must accept a
  /// *different* timebase as its actual first push. (Pre-fix
  /// behavior committed the rejected timebase, so this would
  /// have tripped InconsistentTimebase.)
  #[test]
  fn first_push_backpressure_allows_different_timebase_on_retry() {
    let mut b = SampleBuffer::new(150, 3200);
    let _ = b.append(ts_at_48k(0), &[0.0; 200], 0); // rejected
    let other_tb = Timebase::new(1, NonZeroU32::new(96_000).unwrap());
    // Different timebase + smaller packet must succeed.
    b.append(Timestamp::new(0, other_tb), &[0.0; 100], 0)
      .unwrap();
    assert_eq!(b.output_timebase(), Some(other_tb));
    assert_eq!(b.absolute_sample_offset(), 100);
  }

  #[test]
  fn inconsistent_timebase_errors() {
    let mut b = SampleBuffer::new(1_000_000, 3200);
    b.append(ts_at_48k(0), &[0.0; 100], 0).unwrap();
    let other_tb = Timebase::new(1, NonZeroU32::new(1000).unwrap());
    let r = b.append(Timestamp::new(0, other_tb), &[0.0; 100], 0);
    assert!(matches!(
      r,
      Err(TranscriberError::InconsistentTimebase { .. })
    ));
  }

  #[test]
  fn extract_returns_correct_slice() {
    use crate::core::cut::SampleRange;
    let mut b = SampleBuffer::new(1_000_000, 3200);
    let mut samples = Vec::with_capacity(1000);
    for i in 0..1000 {
      samples.push(i as f32);
    }
    b.append(ts_at_48k(0), &samples, 0).unwrap();
    let arc = b.extract(SampleRange::new(100, 200));
    assert_eq!(arc.len(), 100);
    assert_eq!(arc[0], 100.0);
    assert_eq!(arc[99], 199.0);
  }

  #[test]
  fn samples_to_output_range_drift_free_across_trims() {
    use crate::core::cut::SampleRange;
    let mut b = SampleBuffer::new(1_000_000, 3200);
    b.append(ts_at_48k(0), &[0.0; 16_000], 0).unwrap();
    let range_before = b.samples_to_output_range(SampleRange::new(8_000, 12_000));
    b.trim_to(4_000);
    let range_after = b.samples_to_output_range(SampleRange::new(8_000, 12_000));
    assert_eq!(
      range_before, range_after,
      "samples_to_output_range must not drift across trims"
    );
  }

  #[test]
  fn trim_to_below_drop_offset_is_noop() {
    let mut b = SampleBuffer::new(1_000_000, 3200);
    b.append(ts_at_48k(0), &[0.0; 1000], 0).unwrap();
    b.trim_to(500);
    assert_eq!(b.buffer_drop_offset(), 500);
    b.trim_to(300); // below current drop_offset
    assert_eq!(b.buffer_drop_offset(), 500);
  }

  #[test]
  fn handle_restart_resets_offsets_and_anchor() {
    let mut b = SampleBuffer::new(1_000_000, 3200);
    b.append(ts_at_48k(0), &[1.0; 1000], 0).unwrap();
    b.handle_restart(ts_at_48k(50_000_000));
    assert_eq!(b.absolute_sample_offset(), 0);
    assert_eq!(b.buffer_drop_offset(), 0);
    assert_eq!(b.buffered_samples(), 0);
    // Next push at 50_000_000 must succeed without PtsRegression.
    b.append(ts_at_48k(50_000_000), &[2.0; 1000], 0).unwrap();
  }

  // --- Codex round-37: empty packet must not advance the stream ---

  /// An empty packet at a forward PTS within gap-tolerance MUST
  /// NOT zero-fill or advance `absolute_sample_offset`. Pre-fix,
  /// such a "heartbeat" call would commit phantom audio and
  /// reject the next real packet at the originally-expected PTS
  /// as `PtsRegression`. Post-fix, the next real packet is
  /// accepted.
  #[test]
  fn empty_packet_at_forward_delta_does_not_advance_stream() {
    let mut b = SampleBuffer::new(1_000_000, 16_000);
    // First push: 1000 real samples at PTS 0 — establishes anchor.
    b.append(ts_at_48k(0), &[1.0; 1000], 0).unwrap();
    let offset_before = b.absolute_sample_offset();
    let buffered_before = b.buffered_samples();
    let next_expected = b.next_expected_starts_at().unwrap();

    // Empty packet "heartbeat" at a forward PTS — within
    // gap-tolerance but no audio carried.
    let heartbeat_pts = next_expected.pts() + 100; // ~33 ms forward in 48k
    let r = b.append(ts_at_48k(heartbeat_pts), &[], 0);
    assert!(r.is_ok(), "empty heartbeat must succeed; got {r:?}");

    // Critical: the heartbeat MUST NOT have advanced state.
    assert_eq!(
      b.absolute_sample_offset(),
      offset_before,
      "empty heartbeat must not advance absolute_sample_offset"
    );
    assert_eq!(
      b.buffered_samples(),
      buffered_before,
      "empty heartbeat must not zero-fill the live buffer"
    );

    // The next real packet at the originally-expected PTS must
    // succeed (pre-fix this was rejected as PtsRegression).
    let r2 = b.append(next_expected, &[2.0; 500], 0);
    assert!(
      r2.is_ok(),
      "next real packet at original expected PTS must succeed; got {r2:?}"
    );
    assert_eq!(b.absolute_sample_offset(), 1500);
  }

  /// An empty FIRST packet establishes the stream anchor at its
  /// own `starts_at` (delta == 0 by definition for the first
  /// push, since there is no expected next yet). A subsequent
  /// empty-at-forward-delta call is then a no-op, and a real
  /// packet at the originally-expected PTS still succeeds.
  /// This documents that "heartbeat then real audio" sequences
  /// don't trip PtsRegression after the round-37 fix.
  /// Codex round-38 finding 4: the FIRST empty packet must NOT
  /// commit the anchor. A heartbeat-then-real-audio sequence
  /// (empty packet at heartbeat PTS, then real audio at the
  /// actual stream-zero PTS) now succeeds — the anchor is
  /// reserved for the first non-empty push. Pre-fix, the empty
  /// heartbeat claimed the anchor at its own PTS and the real
  /// audio at PTS 0 failed with `PtsRegression` /
  /// `InconsistentTimebase`.
  #[test]
  fn empty_first_packet_does_not_commit_anchor() {
    let mut b = SampleBuffer::new(1_000_000, 16_000);
    let r = b.append(ts_at_48k(50_000), &[], 0);
    assert!(r.is_ok(), "empty first push must succeed; got {r:?}");
    assert!(
      b.output_timebase().is_none(),
      "empty first push must NOT commit timebase"
    );
    assert!(
      b.next_expected_starts_at().is_none(),
      "empty first push must leave the stream un-anchored"
    );
    assert_eq!(b.absolute_sample_offset(), 0);

    // First non-empty push at any PTS becomes the actual first
    // push and anchors the stream there.
    let r = b.append(ts_at_48k(0), &[1.0; 1000], 0);
    assert!(
      r.is_ok(),
      "real first audio at PTS 0 must succeed after empty heartbeat at PTS 50_000; got {r:?}"
    );
    assert_eq!(b.absolute_sample_offset(), 1000);
    let next_expected = b.next_expected_starts_at().unwrap();
    assert!(b.output_timebase().is_some());

    // Subsequent empty heartbeat at a forward PTS does not
    // advance state (round-37 fix).
    let offset_before = b.absolute_sample_offset();
    let r = b.append(ts_at_48k(next_expected.pts() + 100), &[], 0);
    assert!(r.is_ok());
    assert_eq!(b.absolute_sample_offset(), offset_before);
  }

  /// Empty packet at exactly the expected next PTS (delta == 0)
  /// remains a true no-op as before. Ensures the fix doesn't
  /// regress the well-defined heartbeat-on-time case.
  #[test]
  fn empty_packet_at_zero_delta_is_noop() {
    let mut b = SampleBuffer::new(1_000_000, 16_000);
    b.append(ts_at_48k(0), &[1.0; 1000], 0).unwrap();
    let next_expected = b.next_expected_starts_at().unwrap();
    let offset_before = b.absolute_sample_offset();
    let r = b.append(next_expected, &[], 0);
    assert!(r.is_ok());
    assert_eq!(b.absolute_sample_offset(), offset_before);
  }

  /// Codex round-37 round-26 [high]: gigantic forward gaps must
  /// surface as `Backpressure`, not panic on `usize` overflow
  /// in the capacity-pre-check arithmetic. The pre-fix
  /// `samples.len() + delta_samples as usize + packet.len() +
  /// extra_queued` expression could wrap on 64-bit when
  /// `gap_tolerance_samples` was set near `u64::MAX` and the
  /// caller advanced PTS by that much. Post-fix the
  /// `checked_add` chain rejects overflow as
  /// `Backpressure { buffered: usize::MAX, .. }` so the
  /// caller's retry/backoff loop sees a typed error instead
  /// of a process abort.
  #[test]
  fn enormous_forward_gap_returns_backpressure_not_panic() {
    // Tolerance saturated at u64::MAX (the cap-checked public
    // setter rejects this in real use, but `SampleBuffer` is
    // crate-private and an internal caller might still pass a
    // pathological value through if validation regresses).
    let mut b = SampleBuffer::new(/* cap: */ 1_000, /* tolerance: */ u64::MAX);
    b.append(ts_at_48k(0), &[1.0; 100], 0).unwrap();
    // Forward jump by ~6 hours at 48 kHz — way past `cap`.
    let huge_pts = 48_000_i64.saturating_mul(6 * 3600);
    let r = b.append(ts_at_48k(huge_pts), &[1.0; 1], 0);
    match r {
      Err(TranscriberError::Backpressure { .. }) => {}
      other => panic!("expected Backpressure on overflow; got {other:?}"),
    }
  }
}
