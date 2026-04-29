//! `SampleBuffer` — bounded f32 buffer with output-timebase PTS
//! arithmetic anchored at the first push.
//!
//! Round-3 / round-4 invariants: `base_pts_out_anchor` is immutable
//! after the first push (so trim doesn't accumulate drift on
//! non-integer-ratio output timebases); the regression check runs
//! in output-PTS space (so contiguous caller pushes on NTSC-like
//! timebases don't produce spurious `PtsRegression`); trim's
//! low-water is computed from `cut_pending` only, not `in_flight`,
//! because in-flight chunks already hold their audio in their own
//! `Arc<[f32]>` (decoupled from the live buffer).
//!
//! See spec §5.4.

use alloc::vec::Vec;

use mediatime::{Timebase, Timestamp};

use crate::time::ANALYSIS_TIMEBASE;
use crate::types::TranscriberError;

/// Live audio buffer.
pub(crate) struct SampleBuffer {
    /// Output timebase recorded from the first push.
    output_tb: Option<Timebase>,
    /// PTS (in `output_tb`) of stream-zero. **Immutable** after the
    /// first push.
    base_pts_out_anchor: i64,
    /// Total samples ever appended (monotonic; reset only by
    /// `restart_at`).
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

    /// Append a packet of samples whose first sample's PTS is
    /// `starts_at` in the output timebase. Returns `Backpressure`
    /// when the buffer would exceed its cap; `PtsRegression` /
    /// `GapExceedsTolerance` / `InconsistentTimebase` per their
    /// usual contracts.
    pub(crate) fn append(
        &mut self,
        starts_at: Timestamp,
        packet: &[f32],
    ) -> Result<(), TranscriberError> {
        if let Some(expected_tb) = self.output_tb {
            if starts_at.timebase() != expected_tb {
                return Err(TranscriberError::InconsistentTimebase {
                    expected: expected_tb,
                    got: starts_at.timebase(),
                });
            }
        } else {
            self.output_tb = Some(starts_at.timebase());
            self.base_pts_out_anchor = starts_at.pts();
        }
        let output_tb = self.output_tb.expect("just set");

        // Compute expected next-PTS in output-tb space, then the
        // delta against caller's starts_at. This is the round-4
        // M-δ fix: the regression check stays in output-PTS space
        // so contiguous pushes on non-integer-ratio output
        // timebases don't trip spurious regressions through round-trip
        // truncation.
        let expected_pts_out = self.base_pts_out_anchor
            + Timebase::rescale_pts(
                self.absolute_sample_offset as i64,
                ANALYSIS_TIMEBASE,
                output_tb,
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
            let g = Timebase::rescale_pts(delta_pts_out, output_tb, ANALYSIS_TIMEBASE);
            if (g as u64) > self.gap_tolerance_samples {
                return Err(TranscriberError::GapExceedsTolerance {
                    gap_samples: g as u64,
                    tolerance_samples: self.gap_tolerance_samples,
                });
            }
            g as u64
        };

        // Zero-fill any tolerated gap, then append the packet.
        if delta_samples > 0 {
            self.samples.extend(core::iter::repeat(0.0_f32).take(delta_samples as usize));
            self.absolute_sample_offset += delta_samples;
        }
        self.samples.extend_from_slice(packet);
        self.absolute_sample_offset += packet.len() as u64;

        if self.samples.len() > self.cap {
            return Err(TranscriberError::Backpressure {
                buffered: self.samples.len(),
                cap: self.cap,
            });
        }
        Ok(())
    }

    /// Total samples ever appended (after restart_at, this restarts
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
            + Timebase::rescale_pts(
                self.absolute_sample_offset as i64,
                ANALYSIS_TIMEBASE,
                tb,
            );
        Some(Timestamp::new(pts, tb))
    }
}

/// Construct a default `SampleBuffer` with the spec's defaults
/// (60 s × 16 kHz cap, 200 ms gap tolerance). Used by tests and as
/// the default in `TranscriberConfig`.
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
        b.append(ts_at_48k(48_000), &[0.0; 100]).unwrap();
        assert_eq!(b.output_timebase(), Some(tb_48k()));
        assert_eq!(b.absolute_sample_offset(), 100);
        // Next expected: 48_000 + rescale(100, 1/16k, 1/48k) = 48_000 + 300 = 48_300
        assert_eq!(b.next_expected_starts_at().unwrap().pts(), 48_300);
    }

    #[test]
    fn contiguous_push_succeeds() {
        let mut b = SampleBuffer::new(1_000_000, 3200);
        b.append(ts_at_48k(0), &[0.0; 1000]).unwrap();
        let next = b.next_expected_starts_at().unwrap();
        b.append(next, &[0.0; 500]).unwrap();
        assert_eq!(b.absolute_sample_offset(), 1500);
    }

    #[test]
    fn pts_regression_returns_error() {
        let mut b = SampleBuffer::new(1_000_000, 3200);
        b.append(ts_at_48k(48_000), &[0.0; 100]).unwrap();
        let result = b.append(ts_at_48k(47_000), &[0.0; 100]);
        assert!(matches!(
            result,
            Err(TranscriberError::PtsRegression { kind: crate::types::PushKind::Samples, .. })
        ));
    }

    #[test]
    fn forward_gap_within_tolerance_zero_fills() {
        let mut b = SampleBuffer::new(1_000_000, 3200);
        b.append(ts_at_48k(0), &[1.0; 100]).unwrap();
        // Skip 300 PTS at 1/48000 = 100 16 kHz samples (within tolerance).
        b.append(ts_at_48k(600), &[2.0; 100]).unwrap();
        // First 100 samples = 1.0; next 100 = zero-fill; next 100 = 2.0.
        assert_eq!(b.absolute_sample_offset(), 300);
    }

    #[test]
    fn forward_gap_above_tolerance_errors() {
        // gap_tolerance_samples is in 16 kHz.
        let mut b = SampleBuffer::new(1_000_000, 100);
        b.append(ts_at_48k(0), &[0.0; 100]).unwrap();
        // 1300 PTS at 1/48000 = 1300 * 16 / 48 ≈ 433 samples > 100.
        let r = b.append(ts_at_48k(1300), &[0.0; 100]);
        assert!(matches!(r, Err(TranscriberError::GapExceedsTolerance { .. })));
    }

    #[test]
    fn backpressure_at_cap() {
        let mut b = SampleBuffer::new(150, 3200);
        let r = b.append(ts_at_48k(0), &[0.0; 200]);
        assert!(matches!(r, Err(TranscriberError::Backpressure { buffered, cap }) if buffered == 200 && cap == 150));
    }

    #[test]
    fn inconsistent_timebase_errors() {
        let mut b = SampleBuffer::new(1_000_000, 3200);
        b.append(ts_at_48k(0), &[0.0; 100]).unwrap();
        let other_tb = Timebase::new(1, NonZeroU32::new(1000).unwrap());
        let r = b.append(Timestamp::new(0, other_tb), &[0.0; 100]);
        assert!(matches!(r, Err(TranscriberError::InconsistentTimebase { .. })));
    }
}
