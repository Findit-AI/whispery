//! Silence-mask construction stage of the alignment algorithm.

#[cfg(test)]
use Vec;

#[cfg(test)]
use mediatime::TimeRange;

/// Zero-mask the chunk audio outside the union of sub-VAD-segments.
///
/// **Why mask, not skip.** wav2vec2 distributes near-all probability
/// to the blank token in long silence regions; CTC Viterbi is
/// robust under uniform silence but smears word boundaries when
/// non-speech regions carry phoneme-like noise. Zeroing the
/// non-speech samples produces uniform silence the CTC path can
/// step over without spurious emission.
///
/// **Coordinate space.** The chunk's `samples` are 16 kHz f32 mono;
/// indices are chunk-local (0..samples.len()). The
/// `sub_segments` come from `MergedChunk.sub_segments`, which are
/// `TimeRange`s in the *output timebase* — they could be 48 kHz
/// or 90 kHz or anything else the caller chose.
///
/// `chunk_first_sample_in_stream` is the chunk's first 16 kHz
/// sample index in stream coordinates; `output_range_to_chunk_local`
/// converts an output-timebase `TimeRange` into a chunk-local
/// `(start_sample, end_sample)` pair (clamped to the chunk's
/// sample bounds).
///
/// Returns a fresh `Vec<f32>` of the same length as `samples`, with
/// every sample outside the union of converted sub-segment ranges
/// replaced with `0.0_f32`. The original `samples` slice is not
/// mutated (the wav2vec2 input is short-lived; allocating a fresh
/// vec keeps the caller's audio cache pure).
// Test-only helper: production callers use `build_speech_mask`
// (the boolean-mask form) which feeds straight into the silence-
// aware normaliser. `build_masked_samples` predates that
// migration; it stays in tree because the unit tests below
// pin the masking semantics, but it's no longer reachable
// from the alignment pipeline.
#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn build_masked_samples<F>(
  samples: &[f32],
  sub_segments: &[TimeRange],
  output_range_to_chunk_local: F,
) -> Vec<f32>
where
  F: Fn(TimeRange) -> (u64, u64),
{
  let n = samples.len();
  let mut out = vec![0.0_f32; n];

  for &seg in sub_segments {
    let (start, end) = output_range_to_chunk_local(seg);
    let start = (start as usize).min(n);
    let end = (end as usize).min(n);
    if end <= start {
      continue;
    }
    out[start..end].copy_from_slice(&samples[start..end]);
  }
  out
}

#[cfg(test)]
mod tests {
  use super::*;
  use core::num::NonZeroU32;
  use mediatime::Timebase;

  fn tb_16k() -> Timebase {
    Timebase::new(1, NonZeroU32::new(16_000).unwrap())
  }

  fn ms_to_chunk_local(seg: TimeRange) -> (u64, u64) {
    // Test convenience: when the timebase is 1/16000, PTS units
    // are already 16 kHz samples. Identity conversion.
    (seg.start_pts() as u64, seg.end_pts() as u64)
  }

  #[test]
  fn empty_segments_zero_everything() {
    let samples = vec![1.0_f32; 100];
    let masked = build_masked_samples(&samples, &[], ms_to_chunk_local);
    assert!(masked.iter().all(|&x| x == 0.0));
  }

  #[test]
  fn single_segment_preserves_only_overlap() {
    let samples: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let seg = TimeRange::new(20, 40, tb_16k());
    let masked = build_masked_samples(&samples, &[seg], ms_to_chunk_local);
    for i in 0..20 {
      assert_eq!(masked[i], 0.0, "pre-segment must be zero at {i}");
    }
    for i in 20..40 {
      assert_eq!(masked[i], i as f32, "in-segment must be preserved at {i}");
    }
    for i in 40..100 {
      assert_eq!(masked[i], 0.0, "post-segment must be zero at {i}");
    }
  }

  #[test]
  fn multiple_segments_union() {
    let samples: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let segs = [
      TimeRange::new(10, 20, tb_16k()),
      TimeRange::new(50, 70, tb_16k()),
    ];
    let masked = build_masked_samples(&samples, &segs, ms_to_chunk_local);
    assert_eq!(masked[5], 0.0);
    assert_eq!(masked[15], 15.0);
    assert_eq!(masked[25], 0.0);
    assert_eq!(masked[55], 55.0);
    assert_eq!(masked[75], 0.0);
  }

  #[test]
  fn segment_past_end_clamps() {
    let samples = vec![1.0_f32; 50];
    let seg = TimeRange::new(40, 200, tb_16k());
    let masked = build_masked_samples(&samples, &[seg], ms_to_chunk_local);
    assert!(masked.iter().take(40).all(|&x| x == 0.0));
    assert!(masked[40..].iter().all(|&x| x == 1.0));
  }

  #[test]
  fn overlapping_segments_idempotent() {
    let samples = vec![1.0_f32; 100];
    let segs = [
      TimeRange::new(20, 60, tb_16k()),
      TimeRange::new(40, 80, tb_16k()),
    ];
    let masked = build_masked_samples(&samples, &segs, ms_to_chunk_local);
    // The union 20..80 should be preserved.
    assert!(masked[..20].iter().all(|&x| x == 0.0));
    assert!(masked[20..80].iter().all(|&x| x == 1.0));
    assert!(masked[80..].iter().all(|&x| x == 0.0));
  }

  #[test]
  fn does_not_mutate_input() {
    let samples = vec![1.0_f32; 50];
    let snapshot = samples.clone();
    let _ = build_masked_samples(&samples, &[], ms_to_chunk_local);
    assert_eq!(samples, snapshot);
  }
}
