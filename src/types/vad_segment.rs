//! `VadSegment` — silero-shaped speech segment in 16 kHz analysis
//! indices.

/// Speech segment as 16 kHz analysis-frame indices. Asry accepts
/// silero-shaped input via this type and converts to output-timebase
/// `TimeRange`s internally; the caller never does PTS arithmetic for
/// VAD inputs.
///
/// Carries the invariant `end_sample > start_sample`. The
/// constructor [`VadSegment::new`] enforces it; the
/// `Deserialize` impl below also enforces it (:
/// `derive(Deserialize)` would have skipped construction
/// validation, letting malformed input wrap `sample_count()`
/// to a huge value and trip the `Cut::push_segment` hard-split
/// assertion in release).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct VadSegment {
  start_sample: u64,
  end_sample: u64,
}

impl VadSegment {
  /// Constructs a `VadSegment`.
  ///
  /// **Panics** if `end_sample <= start_sample`. The strict
  /// inequality matters: zero-duration VAD segments would emit
  /// zero-length `MergedChunk`s downstream which break alignment
  /// and confuse downstream consumers. silero never produces
  /// zero-duration segments; the panic surfaces programmer error
  /// at the boundary.
  pub const fn new(start_sample: u64, end_sample: u64) -> Self {
    if end_sample <= start_sample {
      panic!("VadSegment::new requires end_sample > start_sample");
    }
    Self {
      start_sample,
      end_sample,
    }
  }

  /// 16 kHz analysis-frame index of the segment's start, inclusive.
  pub const fn start_sample(&self) -> u64 {
    self.start_sample
  }

  /// 16 kHz analysis-frame index of the segment's end, exclusive.
  pub const fn end_sample(&self) -> u64 {
    self.end_sample
  }

  /// Number of samples in the segment (`end - start`).
  pub const fn sample_count(&self) -> u64 {
    self.end_sample - self.start_sample
  }
}

#[cfg(feature = "serde")]
const _: () = {
  impl<'de> serde::Deserialize<'de> for VadSegment {
    /// Custom impl: a derived `Deserialize` would skip
    /// [`VadSegment::new`]'s `end_sample > start_sample` invariant
    /// check, letting malformed input wrap `sample_count()` to a
    /// huge value and panic deep inside the cut state machine.
    /// Flagged this; here we re-validate at the
    /// serde boundary and surface the violation as a typed
    /// deserialization error instead.
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
      D: serde::Deserializer<'de>,
    {
      use serde::de::Error as _;

      // Mirror the derived shape: a struct with two `u64` fields.
      // We use a private inner type with `derive(Deserialize)` to
      // get serde's standard field-naming and missing-field error
      // text for free, then validate after reading.
      #[derive(serde::Deserialize)]
      struct Raw {
        start_sample: u64,
        end_sample: u64,
      }
      let r = Raw::deserialize(deserializer)?;
      if r.end_sample <= r.start_sample {
        return Err(D::Error::custom(format!(
          "VadSegment requires end_sample > start_sample (got start_sample={}, end_sample={})",
          r.start_sample, r.end_sample
        )));
      }
      Ok(Self {
        start_sample: r.start_sample,
        end_sample: r.end_sample,
      })
    }
  }
};

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn round_trip() {
    let s = VadSegment::new(100, 250);
    assert_eq!(s.start_sample(), 100);
    assert_eq!(s.end_sample(), 250);
    assert_eq!(s.sample_count(), 150);
  }

  #[test]
  #[should_panic(expected = "end_sample > start_sample")]
  fn zero_duration_panics() {
    VadSegment::new(100, 100);
  }

  #[test]
  #[should_panic(expected = "end_sample > start_sample")]
  fn negative_duration_panics() {
    VadSegment::new(200, 100);
  }

  /// : serde's derived `Deserialize` skips the
  /// constructor's invariant. A reversed range would wrap
  /// `sample_count()` to ~u64::MAX and trip the cut state
  /// machine's assertions deep in the pipeline. The custom
  /// impl rejects at the serde boundary instead.
  #[cfg(feature = "serde")]
  #[test]
  fn deserialize_rejects_reversed_range() {
    let json = r#"{"start_sample":200,"end_sample":100}"#;
    let res: Result<VadSegment, _> = serde_json::from_str(json);
    assert!(
      res.is_err(),
      "reversed range must fail deserialization; got {res:?}"
    );
    let err = res.err().unwrap().to_string();
    assert!(
      err.contains("end_sample > start_sample"),
      "expected invariant in error message, got {err:?}"
    );
  }

  #[cfg(feature = "serde")]
  #[test]
  fn deserialize_rejects_zero_duration() {
    let json = r#"{"start_sample":100,"end_sample":100}"#;
    let res: Result<VadSegment, _> = serde_json::from_str(json);
    assert!(res.is_err(), "zero-duration must fail; got {res:?}");
  }

  #[cfg(feature = "serde")]
  #[test]
  fn deserialize_accepts_valid_range() {
    let json = r#"{"start_sample":100,"end_sample":250}"#;
    let s: VadSegment = serde_json::from_str(json).expect("valid range");
    assert_eq!(s.start_sample(), 100);
    assert_eq!(s.end_sample(), 250);
  }

  #[cfg(feature = "serde")]
  #[test]
  fn serde_round_trip() {
    let original = VadSegment::new(100, 250);
    let json = serde_json::to_string(&original).expect("serialize");
    let back: VadSegment = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(original, back);
  }
}
