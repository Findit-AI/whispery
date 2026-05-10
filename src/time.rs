//! Time constants for whispery.
//!
//! whispery operates on **two timebases**:
//!
//! - **Internal (analysis) timebase = `1/16_000`.** All cut decisions,
//!   `SampleBuffer` indexing, and CTC alignment happen in 16 kHz
//!   sample-index space.
//! - **External (output) timebase = caller-chosen.** Every public
//!   [`mediatime::TimeRange`] whispery emits is in the timebase of
//!   the caller's first `handle_samples` call.

use core::num::NonZeroU32;
use mediatime::Timebase;

/// Internal analysis sample rate. All audio fed to whispery must
/// already be resampled to this rate (caller's responsibility).
pub const SAMPLE_RATE_HZ: u32 = 16_000;

/// `const fn` helper for `NonZeroU32` conversion. Panics on zero
/// input — only used at compile time on statically-nonzero values,
/// so the panic is unreachable in practice. Avoids depending on
/// `Option::unwrap` const stability.
#[cfg_attr(not(tarpaulin), inline(always))]
const fn nz(n: u32) -> NonZeroU32 {
  match NonZeroU32::new(n) {
    Some(n) => n,
    None => panic!("expected nonzero u32"),
  }
}

const SAMPLE_RATE_NZ: NonZeroU32 = nz(SAMPLE_RATE_HZ);

/// Internal analysis timebase (`1 / 16_000`). Used by the cut state
/// machine, the sample buffer, and the alignment pipeline. Not part
/// of whispery's public output surface — every emitted `TimeRange`
/// is in the caller's external timebase.
pub const ANALYSIS_TIMEBASE: Timebase = Timebase::new(1, SAMPLE_RATE_NZ);

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn analysis_timebase_is_one_over_16k() {
    assert_eq!(ANALYSIS_TIMEBASE.num(), 1);
    assert_eq!(ANALYSIS_TIMEBASE.den().get(), 16_000);
  }

  #[test]
  fn sample_rate_constant_matches_timebase() {
    assert_eq!(SAMPLE_RATE_HZ, ANALYSIS_TIMEBASE.den().get());
  }
}
