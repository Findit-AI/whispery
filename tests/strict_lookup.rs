//! Strict lookup-order regression. Spec §6.3.1.
//!
//! AlignmentSet::lookup returns:
//!   - Hit { matched: Lang(L), .. } when Lang(L) is registered.
//!   - AnyFallback when Lang(L) is missing AND Any is registered.
//!   - Miss { fallback } when both are missing.
//!
//! The strictness contract — "failure on a registered Lang(L)
//! does NOT consult Any" — lives at the worker level
//! (run_one_alignment in alignment_pool.rs). We can't directly
//! exercise the worker without a real Aligner; instead, we
//! exercise the lookup boundary and assert the documented
//! contract is reflected at the type level (Hit / AnyFallback /
//! Miss are distinct variants — there is no "FellThroughOnFailure"
//! variant).

#![cfg(feature = "alignment")]

// Plan note: Task 28's example imports these names from `whispery::`
// directly; the crate-root re-exports land in Task 29 (§3.3). For
// Task 28 we name them via the existing `whispery::runner` path to
// keep the test self-contained (no lib.rs change in this task), the
// same workaround Task 25 (alignment_e2e.rs) used.
use whispery::runner::{AlignmentFallback, AlignmentLookup, AlignmentSetBuilder};
use whispery::Lang;

#[test]
fn empty_set_misses_with_default_skip_chunk() {
    let set = AlignmentSetBuilder::new().build();
    match set.lookup(&Lang::En) {
        AlignmentLookup::Miss { fallback } => {
            assert_eq!(fallback, AlignmentFallback::SkipChunk);
        }
        _ => panic!("expected Miss"),
    }
}

#[test]
fn empty_set_with_error_fallback_misses_to_error() {
    let set = AlignmentSetBuilder::new()
        .with_fallback(AlignmentFallback::Error)
        .build();
    match set.lookup(&Lang::Zh) {
        AlignmentLookup::Miss { fallback } => {
            assert_eq!(fallback, AlignmentFallback::Error);
        }
        _ => panic!("expected Miss"),
    }
}

#[test]
fn variants_are_distinct_documented_strictness() {
    // Compile-time documentation: AlignmentLookup has exactly
    // three variants — Hit, AnyFallback, Miss. There is NO
    // "RegisteredFailedFellThroughToAny" variant; the worker's
    // `run_one_alignment` does not retry on registered failure.
    fn _exhaustive_match(l: AlignmentLookup<'_>) {
        match l {
            AlignmentLookup::Hit { .. } => {}
            AlignmentLookup::AnyFallback { .. } => {}
            AlignmentLookup::Miss { .. } => {}
        }
    }
}
