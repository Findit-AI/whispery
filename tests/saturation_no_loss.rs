//! v3-v5 regression test: NB-β saturation-result-loss.
//!
//! Drives the runner with max_queued_chunks=1 and many chunks so the
//! dispatch loop's saturation wait fires repeatedly. Asserts every
//! chunk_id emits exactly one Transcript (or Error). The pre-fix
//! `select! { recv -> _ => {} }` form would lose 1 result per
//! saturation cycle and miss transcripts.

#![cfg(feature = "runner")]

use core::num::NonZeroU32;
use core::time::Duration;

use mediatime::{Timebase, Timestamp};
// Plan note: the plan's example imports `ManagedTranscriber` and
// `WhisperPoolConfig` from `whispery::` directly; those crate-root
// re-exports land in Task 24 (§3.3). For Task 21 we name them via
// the existing `whispery::runner` path to keep the test self-contained
// (no lib.rs change in this task's file list), mirroring the same
// workaround used in `tests/runner_e2e.rs`.
use whispery::{LanguagePolicy, VadSegment};
use whispery::runner::{ManagedTranscriber, WhisperPoolConfig};

const MODEL_PATH: Option<&str> = option_env!("WHISPERY_TINY_EN_MODEL");

#[test]
fn saturation_emits_all_chunks_in_order() {
    let model_path = match MODEL_PATH {
        Some(p) => p,
        None => return,
    };

    // 12 chunks worth of audio + max_queued_chunks=1 forces the
    // saturation wait to fire 11+ times. If a single result is lost
    // per saturation cycle, the final count would be < 12.
    let pool = WhisperPoolConfig::new(model_path)
        .with_worker_count(1)
        .with_max_queued_chunks(1);
    let mut runner = ManagedTranscriber::from_config(pool)
        .expect("build pool config")
        .chunk_size(Duration::from_secs(2))
        .language_policy(LanguagePolicy::Lock { hint: whispery::Lang::En })
        .build()
        .expect("build runner");

    let tb = Timebase::new(1, NonZeroU32::new(48_000).unwrap());
    // 24 s of zero audio at 16 kHz internal = 384 000 samples; 12 chunks
    // of 2 s each.
    let samples = vec![0.0_f32; 384_000];
    let mut vads = Vec::new();
    for i in 0..12u64 {
        vads.push(VadSegment::new(i * 32_000, (i + 1) * 32_000));
    }
    runner
        .process_packet(Timestamp::new(0, tb), &samples, &vads, None)
        .expect("process_packet");
    runner.signal_eof().expect("signal_eof");
    runner.drain().expect("drain");

    let mut chunk_ids = Vec::new();
    while let Some(t) = runner.poll_transcript() {
        chunk_ids.push(t.chunk_id().as_u64());
    }
    while let Some((id, _err)) = runner.poll_error() {
        chunk_ids.push(id.as_u64());
    }
    chunk_ids.sort();
    assert_eq!(
        chunk_ids,
        (0..12u64).collect::<Vec<_>>(),
        "every chunk must emit exactly once; got chunk_ids = {chunk_ids:?}"
    );
}
