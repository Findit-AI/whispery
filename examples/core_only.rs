//! Example: drive the Sans-I/O core directly.
//!
//! This example uses NO ML backends — every "ASR result" is
//! synthesised on the fly. It demonstrates the push/poll/inject
//! contract end-to-end.
//!
//! Run with: `cargo run --example core_only`

use core::{num::NonZeroU32, time::Duration};

use asry::{AsrResult, Command, Event, Lang, Transcriber, TranscriberOptions, VadSegment};
use mediatime::{Timebase, Timestamp};

fn main() {
  // Output timebase: original media at 48 kHz.
  let output_tb = Timebase::new(1, NonZeroU32::new(48_000).unwrap());

  let config = TranscriberOptions::default().with_chunk_size(Duration::from_secs(2));
  let mut t = Transcriber::new(config);

  // Push 4 seconds of audio at 16 kHz internal = 64_000 samples.
  let samples = vec![0.0_f32; 64_000];
  t.handle_samples(Timestamp::new(0, output_tb), &samples)
    .unwrap();

  // Two VAD segments, each ~2 s.
  t.handle_vad_segment(VadSegment::new(0, 32_000)).unwrap();
  t.handle_vad_segment(VadSegment::new(32_000, 64_000))
    .unwrap();
  t.handle_eof().unwrap();

  // Drain commands, feed mocked results back.
  while let Some(cmd) = t.poll_command() {
    match cmd {
      Command::Asr {
        chunk_id, samples, ..
      } => {
        println!("[asr] chunk {} ({} samples)", chunk_id, samples.len());
        t.handle_asr(
          chunk_id,
          AsrResult::new(
            format!("(mock transcript for chunk {})", chunk_id).into(),
            Lang::En,
            -0.5,
            0.05,
            0.0,
          ),
        )
        .unwrap();
      }
      Command::Alignment { .. } => {
        unreachable!("alignment off in this example");
      }
    }
  }

  // Drain events.
  while let Some(ev) = t.poll_event() {
    match ev {
      Event::Transcript(tr) => {
        println!(
          "[transcript] chunk {} text={:?} range={:?}",
          tr.chunk_id(),
          tr.text(),
          tr.range()
        );
      }
      Event::Error { chunk_id, error } => {
        println!("[error] chunk {} error={:?}", chunk_id, error);
      }
    }
  }
}
