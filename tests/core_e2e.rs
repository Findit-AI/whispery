//! End-to-end black-box test for the core state machine.

use core::{num::NonZeroU32, time::Duration};

use asry::{
  AsrResult, Command, Event, Lang, LanguagePolicy, Transcriber, TranscriberOptions, VadSegment,
};
use mediatime::{Timebase, Timestamp};

fn tb_48k() -> Timebase {
  Timebase::new(1, NonZeroU32::new(48_000).unwrap())
}

fn ts(pts: i64) -> Timestamp {
  Timestamp::new(pts, tb_48k())
}

fn happy_asr_result(text: &str) -> AsrResult {
  AsrResult::new(smol_str::SmolStr::new(text), Lang::En, -0.5, 0.05, 0.0)
}

#[test]
fn happy_path_three_chunks_emit_in_order() {
  let config = TranscriberOptions::default()
    .with_chunk_size(Duration::from_secs(2))
    .with_max_in_flight(4);

  let mut t = Transcriber::new(config);

  // Push 6 seconds of audio at 16 kHz = 96_000 samples, anchored
  // at output PTS 0 with timebase 1/48000.
  t.handle_samples(ts(0), &vec![0.0_f32; 96_000]).unwrap();

  // Three VAD segments, each ~2 seconds long. cut_size = 2s, so
  // each segment closes one chunk on the *next* segment's push.
  t.handle_vad_segment(VadSegment::new(0, 32_000)).unwrap();
  t.handle_vad_segment(VadSegment::new(32_000, 64_000))
    .unwrap();
  t.handle_vad_segment(VadSegment::new(64_000, 96_000))
    .unwrap();
  t.handle_eof().unwrap();

  // Drain commands and feed back results.
  let mut chunk_ids = Vec::new();
  while let Some(cmd) = t.poll_command() {
    match cmd {
      Command::Asr { chunk_id, .. } => {
        chunk_ids.push(chunk_id);
        t.handle_asr(chunk_id, happy_asr_result(&format!("c{}", chunk_id)))
          .unwrap();
      }
      Command::Alignment { .. } => panic!("alignment off in this test"),
    }
  }

  // Drain events; expect three Transcripts in chunk-id order.
  let mut texts = Vec::new();
  while let Some(ev) = t.poll_event() {
    match ev {
      Event::Transcript(tr) => texts.push((tr.chunk_id().as_u64(), tr.text().to_owned())),
      Event::Error { .. } => panic!("no errors expected"),
    }
  }
  assert_eq!(texts.len(), 3);
  assert_eq!(texts[0].0, 0);
  assert_eq!(texts[1].0, 1);
  assert_eq!(texts[2].0, 2);
}

#[test]
fn out_of_order_completion_emits_in_chunk_id_order() {
  // Auto policy: this test exercises out-of-order completion, not
  // language locking. Default AutoLockAfter(1) would gate chunks
  // 1, 2 until chunk 0 resolves, defeating the out-of-order
  // scenario.
  let config = TranscriberOptions::default()
    .with_chunk_size(Duration::from_secs(1))
    .with_language_policy(LanguagePolicy::Auto);
  let mut t = Transcriber::new(config);

  t.handle_samples(ts(0), &vec![0.0_f32; 64_000]).unwrap();
  t.handle_vad_segment(VadSegment::new(0, 16_000)).unwrap();
  t.handle_vad_segment(VadSegment::new(16_000, 32_000))
    .unwrap();
  t.handle_vad_segment(VadSegment::new(32_000, 48_000))
    .unwrap();
  t.handle_eof().unwrap();

  // Issue all Asr commands.
  let mut commands = Vec::new();
  while let Some(cmd) = t.poll_command() {
    commands.push(cmd);
  }
  assert_eq!(commands.len(), 3);

  // Resolve in reverse order.
  for cmd in commands.into_iter().rev() {
    if let Command::Asr { chunk_id, .. } = cmd {
      t.handle_asr(chunk_id, happy_asr_result("x")).unwrap();
    }
  }

  let mut ids = Vec::new();
  while let Some(ev) = t.poll_event() {
    match ev {
      Event::Transcript(tr) => ids.push(tr.chunk_id().as_u64()),
      Event::Error { .. } => panic!(),
    }
  }
  assert_eq!(ids, vec![0, 1, 2]);
}
