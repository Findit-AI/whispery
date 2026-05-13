//! Throughput bench: dispatch state machine with mocked inference.

use core::time::Duration;
use std::hint::black_box;

use asry::{AsrResult, Command, Lang, Transcriber, TranscriberOptions, VadSegment};
use criterion::{Criterion, criterion_group, criterion_main};

fn bench_dispatch(c: &mut Criterion) {
  c.bench_function("e2e_300_chunks_mocked", |b| {
    b.iter(|| {
      let config = TranscriberOptions::default()
                .with_chunk_size(Duration::from_millis(125)) // 2_000 samples
                .with_buffer_cap_samples(64_000_000)
                .with_max_in_flight(32);
      let mut t = Transcriber::new(config);
      let tb = mediatime::Timebase::new(1, core::num::NonZeroU32::new(48_000).unwrap());
      t.handle_samples(mediatime::Timestamp::new(0, tb), &vec![0.0_f32; 600_000])
        .unwrap();
      for i in 0..300u64 {
        let s = i * 2_000;
        let e = s + 1_900;
        t.handle_vad_segment(VadSegment::new(s, e)).unwrap();
      }
      t.handle_eof().unwrap();
      while let Some(cmd) = t.poll_command() {
        if let Command::Asr { chunk_id, .. } = cmd {
          t.handle_asr(
            chunk_id,
            AsrResult::new("x".into(), Lang::En, -0.5, 0.05, 0.0),
          )
          .unwrap();
        }
      }
      while black_box(t.poll_event()).is_some() {}
    });
  });
}

criterion_group!(benches, bench_dispatch);
criterion_main!(benches);
