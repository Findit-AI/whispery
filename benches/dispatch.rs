//! Throughput bench: dispatch state machine with mocked inference.

use core::time::Duration;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use whispery::{AsrResult, Command, Lang, Transcriber, TranscriberConfig, VadSegment};

fn bench_dispatch(c: &mut Criterion) {
    c.bench_function("e2e_300_chunks_mocked", |b| {
        b.iter(|| {
            let mut config = TranscriberConfig::default();
            config.chunk_size = Duration::from_millis(125); // 2_000 samples
            config.buffer_cap_samples = 64_000_000;
            config.max_in_flight = 32;
            let mut t = Transcriber::new(config);
            let tb = mediatime::Timebase::new(1, core::num::NonZeroU32::new(48_000).unwrap());
            t.push_samples(mediatime::Timestamp::new(0, tb), &vec![0.0_f32; 600_000]).unwrap();
            for i in 0..300u64 {
                let s = i * 2_000;
                let e = s + 1_900;
                t.push_vad_segment(VadSegment::new(s, e)).unwrap();
            }
            t.signal_eof().unwrap();
            while let Some(cmd) = t.poll_command() {
                if let Command::RunAsr { chunk_id, .. } = cmd {
                    t.inject_asr_result(
                        chunk_id,
                        AsrResult {
                            text: "x".into(),
                            language: Lang::En,
                            avg_logprob: -0.5,
                            no_speech_prob: 0.05,
                            temperature: 0.0,
                        },
                    ).unwrap();
                }
            }
            while let Some(_) = black_box(t.poll_event()) {}
        });
    });
}

criterion_group!(benches, bench_dispatch);
criterion_main!(benches);
