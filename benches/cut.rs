//! Throughput bench: cut state machine driven through the public
//! Transcriber surface.

use core::num::NonZeroU32;
use core::time::Duration;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use mediatime::{Timebase, Timestamp};
use whispery::{Transcriber, TranscriberConfig, VadSegment};

fn bench_push_vad(c: &mut Criterion) {
    c.bench_function("push_vad_segment_x1000", |b| {
        b.iter(|| {
            let mut config = TranscriberConfig::default();
            config.chunk_size = Duration::from_secs(30);
            config.buffer_cap_samples = 100_000_000;
            let mut t = Transcriber::new(config);
            let tb = Timebase::new(1, NonZeroU32::new(48_000).unwrap());
            t.push_samples(Timestamp::new(0, tb), &vec![0.0_f32; 1000]).unwrap();
            for i in 0..1000u64 {
                let s = i * 100;
                let e = s + 99;
                let _ = black_box(t.push_vad_segment(VadSegment::new(s, e)));
            }
        });
    });
}

criterion_group!(benches, bench_push_vad);
criterion_main!(benches);
