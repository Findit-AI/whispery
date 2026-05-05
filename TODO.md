# TODO

## OOM-prevention: extract a shared spill-buffer crate

dia ships `SpillBytes` / `SpillBytesMut` in `dia/src/ops/spill.rs`. They
solve the "size-known-upfront, large flat buffer of `T: Pod` cells" case
by picking heap below a configurable threshold and a tempfile-backed
mmap above it. Two-phase ownership (`SpillBytesMut` writer → `freeze` →
`SpillBytes` reader, `Arc`-cloneable + `Send + Sync`) gives mutable fill
plus cheap-clone fan-out.

We want both dia and whispery to use the same implementation. A new
crate (working name **TBD** — `spillbuf`? `spill-bytes`?) hosts the type
and the `SpillOptions` config; dia and whispery depend on it.

### Migration steps

- [ ] Create new crate. Move `dia::ops::spill` into it verbatim. Public
      surface: `SpillBytes<T>`, `SpillBytesMut<T>`, `SpillOptions`,
      `SpillError`. `T: bytemuck::Pod`. Keep the existing tests; they
      exercise the heap path, the mmap path, and the freeze handoff.
- [ ] dia: replace internal `crate::ops::spill::*` references with the
      external crate. Re-export at the original module path so the
      existing call sites (`pdist`, `reconstruct/algo.rs`, `aggregate/count.rs`,
      `streaming/offline_diarizer.rs`) need no change. Bump dia's
      Cargo.toml.
- [ ] whispery: add the crate as a dependency under the existing
      `runner` feature (it's `std`-only). No-std pure-core stays clean.

### Whispery integration — what actually spills

Inventory of whispery allocations and the verdict for each:

| Allocation | Size | Sized once? | `T: Pod`? | Verdict |
|---|---|---|---|---|
| `ExtractedChunk.samples` | up to `chunk_size_samples × 4` per chunk | yes (at `extract_from`) | f32 ✓ | **Spill — drop-in `SpillBytes<f32>`** |
| Trellis `Vec<f32>` (`get_trellis`) | up to 128 MB (capped) | yes | f32 ✓ | Keep fail-fast. The 32 M-cell cap turns hallucinated long token lists into an in-band `WorkFailure`; spilling instead would mask the upstream bug. |
| Emission / log_probs | T × V = ~192 KB for default | yes | f32 ✓ | Too small to bother. |
| `SampleBuffer.samples` | bounded by `chunk_size_samples` + push burst (trims continuously) | no — push-grows + front-trim ring | f32 ✓ | Bounded in practice; deferred. See "zuffer" section. |
| `pending_transcripts` `VecDeque<Transcript>` | unbounded if caller doesn't poll | no | `Transcript` not `Pod` | Backpressure already covers this. |

### Concrete migration target: `ExtractedChunk.samples`

```rust
// src/core/dispatch.rs
- pub samples: Arc<[f32]>,
+ pub samples: SpillBytes<f32>,
```

Construction in `extract_from` becomes:

```rust
let n = (chunk.range.end - chunk.range.start) as usize;
let mut buf = SpillBytesMut::<f32>::zeros(n, &spill_options)?;
buf.as_mut_slice().copy_from_slice(&buffer.samples[lo..hi]);
let samples = buf.freeze();
```

Read sites (`alignment_pool.rs`, `whisper_pool.rs`) already work with
`&[f32]` via `Arc::as_ref()`; they switch to `SpillBytes::as_slice()`
unchanged.

`SpillOptions` lives on `TranscriberOptions` (or `ManagedTranscriberBuilder`);
default = 256 MiB threshold, default temp dir. Wire it through `Dispatch`
so `extract_from` has access.

### Concrete trigger: when does this earn its keep

- 16 kHz f32 = 64 KB/sec.
- Single chunk crosses 256 MB threshold at chunk_size > ~70 minutes.
- `cut_pending` (max_in_flight = 6) crosses 256 MB total at chunk_size
  > ~12 minutes per chunk.
- For default 30-second chunks (~1.9 MB each), the buffer stays on the
  heap; `SpillBytes` is a no-op.

So the migration is **future-proofing**, not an active OOM fix. Worth
doing once for downstream users running large `chunk_size` configs;
not urgent.

### Cheap defenses to do first (independent of spill)

Even without `SpillBytes`, validate the public knobs that gate runtime
allocation size:

- [ ] `TranscriberOptions::set_buffer_cap_samples`: panic on
      `value > MAX_BUFFER_CAP_SAMPLES` (suggested cap: 1 hour =
      57.6 M samples = ~230 MB). Match the existing `set_chunk_size`
      / `set_max_attempts` / `set_n_threads` panic-on-bad-input style.
- [ ] `TranscriberOptions::set_chunk_size`: panic on
      `value > MAX_CHUNK_SIZE` (suggested cap: 1 hour). whisper.cpp's
      internal window is 30 s; chunks >> that overflow no realistic
      decoder budget.
- [ ] Document the memory cost of each setter in its rustdoc.

These land the OOM defense in one PR with no new dependency.

---

## zuffer's idea — verdict: do not pursue

zuffer (https://github.com/al8n/zuffer) is a Go-`bytes.Buffer`-style
**growable** byte buffer with auto-mmap-on-grow: starts on the heap,
transparently copies itself to a tempfile-mmap when `grow()` would push
past a runtime threshold. Untyped (`&[u8]`), single-threaded,
length-prefixed slice records.

**Why it's not a fit for whispery:**

| Aspect | What whispery needs | What zuffer offers |
|---|---|---|
| The only push-grow buffer (`SampleBuffer.samples`) | ring buffer (append + **trim-from-front** + random read by absolute sample index) | stack (append + reset only — no front-trim) |
| All large allocations | sized once at construction (per-chunk audio, trellis, emission) | growable |
| Element type | `f32` (cheap with bytemuck) | `&[u8]` |
| Concurrency | per-chunk audio is `Send + Sync` once frozen | not thread-safe |

Even if we wanted zuffer's auto-mmap-on-grow behaviour for
`SampleBuffer`, we couldn't use zuffer as-is because `SampleBuffer`
trims a prefix on every chunk emit — the zuffer API has no equivalent
operation. Reshaping zuffer into a ring buffer would be a redesign,
not a polish.

**Where zuffer's *idea* would matter for whispery:** if a future
feature lets users buffer arbitrarily long audio without VAD-driven
chunk emission, peak `SampleBuffer.samples` size would no longer be
bounded by `chunk_size_samples`. Today it is — the cut state machine
emits whenever `chunk_size_samples` is reached, which triggers
`after_inject → trim_to`. Peak buffer = `chunk_size_samples + push burst`.

**Conclusion:** the dia spill crate covers our concrete cases
(`ExtractedChunk.samples` is the only one that crosses the threshold
at realistic configs). zuffer's growable-with-spill model is overkill
for sized-once buffers and underkill for ring-buffer semantics. Skip.
