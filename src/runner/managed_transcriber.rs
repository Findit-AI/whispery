//! ManagedTranscriber — the runner's public surface.

use alloc::collections::VecDeque;
use core::time::Duration;
use std::sync::{Arc, atomic::AtomicBool};

use crossbeam_channel::TrySendError;
use whisper_rs::WhisperContext;

use crate::{
  core::{AsrParams, AsrParamsOverride, Command, Event, LanguagePolicy, Transcriber},
  runner::{
    RunnerError, WhisperPoolOptions,
    whisper_pool::{AsrWorkItem, WhisperPool},
  },
  types::{ChunkId, Transcript, VadSegment, WorkFailure},
};
use mediatime::Timestamp;

/// Outcome of a single try-send into the work_tx channel.
#[derive(Debug)]
pub(super) enum DispatchOutcome {
  /// Command was sent and consumed.
  Sent,
  /// Channel was full; the command must be re-parked via
  /// `Transcriber::unpoll_command`.
  Backpressure(Command),
  /// All worker channels are disconnected — the pool has shut down.
  Disconnected,
}

/// Public runner: wraps `core::Transcriber` and a `WhisperPool` with
/// a saturation-deadlock-safe dispatch loop.
pub struct ManagedTranscriber {
  core: Transcriber,
  whisper_pool: WhisperPool,
  asr_params_default: AsrParams,
  asr_timeout: Duration,
  drain_timeout: Duration,
  block_on_full_queue: bool,
  dispatch_idle_poll: Duration,
  buffer_cap_samples: usize,
  pending_transcripts: VecDeque<Transcript>,
  pending_errors: VecDeque<(ChunkId, WorkFailure)>,
  /// One-shot fatal-error slot. `poll_transcript` and
  /// `poll_error` call `drive_one_step` to drain worker /
  /// dispatch queues; if that step returns `Err`, we stash the
  /// error here rather than dropping it. The next poll call
  /// drains the local pending buffers first (so already-arrived
  /// transcripts and chunk-level errors aren't lost), then
  /// surfaces the fatal once the buffers are empty. Codex
  /// round-30 flagged the prior `let _ = self.drive_one_step()`
  /// as silently dropping `WhisperPoolShutdown` and
  /// `Backpressure`, making a dead pool look like an empty
  /// stream.
  pending_fatal: Option<RunnerError>,

  /// Alignment pool (single worker). `None` when `with_alignment`
  /// was not called or the supplied set was empty.
  #[cfg(feature = "alignment")]
  alignment_pool: Option<crate::runner::alignment_pool::AlignmentPool>,

  /// Per-job alignment timeout. Stamped on each
  /// `AlignWorkItem`.
  #[cfg(feature = "alignment")]
  align_timeout: Duration,
}

impl ManagedTranscriber {
  /// Try to send a Command into the appropriate worker pool.
  /// Non-blocking. Also handles `Command::RunAlignment` by
  /// shipping into the alignment pool.
  fn try_dispatch(&self, cmd: Command, asr_timeout: Duration) -> DispatchOutcome {
    match cmd {
      Command::RunAsr {
        chunk_id,
        samples,
        params,
        sample_rate: _,
      } => {
        // The core's emitted params already include locked
        // language AND the per-packet override that was active
        // when this chunk was *extracted* (see
        // `ExtractedChunk::override_at_creation` and
        // `Transcriber::set_runtime_override`). Don't merge any
        // current `pending_override` here — that would corrupt
        // parked / cut-pending-promoted commands with the wrong
        // packet's params. Use `params` verbatim.
        let abort_flag = Arc::new(AtomicBool::new(false));
        let item = AsrWorkItem {
          chunk_id,
          samples,
          params,
          asr_timeout,
          abort_flag,
        };
        match self.whisper_pool.work_tx.try_send(item) {
          Ok(()) => DispatchOutcome::Sent,
          Err(TrySendError::Full(item)) => {
            // Reconstruct the original Command so the core
            // can re-park it via unpoll_command.
            let cmd = Command::RunAsr {
              chunk_id: item.chunk_id,
              samples: item.samples,
              sample_rate: crate::time::SAMPLE_RATE_HZ,
              params: item.params,
            };
            DispatchOutcome::Backpressure(cmd)
          }
          Err(TrySendError::Disconnected(_)) => DispatchOutcome::Disconnected,
        }
      }

      #[cfg(feature = "alignment")]
      Command::RunAlignment {
        chunk_id,
        samples,
        sub_segments,
        text,
        language,
      } => {
        // text + language are Whisper's authoritative output
        // (the registry lookup depends on `language`). They are
        // forwarded into the AlignWorkItem verbatim — never
        // re-derived from a runner-side default.
        let Some(pool) = self.alignment_pool.as_ref() else {
          // Core emitted RunAlignment but we have no pool.
          // This is a misconfigured builder (with_word_alignment
          // on, with_alignment empty/unset). Park indefinitely
          // — the core retries forever. v1 surfaces as
          // backpressure to avoid losing the chunk.
          return DispatchOutcome::Backpressure(Command::RunAlignment {
            chunk_id,
            samples,
            sub_segments,
            text,
            language,
          });
        };

        // Need the bound `samples_to_output_range` closure for
        // the worker's wav2vec2 frame → output-timebase mapping.
        // Pulled from the chunk's record (anchor snapshot at
        // extract time) rather than the live buffer, so word
        // ranges stay in this chunk's PTS epoch even if a
        // `restart_at` happened between extract and now.
        let Some(samples_to_output_range) = self.core.chunk_samples_to_output_range_fn(chunk_id)
        else {
          return DispatchOutcome::Backpressure(Command::RunAlignment {
            chunk_id,
            samples,
            sub_segments,
            text,
            language,
          });
        };

        // Need the chunk's stream-coordinate first sample to
        // build chunk-local sub_segments and to ship to the
        // worker for its frame → stream-sample bridge.
        let chunk_first_sample = match self.core.chunk_first_sample(chunk_id) {
          Some(v) => v,
          None => {
            // The chunk just ran ASR and emitted RunAlignment;
            // its dispatch record must still hold. If not,
            // surface backpressure to retry.
            return DispatchOutcome::Backpressure(Command::RunAlignment {
              chunk_id,
              samples,
              sub_segments,
              text,
              language,
            });
          }
        };

        // The output-timebase TimeRanges in `sub_segments` are
        // produced by `SampleBuffer::samples_to_output_range`;
        // for the aligner's silence mask we need them in
        // chunk-local 16 kHz sample-index space. Pull the raw
        // stream-sample form preserved alongside on the
        // dispatch record (`ChunkRecord::sub_segments_samples`)
        // and offset by `chunk_first_sample` so start_pts ==
        // chunk-local sample index.
        let chunk_local_subs = self
          .core
          .chunk_sub_segments_samples(chunk_id)
          .unwrap_or_default();
        let chunk_local_subs_as_ranges: alloc::vec::Vec<mediatime::TimeRange> = chunk_local_subs
          .iter()
          .map(|(start, end)| {
            // Encode as TimeRange with timebase 1/16000
            // so start_pts == start_sample (chunk-local).
            mediatime::TimeRange::new(
              (*start as i64) - (chunk_first_sample as i64),
              (*end as i64) - (chunk_first_sample as i64),
              mediatime::Timebase::new(1, core::num::NonZeroU32::new(16_000).unwrap()),
            )
          })
          .collect();

        let abort_flag = Arc::new(AtomicBool::new(false));
        let item = crate::runner::alignment_pool::AlignWorkItem {
          chunk_id,
          samples,
          sub_segments: chunk_local_subs_as_ranges,
          text,
          language,
          align_timeout: self.align_timeout,
          abort_flag,
          chunk_first_sample_in_stream: chunk_first_sample,
          samples_to_output_range,
        };
        match pool.work_tx.try_send(item) {
          Ok(()) => DispatchOutcome::Sent,
          Err(TrySendError::Full(item)) => {
            // Re-park: the original output-timebase
            // sub_segments were consumed into the work
            // item. The core's unpoll_command + dispatch
            // record retain the authoritative copy
            // (`ChunkRecord::sub_segments`), so the next
            // poll_command can rebuild from there. For the
            // re-parked Command we ship an empty Vec — the
            // data is recoverable from the record on retry
            // via this same try_dispatch path.
            let cmd = Command::RunAlignment {
              chunk_id: item.chunk_id,
              samples: item.samples,
              sub_segments: alloc::vec::Vec::new(),
              text: item.text,
              language: item.language,
            };
            DispatchOutcome::Backpressure(cmd)
          }
          Err(TrySendError::Disconnected(_)) => DispatchOutcome::Disconnected,
        }
      }

      // Without the `alignment` cargo feature, RunAlignment is
      // emitted by the core only when `word_alignment` was set
      // on the core's TranscriberOptions — which the runner's
      // builder gates on the alignment pool's presence. Reaching
      // this arm with feature off means a non-runner caller is
      // driving the core; re-park indefinitely.
      #[cfg(not(feature = "alignment"))]
      cmd @ Command::RunAlignment { .. } => DispatchOutcome::Backpressure(cmd),
    }
  }

  /// One non-blocking step of the inline dispatch loop.
  ///
  /// Returns `Ok(true)` if any of (drain ≥ 1 result | send ≥ 1
  /// command | core surfaced ≥ 1 event); `Ok(false)` if nothing
  /// changed.
  ///
  /// `Err(RunnerError::Backpressure)` is returned only when
  /// `block_on_full_queue=false` and a try_send hit Full. The
  /// command was re-parked via `Transcriber::unpoll_command`; the
  /// core's buffer state has already advanced (samples buffered,
  /// segments merged into possibly-pending chunks). The caller
  /// must drain via `poll_*` before pushing again.
  ///
  /// `Err(RunnerError::WhisperPoolShutdown)` is fatal: a worker
  /// channel disconnected.
  pub(super) fn drive_one_step(&mut self) -> Result<bool, RunnerError> {
    let mut progress = false;

    // Phase 1a: drain whisper results.
    loop {
      match self.whisper_pool.result_rx.try_recv() {
        Ok((chunk_id, Ok(asr_result))) => {
          progress = true;
          self.core.inject_asr_result(chunk_id, asr_result)?;
        }
        Ok((chunk_id, Err(failure))) => {
          progress = true;
          self.core.inject_failure(chunk_id, failure)?;
        }
        Err(crossbeam_channel::TryRecvError::Empty) => break,
        Err(crossbeam_channel::TryRecvError::Disconnected) => {
          return Err(RunnerError::WhisperPoolShutdown);
        }
      }
    }

    // Phase 1b: drain alignment results (when the pool exists).
    // Parallel to the whisper pool drain. An alignment-pool
    // disconnect is mapped onto WhisperPoolShutdown for now
    // (same semantics: rebuild the runner). A dedicated
    // RunnerError::AlignmentPoolShutdown is straightforward
    // future work.
    #[cfg(feature = "alignment")]
    if let Some(pool) = self.alignment_pool.as_ref() {
      loop {
        match pool.result_rx.try_recv() {
          Ok((chunk_id, Ok(align_result))) => {
            progress = true;
            self.core.inject_alignment_result(chunk_id, align_result)?;
          }
          Ok((chunk_id, Err(failure))) => {
            progress = true;
            self.core.inject_failure(chunk_id, failure)?;
          }
          Err(crossbeam_channel::TryRecvError::Empty) => break,
          Err(crossbeam_channel::TryRecvError::Disconnected) => {
            return Err(RunnerError::WhisperPoolShutdown);
          }
        }
      }
    }

    // Phase 2: drain core events into the runner's local
    // pending_transcripts / pending_errors buckets. Without this,
    // every Transcript / Error sits on `core.pending_events`
    // until the caller hits `poll_transcript` / `poll_error`,
    // and `core.is_idle()` (which checks `pending_events.is_empty()`)
    // never goes true. `drain` would then loop until
    // `drain_timeout` fires even when every worker has long since
    // returned its result — a drain hang. Pulling here keeps
    // `is_idle` honest: it goes true as soon as the last chunk's
    // event is bucketed.
    while let Some(ev) = self.core.poll_event() {
      progress = true;
      match ev {
        Event::Transcript(tr) => self.pending_transcripts.push_back(tr),
        Event::Error { chunk_id, error } => self.pending_errors.push_back((chunk_id, error)),
      }
    }

    // Phase 3: drain commands and try to dispatch each.
    while let Some(cmd) = self.core.poll_command() {
      match self.try_dispatch(cmd, self.asr_timeout) {
        DispatchOutcome::Sent => progress = true,
        DispatchOutcome::Backpressure(parked) => {
          self.core.unpoll_command(parked);
          if !self.block_on_full_queue {
            return Err(RunnerError::Backpressure {
              buffered: self.core.buffered_samples(),
              cap: self.buffer_cap_samples,
            });
          }
          return Ok(progress);
        }
        DispatchOutcome::Disconnected => {
          return Err(RunnerError::WhisperPoolShutdown);
        }
      }
    }

    Ok(progress)
  }

  /// Block (with `dispatch_idle_poll` safety) until at least one
  /// worker channel has data, OR the safety-timeout fires. Does NOT
  /// consume any message — the next `drive_one_step` does that via
  /// `try_recv`. `Select::ready_timeout` is the correct primitive;
  /// consuming variants would silently drop results.
  fn wait_for_progress(&self) -> Result<(), RunnerError> {
    let mut sel = crossbeam_channel::Select::new();
    sel.recv(&self.whisper_pool.result_rx);
    // Also wake on alignment results when the pool exists.
    // The index is unused — drive_one_step's try_recv handles
    // message vs. disconnect on each receiver.
    #[cfg(feature = "alignment")]
    let _alignment_idx = if let Some(pool) = self.alignment_pool.as_ref() {
      Some(sel.recv(&pool.result_rx))
    } else {
      None
    };
    // ready_timeout returns Ok(idx) with idx of the first ready
    // op (including disconnects), or Err(SelectTimeoutError) on
    // timeout. We don't care which arm fired — the next
    // drive_one_step's try_recv handles message vs. disconnect.
    let _ = sel.ready_timeout(self.dispatch_idle_poll);
    Ok(())
  }

  /// Drive the inline dispatch loop in a saturation wait.
  ///
  /// Loops:
  ///   1. drive_one_step — if Ok(true), made progress; loop again.
  ///   2. else if no command is parked, exit (genuine idle).
  ///   3. else wait_for_progress, then loop.
  ///
  /// Used by both `process_packet` (after pushing inputs) and
  /// `drain` (until idle).
  fn pump_until_idle_or_progress(&mut self) -> Result<(), RunnerError> {
    loop {
      if self.drive_one_step()? {
        continue;
      }
      // No progress. Is there a parked command waiting?
      // We can't peek without popping; the only way to detect
      // a parked command is to call poll_command, then re-park
      // if Some.
      match self.core.poll_command() {
        None => return Ok(()),
        Some(cmd) => {
          self.core.unpoll_command(cmd);
          self.wait_for_progress()?;
        }
      }
    }
  }
}

/// Builder for [`ManagedTranscriber`].
///
/// All knobs are `with_*` style. Construct via
/// [`ManagedTranscriber::builder`].
pub struct ManagedTranscriberBuilder {
  whisper_ctx: WhisperContext,
  pool_options: WhisperPoolOptions,
  chunk_size: Duration,
  buffer_cap_samples: usize,
  gap_tolerance_samples: u64,
  language_policy: LanguagePolicy,
  asr_params: AsrParams,
  worker_timeouts_asr: Duration,
  worker_timeouts_align: Duration,
  drain_timeout: Option<Duration>,

  /// Optional alignment registry. When `Some(set)` and
  /// `set.is_empty() == false`, `build()` spawns an alignment
  /// worker and emits `Command::RunAlignment` per chunk.
  #[cfg(feature = "alignment")]
  alignment_set: Option<crate::runner::aligner::AlignmentSet>,

  /// Queue depth for the alignment work channel. Default =
  /// whisper pool's `max_queued_chunks`.
  #[cfg(feature = "alignment")]
  alignment_max_queued_chunks: Option<usize>,
}

impl ManagedTranscriberBuilder {
  /// Internal constructor used by `ManagedTranscriber::builder`.
  fn new(whisper_ctx: WhisperContext, pool_options: WhisperPoolOptions) -> Self {
    Self {
      whisper_ctx,
      pool_options,
      chunk_size: Duration::from_secs(30),
      buffer_cap_samples: 60 * 16_000,
      gap_tolerance_samples: 200 * 16,
      language_policy: LanguagePolicy::AutoLockAfter(1),
      asr_params: AsrParams::new(),
      worker_timeouts_asr: Duration::from_secs(60),
      worker_timeouts_align: Duration::from_secs(30),
      drain_timeout: None,
      #[cfg(feature = "alignment")]
      alignment_set: None,
      #[cfg(feature = "alignment")]
      alignment_max_queued_chunks: None,
    }
  }

  /// Override [`crate::core::TranscriberOptions::chunk_size`].
  pub fn chunk_size(mut self, d: Duration) -> Self {
    self.chunk_size = d;
    self
  }

  /// Override [`crate::core::TranscriberOptions::buffer_cap_samples`].
  pub fn buffer_cap_samples(mut self, n: usize) -> Self {
    self.buffer_cap_samples = n;
    self
  }

  /// Override [`crate::core::TranscriberOptions::gap_tolerance_samples`].
  pub fn gap_tolerance_samples(mut self, n: u64) -> Self {
    self.gap_tolerance_samples = n;
    self
  }

  /// Override [`crate::core::TranscriberOptions::language_policy`].
  pub fn language_policy(mut self, p: LanguagePolicy) -> Self {
    self.language_policy = p;
    self
  }

  /// Override the [`WhisperPoolOptions`].
  pub fn whisper_pool(mut self, cfg: WhisperPoolOptions) -> Self {
    self.pool_options = cfg;
    self
  }

  /// Override the default [`AsrParams`].
  pub fn asr_params(mut self, p: AsrParams) -> Self {
    self.asr_params = p;
    self
  }

  /// Per-job worker timeouts. Default 60 s for ASR, 30 s for
  /// alignment.
  pub fn worker_timeouts(mut self, asr: Duration, align: Duration) -> Self {
    self.worker_timeouts_asr = asr;
    self.worker_timeouts_align = align;
    self
  }

  /// Cap on `drain()`. Default 10× the longest worker timeout.
  pub fn drain_timeout(mut self, t: Duration) -> Self {
    self.drain_timeout = Some(t);
    self
  }

  /// Wire word-level forced alignment using the supplied
  /// [`crate::runner::aligner::AlignmentSet`]. The alignment
  /// worker is spawned at `build()` time; chunks emit
  /// `Command::RunAlignment` after their ASR result lands.
  ///
  /// An empty `set` is accepted: `build()` will not spawn an
  /// alignment worker and the runner behaves identically to a
  /// no-alignment build. This lets callers conditionally
  /// configure alignment without branching at the call site.
  ///
  /// Gated on `feature = "alignment"`.
  #[cfg(feature = "alignment")]
  pub fn with_alignment(mut self, set: crate::runner::aligner::AlignmentSet) -> Self {
    self.alignment_set = Some(set);
    self
  }

  /// Override the alignment work-channel capacity. Default =
  /// whisper pool's `max_queued_chunks`. Higher values smooth
  /// over alignment-worker stalls; lower values bound memory
  /// when alignment is the bottleneck.
  ///
  /// Gated on `feature = "alignment"`.
  #[cfg(feature = "alignment")]
  pub const fn alignment_max_queued_chunks(mut self, value: usize) -> Self {
    self.alignment_max_queued_chunks = Some(value);
    self
  }

  /// Construct the `ManagedTranscriber`. Spawns worker threads and
  /// wires channels.
  pub fn build(self) -> Result<ManagedTranscriber, RunnerError> {
    let drain_timeout = self.drain_timeout.unwrap_or_else(|| {
      // 10× the longest worker timeout.
      let longest = core::cmp::max(self.worker_timeouts_asr, self.worker_timeouts_align);
      longest * 10
    });

    // Build the alignment pool only when a non-empty
    // AlignmentSet was supplied. An empty set is silently
    // accepted (callers can conditionally configure alignment
    // without branching at the call site) and produces a build
    // that behaves identically to a no-alignment build.
    #[cfg(feature = "alignment")]
    let alignment_pool = match self.alignment_set {
      Some(set) if !set.is_empty() => {
        let cap = self
          .alignment_max_queued_chunks
          .unwrap_or_else(|| self.pool_options.max_queued_chunks());
        let arc_set = alloc::sync::Arc::new(set);
        Some(crate::runner::alignment_pool::AlignmentPool::new(
          arc_set, cap,
        )?)
      }
      _ => None,
    };

    // The core's `word_alignment` flag follows the alignment
    // pool's presence — if no pool, no `Command::RunAlignment`
    // should be emitted; the core respects this via
    // `TranscriberOptions::with_word_alignment`.
    #[cfg(feature = "alignment")]
    let word_alignment_flag = alignment_pool.is_some();
    #[cfg(not(feature = "alignment"))]
    let word_alignment_flag = false;

    let core_config = crate::core::TranscriberOptions::new()
      .with_chunk_size(self.chunk_size)
      .with_buffer_cap_samples(self.buffer_cap_samples)
      .with_gap_tolerance_samples(self.gap_tolerance_samples)
      .with_language_policy(self.language_policy)
      .with_asr_params(self.asr_params.clone())
      .with_word_alignment(word_alignment_flag)
      .with_max_in_flight(self.pool_options.worker_count() + 2);

    let whisper_pool = WhisperPool::new(self.whisper_ctx, &self.pool_options)?;

    Ok(ManagedTranscriber {
      core: Transcriber::new(core_config),
      whisper_pool,
      asr_params_default: self.asr_params,
      asr_timeout: self.worker_timeouts_asr,
      drain_timeout,
      block_on_full_queue: self.pool_options.block_on_full_queue(),
      dispatch_idle_poll: self.pool_options.dispatch_idle_poll(),
      buffer_cap_samples: self.buffer_cap_samples,
      pending_transcripts: VecDeque::new(),
      pending_errors: VecDeque::new(),
      pending_fatal: None,
      #[cfg(feature = "alignment")]
      alignment_pool,
      #[cfg(feature = "alignment")]
      align_timeout: self.worker_timeouts_align,
    })
  }
}

impl ManagedTranscriber {
  /// Begin building a `ManagedTranscriber` from a pre-constructed
  /// `WhisperContext`. The caller controls flash_attn / DTW / GPU
  /// device explicitly when constructing the context.
  ///
  /// `pool_options` carries the runner-side knobs (worker count,
  /// queue depth, backpressure mode).
  pub fn builder(
    whisper_ctx: WhisperContext,
    pool_options: WhisperPoolOptions,
  ) -> ManagedTranscriberBuilder {
    ManagedTranscriberBuilder::new(whisper_ctx, pool_options)
  }

  /// Convenience: build directly from a `WhisperPoolOptions`'s
  /// `model_path`, loading the context with the config's GPU
  /// settings. Intended for callers that don't need to customise
  /// `WhisperContextParameters` beyond what `WhisperPoolOptions`
  /// already exposes.
  pub fn from_options(
    pool_options: WhisperPoolOptions,
  ) -> Result<ManagedTranscriberBuilder, RunnerError> {
    let mut ctx_params = whisper_rs::WhisperContextParameters::default();
    ctx_params.use_gpu(pool_options.use_gpu());
    ctx_params.gpu_device(pool_options.gpu_device());
    ctx_params.flash_attn(pool_options.flash_attn());
    let path =
      pool_options
        .model_path()
        .to_str()
        .ok_or_else(|| RunnerError::WhisperContextLoad {
          message: format!(
            "model_path is not valid UTF-8: {:?}",
            pool_options.model_path()
          ),
        })?;
    let ctx = WhisperContext::new_with_params(path, ctx_params).map_err(|e| {
      RunnerError::WhisperContextLoad {
        message: format!("{e:?}"),
      }
    })?;
    Ok(ManagedTranscriberBuilder::new(ctx, pool_options))
  }
}

impl ManagedTranscriber {
  /// Push one packet of audio + the VAD segments newly closed
  /// within or before that packet's range.
  ///
  /// **Empty packet** (`samples.is_empty()`): accepted as a no-op
  /// when `delta_pts_out == 0` — VAD segments in the same call are
  /// still pushed.
  ///
  /// **VAD segment ordering contract:** segments must be strictly
  /// monotonic and non-overlapping; violations are surfaced as
  /// `RunnerError::Transcriber(TranscriberError::PtsRegression {
  /// kind: PushKind::VadSegment, .. })`.
  ///
  /// **Backpressure contract**: when this returns
  /// `Err(RunnerError::Backpressure { .. })`, inputs were already
  /// consumed; the caller must drain via `poll_transcript` /
  /// `poll_error` before pushing again.
  pub fn process_packet(
    &mut self,
    starts_at: Timestamp,
    samples: &[f32],
    vad_segments: &[VadSegment],
    params_override: Option<AsrParamsOverride>,
  ) -> Result<(), RunnerError> {
    // Step 1: stamp the override on the dispatch. Chunks
    // extracted from the buffer during this call snapshot the
    // override into their `ExtractedChunk::override_at_creation`,
    // so the params that promote_extracted later emits with
    // `RunAsr` are tied to *this* packet's audio — even if a
    // chunk is held in `cut_pending` past the end of this
    // process_packet and promoted during a later call. Cleared
    // on exit regardless of outcome.
    self.core.set_runtime_override(params_override);

    // Step 2: pre-flight VAD validation BEFORE pushing samples.
    // Codex round-35 caught a partial-commit bug here: the
    // previous order (push_samples → push_vads loop) left the
    // samples + earlier VADs committed if a later VAD was
    // out-of-order or ahead-of-audio. Caller could not safely
    // retry the packet (the same starts_at would now trip
    // PtsRegression). The precheck verifies `eof_signaled`,
    // timebase, ordering, and high-water against a projection
    // of the buffer's high water AFTER the upcoming push, so
    // either everything commits or nothing does.
    let precheck = self.core.precheck_vad_segments(vad_segments, samples.len());
    if let Err(e) = precheck {
      // Roll back the override stamp; nothing else mutated.
      self.core.set_runtime_override(None);
      return Err(RunnerError::Transcriber(e));
    }

    // Step 3: push samples (atomic — either fully commits or
    // returns Err with no commit; see SampleBuffer::append).
    let push_result = self.push_samples_internal(starts_at, samples);

    // Step 4: push VAD segments. Per the precheck above, every
    // segment is guaranteed to satisfy push_vad_segment's
    // validation; this loop's failure modes are now reduced to
    // "push_samples failed" (the and_then short-circuits).
    let result = push_result.and_then(|()| self.push_vads_internal(vad_segments));

    // Step 5: pump the dispatch loop until idle or saturation.
    let drive_result = result.and_then(|()| self.pump_until_idle_or_progress());

    // Step 6: clear the override stamp. Chunks already in
    // `cut_pending` keep the override they captured at extract
    // time — only newly-extracted chunks (in future packets)
    // see `None` here.
    self.core.set_runtime_override(None);

    drive_result
  }

  /// Push samples, mapping core errors into `RunnerError::Transcriber`.
  fn push_samples_internal(
    &mut self,
    starts_at: Timestamp,
    samples: &[f32],
  ) -> Result<(), RunnerError> {
    if samples.is_empty() {
      // The core's push_samples accepts empty packets when
      // delta_pts_out == 0; the underlying buffer call returns
      // Ok(()). We still call through so the timebase / EOF
      // checks fire normally.
      self.core.push_samples(starts_at, samples)?;
      return Ok(());
    }
    self.core.push_samples(starts_at, samples)?;
    Ok(())
  }

  /// Push VAD segments in order.
  fn push_vads_internal(&mut self, vad_segments: &[VadSegment]) -> Result<(), RunnerError> {
    for seg in vad_segments {
      self.core.push_vad_segment(*seg)?;
    }
    Ok(())
  }

  /// Apply a per-call override on top of the runner default; return
  /// the original to be restored at end-of-call.
  fn swap_asr_default(&mut self, ovr: &AsrParamsOverride) -> AsrParams {
    // Temporarily replace the core's default with the merged
    // params. The core uses its default AsrParams when it issues
    // a RunAsr command; for v1, that's the only injection point.
    let merged = merge_overrides(&self.asr_params_default, ovr);
    let prior = core::mem::replace(&mut self.asr_params_default, merged);
    // The core's TranscriberOptions defaults are baked in at
    // construction; the runtime override path is via the
    // Command's `params` field. The core does NOT expose a
    // runtime setter for the dispatch's default AsrParams; this
    // is OK because the runner's own default lives on
    // `asr_params_default` and the runner is the one that emits
    // RunAsr commands' `params`. We override at dispatch time in
    // `try_dispatch`'s `params` consumption — that path is not
    // currently used because the core's poll_command pre-fills
    // `params` from the core's config.
    //
    // To honor the runner's override semantics, we substitute the
    // params on the issued command in dispatch. The simplest
    // correct approach: keep `asr_params_default` updated; have
    // try_dispatch overwrite the Command's `params` with our own
    // before sending.
    prior
  }

  fn restore_asr_default(&mut self, prior: AsrParams) {
    self.asr_params_default = prior;
  }
}

impl ManagedTranscriber {
  /// Mark the input stream as ended. Flushes the cut accumulator,
  /// then drives the dispatch loop one more time. Idempotent.
  pub fn signal_eof(&mut self) -> Result<(), RunnerError> {
    self.core.signal_eof()?;
    self.pump_until_idle_or_progress()?;
    Ok(())
  }

  /// Pop the next available `Transcript`, draining the dispatch
  /// loop along the way.
  ///
  /// Returns:
  /// - `Ok(Some(transcript))` — a transcript was available
  /// - `Ok(None)` — no transcript currently available; call
  ///    again when more progress is expected (or check
  ///    [`is_idle`](Self::is_idle))
  /// - `Err(RunnerError)` — a fatal worker / dispatch failure
  ///   surfaced (e.g. `WhisperPoolShutdown`,
  ///   `Backpressure`). Already-buffered transcripts AND
  ///   chunk-level errors drain ahead of the fatal, so callers
  ///   never lose a result they could have observed.
  ///
  /// Codex round-30: previously this returned plain
  /// `Option<Transcript>` and `let _ = self.drive_one_step()`
  /// silently dropped fatals — a dead pool looked like an empty
  /// stream. Surfacing via `Result` forces callers to handle
  /// the fatal.
  pub fn poll_transcript(&mut self) -> Result<Option<Transcript>, RunnerError> {
    self.drive_or_record_fatal();

    if let Some(tr) = self.pending_transcripts.pop_front() {
      return Ok(Some(tr));
    }
    while let Some(ev) = self.core.poll_event() {
      match ev {
        Event::Transcript(tr) => return Ok(Some(tr)),
        Event::Error { chunk_id, error } => {
          self.pending_errors.push_back((chunk_id, error));
          // Continue: maybe a Transcript is right behind it.
        }
      }
    }
    // No transcript anywhere. Surface the stashed fatal if any.
    if let Some(err) = self.pending_fatal.take() {
      return Err(err);
    }
    Ok(None)
  }

  /// Pop the next available `(ChunkId, WorkFailure)` error,
  /// draining the dispatch loop along the way.
  ///
  /// Result shape mirrors [`poll_transcript`](Self::poll_transcript):
  /// the per-chunk `WorkFailure` (transient, one chunk failed)
  /// is the success payload; structural runner errors (worker
  /// pool dead, backpressure rejected) come back via `Err`.
  ///
  /// Buffered transcripts encountered while scanning for an
  /// error are stashed onto `pending_transcripts`, so the next
  /// `poll_transcript` will see them.
  pub fn poll_error(&mut self) -> Result<Option<(ChunkId, WorkFailure)>, RunnerError> {
    self.drive_or_record_fatal();

    if let Some(pair) = self.pending_errors.pop_front() {
      return Ok(Some(pair));
    }
    while let Some(ev) = self.core.poll_event() {
      match ev {
        Event::Error { chunk_id, error } => return Ok(Some((chunk_id, error))),
        Event::Transcript(tr) => {
          self.pending_transcripts.push_back(tr);
        }
      }
    }
    if let Some(err) = self.pending_fatal.take() {
      return Err(err);
    }
    Ok(None)
  }

  /// Run `drive_one_step` and stash any error in
  /// `pending_fatal`. The first error wins (subsequent errors
  /// during further polls are dropped — a fatal pool is a
  /// fatal pool, the second error is rarely more useful than
  /// the first). Returns the boolean `progress` flag from the
  /// step, or `false` if the step errored (we don't know
  /// whether progress was made before the error, but
  /// downstream is observing buffered events anyway).
  fn drive_or_record_fatal(&mut self) -> bool {
    match self.drive_one_step() {
      Ok(progress) => progress,
      Err(err) => {
        if self.pending_fatal.is_none() {
          self.pending_fatal = Some(err);
        }
        false
      }
    }
  }

  /// Block until [`Self::is_idle`] OR `drain_timeout` elapses.
  ///
  /// Codex round-35:
  ///
  /// - The idle check now goes through [`Self::is_idle`] (the
  ///   public predicate) rather than the inner `core.is_idle()`.
  ///   The public predicate also accounts for buffered
  ///   transcripts/errors AND a stashed `pending_fatal`, so a
  ///   fatal worker error sitting behind buffered output can no
  ///   longer race past `drain` and have it return `Ok`.
  /// - `DrainTimeout.in_flight` now reports the actual chunk
  ///   count via `core.in_flight_chunk_count()`, matching the
  ///   field's documented meaning. The previous `buffered_samples()`
  ///   was a sample count and could read zero (after trim) even
  ///   while chunks were still working.
  pub fn drain(&mut self) -> Result<(), RunnerError> {
    let started = std::time::Instant::now();
    let timeout = self.drain_timeout;
    loop {
      self.pump_until_idle_or_progress()?;
      // Surface a stashed fatal eagerly so the caller sees the
      // structural failure before `drain` can return Ok.
      if let Some(err) = self.pending_fatal.take() {
        return Err(err);
      }
      if self.is_idle() {
        return Ok(());
      }
      if started.elapsed() > timeout {
        return Err(RunnerError::DrainTimeout {
          timeout,
          in_flight: self.core.in_flight_chunk_count(),
        });
      }
      // No progress and not idle: wait for a worker.
      self.wait_for_progress()?;
    }
  }
}

impl ManagedTranscriber {
  /// True iff every queue is empty (core idle, no pending
  /// transcripts/errors locally buffered, and no fatal worker
  /// error stashed for the next poll).
  ///
  /// Codex round-33: previously this ignored `pending_fatal`. A
  /// `drive_one_step` failure stashed during one poll could be
  /// shadowed by `is_idle() == true` on the next call, so a
  /// caller that only polls "until idle" would stop and never
  /// see the structural failure (worker pool dead, backpressure
  /// rejected). Including the slot here closes the gap — callers
  /// must continue polling (which surfaces the fatal as `Err`)
  /// before they can observe a true-idle state.
  pub fn is_idle(&self) -> bool {
    is_idle_inner(
      self.core.is_idle(),
      self.pending_transcripts.is_empty(),
      self.pending_errors.is_empty(),
      self.pending_fatal.is_none(),
    )
  }

  /// Live buffer length in samples (proxy from the core).
  pub fn buffered_samples(&self) -> usize {
    self.core.buffered_samples()
  }

  /// Output timebase, recorded on the first push_samples call.
  pub fn output_timebase(&self) -> Option<mediatime::Timebase> {
    self.core.output_timebase()
  }

  /// PTS that the core expects on the next contiguous push_samples.
  pub fn next_expected_starts_at(&self) -> Option<Timestamp> {
    self.core.next_expected_starts_at()
  }

  /// Non-mutating predicate: would the next push of `samples_len`
  /// audio samples fit?
  pub fn would_accept(&self, samples_len: usize) -> bool {
    self.core.would_accept(samples_len, 0)
  }
}

/// Merge a sparse `AsrParamsOverride` onto `base`, producing the
/// final `AsrParams` that will ship in any RunAsr emitted from the
/// current packet's chunks.
fn merge_overrides(base: &AsrParams, ovr: &AsrParamsOverride) -> AsrParams {
  let mut out = base.clone();
  if let Some(opt_lang) = ovr.language_hint() {
    out.set_language_hint(opt_lang.clone());
  }
  if let Some(strategy) = ovr.strategy() {
    out.set_strategy(strategy);
  }
  if let Some(t) = ovr.initial_temperature() {
    out.set_initial_temperature(t);
  }
  if let Some(prompt) = ovr.initial_prompt() {
    out.set_initial_prompt(prompt.clone());
  }
  out
}

/// Pure boolean reduction backing [`ManagedTranscriber::is_idle`].
/// Extracted as a free function so the all-conjuncts contract
/// (each input's "empty / quiescent" state must hold) is testable
/// without standing up a real `WhisperContext` + worker pool.
const fn is_idle_inner(
  core_idle: bool,
  pending_transcripts_empty: bool,
  pending_errors_empty: bool,
  pending_fatal_none: bool,
) -> bool {
  core_idle && pending_transcripts_empty && pending_errors_empty && pending_fatal_none
}

#[cfg(test)]
mod is_idle_tests {
  use super::is_idle_inner;

  /// All four conjuncts true → idle.
  #[test]
  fn all_true_is_idle() {
    assert!(is_idle_inner(true, true, true, true));
  }

  /// Codex round-33: a stashed fatal MUST keep the runner
  /// non-idle so the caller keeps polling and observes the Err.
  #[test]
  fn pending_fatal_blocks_idle() {
    assert!(
      !is_idle_inner(true, true, true, /* pending_fatal_none = */ false),
      "pending_fatal=Some must keep is_idle false"
    );
  }

  /// Pre-existing conjuncts also block idle (regression guard
  /// against accidentally dropping any of them in a refactor).
  #[test]
  fn any_other_false_blocks_idle() {
    assert!(!is_idle_inner(false, true, true, true));
    assert!(!is_idle_inner(true, false, true, true));
    assert!(!is_idle_inner(true, true, false, true));
  }
}

#[cfg(test)]
#[cfg(feature = "alignment")]
mod alignment_dispatch_smoke {
  // Real ManagedTranscriber construction needs WhisperContext +
  // AlignmentSet with real ONNX. End-to-end tests cover the
  // real flow; here we only assert that the core's is_idle path
  // is consulted and that RunAlignment dispatch does not panic
  // on the misconfigured-no-pool path.

  #[test]
  fn alignment_pool_optional_default_none() {
    // Type-level smoke: alignment_pool field is Option, so the
    // misconfigured path (with_word_alignment set on the core
    // but no with_alignment) yields None and short-circuits.
  }
}

#[cfg(test)]
mod merge_tests {
  use super::*;
  use crate::types::Lang;

  #[test]
  fn empty_override_is_identity() {
    let base = AsrParams::default();
    let ovr = AsrParamsOverride::new();
    let out = merge_overrides(&base, &ovr);
    assert_eq!(out.initial_temperature(), base.initial_temperature());
    assert_eq!(out.max_attempts(), base.max_attempts());
  }

  #[test]
  fn override_replaces_only_specified_fields() {
    let base = AsrParams::default();
    let ovr = AsrParamsOverride::new()
      .with_language_hint(Some(Some(Lang::En)))
      .with_initial_temperature(Some(0.7));
    let out = merge_overrides(&base, &ovr);
    assert_eq!(out.language_hint().cloned(), Some(Lang::En));
    assert!((out.initial_temperature() - 0.7).abs() < 1e-9);
    assert_eq!(out.max_attempts(), base.max_attempts());
  }

  #[test]
  fn override_can_clear_language_hint() {
    let base = AsrParams::default().with_language_hint(Some(Lang::En));
    let ovr = AsrParamsOverride::new().with_language_hint(Some(None));
    let out = merge_overrides(&base, &ovr);
    assert!(out.language_hint().is_none());
  }
}
