//! Dispatch state machine — per-chunk lifecycle, in-order emission.

use alloc::collections::{BTreeMap, VecDeque};
use alloc::sync::Arc;
use alloc::vec::Vec;

use mediatime::TimeRange;

use crate::core::buffer::SampleBuffer;
use crate::core::command::{AsrParams, AsrResult, Command};
use crate::core::cut::{MergedChunk, SampleRange, SubOrigin, SubRange};
use crate::core::event::Event;
use crate::types::{ChunkId, Lang, Transcript, TranscriberError, WorkFailure};

#[allow(dead_code)] // alignment fields land in Plan C
#[derive(Debug)]
pub(crate) enum ChunkPhase {
    AwaitingAsr,
    AwaitingAlignment,
    Ready { transcript: Transcript },
    FailedReady { failure: WorkFailure },
}

#[derive(Debug)]
pub(crate) struct ChunkRecord {
    pub chunk_id: ChunkId,
    pub range: TimeRange,
    pub samples: Arc<[f32]>,
    pub sample_range: SampleRange,
    pub sub_segments: Vec<TimeRange>,
    #[allow(dead_code)] // used by alignment in Plan C
    pub sub_origins: Vec<SubOrigin>,
    pub phase: ChunkPhase,
    pub asr_result: Option<AsrResult>,
}

pub(crate) struct Dispatch {
    pub cut_pending: VecDeque<(ChunkId, MergedChunk)>,
    pub in_flight: BTreeMap<ChunkId, ChunkRecord>,
    pub next_emit_chunk_id: ChunkId,
    pub pending_commands: VecDeque<Command>,
    pub pending_events: VecDeque<Event>,
    pub word_alignment: bool,
    pub max_in_flight: usize,
    pub asr_params: AsrParams,
    /// Set true while `restart_at` is draining `cut_pending`. While
    /// true, the promotion guard `in_flight.len() < max_in_flight`
    /// is suspended (per §5.5 invariant 4 exception). Reset to
    /// false at the end of restart_at.
    pub draining_for_restart: bool,
    /// Single-slot undo for the runner's dispatch loop. Set by
    /// `unpoll_command`, consumed by the next `poll_command` (which
    /// returns the parked command first).
    pub parked_command: Option<Command>,
}

impl Dispatch {
    pub(crate) fn new(asr_params: AsrParams, word_alignment: bool, max_in_flight: usize) -> Self {
        Self {
            cut_pending: VecDeque::new(),
            in_flight: BTreeMap::new(),
            next_emit_chunk_id: ChunkId::from_raw(0),
            pending_commands: VecDeque::new(),
            pending_events: VecDeque::new(),
            word_alignment,
            max_in_flight,
            asr_params,
            draining_for_restart: false,
            parked_command: None,
        }
    }

    /// Called by `Transcriber` whenever the cut state machine emits
    /// a `MergedChunk`. Either promotes the chunk to `in_flight`
    /// immediately (and emits a `RunAsr` command) or queues it on
    /// `cut_pending` if `max_in_flight` is saturated.
    pub(crate) fn on_emit(
        &mut self,
        chunk: MergedChunk,
        chunk_id: ChunkId,
        buffer: &SampleBuffer,
    ) {
        if self.draining_for_restart || self.in_flight.len() < self.max_in_flight {
            self.promote(chunk_id, chunk, buffer);
        } else {
            self.cut_pending.push_back((chunk_id, chunk));
        }
    }

    /// Move a chunk from "just produced by Cut" or "pending" to
    /// "in_flight" by extracting its samples and queuing a
    /// `RunAsr` command. Crate-private; the trim path also calls it.
    fn promote(&mut self, chunk_id: ChunkId, chunk: MergedChunk, buffer: &SampleBuffer) {
        let samples = buffer.extract(chunk.range);
        let range = buffer.samples_to_output_range(chunk.range);
        let sub_segments: Vec<TimeRange> = chunk
            .subs
            .iter()
            .map(|s| buffer.samples_to_output_range(s.range))
            .collect();
        let sub_origins: Vec<SubOrigin> = chunk.subs.iter().map(|s| s.origin).collect();

        let record = ChunkRecord {
            chunk_id,
            range,
            samples: samples.clone(),
            sample_range: chunk.range,
            sub_segments,
            sub_origins,
            phase: ChunkPhase::AwaitingAsr,
            asr_result: None,
        };
        self.in_flight.insert(chunk_id, record);

        self.pending_commands.push_back(Command::RunAsr {
            chunk_id,
            samples,
            sample_rate: crate::time::SAMPLE_RATE_HZ,
            params: self.asr_params.clone(),
        });
    }

    /// Drain pending events to the caller in chunk-id order.
    /// Idempotent / re-entrant: stops when the head of `in_flight`
    /// is not yet `Ready` / `FailedReady`, or when `next_emit_chunk_id`
    /// is past every record in `in_flight`.
    fn flush_in_order_events(&mut self) {
        loop {
            let head_id = self.next_emit_chunk_id;
            let entry = match self.in_flight.get(&head_id) {
                Some(e) => e,
                None => break,
            };
            match &entry.phase {
                ChunkPhase::Ready { .. } | ChunkPhase::FailedReady { .. } => {}
                _ => break,
            }
            let mut record = self.in_flight.remove(&head_id).expect("just got");
            let phase = core::mem::replace(&mut record.phase, ChunkPhase::AwaitingAsr);
            let event = match phase {
                ChunkPhase::Ready { transcript } => Event::Transcript(transcript),
                ChunkPhase::FailedReady { failure } => Event::Error {
                    chunk_id: head_id,
                    error: failure,
                },
                _ => unreachable!("phase guarded above"),
            };
            self.pending_events.push_back(event);
            self.next_emit_chunk_id = ChunkId::from_raw(head_id.as_u64() + 1);
        }
    }

    /// Compute trim's low-water from `cut_pending` only — in-flight
    /// chunks have their audio in their own Arc<[f32]>s and are
    /// decoupled from the live buffer. If `cut_pending` is empty,
    /// the buffer can be trimmed all the way to its high-water
    /// (caller passes `absolute_sample_offset`).
    pub(crate) fn low_water_samples(&self, fallback_high_water: u64) -> u64 {
        self.cut_pending
            .iter()
            .map(|(_, c)| c.range.start)
            .min()
            .unwrap_or(fallback_high_water)
    }

    /// After an inject_* path, try to land any newly-eligible
    /// in-flight chunks as events, then promote pending chunks if
    /// slots have opened. The caller (`Transcriber`) must invoke
    /// `flush_in_order_events()` then `trim()` in this order on
    /// every inject path (§5.5 invariant 3).
    pub(crate) fn after_inject(&mut self, buffer: &mut SampleBuffer) {
        self.flush_in_order_events();
        // Trim the buffer to the lowest pending-chunk start.
        let low = self.low_water_samples(buffer.absolute_sample_offset());
        buffer.trim_to(low);
        // Promote pending chunks if slots are open.
        while !self.draining_for_restart
            && self.in_flight.len() < self.max_in_flight
            && !self.cut_pending.is_empty()
        {
            let (chunk_id, chunk) = self.cut_pending.pop_front().expect("just checked non-empty");
            self.promote(chunk_id, chunk, buffer);
        }
    }
}
