//! Diagnostic: read WhisperX-side emissions + trellis dumps, run
//! whispery's `get_trellis` and `backtrack_beam` against the same
//! emissions, and print cell-level diffs.
//!
//! Inputs (created by `python/dump_wx_segment.py`):
//! - `<basename>.emission.bin`: T u32 LE, V u32 LE, T*V f32 LE row-major.
//! - `<basename>.trellis.bin`:  T u32 LE, NTOK u32 LE, T*NTOK f32 LE row-major.
//! - `<basename>.json`: must contain `tokens`, `blank_id` (and other
//!   fields are unused here).
//!
//! Usage:
//!   whispery-trellis-diff <basename> [--top N]
//!
//! Where `<basename>` is e.g.
//! `tests/parity_whisperx/out/wx_seg20_dump`. The `.emission.bin`,
//! `.trellis.bin`, and `.json` companions are read in.

use std::{
  fs,
  io::Read,
  path::{Path, PathBuf},
  sync::atomic::AtomicBool,
};

use anyhow::{Context, Result, bail};
use clap::Parser;
use serde_json::Value;
use whispery::{
  Lang,
  __bench::{
    ALIGN_BEAM_WIDTH, LogProbsTV, PathPointPublic, backtrack_beam, get_trellis,
  },
};

#[derive(Parser, Debug)]
struct Args {
  /// Basename without the `.emission.bin`/`.trellis.bin`/`.json` suffix.
  /// E.g. `tests/parity_whisperx/out/wx_seg20_dump`.
  basename: PathBuf,

  /// How many of the largest-diff cells to dump.
  #[arg(long, default_value_t = 10)]
  top: usize,
}

fn read_u32_le(buf: &[u8], off: usize) -> u32 {
  u32::from_le_bytes(buf[off..off + 4].try_into().unwrap())
}

fn read_bin_2d(path: &Path) -> Result<(usize, usize, Vec<f32>)> {
  let mut f = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
  let mut buf = Vec::new();
  f.read_to_end(&mut buf)?;
  if buf.len() < 8 {
    bail!("{}: too small ({} bytes)", path.display(), buf.len());
  }
  let dim0 = read_u32_le(&buf, 0) as usize;
  let dim1 = read_u32_le(&buf, 4) as usize;
  let n = dim0 * dim1;
  if buf.len() != 8 + 4 * n {
    bail!(
      "{}: header says {}*{} = {} cells = {} bytes; buffer is {} bytes (want {})",
      path.display(),
      dim0,
      dim1,
      n,
      4 * n,
      buf.len() - 8,
      4 * n
    );
  }
  let mut data = Vec::with_capacity(n);
  for i in 0..n {
    let off = 8 + 4 * i;
    let bytes: [u8; 4] = buf[off..off + 4].try_into().unwrap();
    data.push(f32::from_le_bytes(bytes));
  }
  Ok((dim0, dim1, data))
}

fn main() -> Result<()> {
  let args = Args::parse();
  let emission_path = args.basename.with_extension("emission.bin");
  let trellis_path = args.basename.with_extension("trellis.bin");
  let json_path = args.basename.with_extension("json");

  // Python's json dumps `float('inf')` as the JSON-extension literal
  // `Infinity` (and `-Infinity`, `NaN`), which `serde_json` rejects.
  // We don't read those fields here — but the fields we DO read sit
  // beyond them in the document, so just substitute textual sentinels
  // before parsing. This is a one-shot diagnostic; correctness on
  // those fields doesn't matter to us.
  let raw = fs::read_to_string(&json_path)
    .with_context(|| format!("read {}", json_path.display()))?;
  let cleaned = raw
    .replace("-Infinity", "-1e308")
    .replace("Infinity", "1e308")
    .replace("NaN", "0.0");
  let json: Value = serde_json::from_str(&cleaned)
    .with_context(|| format!("parse {}", json_path.display()))?;
  let blank_id = json
    .get("blank_id")
    .and_then(Value::as_u64)
    .context("blank_id missing")? as u32;
  let tokens: Vec<i32> = json
    .get("tokens")
    .and_then(Value::as_array)
    .context("tokens missing")?
    .iter()
    .map(|v| v.as_i64().unwrap_or(0) as i32)
    .collect();

  let (em_t, em_v, em_data) = read_bin_2d(&emission_path)?;
  println!("emission: T={} V={}", em_t, em_v);

  let (tr_t, tr_n, tr_data) = read_bin_2d(&trellis_path)?;
  println!("WhisperX trellis: T={} NTOK={}", tr_t, tr_n);

  if tr_t != em_t {
    bail!("T mismatch: emission T={} vs trellis T={}", em_t, tr_t);
  }
  if tr_n != tokens.len() {
    bail!("NTOK mismatch: trellis {} vs tokens {}", tr_n, tokens.len());
  }

  let log_probs = LogProbsTV {
    t: em_t,
    v: em_v,
    data: em_data,
  };

  let abort = AtomicBool::new(false);
  let lang = Lang::En;
  let wy_trellis = get_trellis(&log_probs, &tokens, blank_id, &abort, &lang)
    .map_err(|e| anyhow::anyhow!("whispery get_trellis failed: {:?}", e))?;
  println!("whispery trellis: cells={}", wy_trellis.len());
  if wy_trellis.len() != tr_data.len() {
    bail!(
      "size mismatch: whispery {} vs whisperx {}",
      wy_trellis.len(),
      tr_data.len()
    );
  }

  // Cell-by-cell comparison. Treat -inf/+inf as exact-match category;
  // they're meant to be sentinels and any mismatch there is a structural
  // bug not a numerical one.
  let mut max_abs_diff = 0.0_f32;
  let mut sum_abs_diff = 0.0_f64;
  let mut count_finite_pairs = 0_usize;
  let mut count_gt_1e_3 = 0_usize;
  let mut count_gt_1e_2 = 0_usize;
  let mut count_gt_1e_1 = 0_usize;
  let mut count_inf_mismatch = 0_usize;
  let mut top: Vec<(f32, usize, usize, f32, f32)> = Vec::new(); // (abs_diff, t, j, wy, wx)

  for ti in 0..tr_t {
    for ji in 0..tr_n {
      let idx = ti * tr_n + ji;
      let wy = wy_trellis[idx];
      let wx = tr_data[idx];
      let both_finite = wy.is_finite() && wx.is_finite();
      if !both_finite {
        // -inf vs -inf or +inf vs +inf is OK; otherwise mark a sentinel mismatch.
        if wy.is_finite() != wx.is_finite() || (wy.is_infinite() && wx.is_infinite() && wy != wx) {
          count_inf_mismatch += 1;
        }
        continue;
      }
      let d = (wy - wx).abs();
      if d > max_abs_diff {
        max_abs_diff = d;
      }
      sum_abs_diff += d as f64;
      count_finite_pairs += 1;
      if d > 1e-3 {
        count_gt_1e_3 += 1;
      }
      if d > 1e-2 {
        count_gt_1e_2 += 1;
      }
      if d > 1e-1 {
        count_gt_1e_1 += 1;
      }
      if top.len() < args.top {
        top.push((d, ti, ji, wy, wx));
      } else {
        // Replace the smallest in top with this if larger.
        let (mn_idx, mn_d) = top
          .iter()
          .enumerate()
          .min_by(|(_, a), (_, b)| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
          .map(|(i, x)| (i, x.0))
          .unwrap();
        if d > mn_d {
          top[mn_idx] = (d, ti, ji, wy, wx);
        }
      }
    }
  }

  let mean_abs_diff = if count_finite_pairs > 0 {
    sum_abs_diff / (count_finite_pairs as f64)
  } else {
    0.0
  };

  println!();
  println!("=== TRELLIS CELL DIFF (finite pairs only) ===");
  println!("finite_pair_count = {}", count_finite_pairs);
  println!("inf_sentinel_mismatch_count = {}", count_inf_mismatch);
  println!("max |diff| = {:.6e}", max_abs_diff);
  println!("mean |diff| = {:.6e}", mean_abs_diff);
  println!("count > 1e-3 = {}", count_gt_1e_3);
  println!("count > 1e-2 = {}", count_gt_1e_2);
  println!("count > 1e-1 = {}", count_gt_1e_1);

  println!();
  println!("=== TOP-{} DIFFERING CELLS ===", args.top);
  top.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
  for (d, ti, ji, wy, wx) in &top {
    println!(
      "  (t={:>5}, j={:>5})  wy={:.6}  wx={:.6}  |diff|={:.6e}",
      ti, ji, wy, wx, d
    );
  }

  // Also run whispery's beam on its own trellis and on whisperx's
  // trellis, and on whisperx's emissions either way. Then dump the
  // first-divergence frame between whispery's path and the WhisperX
  // path stored in JSON.
  println!();
  println!("=== BEAM PATH DIFF ===");
  let wy_path = backtrack_beam(
    &wy_trellis,
    &log_probs,
    &tokens,
    blank_id,
    ALIGN_BEAM_WIDTH,
    &abort,
    &lang,
  )
  .map_err(|e| anyhow::anyhow!("whispery backtrack_beam (wy trellis): {:?}", e))?;
  let wy_path_on_wx_trellis = backtrack_beam(
    &tr_data,
    &log_probs,
    &tokens,
    blank_id,
    ALIGN_BEAM_WIDTH,
    &abort,
    &lang,
  )
  .map_err(|e| anyhow::anyhow!("whispery backtrack_beam (wx trellis): {:?}", e))?;

  // Read WhisperX path from JSON.
  let wx_path_json = json
    .get("path")
    .and_then(Value::as_array)
    .context("path missing in JSON")?;
  let wx_path: Vec<PathPointPublic> = wx_path_json
    .iter()
    .map(|p| PathPointPublic {
      token_index: p.get("token_index").and_then(Value::as_u64).unwrap() as usize,
      time_index: p.get("time_index").and_then(Value::as_u64).unwrap() as usize,
      score: p.get("score").and_then(Value::as_f64).unwrap_or(0.0) as f32,
    })
    .collect();

  println!("whispery path on wy trellis: len={}", wy_path.len());
  println!("whispery path on wx trellis: len={}", wy_path_on_wx_trellis.len());
  println!("whisperx path: len={}", wx_path.len());

  // Compare wy_path on wy_trellis vs wx_path: where do they first diverge?
  let n_min = wy_path.len().min(wx_path.len());
  let mut first_diverge = None;
  for i in 0..n_min {
    if wy_path[i].token_index != wx_path[i].token_index
      || wy_path[i].time_index != wx_path[i].time_index
    {
      first_diverge = Some(i);
      break;
    }
  }
  match first_diverge {
    Some(i) => {
      println!(
        "wy(wy-trellis) vs wx FIRST DIVERGE at i={}: wy=(t={}, j={}) wx=(t={}, j={})",
        i, wy_path[i].time_index, wy_path[i].token_index, wx_path[i].time_index, wx_path[i].token_index
      );
      // Dump 8 around it on each side
      let lo = i.saturating_sub(4);
      let hi = (i + 6).min(n_min);
      for k in lo..hi {
        println!(
          "  {:>4}: wy=(t={:>5}, j={:>5}, score={:.6})  wx=(t={:>5}, j={:>5}, score={:.6})",
          k,
          wy_path[k].time_index,
          wy_path[k].token_index,
          wy_path[k].score,
          wx_path[k].time_index,
          wx_path[k].token_index,
          wx_path[k].score,
        );
      }
    }
    None if wy_path.len() == wx_path.len() => println!("wy(wy-trellis) vs wx: IDENTICAL"),
    None => println!(
      "wy(wy-trellis) vs wx: same prefix, differ in length ({} vs {})",
      wy_path.len(),
      wx_path.len()
    ),
  }

  // Compare wy_path-on-wx-trellis vs wx_path: same, but driven by
  // identical trellis values. If this matches whisperx, the trellis
  // values aren't the issue — it's the beam/order of stay-vs-change.
  let n_min2 = wy_path_on_wx_trellis.len().min(wx_path.len());
  let mut first_diverge2 = None;
  for i in 0..n_min2 {
    if wy_path_on_wx_trellis[i].token_index != wx_path[i].token_index
      || wy_path_on_wx_trellis[i].time_index != wx_path[i].time_index
    {
      first_diverge2 = Some(i);
      break;
    }
  }
  match first_diverge2 {
    Some(i) => {
      println!(
        "wy(WX-trellis) vs wx FIRST DIVERGE at i={}: wy=(t={}, j={}) wx=(t={}, j={})",
        i,
        wy_path_on_wx_trellis[i].time_index,
        wy_path_on_wx_trellis[i].token_index,
        wx_path[i].time_index,
        wx_path[i].token_index
      );
      let lo = i.saturating_sub(4);
      let hi = (i + 6).min(n_min2);
      for k in lo..hi {
        println!(
          "  {:>4}: wy=(t={:>5}, j={:>5}, score={:.6})  wx=(t={:>5}, j={:>5}, score={:.6})",
          k,
          wy_path_on_wx_trellis[k].time_index,
          wy_path_on_wx_trellis[k].token_index,
          wy_path_on_wx_trellis[k].score,
          wx_path[k].time_index,
          wx_path[k].token_index,
          wx_path[k].score,
        );
      }
    }
    None if wy_path_on_wx_trellis.len() == wx_path.len() => {
      println!("wy(WX-trellis) vs wx: IDENTICAL — beam logic matches given identical trellis");
    }
    None => println!(
      "wy(WX-trellis) vs wx: same prefix, differ in length ({} vs {})",
      wy_path_on_wx_trellis.len(),
      wx_path.len()
    ),
  }

  Ok(())
}
