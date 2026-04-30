//! Steps 5-6 of the alignment algorithm: CTC lattice + Viterbi.

use alloc::string::String;
use alloc::vec::Vec;

use crate::runner::aligner::algorithm::encode::LogProbsTV;
use crate::types::{AlignmentFailureKind, Lang, WorkFailure};

/// Result of CTC Viterbi alignment.
#[derive(Debug)]
pub(crate) struct ViterbiPath {
    /// Length-T vector of state indices in the (2|Y|+1)-wide lattice.
    /// State `2k` is the blank between y_{k-1} and y_k (k=0 is the
    /// leading blank); state `2k+1` is symbol y_k itself.
    pub state_per_frame: Vec<usize>,
    /// Convenience: the original token sequence Y (vocab ids).
    /// State `2k+1` corresponds to `tokens[k]`; state `2k` is blank.
    pub tokens: Vec<u32>,
}

/// Run CTC Viterbi alignment of `tokens` (Y) to `log_probs` (T, V).
///
/// `blank_id` is the CTC blank-token vocab id (read at
/// `Aligner::from_paths` time). `tokens` is the tokenised
/// normalised text from step 2.
///
/// Returns the highest-probability monotonic path through the
/// (2|Y|+1)-state CTC lattice. The state-per-frame vector lets
/// the next stage (step 7) walk frame-by-frame and accumulate
/// per-word state.
///
/// Returns `WorkFailure::AlignmentFailed { kind: NoAlignmentPath,
/// .. }` if the lattice is empty (T < 2|Y|+1, i.e., the audio is
/// too short to fit the symbol sequence even with no repeats).
pub(crate) fn ctc_viterbi(
    log_probs: &LogProbsTV,
    tokens: &[u32],
    blank_id: u32,
    language: &Lang,
) -> Result<ViterbiPath, WorkFailure> {
    let t = log_probs.t;
    let m = tokens.len();
    if m == 0 {
        return Err(WorkFailure::AlignmentFailed {
            kind: AlignmentFailureKind::NoAlignmentPath,
            message: String::from("token sequence is empty"),
            language: language.clone(),
        });
    }
    let n_states = 2 * m + 1;

    // CTC requires at least one frame per (token + blank gap).
    // Conservative lower bound: T >= 2*m + 1 (one frame per state).
    // In practice models emit ~50 frames/sec @ 16 kHz, so audio of
    // length << m * 20 ms is too short.
    if t < n_states {
        return Err(WorkFailure::AlignmentFailed {
            kind: AlignmentFailureKind::NoAlignmentPath,
            message: alloc::format!(
                "audio too short: T={} frames < required {} states",
                t,
                n_states
            ),
            language: language.clone(),
        });
    }

    // Lattice DP. dp[state] = best log-prob to reach `state` at
    // current `t`; back[t][state] = predecessor state at t-1.
    let mut dp_prev = alloc::vec![f32::NEG_INFINITY; n_states];
    let mut dp_curr = alloc::vec![f32::NEG_INFINITY; n_states];
    let mut back: Vec<Vec<usize>> = (0..t).map(|_| alloc::vec![usize::MAX; n_states]).collect();

    // State id helpers:
    //   state 2k     => blank
    //   state 2k+1   => tokens[k]
    let state_token = |state: usize| -> u32 {
        if state % 2 == 0 {
            blank_id
        } else {
            tokens[state / 2]
        }
    };

    // Initialise t=0: only states 0 (leading blank) and 1 (y_0) are
    // reachable.
    dp_prev[0] = log_probs.at(0, blank_id as usize);
    if n_states >= 2 {
        dp_prev[1] = log_probs.at(0, tokens[0] as usize);
    }

    for t_idx in 1..t {
        for s in 0..n_states {
            let emit = log_probs.at(t_idx, state_token(s) as usize);
            // Predecessors of state s:
            //   - self-loop:        s        (same symbol)
            //   - one step:         s - 1    (always valid if s > 0)
            //   - two steps:        s - 2    (only if s is non-blank
            //                                 AND tokens[s/2] != tokens[(s-2)/2],
            //                                 to avoid skipping a
            //                                 needed blank between
            //                                 same-symbol repeats)
            let mut best = f32::NEG_INFINITY;
            let mut best_pred = usize::MAX;

            // Self-loop.
            if dp_prev[s] > best {
                best = dp_prev[s];
                best_pred = s;
            }
            // One step.
            if s >= 1 && dp_prev[s - 1] > best {
                best = dp_prev[s - 1];
                best_pred = s - 1;
            }
            // Two steps (skip a blank). Only legal for non-blank
            // state s where s >= 2 AND tokens[s/2] != tokens[(s-2)/2].
            if s >= 2
                && s % 2 == 1
                && tokens[s / 2] != tokens[(s - 2) / 2]
                && dp_prev[s - 2] > best
            {
                best = dp_prev[s - 2];
                best_pred = s - 2;
            }

            dp_curr[s] = best + emit;
            back[t_idx][s] = best_pred;
        }
        core::mem::swap(&mut dp_prev, &mut dp_curr);
        for slot in dp_curr.iter_mut() {
            *slot = f32::NEG_INFINITY;
        }
    }

    // The valid end states are the last symbol (n_states-2) and the
    // trailing blank (n_states-1).
    let end_a = n_states - 2;
    let end_b = n_states - 1;
    let final_state = if dp_prev[end_b] >= dp_prev[end_a] {
        end_b
    } else {
        end_a
    };
    if !dp_prev[final_state].is_finite() {
        return Err(WorkFailure::AlignmentFailed {
            kind: AlignmentFailureKind::NoAlignmentPath,
            message: alloc::format!(
                "no finite-probability path from t=0 to T={}; final dp = {:?}",
                t,
                dp_prev[final_state]
            ),
            language: language.clone(),
        });
    }

    // Backtrack.
    let mut state_per_frame = alloc::vec![0_usize; t];
    state_per_frame[t - 1] = final_state;
    let mut s = final_state;
    for t_idx in (1..t).rev() {
        let pred = back[t_idx][s];
        if pred == usize::MAX {
            return Err(WorkFailure::AlignmentFailed {
                kind: AlignmentFailureKind::NoAlignmentPath,
                message: alloc::format!(
                    "backtrack hit dead-end at t={}, state={}",
                    t_idx,
                    s
                ),
                language: language.clone(),
            });
        }
        state_per_frame[t_idx - 1] = pred;
        s = pred;
    }

    Ok(ViterbiPath {
        state_per_frame,
        tokens: tokens.to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Lang;

    fn lp(t: usize, v: usize, vals: Vec<f32>) -> LogProbsTV {
        assert_eq!(vals.len(), t * v);
        LogProbsTV { t, v, data: vals }
    }

    #[test]
    fn empty_tokens_errors() {
        let log_probs = lp(5, 3, alloc::vec![0.0; 15]);
        let err = ctc_viterbi(&log_probs, &[], 0, &Lang::En).unwrap_err();
        assert!(matches!(
            err,
            WorkFailure::AlignmentFailed {
                kind: AlignmentFailureKind::NoAlignmentPath,
                ..
            }
        ));
    }

    #[test]
    fn audio_too_short_errors() {
        // 1 token => need 2*1+1 = 3 states; T=2 is too short.
        let log_probs = lp(2, 3, alloc::vec![0.0; 6]);
        let err = ctc_viterbi(&log_probs, &[1], 0, &Lang::En).unwrap_err();
        assert!(matches!(
            err,
            WorkFailure::AlignmentFailed {
                kind: AlignmentFailureKind::NoAlignmentPath,
                ..
            }
        ));
    }

    #[test]
    fn simple_two_token_path() {
        // Tokens = [1, 2]; blank = 0. Vocab size 3.
        // T = 5 frames, with synthetic log-probs that strongly favour
        // [blank, 1, blank, 2, blank].
        let mut data = alloc::vec![-100.0_f32; 5 * 3];
        // Frame 0: prefer blank.
        data[0 * 3 + 0] = -0.1;
        data[0 * 3 + 1] = -100.0;
        data[0 * 3 + 2] = -100.0;
        // Frame 1: prefer token 1.
        data[1 * 3 + 0] = -100.0;
        data[1 * 3 + 1] = -0.1;
        data[1 * 3 + 2] = -100.0;
        // Frame 2: prefer blank.
        data[2 * 3 + 0] = -0.1;
        data[2 * 3 + 1] = -100.0;
        data[2 * 3 + 2] = -100.0;
        // Frame 3: prefer token 2.
        data[3 * 3 + 0] = -100.0;
        data[3 * 3 + 1] = -100.0;
        data[3 * 3 + 2] = -0.1;
        // Frame 4: prefer blank.
        data[4 * 3 + 0] = -0.1;
        data[4 * 3 + 1] = -100.0;
        data[4 * 3 + 2] = -100.0;

        let log_probs = lp(5, 3, data);
        let path = ctc_viterbi(&log_probs, &[1, 2], 0, &Lang::En).expect("path");
        // Expected state sequence: [0, 1, 2, 3, 4]
        assert_eq!(path.state_per_frame, alloc::vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn repeated_token_requires_blank_between() {
        // tokens = [1, 1]; without an intervening blank, CTC must
        // pass through state 2 (the blank between the two 1s).
        // n_states = 5: blank, 1, blank, 1, blank.
        let mut data = alloc::vec![-100.0_f32; 6 * 3];
        // Frames slightly favour blank then 1 then blank then 1
        // then 1 (repeat) then blank.
        for t in 0..6 {
            data[t * 3 + 0] = -1.0; // blank
            data[t * 3 + 1] = -1.5; // 1
        }
        // Strong preference for token 1 at frames 1, 3.
        data[1 * 3 + 1] = -0.1;
        data[3 * 3 + 1] = -0.1;
        let log_probs = lp(6, 3, data);
        let path = ctc_viterbi(&log_probs, &[1, 1], 0, &Lang::En).expect("path");
        // The path must visit state 2 (the inter-token blank) before
        // state 3 (the second token).
        let visited_2 = path.state_per_frame.contains(&2);
        let visited_3 = path.state_per_frame.contains(&3);
        assert!(visited_2 && visited_3);
    }
}
