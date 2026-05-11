//! Script → [`Lang`] mapping.
//!
//! Pure-Rust per-character script classification used by the
//! script-dispatch pass to split a single mixed-script Whisper
//! segment into language-tagged runs. The Q1 disambiguation rules
//! that drive Han / Latin assignment live here; `script_dispatch`
//! is the iteration shell that calls into them.
//!
//! No FFI, no feature gating — the whole surface is always
//! available so non-runner callers (parity tests, offline tooling)
//! can re-use the mapping without pulling in the `whispercpp`
//! dependency.

use unicode_script::Script;

use crate::types::Lang;

/// Per-segment context flags collected by walking the segment's
/// characters once. Drives the Han disambiguation: a Han ideograph
/// in a segment that also contains kana is Japanese; one alongside
/// Hangul is Korean; otherwise it falls through to Chinese.
///
/// Cheap to compute (one linear pass over `segment.text()`) and
/// re-used for every character of that same segment, so the caller
/// builds it once per segment rather than once per character.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SegmentContext {
  /// `true` when the segment contains at least one Hiragana or
  /// Katakana character. Forces every Han ideograph in this
  /// segment to map to [`Lang::Ja`].
  has_kana: bool,

  /// `true` when the segment contains at least one Hangul
  /// character. Forces every Han ideograph in this segment to
  /// map to [`Lang::Ko`] (only when `has_kana` is `false`).
  has_hangul: bool,
}

impl SegmentContext {
  /// Construct from the two flags directly. Most callers prefer
  /// [`Self::from_text`].
  #[must_use]
  pub const fn new(has_kana: bool, has_hangul: bool) -> Self {
    Self {
      has_kana,
      has_hangul,
    }
  }

  /// Whether the segment contains at least one kana character.
  #[must_use]
  pub const fn has_kana(&self) -> bool {
    self.has_kana
  }

  /// Whether the segment contains at least one Hangul character.
  #[must_use]
  pub const fn has_hangul(&self) -> bool {
    self.has_hangul
  }

  /// Build the per-segment context flags by scanning every
  /// character of `text` once. Subsequent per-character mapping
  /// calls re-use the result.
  #[must_use]
  pub fn from_text(text: &str) -> Self {
    let mut ctx = Self::default();
    for ch in text.chars() {
      match Script::from(ch) {
        Script::Hiragana | Script::Katakana => ctx.has_kana = true,
        Script::Hangul => ctx.has_hangul = true,
        _ => {}
      }
      // Early-out: both flags set, no further information to gain.
      if ctx.has_kana && ctx.has_hangul {
        break;
      }
    }
    ctx
  }
}

/// Whether `lang` writes primarily in the Latin script.
///
/// Used by [`script_to_lang`] to decide whether a Latin-script
/// character can carry the `state_lang` hint. Languages outside
/// this set (Zh, Ja, Ko, Ar, Ru, He, Hi, ...) fall back to
/// [`Lang::En`] for stray Latin characters — typically loanwords or
/// brand names, where defaulting to English is the least-wrong
/// choice when the surrounding state language doesn't share the
/// script.
///
/// True when `lang` is a no-space CJK language whose normalizer
/// handles embedded Latin per-character (matching WhisperX's
/// `LANGUAGES_WITHOUT_SPACES` contract). Retained for callers
/// that may want this classification independent of script
/// dispatch; the dispatcher itself no longer uses it (Latin
/// chars under CJK `state_lang` route to `Lang::En` to preserve
/// code-switches).
#[must_use]
pub const fn is_no_space_cjk_lang(lang: &Lang) -> bool {
  matches!(lang, Lang::Ja | Lang::Zh | Lang::Yue | Lang::Ko)
}

/// The set is intentionally inclusive: every CJK / RTL / Indic /
/// Cyrillic-leaning language returns `false`. Adding a new Latin
/// language is one match arm here.
#[must_use]
pub const fn is_latin_script_lang(lang: &Lang) -> bool {
  matches!(
    lang,
    Lang::En
      | Lang::Es
      | Lang::Fr
      | Lang::De
      | Lang::It
      | Lang::Pt
      | Lang::Nl
      | Lang::Sv
      | Lang::No
      | Lang::Da
      | Lang::Fi
      | Lang::Pl
      | Lang::Ro
      | Lang::Cs
      | Lang::Hu
      | Lang::Tr
      | Lang::Vi
      | Lang::Ca
      | Lang::Sk
      | Lang::Sl
      | Lang::Hr
      | Lang::Lt
      | Lang::Lv
      | Lang::Et
      | Lang::Id
      | Lang::Ms
      | Lang::Sw
      | Lang::Af
      | Lang::Eu
      | Lang::Gl
      | Lang::Cy
      | Lang::Is
      | Lang::Mt
      | Lang::Sq
      | Lang::Tl
      | Lang::Haw
      | Lang::Ln
      | Lang::Ha
      | Lang::Yo
      | Lang::So
      | Lang::Oc
      | Lang::Br
      | Lang::Lb
      | Lang::Nn
      | Lang::Fo
      | Lang::Ht
      | Lang::Tk
      | Lang::Jw
      | Lang::Su
      | Lang::Mg
      | Lang::Mi
      | Lang::Sn
      | Lang::La
  )
}

/// Result of classifying a single character against the
/// per-segment script context.
///
/// `Carry` means "this character has no script signal of its own
/// (digits, punctuation, whitespace, or a script the dispatcher
/// can't disambiguate without more context); reuse whatever
/// language the preceding run is using." The dispatcher resolves
/// `Carry` against its own state — at run start, leading carries
/// fold into the first concrete classification.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CharClass {
  /// Character pinned to a specific language by its script (and,
  /// for Han, the segment context).
  Lang(Lang),
  /// No standalone signal — extend the surrounding run.
  Carry,
}

/// Map a single character to a language, given the segment-level
/// kana/Hangul context and an optional `state_lang` hint from the
/// transcriber.
///
/// Implements the Q1 rules:
///
/// * Hiragana / Katakana → [`Lang::Ja`]
/// * Hangul              → [`Lang::Ko`]
/// * Han                 → [`Lang::Ja`] when the segment contains
///   kana; else [`Lang::Ko`] when it contains Hangul; else
///   [`Lang::Zh`].
/// * Latin               → `state_lang` when it is set and is a
///   Latin-script language ([`is_latin_script_lang`]); else
///   [`Lang::En`] as a defensive fallback.
/// * Cyrillic, Arabic, and any other concrete script             →
///   `state_lang` when it is set; else `Carry` (let the
///   surrounding run win).
/// * Digits, punctuation, whitespace, `Common`, `Inherited`,
///   `Unknown`                                                   →
///   `Carry`.
#[must_use]
pub fn script_to_lang(ch: char, ctx: SegmentContext, state_lang: Option<&Lang>) -> CharClass {
  match Script::from(ch) {
    Script::Hiragana | Script::Katakana => CharClass::Lang(Lang::Ja),
    Script::Hangul => CharClass::Lang(Lang::Ko),
    Script::Han => {
      if ctx.has_kana {
        CharClass::Lang(Lang::Ja)
      } else if ctx.has_hangul {
        CharClass::Lang(Lang::Ko)
      } else {
        CharClass::Lang(Lang::Zh)
      }
    }
    Script::Latin => match state_lang {
      Some(l) if is_latin_script_lang(l) => CharClass::Lang(l.clone()),
      // route Latin chars to
      // `Lang::En` even when `state_lang` is a no-space CJK
      // language. Round 9 had this fold Latin INTO the CJK
      // run so embedded loanwords ("USAで", "Python") could
      // ride the CJK normalizer's per-char Latin handling,
      // but that rule applied globally — it suppressed
      // legitimate code-switches like `"hello 你好"` (detected
      // as Zh) into a single Zh run, sending the Latin span
      // through the wrong aligner / normalizer and silently
      // contradicting the per-language code-switch claim. The
      // dispatcher now produces SEPARATE En + CJK runs for
      // mixed input; callers who only have a CJK aligner
      // registered can use `AlignmentFallback::Any` (or
      // register `Lang::En` against the same aligner) to
      // route the Latin span back to a CJK normalizer if
      // they prefer the loanword behaviour.
      _ => CharClass::Lang(Lang::En),
    },
    // `Common` covers ASCII digits, punctuation, whitespace, and
    // most symbols. `Inherited` covers combining marks. `Unknown`
    // covers private-use / unassigned codepoints. None of those
    // carry a language signal — fold them into the surrounding
    // run.
    Script::Common | Script::Inherited | Script::Unknown => CharClass::Carry,
    // Concrete scripts the dispatcher can't disambiguate without
    // a state-language hint. Cyrillic could be Ru / Uk / Bg / Sr;
    // Arabic could be Ar / Fa / Ur / Ps; etc. Honour the hint
    // when it is set, otherwise carry — the run's preceding
    // language is the best heuristic we have.
    _ => match state_lang {
      Some(l) => CharClass::Lang(l.clone()),
      None => CharClass::Carry,
    },
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn segment_context_pure_english() {
    let ctx = SegmentContext::from_text("hello world");
    assert!(!ctx.has_kana);
    assert!(!ctx.has_hangul);
  }

  #[test]
  fn segment_context_pure_chinese() {
    let ctx = SegmentContext::from_text("你好世界");
    assert!(!ctx.has_kana);
    assert!(!ctx.has_hangul);
  }

  #[test]
  fn segment_context_jp_with_kana() {
    let ctx = SegmentContext::from_text("これは日本語です");
    assert!(ctx.has_kana);
    assert!(!ctx.has_hangul);
  }

  #[test]
  fn segment_context_ko_with_hangul() {
    let ctx = SegmentContext::from_text("안녕하세요");
    assert!(!ctx.has_kana);
    assert!(ctx.has_hangul);
  }

  #[test]
  fn hiragana_maps_to_ja() {
    let ctx = SegmentContext::default();
    assert_eq!(script_to_lang('あ', ctx, None), CharClass::Lang(Lang::Ja),);
  }

  #[test]
  fn katakana_maps_to_ja() {
    let ctx = SegmentContext::default();
    assert_eq!(script_to_lang('カ', ctx, None), CharClass::Lang(Lang::Ja),);
  }

  #[test]
  fn hangul_maps_to_ko() {
    let ctx = SegmentContext::default();
    assert_eq!(script_to_lang('한', ctx, None), CharClass::Lang(Lang::Ko),);
  }

  #[test]
  fn han_default_is_zh() {
    let ctx = SegmentContext::default();
    assert_eq!(script_to_lang('語', ctx, None), CharClass::Lang(Lang::Zh),);
  }

  #[test]
  fn han_with_kana_context_is_ja() {
    let ctx = SegmentContext {
      has_kana: true,
      has_hangul: false,
    };
    assert_eq!(script_to_lang('語', ctx, None), CharClass::Lang(Lang::Ja),);
  }

  #[test]
  fn han_with_hangul_context_is_ko() {
    let ctx = SegmentContext {
      has_kana: false,
      has_hangul: true,
    };
    assert_eq!(script_to_lang('語', ctx, None), CharClass::Lang(Lang::Ko),);
  }

  #[test]
  fn han_kana_beats_hangul_when_both_present() {
    let ctx = SegmentContext {
      has_kana: true,
      has_hangul: true,
    };
    assert_eq!(script_to_lang('語', ctx, None), CharClass::Lang(Lang::Ja),);
  }

  #[test]
  fn latin_no_hint_defaults_to_en() {
    let ctx = SegmentContext::default();
    assert_eq!(script_to_lang('h', ctx, None), CharClass::Lang(Lang::En),);
  }

  #[test]
  fn latin_with_es_hint_uses_es() {
    let ctx = SegmentContext::default();
    assert_eq!(
      script_to_lang('h', ctx, Some(&Lang::Es)),
      CharClass::Lang(Lang::Es),
    );
  }

  /// a Latin char under a
  /// CJK `state_lang` must route to `Lang::En` so genuine
  /// code-switches (e.g. `"hello 你好"` detected as Zh) split
  /// into separate per-language runs instead of collapsing
  /// into one CJK run that would route the Latin span through
  /// the wrong aligner / normalizer. Round 9 had the opposite
  /// rule (Latin folded into the active CJK run for embedded
  /// loanword handling); the global behaviour suppressed
  /// code-switch alignment everywhere it fired.
  #[test]
  fn latin_with_zh_hint_routes_to_en() {
    let ctx = SegmentContext::default();
    assert_eq!(
      script_to_lang('h', ctx, Some(&Lang::Zh)),
      CharClass::Lang(Lang::En),
    );
  }

  #[test]
  fn latin_with_ja_hint_routes_to_en() {
    let ctx = SegmentContext::default();
    assert_eq!(
      script_to_lang('U', ctx, Some(&Lang::Ja)),
      CharClass::Lang(Lang::En),
    );
  }

  #[test]
  fn latin_with_ko_hint_routes_to_en() {
    let ctx = SegmentContext::default();
    assert_eq!(
      script_to_lang('K', ctx, Some(&Lang::Ko)),
      CharClass::Lang(Lang::En),
    );
  }

  #[test]
  fn latin_with_no_state_lang_falls_back_to_en() {
    let ctx = SegmentContext::default();
    assert_eq!(script_to_lang('h', ctx, None), CharClass::Lang(Lang::En),);
  }

  #[test]
  fn punctuation_carries() {
    let ctx = SegmentContext::default();
    assert_eq!(script_to_lang(',', ctx, None), CharClass::Carry);
    assert_eq!(script_to_lang('.', ctx, None), CharClass::Carry);
    assert_eq!(script_to_lang('?', ctx, None), CharClass::Carry);
  }

  #[test]
  fn whitespace_carries() {
    let ctx = SegmentContext::default();
    assert_eq!(script_to_lang(' ', ctx, None), CharClass::Carry);
    assert_eq!(script_to_lang('\t', ctx, None), CharClass::Carry);
  }

  #[test]
  fn digits_carry() {
    let ctx = SegmentContext::default();
    assert_eq!(script_to_lang('0', ctx, None), CharClass::Carry);
    assert_eq!(script_to_lang('9', ctx, None), CharClass::Carry);
  }

  #[test]
  fn cyrillic_with_ru_hint_uses_ru() {
    let ctx = SegmentContext::default();
    assert_eq!(
      script_to_lang('п', ctx, Some(&Lang::Ru)),
      CharClass::Lang(Lang::Ru),
    );
  }

  #[test]
  fn cyrillic_without_hint_carries() {
    let ctx = SegmentContext::default();
    assert_eq!(script_to_lang('п', ctx, None), CharClass::Carry);
  }

  #[test]
  fn is_latin_script_lang_known_set() {
    assert!(is_latin_script_lang(&Lang::En));
    assert!(is_latin_script_lang(&Lang::Es));
    assert!(is_latin_script_lang(&Lang::Vi));
    assert!(is_latin_script_lang(&Lang::Tr));
    assert!(!is_latin_script_lang(&Lang::Zh));
    assert!(!is_latin_script_lang(&Lang::Ja));
    assert!(!is_latin_script_lang(&Lang::Ko));
    assert!(!is_latin_script_lang(&Lang::Ar));
    assert!(!is_latin_script_lang(&Lang::Ru));
    assert!(!is_latin_script_lang(&Lang::He));
  }
}
