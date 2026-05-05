//! `Lang` — typed enum over whisper.cpp's supported languages, with
//! an `Other(SmolStr)` escape hatch for unknown ISO codes.

use smol_str::SmolStr;

/// Language code. Marked `#[non_exhaustive]` so new variants can be
/// added when whisper.cpp adds languages without forcing a
/// semver-major bump; carries an `Other(SmolStr)` variant so unknown
/// ISO codes flowing in from whisper's auto-detect don't fail an
/// indexing run.
///
/// **Canonicalisation invariant.** [`Lang::from_iso639_1`] maps known
/// codes to named variants and never produces `Other` for an
/// enum-known code. This keeps structural `PartialEq`/`Hash` correct:
/// `Lang::En != Lang::Other("en")` is fine because no API path
/// constructs `Lang::Other("en")`.
///
/// **Serde wire format.** Lowercase ISO-639-1 strings: `"en"`,
/// `"yue"`, etc. (Codex round-34: a previous `derive(Serialize,
/// Deserialize)` produced Rust variant names like `"En"` and
/// `{"Other":"xx"}`, which contradicted documented config shapes
/// and made human-edited configs brittle. The custom impls
/// below canonicalise through [`Lang::from_iso639_1`] /
/// [`Lang::as_str`] so the in-memory representation stays as-is
/// while the wire format matches the docs.)
#[non_exhaustive]
#[allow(missing_docs)] // variants are ISO 639-1 codes; self-documenting by name
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Lang {
  En,
  Zh,
  De,
  Es,
  Ru,
  Ko,
  Fr,
  Ja,
  Pt,
  Tr,
  Pl,
  Ca,
  Nl,
  Ar,
  Sv,
  It,
  Id,
  Hi,
  Fi,
  Vi,
  He,
  Uk,
  El,
  Ms,
  Cs,
  Ro,
  Da,
  Hu,
  Ta,
  No,
  Th,
  Ur,
  Hr,
  Bg,
  Lt,
  La,
  Mi,
  Ml,
  Cy,
  Sk,
  Te,
  Fa,
  Lv,
  Bn,
  Sr,
  Az,
  Sl,
  Kn,
  Et,
  Mk,
  Br,
  Eu,
  Is,
  Hy,
  Ne,
  Mn,
  Bs,
  Kk,
  Sq,
  Sw,
  Gl,
  Mr,
  Pa,
  Si,
  Km,
  Sn,
  Yo,
  So,
  Af,
  Oc,
  Ka,
  Be,
  Tg,
  Sd,
  Gu,
  Am,
  Yi,
  Lo,
  Uz,
  Fo,
  Ht,
  Ps,
  Tk,
  Nn,
  Mt,
  Sa,
  Lb,
  My,
  Bo,
  Tl,
  Mg,
  As,
  Tt,
  Haw,
  Ln,
  Ha,
  Ba,
  Jw,
  Su,
  Yue,
  /// ISO 639-1 (or whisper-supplied) code that did not match any
  /// known variant. `from_iso639_1` and `as_str` round-trip
  /// through this for unknown codes; the indexer can log the
  /// SmolStr value and continue.
  Other(SmolStr),
}

impl Lang {
  /// Stable round-trip with [`Lang::from_iso639_1`]. Named variants
  /// emit their canonical lowercase ISO code; `Other(s)` emits `s`.
  pub fn as_str(&self) -> &str {
    match self {
      Self::En => "en",
      Self::Zh => "zh",
      Self::De => "de",
      Self::Es => "es",
      Self::Ru => "ru",
      Self::Ko => "ko",
      Self::Fr => "fr",
      Self::Ja => "ja",
      Self::Pt => "pt",
      Self::Tr => "tr",
      Self::Pl => "pl",
      Self::Ca => "ca",
      Self::Nl => "nl",
      Self::Ar => "ar",
      Self::Sv => "sv",
      Self::It => "it",
      Self::Id => "id",
      Self::Hi => "hi",
      Self::Fi => "fi",
      Self::Vi => "vi",
      Self::He => "he",
      Self::Uk => "uk",
      Self::El => "el",
      Self::Ms => "ms",
      Self::Cs => "cs",
      Self::Ro => "ro",
      Self::Da => "da",
      Self::Hu => "hu",
      Self::Ta => "ta",
      Self::No => "no",
      Self::Th => "th",
      Self::Ur => "ur",
      Self::Hr => "hr",
      Self::Bg => "bg",
      Self::Lt => "lt",
      Self::La => "la",
      Self::Mi => "mi",
      Self::Ml => "ml",
      Self::Cy => "cy",
      Self::Sk => "sk",
      Self::Te => "te",
      Self::Fa => "fa",
      Self::Lv => "lv",
      Self::Bn => "bn",
      Self::Sr => "sr",
      Self::Az => "az",
      Self::Sl => "sl",
      Self::Kn => "kn",
      Self::Et => "et",
      Self::Mk => "mk",
      Self::Br => "br",
      Self::Eu => "eu",
      Self::Is => "is",
      Self::Hy => "hy",
      Self::Ne => "ne",
      Self::Mn => "mn",
      Self::Bs => "bs",
      Self::Kk => "kk",
      Self::Sq => "sq",
      Self::Sw => "sw",
      Self::Gl => "gl",
      Self::Mr => "mr",
      Self::Pa => "pa",
      Self::Si => "si",
      Self::Km => "km",
      Self::Sn => "sn",
      Self::Yo => "yo",
      Self::So => "so",
      Self::Af => "af",
      Self::Oc => "oc",
      Self::Ka => "ka",
      Self::Be => "be",
      Self::Tg => "tg",
      Self::Sd => "sd",
      Self::Gu => "gu",
      Self::Am => "am",
      Self::Yi => "yi",
      Self::Lo => "lo",
      Self::Uz => "uz",
      Self::Fo => "fo",
      Self::Ht => "ht",
      Self::Ps => "ps",
      Self::Tk => "tk",
      Self::Nn => "nn",
      Self::Mt => "mt",
      Self::Sa => "sa",
      Self::Lb => "lb",
      Self::My => "my",
      Self::Bo => "bo",
      Self::Tl => "tl",
      Self::Mg => "mg",
      Self::As => "as",
      Self::Tt => "tt",
      Self::Haw => "haw",
      Self::Ln => "ln",
      Self::Ha => "ha",
      Self::Ba => "ba",
      Self::Jw => "jw",
      Self::Su => "su",
      Self::Yue => "yue",
      Self::Other(s) => s.as_str(),
    }
  }
}

impl Lang {
  /// Total-function constructor: every `&str` produces a `Lang`.
  /// Known whisper.cpp codes canonicalise to their named variant;
  /// unknown codes go to `Lang::Other`. Never produces
  /// `Lang::Other("en")` for an enum-known code "en" — see the
  /// canonicalisation invariant on the type doc.
  pub fn from_iso639_1(s: &str) -> Self {
    match s {
      "en" => Self::En,
      "zh" => Self::Zh,
      "de" => Self::De,
      "es" => Self::Es,
      "ru" => Self::Ru,
      "ko" => Self::Ko,
      "fr" => Self::Fr,
      "ja" => Self::Ja,
      "pt" => Self::Pt,
      "tr" => Self::Tr,
      "pl" => Self::Pl,
      "ca" => Self::Ca,
      "nl" => Self::Nl,
      "ar" => Self::Ar,
      "sv" => Self::Sv,
      "it" => Self::It,
      "id" => Self::Id,
      "hi" => Self::Hi,
      "fi" => Self::Fi,
      "vi" => Self::Vi,
      "he" => Self::He,
      "uk" => Self::Uk,
      "el" => Self::El,
      "ms" => Self::Ms,
      "cs" => Self::Cs,
      "ro" => Self::Ro,
      "da" => Self::Da,
      "hu" => Self::Hu,
      "ta" => Self::Ta,
      "no" => Self::No,
      "th" => Self::Th,
      "ur" => Self::Ur,
      "hr" => Self::Hr,
      "bg" => Self::Bg,
      "lt" => Self::Lt,
      "la" => Self::La,
      "mi" => Self::Mi,
      "ml" => Self::Ml,
      "cy" => Self::Cy,
      "sk" => Self::Sk,
      "te" => Self::Te,
      "fa" => Self::Fa,
      "lv" => Self::Lv,
      "bn" => Self::Bn,
      "sr" => Self::Sr,
      "az" => Self::Az,
      "sl" => Self::Sl,
      "kn" => Self::Kn,
      "et" => Self::Et,
      "mk" => Self::Mk,
      "br" => Self::Br,
      "eu" => Self::Eu,
      "is" => Self::Is,
      "hy" => Self::Hy,
      "ne" => Self::Ne,
      "mn" => Self::Mn,
      "bs" => Self::Bs,
      "kk" => Self::Kk,
      "sq" => Self::Sq,
      "sw" => Self::Sw,
      "gl" => Self::Gl,
      "mr" => Self::Mr,
      "pa" => Self::Pa,
      "si" => Self::Si,
      "km" => Self::Km,
      "sn" => Self::Sn,
      "yo" => Self::Yo,
      "so" => Self::So,
      "af" => Self::Af,
      "oc" => Self::Oc,
      "ka" => Self::Ka,
      "be" => Self::Be,
      "tg" => Self::Tg,
      "sd" => Self::Sd,
      "gu" => Self::Gu,
      "am" => Self::Am,
      "yi" => Self::Yi,
      "lo" => Self::Lo,
      "uz" => Self::Uz,
      "fo" => Self::Fo,
      "ht" => Self::Ht,
      "ps" => Self::Ps,
      "tk" => Self::Tk,
      "nn" => Self::Nn,
      "mt" => Self::Mt,
      "sa" => Self::Sa,
      "lb" => Self::Lb,
      "my" => Self::My,
      "bo" => Self::Bo,
      "tl" => Self::Tl,
      "mg" => Self::Mg,
      "as" => Self::As,
      "tt" => Self::Tt,
      "haw" => Self::Haw,
      "ln" => Self::Ln,
      "ha" => Self::Ha,
      "ba" => Self::Ba,
      "jw" => Self::Jw,
      "su" => Self::Su,
      "yue" => Self::Yue,
      other => Self::Other(SmolStr::new(other)),
    }
  }
}

impl core::fmt::Display for Lang {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.write_str(self.as_str())
  }
}

#[cfg(feature = "serde")]
impl serde::Serialize for Lang {
  /// Serialize as the lowercase ISO-639-1 (or whisper-supplied)
  /// string code. Matches what [`Lang::as_str`] returns —
  /// `Lang::En` → `"en"`, `Lang::Other(SmolStr::new("xx"))` →
  /// `"xx"`. Codex round-34: the previous `derive(Serialize)`
  /// produced Rust variant names like `"En"` and
  /// `{"Other":"xx"}`, contradicting the config docs.
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: serde::Serializer,
  {
    serializer.serialize_str(self.as_str())
  }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for Lang {
  /// Deserialize from an ISO-639-1 string code, **case-insensitive**.
  ///
  /// Accepts any ASCII-letter case (`"en"`, `"EN"`, `"En"`,
  /// `"eN"` all canonicalise to `Lang::En`); whisper.cpp's
  /// language codes are conventionally lowercase but the ISO
  /// standard treats them as case-insensitive, and human-edited
  /// configs naturally use mixed case. The accepted alphabet
  /// after lowercasing is `[a-z]{1,8}` — matches the
  /// alignment-stage validation in `runner/whisper_pool.rs`'s
  /// `validate_language_code` (round-32) so an "EN" config
  /// produces a Lang that the FFI layer happily accepts.
  ///
  /// Routes through [`Lang::from_iso639_1`] *after* lowercasing
  /// so input matching a named variant canonicalises to that
  /// variant rather than landing in `Other`. Unknown codes pass
  /// through `Lang::Other(SmolStr::new(lowered))` — the inner
  /// string is always lowercase, preserving the canonicalisation
  /// invariant across the serde boundary AND keeping the
  /// language-string intern table bounded.
  ///
  /// Round-trip asymmetry note: `"EN"` deserialises to
  /// `Lang::En` which then *serialises* as `"en"`. This is
  /// intentional — the on-disk canonical form is lowercase.
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: serde::Deserializer<'de>,
  {
    use serde::de::Error as _;

    let raw = <alloc::borrow::Cow<'_, str> as serde::Deserialize>::deserialize(deserializer)?;
    let s: &str = &raw;
    if s.is_empty() {
      return Err(D::Error::custom("Lang code is empty"));
    }
    if s.len() > 8 {
      return Err(D::Error::custom(alloc::format!(
        "Lang code longer than 8 bytes ({} bytes); whisper.cpp codes are 2–3 ASCII letters",
        s.len()
      )));
    }
    if !s.bytes().all(|b| b.is_ascii_alphabetic()) {
      return Err(D::Error::custom(
        "Lang code must be ASCII letters [a-zA-Z] only (no digits, dashes, or non-ASCII)",
      ));
    }
    // Avoid the lowercasing allocation when input is already canonical.
    if s.bytes().all(|b| b.is_ascii_lowercase()) {
      Ok(Lang::from_iso639_1(s))
    } else {
      let lowered: alloc::string::String = s.to_ascii_lowercase();
      Ok(Lang::from_iso639_1(&lowered))
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Every named variant round-trips through `from_iso639_1(as_str())`
  /// AND does not match `Lang::Other(_)`. This is the
  /// canonicalisation invariant.
  #[test]
  fn named_variants_canonicalise() {
    let known = [
      Lang::En,
      Lang::Zh,
      Lang::De,
      Lang::Es,
      Lang::Ru,
      Lang::Ko,
      Lang::Fr,
      Lang::Ja,
      Lang::Pt,
      Lang::Tr,
      Lang::Pl,
      Lang::Ca,
      Lang::Nl,
      Lang::Ar,
      Lang::Sv,
      Lang::It,
      Lang::Id,
      Lang::Hi,
      Lang::Fi,
      Lang::Vi,
      Lang::He,
      Lang::Uk,
      Lang::El,
      Lang::Ms,
      Lang::Cs,
      Lang::Ro,
      Lang::Da,
      Lang::Hu,
      Lang::Ta,
      Lang::No,
      Lang::Th,
      Lang::Ur,
      Lang::Hr,
      Lang::Bg,
      Lang::Lt,
      Lang::La,
      Lang::Mi,
      Lang::Ml,
      Lang::Cy,
      Lang::Sk,
      Lang::Te,
      Lang::Fa,
      Lang::Lv,
      Lang::Bn,
      Lang::Sr,
      Lang::Az,
      Lang::Sl,
      Lang::Kn,
      Lang::Et,
      Lang::Mk,
      Lang::Br,
      Lang::Eu,
      Lang::Is,
      Lang::Hy,
      Lang::Ne,
      Lang::Mn,
      Lang::Bs,
      Lang::Kk,
      Lang::Sq,
      Lang::Sw,
      Lang::Gl,
      Lang::Mr,
      Lang::Pa,
      Lang::Si,
      Lang::Km,
      Lang::Sn,
      Lang::Yo,
      Lang::So,
      Lang::Af,
      Lang::Oc,
      Lang::Ka,
      Lang::Be,
      Lang::Tg,
      Lang::Sd,
      Lang::Gu,
      Lang::Am,
      Lang::Yi,
      Lang::Lo,
      Lang::Uz,
      Lang::Fo,
      Lang::Ht,
      Lang::Ps,
      Lang::Tk,
      Lang::Nn,
      Lang::Mt,
      Lang::Sa,
      Lang::Lb,
      Lang::My,
      Lang::Bo,
      Lang::Tl,
      Lang::Mg,
      Lang::As,
      Lang::Tt,
      Lang::Haw,
      Lang::Ln,
      Lang::Ha,
      Lang::Ba,
      Lang::Jw,
      Lang::Su,
      Lang::Yue,
    ];
    assert_eq!(
      known.len(),
      100,
      "must keep the 100-variant Appendix C list in sync"
    );
    for v in known.iter() {
      let round = Lang::from_iso639_1(v.as_str());
      assert_eq!(&round, v, "round-trip failed for {:?}", v);
      assert!(
        !matches!(round, Lang::Other(_)),
        "{:?} canonicalised to Other; this breaks Eq/Hash",
        v
      );
    }
  }

  #[test]
  fn unknown_codes_land_in_other() {
    let r = Lang::from_iso639_1("zzz");
    assert_eq!(r, Lang::Other(SmolStr::new("zzz")));
    assert_eq!(r.as_str(), "zzz");
  }

  #[test]
  fn other_round_trips_via_as_str() {
    let r = Lang::Other(SmolStr::new("xx"));
    assert_eq!(r.as_str(), "xx");
    assert_eq!(Lang::from_iso639_1(r.as_str()), r);
  }

  // --- Codex round-34: custom serde wire format ---

  #[cfg(feature = "serde")]
  #[test]
  fn serde_named_variant_serializes_as_lowercase_iso() {
    let json = serde_json::to_string(&Lang::En).expect("serialize");
    assert_eq!(
      json, "\"en\"",
      "Lang::En must serialize as \"en\", not \"En\""
    );
    let json = serde_json::to_string(&Lang::Yue).expect("serialize");
    assert_eq!(json, "\"yue\"");
  }

  #[cfg(feature = "serde")]
  #[test]
  fn serde_other_serializes_as_inner_string() {
    let v = Lang::Other(SmolStr::new("xx"));
    let json = serde_json::to_string(&v).expect("serialize");
    assert_eq!(
      json, "\"xx\"",
      "Lang::Other(\"xx\") must serialize as \"xx\""
    );
  }

  #[cfg(feature = "serde")]
  #[test]
  fn serde_named_variant_round_trips() {
    let json = "\"en\"";
    let lang: Lang = serde_json::from_str(json).expect("deserialize");
    assert_eq!(lang, Lang::En);
    // Re-serialize and verify identical wire form.
    assert_eq!(serde_json::to_string(&lang).unwrap(), json);
  }

  #[cfg(feature = "serde")]
  #[test]
  fn serde_unknown_iso_code_round_trips_via_other() {
    let json = "\"xx\"";
    let lang: Lang = serde_json::from_str(json).expect("deserialize");
    assert_eq!(lang, Lang::Other(SmolStr::new("xx")));
    assert_eq!(serde_json::to_string(&lang).unwrap(), json);
  }

  /// Canonicalisation invariant must hold across serde:
  /// deserializing a code that matches a named variant lands in
  /// the named variant, not in `Other`.
  #[cfg(feature = "serde")]
  #[test]
  fn serde_deserializes_known_codes_to_named_variants() {
    let lang: Lang = serde_json::from_str("\"en\"").unwrap();
    assert!(matches!(lang, Lang::En), "must canonicalise to Lang::En");
    let lang: Lang = serde_json::from_str("\"yue\"").unwrap();
    assert!(matches!(lang, Lang::Yue));
  }

  /// Case-insensitive deserialization (UX win — users editing
  /// configs naturally use mixed case): `"EN"`, `"En"`, `"eN"`,
  /// `"en"` all canonicalise to `Lang::En`. The on-disk
  /// canonical form is lowercase (so re-serialization always
  /// emits `"en"`), but reading is permissive.
  #[cfg(feature = "serde")]
  #[test]
  fn serde_accepts_any_case_for_named_variant() {
    for input in ["\"en\"", "\"EN\"", "\"En\"", "\"eN\""] {
      let lang: Lang = serde_json::from_str(input).expect(input);
      assert_eq!(
        lang,
        Lang::En,
        "input {input} must canonicalise to Lang::En"
      );
      // Re-serialisation always emits the lowercase form.
      assert_eq!(serde_json::to_string(&lang).unwrap(), "\"en\"");
    }
  }

  /// Mixed-case unknown codes also canonicalise — `"XX"`
  /// deserialises to `Lang::Other(SmolStr::new("xx"))`,
  /// preserving the canonicalisation invariant (no
  /// `Lang::Other("XX")` ever exists in the type).
  #[cfg(feature = "serde")]
  #[test]
  fn serde_lowercases_unknown_code_into_other() {
    let lang: Lang = serde_json::from_str("\"XX\"").expect("deserialize");
    assert_eq!(lang, Lang::Other(SmolStr::new("xx")));
    let lang: Lang = serde_json::from_str("\"Xx\"").expect("deserialize");
    assert_eq!(lang, Lang::Other(SmolStr::new("xx")));
  }

  #[cfg(feature = "serde")]
  #[test]
  fn serde_rejects_empty_string() {
    let res: Result<Lang, _> = serde_json::from_str("\"\"");
    assert!(res.is_err());
  }

  #[cfg(feature = "serde")]
  #[test]
  fn serde_rejects_overlong_code() {
    let res: Result<Lang, _> = serde_json::from_str("\"abcdefghi\"");
    assert!(res.is_err(), "9-byte code must be rejected");
  }

  #[cfg(feature = "serde")]
  #[test]
  fn serde_rejects_non_ascii_letters() {
    let res: Result<Lang, _> = serde_json::from_str("\"français\"");
    assert!(res.is_err(), "non-ASCII must be rejected");
    let res: Result<Lang, _> = serde_json::from_str("\"a-b\"");
    assert!(res.is_err(), "dash must be rejected");
    let res: Result<Lang, _> = serde_json::from_str("\"a1b\"");
    assert!(res.is_err(), "digits must be rejected");
  }

  /// Old derive-shaped JSON for `Other` (`{"Other":"xx"}`) must
  /// fail with the new custom impl — it's an externally-tagged
  /// object, not a string. Documents the breaking wire-format
  /// change for migrators.
  ///
  /// Note: legacy `"En"` (Rust variant name) is now ACCEPTED as
  /// a side-effect of case-insensitive deserialization. That's a
  /// happy accident for migration — old configs that happened to
  /// use the variant-name form continue to work, just with the
  /// canonical lowercase form on round-trip. No special handling
  /// needed.
  #[cfg(feature = "serde")]
  #[test]
  fn serde_rejects_legacy_other_as_map() {
    let res: Result<Lang, _> = serde_json::from_str(r#"{"Other":"xx"}"#);
    assert!(
      res.is_err(),
      "legacy Other-as-map encoding must be rejected"
    );
  }
}
