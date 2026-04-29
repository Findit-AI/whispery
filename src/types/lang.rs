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
/// See spec §4.4 and Appendix C for the variant table.
#[non_exhaustive]
#[allow(missing_docs)] // variants are ISO 639-1 codes; self-documenting by name
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Lang {
    En, Zh, De, Es, Ru, Ko, Fr, Ja, Pt, Tr,
    Pl, Ca, Nl, Ar, Sv, It, Id, Hi, Fi, Vi,
    He, Uk, El, Ms, Cs, Ro, Da, Hu, Ta, No,
    Th, Ur, Hr, Bg, Lt, La, Mi, Ml, Cy, Sk,
    Te, Fa, Lv, Bn, Sr, Az, Sl, Kn, Et, Mk,
    Br, Eu, Is, Hy, Ne, Mn, Bs, Kk, Sq, Sw,
    Gl, Mr, Pa, Si, Km, Sn, Yo, So, Af, Oc,
    Ka, Be, Tg, Sd, Gu, Am, Yi, Lo, Uz, Fo,
    Ht, Ps, Tk, Nn, Mt, Sa, Lb, My, Bo, Tl,
    Mg, As, Tt, Haw, Ln, Ha, Ba, Jw, Su, Yue,
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
            Self::En => "en", Self::Zh => "zh", Self::De => "de", Self::Es => "es",
            Self::Ru => "ru", Self::Ko => "ko", Self::Fr => "fr", Self::Ja => "ja",
            Self::Pt => "pt", Self::Tr => "tr", Self::Pl => "pl", Self::Ca => "ca",
            Self::Nl => "nl", Self::Ar => "ar", Self::Sv => "sv", Self::It => "it",
            Self::Id => "id", Self::Hi => "hi", Self::Fi => "fi", Self::Vi => "vi",
            Self::He => "he", Self::Uk => "uk", Self::El => "el", Self::Ms => "ms",
            Self::Cs => "cs", Self::Ro => "ro", Self::Da => "da", Self::Hu => "hu",
            Self::Ta => "ta", Self::No => "no", Self::Th => "th", Self::Ur => "ur",
            Self::Hr => "hr", Self::Bg => "bg", Self::Lt => "lt", Self::La => "la",
            Self::Mi => "mi", Self::Ml => "ml", Self::Cy => "cy", Self::Sk => "sk",
            Self::Te => "te", Self::Fa => "fa", Self::Lv => "lv", Self::Bn => "bn",
            Self::Sr => "sr", Self::Az => "az", Self::Sl => "sl", Self::Kn => "kn",
            Self::Et => "et", Self::Mk => "mk", Self::Br => "br", Self::Eu => "eu",
            Self::Is => "is", Self::Hy => "hy", Self::Ne => "ne", Self::Mn => "mn",
            Self::Bs => "bs", Self::Kk => "kk", Self::Sq => "sq", Self::Sw => "sw",
            Self::Gl => "gl", Self::Mr => "mr", Self::Pa => "pa", Self::Si => "si",
            Self::Km => "km", Self::Sn => "sn", Self::Yo => "yo", Self::So => "so",
            Self::Af => "af", Self::Oc => "oc", Self::Ka => "ka", Self::Be => "be",
            Self::Tg => "tg", Self::Sd => "sd", Self::Gu => "gu", Self::Am => "am",
            Self::Yi => "yi", Self::Lo => "lo", Self::Uz => "uz", Self::Fo => "fo",
            Self::Ht => "ht", Self::Ps => "ps", Self::Tk => "tk", Self::Nn => "nn",
            Self::Mt => "mt", Self::Sa => "sa", Self::Lb => "lb", Self::My => "my",
            Self::Bo => "bo", Self::Tl => "tl", Self::Mg => "mg", Self::As => "as",
            Self::Tt => "tt", Self::Haw => "haw", Self::Ln => "ln", Self::Ha => "ha",
            Self::Ba => "ba", Self::Jw => "jw", Self::Su => "su", Self::Yue => "yue",
            Self::Other(s) => s.as_str(),
        }
    }
}

impl core::fmt::Display for Lang {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_str())
    }
}
