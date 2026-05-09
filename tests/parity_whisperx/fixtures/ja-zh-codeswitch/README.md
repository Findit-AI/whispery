# `ja-zh-codeswitch` parity fixture

End-to-end Ja+Zh code-switch validation for the script-dispatch
pipeline (steps 3 + 4 of
`docs/superpowers/specs/2026-05-09-script-dispatch-design.md`,
implemented on `feat/script-dispatch-wire`).

## Expected files

```
audio.wav            16 kHz mono PCM, ~10–30 s mixing Japanese and
                     Chinese sentences. Hand-picked or synthesised
                     via TTS so the language boundaries are
                     unambiguous.
ground_truth.json    Hand-labelled per-word reference. Schema:
                     [
                       {
                         "word":  "<surface form>",
                         "lang":  "ja" | "zh",
                         "t0_ms": <int>,
                         "t1_ms": <int>
                       },
                       ...
                     ]
                     Times are in milliseconds, half-open ranges
                     anchored at the start of `audio.wav`.
```

The integration test at `tests/script_dispatch_codeswitch.rs`
loads both files and asserts that the script-dispatcher's
per-word language assignment plus the alignment worker's word
boundaries match the reference within ±50 ms on at least 95 %
of words. When either file is missing the test skips with a
diagnostic message rather than failing — keeping the gate green
for callers who haven't materialised the fixture yet.

## Why no fixture is committed yet

Generating the audio is a manual / TTS step that requires
external tooling and (for high-quality alignment) a quiet studio
recording. The harness lands first so the contract is obvious;
the audio + labels can be produced and dropped in here without
any code change.
