//! Build script: fetch the production whisper checkpoint
//! (ggml-large-v3-turbo.bin, ~1.6 GB) into the in-tree
//! `models/` directory once, with SHA-256 verification, and
//! re-run when the env vars below change.

use std::{
  fs,
  io::{Read, Write},
  path::PathBuf,
};

const MODEL_URL: &str =
  "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin";
const MODEL_FILENAME: &str = "ggml-large-v3-turbo.bin";
// Verified SHA-256 from huggingface.co/ggerganov/whisper.cpp at the
// time of writing (matches HF's `x-linked-etag` header for the LFS
// blob). If the upstream rotates, update this constant and re-run
// the test fetch. The asry product runs `large-v3-turbo`, so the
// build.rs fixture matches what we ship — no separate "tiny for tests,
// large for prod" split.
const MODEL_SHA256: &str = "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69";

const WAV_URL: &str = "https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav";
const WAV_FILENAME: &str = "jfk.wav";
// 11-second JFK quote, mono, 16 kHz. SHA-256 of the upstream file at
// the time of writing.
const WAV_SHA256: &str = "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e";

const MODEL_W2V_URL: &str =
  "https://huggingface.co/onnx-community/wav2vec2-base-960h-ONNX/resolve/main/onnx/model.onnx";
const MODEL_W2V_FILENAME: &str = "wav2vec2-base-960h.onnx";
// SHA-256 of the upstream model.onnx, computed via:
//   curl -sSL <URL> | sha256sum
const MODEL_W2V_SHA256: &str = "00b7cc69516c1ab63c429e63a2b543e4d42bb77441ec5b98ee935de175b00de1";

const TOKENIZER_W2V_URL: &str =
  "https://huggingface.co/onnx-community/wav2vec2-base-960h-ONNX/resolve/main/tokenizer.json";
const TOKENIZER_W2V_FILENAME: &str = "wav2vec2-base-960h-tokenizer.json";
// SHA-256 of the upstream tokenizer.json, computed via:
//   curl -sSL <URL> | sha256sum
const TOKENIZER_W2V_SHA256: &str =
  "df57f576f5ef16a454ae2776dcc777ffef0bc824113043043b7218c829fc7405";

// Multi-language alignment models. The upstream `jonatasgrosman/`
// HF repos ship PyTorch weights only — no ONNX. We re-export
// once via `tests/parity_whisperx/python/fetch_align_model.py`
// and mirror the result under `FinDIT-Studio/...-onnx`. From
// build.rs's perspective, that mirror is just another
// SHA-verified direct-download — same shape as English.
const MODEL_W2V_JA_URL: &str = "https://huggingface.co/FinDIT-Studio/wav2vec2-large-xlsr-53-japanese-onnx/resolve/main/model.onnx";
const MODEL_W2V_JA_FILENAME: &str = "jonatasgrosman--wav2vec2-large-xlsr-53-japanese.onnx";
const MODEL_W2V_JA_SHA256: &str =
  "1157d2e1078392f6469e87993d879e3af569fb9754a443c539dd5886cfbd4c5e";
const TOKENIZER_W2V_JA_URL: &str = "https://huggingface.co/FinDIT-Studio/wav2vec2-large-xlsr-53-japanese-onnx/resolve/main/tokenizer.json";
const TOKENIZER_W2V_JA_FILENAME: &str =
  "jonatasgrosman--wav2vec2-large-xlsr-53-japanese-tokenizer.json";
const TOKENIZER_W2V_JA_SHA256: &str =
  "f6390130dea2fc0902dfe5e7b66b249f49d99c26ae08f14265dcd8d67121c4c2";

const MODEL_W2V_ZH_URL: &str = "https://huggingface.co/FinDIT-Studio/wav2vec2-large-xlsr-53-chinese-zh-cn-onnx/resolve/main/model.onnx";
const MODEL_W2V_ZH_FILENAME: &str = "jonatasgrosman--wav2vec2-large-xlsr-53-chinese-zh-cn.onnx";
const MODEL_W2V_ZH_SHA256: &str =
  "4e92f1d33b6bf89b709d5e4512a0c98dcaafd37a9bf7928452b05b01edb83029";
const TOKENIZER_W2V_ZH_URL: &str = "https://huggingface.co/FinDIT-Studio/wav2vec2-large-xlsr-53-chinese-zh-cn-onnx/resolve/main/tokenizer.json";
const TOKENIZER_W2V_ZH_FILENAME: &str =
  "jonatasgrosman--wav2vec2-large-xlsr-53-chinese-zh-cn-tokenizer.json";
const TOKENIZER_W2V_ZH_SHA256: &str =
  "7bb5c156e0ea01980f42ae1904193834132019ae8a6276a9957805cd5a6b37f5";

// Korean alignment fixtures. Upstream `jonatasgrosman/wav2vec2-
// large-xlsr-53-korean` was removed from the Hub; we ship the
// ONNX export of `kresnik/wav2vec2-large-xlsr-korean` instead
// (604k+ downloads, `Wav2Vec2ForCTC`, vocab_size=1205 syllable
// blocks, pad=`[PAD]` at id 1204, unk=`[UNK]` at id 1203). The
// repo URL keeps the `wav2vec2-large-xlsr-53-korean-onnx` slug
// for symmetry with the other languages even though kresnik
// dropped the `-53` from their slug.
const MODEL_W2V_KO_URL: &str =
  "https://huggingface.co/FinDIT-Studio/wav2vec2-large-xlsr-53-korean-onnx/resolve/main/model.onnx";
const MODEL_W2V_KO_FILENAME: &str = "kresnik--wav2vec2-large-xlsr-korean.onnx";
const MODEL_W2V_KO_SHA256: &str =
  "c43c01d7827bda6aaae60b04b722fea9a63399dd94b495166e4ddb529cf81a54";
const TOKENIZER_W2V_KO_URL: &str = "https://huggingface.co/FinDIT-Studio/wav2vec2-large-xlsr-53-korean-onnx/resolve/main/tokenizer.json";
const TOKENIZER_W2V_KO_FILENAME: &str = "kresnik--wav2vec2-large-xlsr-korean-tokenizer.json";
const TOKENIZER_W2V_KO_SHA256: &str =
  "2890d0bbe027b185a4a429f4ca295c1e74f92f792c1517e76405db93ed36cf1c";

// --- Latin-language alignment fixtures (Es, Fr, De, It, Pt) --------
//
// Same shape as Ja / Zh: jonatasgrosman ships PyTorch only, we
// re-export to ONNX once and host the result under
// `FinDIT-Studio/wav2vec2-large-xlsr-53-{lang}-onnx`. Each pair is
// SHA-verified after upload via `curl -L <url> | sha256sum`.
//
// SHAs marked TODO will resolve once the corresponding ONNX file
// is uploaded to the FinDIT-Studio mirror; until then build.rs's
// fetch step short-circuits on the SHA mismatch and the
// dependent integration tests skip via `option_env!()`. List of
// pending uploads is tracked in the branch report.
//
// IMPORTANT: keep this block contiguous so a sibling Korean
// branch's parallel additions merge mechanically.

const MODEL_W2V_ES_URL: &str = "https://huggingface.co/FinDIT-Studio/wav2vec2-large-xlsr-53-spanish-onnx/resolve/main/model.onnx";
const MODEL_W2V_ES_FILENAME: &str = "jonatasgrosman--wav2vec2-large-xlsr-53-spanish.onnx";
// SHAs computed from `optimum-cli export onnx` output of
// `jonatasgrosman/wav2vec2-large-xlsr-53-spanish` (transformers 5.8 /
// optimum 2.1 / onnx opset 14, fp32). Re-derive after any re-export
// via `curl -L <URL> | shasum -a 256`.
const MODEL_W2V_ES_SHA256: &str =
  "3478c4d9beeee5d5f46ef3be4b4cfb896bed6b2baf2498c0b98123a7878e406a";
const TOKENIZER_W2V_ES_URL: &str = "https://huggingface.co/FinDIT-Studio/wav2vec2-large-xlsr-53-spanish-onnx/resolve/main/tokenizer.json";
const TOKENIZER_W2V_ES_FILENAME: &str =
  "jonatasgrosman--wav2vec2-large-xlsr-53-spanish-tokenizer.json";
const TOKENIZER_W2V_ES_SHA256: &str =
  "11f754c360f8fadde294adaeb0aa4d621887b6f1b40a89a447de8dfe4972cee4";

const MODEL_W2V_FR_URL: &str =
  "https://huggingface.co/FinDIT-Studio/wav2vec2-large-xlsr-53-french-onnx/resolve/main/model.onnx";
const MODEL_W2V_FR_FILENAME: &str = "jonatasgrosman--wav2vec2-large-xlsr-53-french.onnx";
const MODEL_W2V_FR_SHA256: &str =
  "a26a555381f6525fbdc155a94664d5eafa0dab48f6c0194d42afe423af7be02b";
const TOKENIZER_W2V_FR_URL: &str = "https://huggingface.co/FinDIT-Studio/wav2vec2-large-xlsr-53-french-onnx/resolve/main/tokenizer.json";
const TOKENIZER_W2V_FR_FILENAME: &str =
  "jonatasgrosman--wav2vec2-large-xlsr-53-french-tokenizer.json";
const TOKENIZER_W2V_FR_SHA256: &str =
  "9e195f634c1bd2dbcc3062b176e482ac3a22653b2a47035819208c73b6895d74";

const MODEL_W2V_DE_URL: &str =
  "https://huggingface.co/FinDIT-Studio/wav2vec2-large-xlsr-53-german-onnx/resolve/main/model.onnx";
const MODEL_W2V_DE_FILENAME: &str = "jonatasgrosman--wav2vec2-large-xlsr-53-german.onnx";
const MODEL_W2V_DE_SHA256: &str =
  "ee286242d24b0b0a07112692cff8a1486fc0373f180b21e6b8c7470ec17a42a2";
const TOKENIZER_W2V_DE_URL: &str = "https://huggingface.co/FinDIT-Studio/wav2vec2-large-xlsr-53-german-onnx/resolve/main/tokenizer.json";
const TOKENIZER_W2V_DE_FILENAME: &str =
  "jonatasgrosman--wav2vec2-large-xlsr-53-german-tokenizer.json";
const TOKENIZER_W2V_DE_SHA256: &str =
  "c722046285ab31f846408457417176d1c9cd3c53e15adff763ccdb746f490e58";

const MODEL_W2V_IT_URL: &str = "https://huggingface.co/FinDIT-Studio/wav2vec2-large-xlsr-53-italian-onnx/resolve/main/model.onnx";
const MODEL_W2V_IT_FILENAME: &str = "jonatasgrosman--wav2vec2-large-xlsr-53-italian.onnx";
const MODEL_W2V_IT_SHA256: &str =
  "4c07d4d3bc86ff0d52a16d60dae69ce6aa7b9cc8363fe3cdc61321eb4ee2cf0f";
const TOKENIZER_W2V_IT_URL: &str = "https://huggingface.co/FinDIT-Studio/wav2vec2-large-xlsr-53-italian-onnx/resolve/main/tokenizer.json";
const TOKENIZER_W2V_IT_FILENAME: &str =
  "jonatasgrosman--wav2vec2-large-xlsr-53-italian-tokenizer.json";
const TOKENIZER_W2V_IT_SHA256: &str =
  "856aa99e17e10afc77c278782d8068f1624123ae2e692764102446b737e1e3ac";

const MODEL_W2V_PT_URL: &str = "https://huggingface.co/FinDIT-Studio/wav2vec2-large-xlsr-53-portuguese-onnx/resolve/main/model.onnx";
const MODEL_W2V_PT_FILENAME: &str = "jonatasgrosman--wav2vec2-large-xlsr-53-portuguese.onnx";
const MODEL_W2V_PT_SHA256: &str =
  "c101cedd8f9c5ade278e5ed8c698975b1f1048545e0eb29744786b0f7159d536";
const TOKENIZER_W2V_PT_URL: &str = "https://huggingface.co/FinDIT-Studio/wav2vec2-large-xlsr-53-portuguese-onnx/resolve/main/tokenizer.json";
const TOKENIZER_W2V_PT_FILENAME: &str =
  "jonatasgrosman--wav2vec2-large-xlsr-53-portuguese-tokenizer.json";
const TOKENIZER_W2V_PT_SHA256: &str =
  "841f77f1a38b2b96629e49df36eb52278f6e0181fa73f9aa934d70a19463c315";

fn main() {
  println!("cargo:rerun-if-changed=build.rs");
  println!("cargo:rerun-if-changed=assets/wav2vec2_base_960h_tokenizer.json");
  println!("cargo:rerun-if-env-changed=ASRY_OFFLINE");
  println!("cargo:rerun-if-env-changed=ASRY_FETCH_MODEL");
  println!("cargo:rerun-if-env-changed=CARGO_FEATURE_ALIGNMENT");
  println!("cargo:rerun-if-env-changed=ASRY_FETCH_W2V");

  // Windows: whisper.cpp's `ggml-cpu` references
  // `RegOpenKeyExA` / `RegQueryValueExA` / `RegCloseKey`
  // (its CPU-feature-detection fallback queries the registry
  // for AVX support). The `whispercpp-sys` build script emits
  // those for the lib build but not for downstream binaries
  // (examples / tests / integration test exes), so the
  // top-level `cargo test` link step on Windows fails with
  // `LNK2019: unresolved external symbol __imp_RegCloseKey`.
  // Re-emit the directive at the asry layer so every
  // binary that pulls our crate also gets `advapi32.lib`
  // linked. Harmless on non-Windows targets.
  if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
    println!("cargo:rustc-link-lib=advapi32");
  }

  // Codegen: parse the bundled wav2vec2-base-960h tokenizer JSON
  // at build time and emit Rust constants into OUT_DIR, so the
  // runtime crate never carries the JSON parse path for our
  // bundled vocab. The runtime accessor in
  // `runner::aligner::bundled::wav2vec2_base_960h` `include!()`s
  // the generated file. Always runs; it's cheap (32-entry vocab).
  if let Err(e) = codegen_wav2vec2_base_960h_tokens() {
    panic!("failed to codegen bundled wav2vec2 tokens: {e}");
  }

  // Fixture fetching is OPT-IN. A previous policy fetched
  // whenever the `runner` feature was active and
  // `ASRY_OFFLINE` was unset — but `runner` is a default
  // feature, so a plain `cargo build` made network requests.
  // That breaks ordinary consumer builds (offline / sandboxed
  // CI) and surprises anyone who didn't expect a build.rs to
  // phone home.
  //
  // New gate: the user must explicitly set `ASRY_FETCH_MODEL`
  // (and `ASRY_FETCH_W2V` for alignment) before any
  // download happens. The `ASRY_OFFLINE` knob stays as a
  // belt-and-braces "definitely don't fetch" override; in the
  // new design it's redundant with "don't set FETCH" but
  // existing scripts that rely on `ASRY_OFFLINE=1` keep
  // working.
  if std::env::var("ASRY_OFFLINE").is_ok() {
    eprintln!("[asry build.rs] ASRY_OFFLINE set; skipping model fetch");
    return;
  }
  let fetch_whisper_opt_in = std::env::var("ASRY_FETCH_MODEL").is_ok();
  if !fetch_whisper_opt_in {
    // Default: skip silently. Only test infrastructure that
    // actually needs the fixture sets the env var.
    return;
  }

  // The 'runner' feature gates whether the test fixture is even
  // applicable. Builds without the runner feature
  // (--no-default-features) skip anyway, even with FETCH_MODEL
  // set.
  let runner_active = std::env::var("CARGO_FEATURE_RUNNER").is_ok();
  if !runner_active {
    return;
  }

  // Two distinct directories:
  // - `models/` (in-tree, gitignored): big ML model files. Lives
  //   alongside the source so a developer can `ls models/` to
  //   see what's been downloaded; survives `cargo clean`.
  // - `target/asry-test-fixtures/`: transient test data
  //   (jfk.wav). Cargo-managed; can be wiped without losing
  //   gigabytes of model weights.
  let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
  let models_dir = manifest_dir.join("models");
  if let Err(e) = fs::create_dir_all(&models_dir) {
    eprintln!("[asry build.rs] cannot create {:?}: {}", models_dir, e);
    return;
  }
  let target_dir = match find_target_dir() {
    Some(p) => p,
    None => {
      eprintln!("[asry build.rs] cannot determine target dir; skipping fetch");
      return;
    }
  };
  let fixture_dir = target_dir.join("asry-test-fixtures");
  if let Err(e) = fs::create_dir_all(&fixture_dir) {
    eprintln!("[asry build.rs] cannot create {:?}: {}", fixture_dir, e);
    return;
  }
  let model_path = models_dir.join(MODEL_FILENAME);

  if model_path.exists() {
    if let Ok(true) = verify_sha256(&model_path, MODEL_SHA256) {
      // Already-good cached file — nothing to do.
      println!(
        "cargo:rustc-env=ASRY_WHISPER_MODEL={}",
        model_path.display()
      );
      fetch_jfk_wav(&fixture_dir);
      fetch_wav2vec2_fixtures(&models_dir);
      return;
    } else {
      eprintln!(
        "[asry build.rs] cached {:?} has wrong checksum; re-downloading",
        model_path
      );
      let _ = fs::remove_file(&model_path);
    }
  }

  eprintln!(
    "[asry build.rs] downloading {} ({})",
    MODEL_FILENAME, MODEL_URL
  );
  if let Err(e) = download(MODEL_URL, &model_path) {
    eprintln!("[asry build.rs] download failed: {}", e);
    let _ = fs::remove_file(&model_path);
    return;
  }
  match verify_sha256(&model_path, MODEL_SHA256) {
    Ok(true) => {
      println!(
        "cargo:rustc-env=ASRY_WHISPER_MODEL={}",
        model_path.display()
      );
      fetch_jfk_wav(&fixture_dir);
      fetch_wav2vec2_fixtures(&models_dir);
    }
    Ok(false) => {
      eprintln!("[asry build.rs] downloaded model has wrong checksum; aborting");
      let _ = fs::remove_file(&model_path);
    }
    Err(e) => {
      eprintln!("[asry build.rs] sha256 verification I/O error: {}", e);
    }
  }
}

fn fetch_jfk_wav(fixture_dir: &std::path::Path) {
  let wav_path = fixture_dir.join(WAV_FILENAME);
  if wav_path.exists() {
    if let Ok(true) = verify_sha256(&wav_path, WAV_SHA256) {
      println!("cargo:rustc-env=ASRY_JFK_WAV={}", wav_path.display());
      return;
    }
    let _ = fs::remove_file(&wav_path);
  }
  eprintln!("[asry build.rs] downloading {} ({})", WAV_FILENAME, WAV_URL);
  if download(WAV_URL, &wav_path).is_err() {
    let _ = fs::remove_file(&wav_path);
    return;
  }
  if let Ok(true) = verify_sha256(&wav_path, WAV_SHA256) {
    println!("cargo:rustc-env=ASRY_JFK_WAV={}", wav_path.display());
  }
}

fn fetch_wav2vec2_fixtures(models_dir: &std::path::Path) {
  // Opt-in via ASRY_FETCH_W2V. Same gate shape as the
  // parent `main`'s ASRY_FETCH_MODEL check — default builds
  // never hit the network, even when the alignment feature is
  // enabled. A user who wants the bundled fixture sets both env
  // vars together.
  let fetch_w2v_opt_in = std::env::var("ASRY_FETCH_W2V").is_ok();
  if !fetch_w2v_opt_in {
    return;
  }
  // Only fetch when the alignment feature is active. (Even with
  // FETCH_W2V set, an alignment-feature-off build doesn't need
  // the wav2vec2 assets.)
  let alignment_active = std::env::var("CARGO_FEATURE_ALIGNMENT").is_ok();
  if !alignment_active {
    return;
  }

  let model_path = models_dir.join(MODEL_W2V_FILENAME);
  if !fetch_with_sha(MODEL_W2V_URL, &model_path, MODEL_W2V_SHA256) {
    return;
  }
  println!("cargo:rustc-env=ASRY_W2V_MODEL={}", model_path.display());

  let tokenizer_path = models_dir.join(TOKENIZER_W2V_FILENAME);
  // Previously this path patched the downloaded tokenizer.json
  // before storing it, so the runtime `Aligner::from_paths`
  // only had to handle the patched form. The compat shim now
  // lives in `Aligner::from_paths` itself
  // (`load_tokenizer_with_compat`), which lets out-of-tree
  // consumers load any HuggingFace wav2vec2 tokenizer.json —
  // patched, unpatched, or in a totally different format. The
  // build.rs path is now a plain SHA-verified fetch, identical
  // to the model fetch shape.
  if !fetch_with_sha(TOKENIZER_W2V_URL, &tokenizer_path, TOKENIZER_W2V_SHA256) {
    return;
  }
  println!(
    "cargo:rustc-env=ASRY_W2V_TOKENIZER={}",
    tokenizer_path.display()
  );

  // Multi-language fixtures (Ja, Zh, ...). Mirror copies live in
  // FinDIT-Studio's HF org as ONNX exports; build.rs fetches +
  // SHA-verifies them under the same ASRY_FETCH_W2V opt-in
  // as English. Each pair is independent — a Ja-only build
  // skips the Zh fetch by env-var.
  fetch_extra_align_fixture(
    models_dir,
    "ASRY_W2V_JA",
    MODEL_W2V_JA_URL,
    MODEL_W2V_JA_FILENAME,
    MODEL_W2V_JA_SHA256,
    TOKENIZER_W2V_JA_URL,
    TOKENIZER_W2V_JA_FILENAME,
    TOKENIZER_W2V_JA_SHA256,
  );
  fetch_extra_align_fixture(
    models_dir,
    "ASRY_W2V_ZH",
    MODEL_W2V_ZH_URL,
    MODEL_W2V_ZH_FILENAME,
    MODEL_W2V_ZH_SHA256,
    TOKENIZER_W2V_ZH_URL,
    TOKENIZER_W2V_ZH_FILENAME,
    TOKENIZER_W2V_ZH_SHA256,
  );
  // Ko fixture: SHA-256 is currently `TODO_…` because the
  // FinDIT-Studio Ko ONNX mirror hasn't been uploaded yet. The
  // function logs + silently returns false on fetch / SHA
  // mismatch, so the build stays green and the Ko smoke test
  // skips via `option_env!()`. Once the upload + checksum step
  // lands, the placeholders flip to real hex digests and Ko
  // joins Ja/Zh as a real fixture.
  fetch_extra_align_fixture(
    models_dir,
    "ASRY_W2V_KO",
    MODEL_W2V_KO_URL,
    MODEL_W2V_KO_FILENAME,
    MODEL_W2V_KO_SHA256,
    TOKENIZER_W2V_KO_URL,
    TOKENIZER_W2V_KO_FILENAME,
    TOKENIZER_W2V_KO_SHA256,
  );

  // Latin-language fixtures (Es, Fr, De, It, Pt). Same opt-in
  // (`ASRY_FETCH_W2V`) as Ja / Zh / Ko. Each entry's SHA is
  // `TODO` until the corresponding ONNX is uploaded to
  // FinDIT-Studio's HF mirror; `fetch_extra_align_fixture`
  // short-circuits on SHA mismatch and the dependent
  // integration tests skip via `option_env!()`.
  fetch_extra_align_fixture(
    models_dir,
    "ASRY_W2V_ES",
    MODEL_W2V_ES_URL,
    MODEL_W2V_ES_FILENAME,
    MODEL_W2V_ES_SHA256,
    TOKENIZER_W2V_ES_URL,
    TOKENIZER_W2V_ES_FILENAME,
    TOKENIZER_W2V_ES_SHA256,
  );
  fetch_extra_align_fixture(
    models_dir,
    "ASRY_W2V_FR",
    MODEL_W2V_FR_URL,
    MODEL_W2V_FR_FILENAME,
    MODEL_W2V_FR_SHA256,
    TOKENIZER_W2V_FR_URL,
    TOKENIZER_W2V_FR_FILENAME,
    TOKENIZER_W2V_FR_SHA256,
  );
  fetch_extra_align_fixture(
    models_dir,
    "ASRY_W2V_DE",
    MODEL_W2V_DE_URL,
    MODEL_W2V_DE_FILENAME,
    MODEL_W2V_DE_SHA256,
    TOKENIZER_W2V_DE_URL,
    TOKENIZER_W2V_DE_FILENAME,
    TOKENIZER_W2V_DE_SHA256,
  );
  fetch_extra_align_fixture(
    models_dir,
    "ASRY_W2V_IT",
    MODEL_W2V_IT_URL,
    MODEL_W2V_IT_FILENAME,
    MODEL_W2V_IT_SHA256,
    TOKENIZER_W2V_IT_URL,
    TOKENIZER_W2V_IT_FILENAME,
    TOKENIZER_W2V_IT_SHA256,
  );
  fetch_extra_align_fixture(
    models_dir,
    "ASRY_W2V_PT",
    MODEL_W2V_PT_URL,
    MODEL_W2V_PT_FILENAME,
    MODEL_W2V_PT_SHA256,
    TOKENIZER_W2V_PT_URL,
    TOKENIZER_W2V_PT_FILENAME,
    TOKENIZER_W2V_PT_SHA256,
  );
}

/// Fetch + SHA-verify a multi-language alignment fixture pair
/// (`.onnx` + `tokenizer.json`) into `models_dir`, then emit
/// `<env_prefix>_MODEL` and `<env_prefix>_TOKENIZER` env vars so
/// `option_env!()` in tests can find them.
///
/// On any fetch / verification failure, this function logs and
/// returns silently — tests guard the fixtures with `option_env!`,
/// so a partial-fixture state just makes the corresponding test
/// skip rather than fail the build.
#[allow(clippy::too_many_arguments)]
fn fetch_extra_align_fixture(
  models_dir: &std::path::Path,
  env_prefix: &str,
  model_url: &str,
  model_filename: &str,
  model_sha: &str,
  tokenizer_url: &str,
  tokenizer_filename: &str,
  tokenizer_sha: &str,
) {
  let model_path = models_dir.join(model_filename);
  if !fetch_with_sha(model_url, &model_path, model_sha) {
    return;
  }
  println!(
    "cargo:rustc-env={env_prefix}_MODEL={}",
    model_path.display()
  );
  let tokenizer_path = models_dir.join(tokenizer_filename);
  if !fetch_with_sha(tokenizer_url, &tokenizer_path, tokenizer_sha) {
    return;
  }
  println!(
    "cargo:rustc-env={env_prefix}_TOKENIZER={}",
    tokenizer_path.display()
  );
}

/// Idempotent fetch + SHA-256 verify. Returns true on success
/// (cached or downloaded), false on any failure (caller skips
/// exporting the env var).
fn fetch_with_sha(url: &str, dest: &std::path::Path, expected_sha: &str) -> bool {
  if dest.exists() {
    if let Ok(true) = verify_sha256(dest, expected_sha) {
      return true;
    }
    eprintln!(
      "[asry build.rs] cached {:?} has wrong checksum; re-downloading",
      dest
    );
    let _ = std::fs::remove_file(dest);
  }
  eprintln!(
    "[asry build.rs] downloading {} ({})",
    dest.file_name().unwrap_or_default().to_string_lossy(),
    url
  );
  if let Err(e) = download(url, dest) {
    eprintln!("[asry build.rs] download failed: {e}");
    let _ = std::fs::remove_file(dest);
    return false;
  }
  match verify_sha256(dest, expected_sha) {
    Ok(true) => true,
    Ok(false) => {
      eprintln!("[asry build.rs] SHA-256 mismatch; aborting");
      let _ = std::fs::remove_file(dest);
      false
    }
    Err(e) => {
      eprintln!("[asry build.rs] SHA-256 verify I/O: {e}");
      false
    }
  }
}

fn find_target_dir() -> Option<PathBuf> {
  let out = std::env::var_os("OUT_DIR")?;
  let mut p = PathBuf::from(&out);
  while let Some(parent) = p.parent().map(PathBuf::from) {
    if parent.file_name().and_then(|s| s.to_str()) == Some("target") || parent.ends_with("target") {
      return Some(parent);
    }
    p = parent;
  }
  None
}

fn download(url: &str, dest: &std::path::Path) -> std::io::Result<()> {
  let resp = ureq::get(url)
    .call()
    .map_err(|e| std::io::Error::other(format!("{e}")))?;
  // ureq 3: response → body → reader (was a single `into_reader()` call in ureq 2).
  let mut reader = resp.into_body().into_reader();
  let mut writer = fs::File::create(dest)?;
  let mut buf = vec![0u8; 64 * 1024];
  loop {
    let n = reader.read(&mut buf)?;
    if n == 0 {
      break;
    }
    writer.write_all(&buf[..n])?;
  }
  writer.flush()
}

fn verify_sha256(path: &std::path::Path, expected: &str) -> std::io::Result<bool> {
  use sha2::{Digest, Sha256};
  // Fail closed on a malformed expected hash so a typo in the
  // pinned constants can never accept a tampered file.
  if expected.len() != 64
    || !expected
      .bytes()
      .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
  {
    return Ok(false);
  }
  let mut f = fs::File::open(path)?;
  let mut hasher = Sha256::new();
  let mut buf = vec![0u8; 64 * 1024];
  loop {
    let n = f.read(&mut buf)?;
    if n == 0 {
      break;
    }
    hasher.update(&buf[..n]);
  }
  let got = hex_encode(&hasher.finalize());
  Ok(got == expected)
}

fn hex_encode(bytes: &[u8]) -> String {
  let mut s = String::with_capacity(bytes.len() * 2);
  for b in bytes {
    s.push_str(&format!("{:02x}", b));
  }
  s
}

/// Parse `assets/wav2vec2_base_960h_tokenizer.json` at build
/// time and emit a Rust source file under `OUT_DIR` containing
/// the bundled vocab as a sorted-by-id `&[(&str, u32)]` slice
/// plus pre-resolved `PAD_TOKEN_ID` / `UNK_TOKEN_ID` /
/// `DELIMITER_TOKEN_ID` constants.
///
/// **Why pre-decode at build time.** The runtime alignment path
/// previously had two reasons to JSON-parse our bundled
/// tokenizer: looking up `<pad>` / `<unk>` / `|` token ids and
/// later, at encode time, mapping characters to ids. Doing the
/// parse once at build time and shipping `&'static [(&'static str, u32)]`
/// cuts the runtime cost to zero — no `serde_json` work, no
/// allocations, no version-rejection edge cases. The runtime
/// `tokenizers` crate is still pulled in for user-supplied
/// non-bundled tokenizer files, but those are now an opt-in
/// path; the bundled English aligner gets parsed-already
/// constants.
fn codegen_wav2vec2_base_960h_tokens() -> Result<(), String> {
  let manifest_dir =
    std::env::var("CARGO_MANIFEST_DIR").map_err(|e| format!("CARGO_MANIFEST_DIR not set: {e}"))?;
  let json_path = std::path::PathBuf::from(&manifest_dir)
    .join("assets")
    .join("wav2vec2_base_960h_tokenizer.json");
  let json_bytes =
    fs::read(&json_path).map_err(|e| format!("read {}: {e}", json_path.display()))?;
  let parsed: serde_json::Value =
    serde_json::from_slice(&json_bytes).map_err(|e| format!("parse tokenizer.json: {e}"))?;
  let vocab = parsed
    .get("model")
    .and_then(|m| m.get("vocab"))
    .and_then(|v| v.as_object())
    .ok_or_else(|| "tokenizer.json missing model.vocab object".to_string())?;

  // Collect (token, id), sort by id ascending. Stable order is
  // load-bearing for the const slice — consumers do
  // `VOCAB[id as usize]` (or linear scan; either way the order
  // matters for reproducibility).
  let mut entries: Vec<(String, u32)> = Vec::with_capacity(vocab.len());
  for (token, id_val) in vocab {
    let id = id_val
      .as_u64()
      .ok_or_else(|| format!("vocab[{token:?}] is not an integer"))?;
    let id_u32 =
      u32::try_from(id).map_err(|e| format!("vocab[{token:?}] id {id} > u32::MAX: {e}"))?;
    entries.push((token.clone(), id_u32));
  }
  entries.sort_by_key(|(_, id)| *id);

  // Resolve the three special-token ids we use elsewhere. If
  // any is missing the bundled file is wrong, not the
  // consumer's fault — fail the build with a clear message.
  let pad_id = lookup_id(&entries, "<pad>").ok_or_else(|| "vocab missing `<pad>`".to_string())?;
  let unk_id = lookup_id(&entries, "<unk>").ok_or_else(|| "vocab missing `<unk>`".to_string())?;
  let delim_id =
    lookup_id(&entries, "|").ok_or_else(|| "vocab missing `|` (word delimiter)".to_string())?;

  // Emit Rust source. `include!`-d at runtime; no external
  // schema, just constants.
  let mut out = String::with_capacity(entries.len() * 24 + 256);
  out.push_str(
    "// Generated by build.rs from assets/wav2vec2_base_960h_tokenizer.json — DO NOT EDIT.\n\
     // The bundled wav2vec2-base-960h vocab, sorted by id ascending.\n\n",
  );
  out.push_str(&format!(
    "/// CTC blank token id (`<pad>` in wav2vec2's vocab).\npub const PAD_TOKEN_ID: u32 = {pad_id};\n\n"
  ));
  out.push_str(&format!(
    "/// `<unk>` (out-of-vocab) token id.\npub const UNK_TOKEN_ID: u32 = {unk_id};\n\n"
  ));
  out.push_str(&format!(
    "/// `|` word-delimiter token id.\npub const DELIMITER_TOKEN_ID: u32 = {delim_id};\n\n"
  ));
  out.push_str(&format!(
    "/// Vocab as `(token, id)` pairs, sorted by id ascending.\n\
     /// {} entries.\n\
     pub const VOCAB: &[(&str, u32)] = &[\n",
    entries.len()
  ));
  for (token, id) in &entries {
    // Escape backslash and double-quote for Rust string literals.
    // The tokens here are short alphabet entries plus `<pad>` /
    // `<s>` / `</s>` / `<unk>` / `|` — no embedded quotes or
    // backslashes in practice, but escape defensively so a
    // future vocab swap can't break the codegen.
    let escaped = token.replace('\\', "\\\\").replace('\"', "\\\"");
    out.push_str(&format!("  (\"{escaped}\", {id}),\n"));
  }
  out.push_str("];\n");

  let out_dir = std::env::var("OUT_DIR").map_err(|e| format!("OUT_DIR not set: {e}"))?;
  let dest = std::path::PathBuf::from(out_dir).join("wav2vec2_base_960h_tokens.rs");
  fs::write(&dest, out).map_err(|e| format!("write {}: {e}", dest.display()))?;
  Ok(())
}

fn lookup_id(entries: &[(String, u32)], token: &str) -> Option<u32> {
  entries.iter().find(|(t, _)| t == token).map(|(_, id)| *id)
}
