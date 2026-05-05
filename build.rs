//! Build script: fetch the tiny whisper test fixture (ggml-tiny.en.bin)
//! into `target/whispery-test-fixtures/` once, with SHA-256
//! verification, and re-run when the env vars below change.

use std::{
  fs,
  io::{Read, Write},
  path::PathBuf,
};

const MODEL_URL: &str =
  "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin";
const MODEL_FILENAME: &str = "ggml-tiny.en.bin";
// Verified SHA-256 from huggingface.co/ggerganov/whisper.cpp at the
// time of writing. If the upstream rotates, update this constant and
// re-run the test fetch.
const MODEL_SHA256: &str = "921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f";

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

fn main() {
  println!("cargo:rerun-if-changed=build.rs");
  println!("cargo:rerun-if-changed=assets/wav2vec2_base_960h_tokenizer.json");
  println!("cargo:rerun-if-env-changed=WHISPERY_OFFLINE");
  println!("cargo:rerun-if-env-changed=WHISPERY_FETCH_MODEL");
  println!("cargo:rerun-if-env-changed=CARGO_FEATURE_ALIGNMENT");
  println!("cargo:rerun-if-env-changed=WHISPERY_FETCH_W2V");

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
  // `WHISPERY_OFFLINE` was unset — but `runner` is a default
  // feature, so a plain `cargo build` made network requests.
  // That breaks ordinary consumer builds (offline / sandboxed
  // CI) and surprises anyone who didn't expect a build.rs to
  // phone home.
  //
  // New gate: the user must explicitly set `WHISPERY_FETCH_MODEL`
  // (and `WHISPERY_FETCH_W2V` for alignment) before any
  // download happens. The `WHISPERY_OFFLINE` knob stays as a
  // belt-and-braces "definitely don't fetch" override; in the
  // new design it's redundant with "don't set FETCH" but
  // existing scripts that rely on `WHISPERY_OFFLINE=1` keep
  // working.
  if std::env::var("WHISPERY_OFFLINE").is_ok() {
    eprintln!("[whispery build.rs] WHISPERY_OFFLINE set; skipping model fetch");
    return;
  }
  let fetch_whisper_opt_in = std::env::var("WHISPERY_FETCH_MODEL").is_ok();
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

  let target_dir = match find_target_dir() {
    Some(p) => p,
    None => {
      eprintln!("[whispery build.rs] cannot determine target dir; skipping fetch");
      return;
    }
  };
  let fixture_dir = target_dir.join("whispery-test-fixtures");
  if let Err(e) = fs::create_dir_all(&fixture_dir) {
    eprintln!("[whispery build.rs] cannot create {:?}: {}", fixture_dir, e);
    return;
  }
  let model_path = fixture_dir.join(MODEL_FILENAME);

  if model_path.exists() {
    if let Ok(true) = verify_sha256(&model_path, MODEL_SHA256) {
      // Already-good cached file — nothing to do.
      println!(
        "cargo:rustc-env=WHISPERY_TINY_EN_MODEL={}",
        model_path.display()
      );
      fetch_jfk_wav(&fixture_dir);
      fetch_wav2vec2_fixtures(&fixture_dir);
      return;
    } else {
      eprintln!(
        "[whispery build.rs] cached {:?} has wrong checksum; re-downloading",
        model_path
      );
      let _ = fs::remove_file(&model_path);
    }
  }

  eprintln!(
    "[whispery build.rs] downloading {} ({})",
    MODEL_FILENAME, MODEL_URL
  );
  if let Err(e) = download(MODEL_URL, &model_path) {
    eprintln!("[whispery build.rs] download failed: {}", e);
    let _ = fs::remove_file(&model_path);
    return;
  }
  match verify_sha256(&model_path, MODEL_SHA256) {
    Ok(true) => {
      println!(
        "cargo:rustc-env=WHISPERY_TINY_EN_MODEL={}",
        model_path.display()
      );
      fetch_jfk_wav(&fixture_dir);
      fetch_wav2vec2_fixtures(&fixture_dir);
    }
    Ok(false) => {
      eprintln!("[whispery build.rs] downloaded model has wrong checksum; aborting");
      let _ = fs::remove_file(&model_path);
    }
    Err(e) => {
      eprintln!("[whispery build.rs] sha256 verification I/O error: {}", e);
    }
  }
}

fn fetch_jfk_wav(fixture_dir: &std::path::Path) {
  let wav_path = fixture_dir.join(WAV_FILENAME);
  if wav_path.exists() {
    if let Ok(true) = verify_sha256(&wav_path, WAV_SHA256) {
      println!("cargo:rustc-env=WHISPERY_JFK_WAV={}", wav_path.display());
      return;
    }
    let _ = fs::remove_file(&wav_path);
  }
  eprintln!(
    "[whispery build.rs] downloading {} ({})",
    WAV_FILENAME, WAV_URL
  );
  if download(WAV_URL, &wav_path).is_err() {
    let _ = fs::remove_file(&wav_path);
    return;
  }
  if let Ok(true) = verify_sha256(&wav_path, WAV_SHA256) {
    println!("cargo:rustc-env=WHISPERY_JFK_WAV={}", wav_path.display());
  }
}

fn fetch_wav2vec2_fixtures(fixture_dir: &std::path::Path) {
  // Opt-in via WHISPERY_FETCH_W2V. Same gate shape as the
  // parent `main`'s WHISPERY_FETCH_MODEL check — default builds
  // never hit the network, even when the alignment feature is
  // enabled. A user who wants the bundled fixture sets both env
  // vars together.
  let fetch_w2v_opt_in = std::env::var("WHISPERY_FETCH_W2V").is_ok();
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

  let model_path = fixture_dir.join(MODEL_W2V_FILENAME);
  if !fetch_with_sha(MODEL_W2V_URL, &model_path, MODEL_W2V_SHA256) {
    return;
  }
  println!(
    "cargo:rustc-env=WHISPERY_W2V_MODEL={}",
    model_path.display()
  );

  let tokenizer_path = fixture_dir.join(TOKENIZER_W2V_FILENAME);
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
    "cargo:rustc-env=WHISPERY_W2V_TOKENIZER={}",
    tokenizer_path.display()
  );

  // Multi-language fixtures (Ja, Zh, ...) are exported by an
  // external `fetch_align_model.py` script, NOT downloaded by
  // build.rs. We just probe for them and emit env vars when
  // present, so multi-lang tests skip-when-fixture-absent
  // identically to the single-language path above.
  detect_extra_align_fixture(fixture_dir, "WHISPERY_W2V_JA",
    "jonatasgrosman--wav2vec2-large-xlsr-53-japanese");
  detect_extra_align_fixture(fixture_dir, "WHISPERY_W2V_ZH",
    "jonatasgrosman--wav2vec2-large-xlsr-53-chinese-zh-cn");
}

/// Detect a multi-language alignment fixture pair (an `.onnx` file
/// and a `<stem>-tokenizer.json`) under `fixture_dir`. When both
/// exist, export `<env_prefix>_MODEL` and `<env_prefix>_TOKENIZER`
/// so `option_env!()` in tests can find them. When missing, do
/// nothing — the test path skips.
fn detect_extra_align_fixture(
  fixture_dir: &std::path::Path,
  env_prefix: &str,
  stem: &str,
) {
  let model_path = fixture_dir.join(format!("{stem}.onnx"));
  let tokenizer_path = fixture_dir.join(format!("{stem}-tokenizer.json"));
  if !model_path.exists() || !tokenizer_path.exists() {
    return;
  }
  println!(
    "cargo:rustc-env={env_prefix}_MODEL={}",
    model_path.display()
  );
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
      "[whispery build.rs] cached {:?} has wrong checksum; re-downloading",
      dest
    );
    let _ = std::fs::remove_file(dest);
  }
  eprintln!(
    "[whispery build.rs] downloading {} ({})",
    dest.file_name().unwrap_or_default().to_string_lossy(),
    url
  );
  if let Err(e) = download(url, dest) {
    eprintln!("[whispery build.rs] download failed: {e}");
    let _ = std::fs::remove_file(dest);
    return false;
  }
  match verify_sha256(dest, expected_sha) {
    Ok(true) => true,
    Ok(false) => {
      eprintln!("[whispery build.rs] SHA-256 mismatch; aborting");
      let _ = std::fs::remove_file(dest);
      false
    }
    Err(e) => {
      eprintln!("[whispery build.rs] SHA-256 verify I/O: {e}");
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
    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{e}")))?;
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
  Ok(got.starts_with(expected))
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
