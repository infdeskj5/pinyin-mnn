#!/usr/bin/env python3
"""
Build the g2pW MNN model + bundle every auxiliary file the Android side
needs for preprocessing/postprocessing.

Source model:
  https://github.com/GitYCC/g2pW (INTERSPEECH 2022)
  Official pretrained ONNX release:
    https://storage.googleapis.com/esun-ai/g2pW/G2PWModel-v2-onnx.zip
  (Same release used by GitYCC/g2pW itself and by mozillazg/pypinyin-g2pW --
  there is exactly one official checkpoint; no PyTorch->ONNX conversion is
  needed here, only ONNX->MNN.)

Verified against g2pW's real source (export_onnx_model.py, api.py,
dataset.py, module.py) rather than assumed:

  ONNX graph inputs (fixed, from export_onnx_model.py's input_names):
    input_ids       int64 [batch, seq_len]   BERT WordPiece token ids, incl [CLS]/[SEP]
    token_type_ids  int64 [batch, seq_len]   all zeros
    attention_mask  int64 [batch, seq_len]   all ones
    phoneme_mask    float [batch, num_labels] 1.0 at this char's valid reading
                                               indices, 0.0 elsewhere -- this
                                               is what makes the softmax
                                               "conditional": it can only ever
                                               choose among that character's
                                               real candidate readings.
    char_ids        int64 [batch]            index of the query char within
                                               the sorted global char list
    position_ids    int64 [batch]            token position of the query char
                                               in input_ids (CLS-offset)
  Output:
    probs           float [batch, num_labels]

  Notably, pos_ids is NOT part of the exported graph -- the released
  checkpoint's conditioning is char-based only, not POS-based, so the
  Android side does not need to map LTP's POS tagset into g2pW's tagset.

  The model's native output vocabulary is Bopomofo (e.g. "ㄒㄧㄥ2"), not
  Pinyin. This script does not assume which raw format
  POLYPHONIC_CHARS.txt inside the official zip actually uses (Bopomofo vs
  Pinyin variants exist across g2pW's own dataset configs) -- it detects
  and logs it instead of guessing, and always bundles the Bopomofo->Pinyin
  conversion table regardless, since it's tiny and harmless to include
  even if unused.

Output files:

  <out-dir>/g2pw.mnn                          the converted model
  <out-dir>/vocab.txt                         bert-base-chinese WordPiece vocab (tokenizer)
  <out-dir>/POLYPHONIC_CHARS.txt              char -> candidate reading(s)
  <out-dir>/MONOPHONIC_CHARS.txt              char -> single fixed reading (no model call needed)
  <out-dir>/char_bopomofo_dict.json           fallback reading table for chars in neither list above
  <out-dir>/bopomofo_to_pinyin_wo_tune_dict.json  Bopomofo component -> Pinyin component (tone handled separately)
  <out-dir>/bert-base-chinese_s2t_dict.txt    simplified -> traditional char map (g2pW's lookup tables are traditional-keyed)
  <out-dir>/manifest.json
  <out-dir>/README.txt

Final zip:

  <zip>

This is intentionally *only* the model + data conversion step (mirrors
build_paddle_models.py / build_ltp_models.py). The native MNN inference
wrapper (WordPiece tokenizer, phoneme_mask/char_ids/position_ids
construction, Bopomofo->Pinyin postprocessing) is a separate, follow-up
piece of work once these files are confirmed to look right.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlretrieve

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print(
        "ERROR: missing dependency. Install with:\n"
        "  python -m pip install -r scripts/python/requirements_g2pw.txt",
        file=sys.stderr,
    )
    sys.exit(1)


G2PW_MODEL_ZIP_URL = "https://storage.googleapis.com/esun-ai/g2pW/G2PWModel-v2-onnx.zip"

# Small auxiliary files that ship inside the g2pW *python package* (not the
# model zip) -- fetched straight from the repo at a pinned commit so the
# build is reproducible.
G2PW_REPO_RAW_BASE = "https://raw.githubusercontent.com/GitYCC/g2pW"
G2PW_REPO_COMMIT = "36c3fcce93aebfcb54803d2ad6677023a28ad950"  # main @ time of writing (verified); override with --g2pw-commit

G2PW_AUX_FILES = [
    "g2pw/char_bopomofo_dict.json",
    "g2pw/bopomofo_to_pinyin_wo_tune_dict.json",
    "g2pw/bert-base-chinese_s2t_dict.txt",
]

BOPOMOFO_RANGE = re.compile(r"[\u3105-\u312F]")


def log(*args):
    print(*args, flush=True)


def fail(msg):
    log(f"ERROR: {msg}")
    sys.exit(1)


def run(cmd):
    import subprocess
    log(f"$ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        fail(f"command failed with exit code {result.returncode}: {' '.join(str(c) for c in cmd)}")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_mnn_convert():
    for name in ("mnnconvert", "MNNConvert"):
        path = shutil.which(name)
        if path:
            return [path]
    # pip-installed `mnn` package exposes a `mnnconvert` console-script;
    # fall back to `python -m mnn.tools.mnnconvert` if the bare binary
    # isn't on PATH for some reason.
    try:
        import mnn  # noqa: F401
        return [sys.executable, "-m", "mnn.tools.mnnconvert"]
    except ImportError:
        pass
    fail("mnnconvert not found. Install with: pip install mnn")


def download_g2pw_model_zip(dest_zip):
    log(f"Downloading {G2PW_MODEL_ZIP_URL} -> {dest_zip}")
    urlretrieve(G2PW_MODEL_ZIP_URL, dest_zip)
    if not dest_zip.exists() or dest_zip.stat().st_size < 1_000_000:
        fail(f"Downloaded file looks too small: {dest_zip}")


def extract_zip(zip_path, dest_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    # The zip contains a single top-level folder (per g2pw/api.py's
    # download_model()); find it.
    entries = [p for p in dest_dir.iterdir() if p.is_dir()]
    if len(entries) == 1:
        return entries[0]
    return dest_dir


def download_g2pw_aux_files(work_dir, commit):
    out = work_dir / "g2pw_repo_files"
    out.mkdir(parents=True, exist_ok=True)
    downloaded = {}
    for rel_path in G2PW_AUX_FILES:
        url = f"{G2PW_REPO_RAW_BASE}/{commit}/{rel_path}"
        dest = out / Path(rel_path).name
        log(f"Downloading {url} -> {dest}")
        urlretrieve(url, dest)
        downloaded[Path(rel_path).name] = dest
    return downloaded


def download_bert_vocab(work_dir):
    log("Downloading bert-base-chinese vocab.txt ...")
    path = hf_hub_download(repo_id="bert-base-chinese", filename="vocab.txt", local_dir=str(work_dir))
    return Path(path)


def detect_polyphonic_format(polyphonic_chars_path):
    sample_readings = []
    with open(polyphonic_chars_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2 and parts[1]:
                sample_readings.append(parts[1])
            if i > 20:
                break

    if not sample_readings:
        return "unknown", sample_readings

    if any(BOPOMOFO_RANGE.search(r) for r in sample_readings):
        return "bopomofo", sample_readings
    if all(re.fullmatch(r"[a-zü]+[1-5]?", r) for r in sample_readings):
        return "pinyin", sample_readings
    return "unknown", sample_readings


def convert_onnx_to_mnn(mnn_cmd, onnx_path, mnn_path, weight_quant_bits, weight_quant_asymmetric):
    cmd = list(mnn_cmd) + [
        "-f", "ONNX",
        "--modelFile", str(onnx_path),
        "--MNNModel", str(mnn_path),
        "--bizCode", "g2pw",
    ]

    if weight_quant_bits:
        cmd += ["--weightQuantBits", str(weight_quant_bits)]
        if weight_quant_asymmetric:
            cmd.append("--weightQuantAsymmetric")

    run(cmd)

    if not mnn_path.exists():
        fail(f"mnnconvert did not produce {mnn_path}")


def write_readme(out_dir, manifest):
    text = f"""
g2pW model pack (polyphone disambiguation for hanzi -> pinyin)
================================================================

Generated at:
  {manifest['generated_at_utc']}

Source:
  {G2PW_MODEL_ZIP_URL}
  g2pW repo commit (for aux files): {manifest['g2pw_repo_commit']}
  weightQuantBits: {manifest['weight_quant_bits']}

Detected POLYPHONIC_CHARS.txt reading format: {manifest['polyphonic_format']}
  Sample readings seen: {manifest['polyphonic_format_samples']}
  (bopomofo = e.g. "ㄒㄧㄥ2", pinyin = e.g. "xing2". The Android
  postprocessing step needs to know which one this is -- see
  manifest.json's "polyphonic_format" field and branch on it, or just
  always run the bopomofo_to_pinyin_wo_tune_dict.json conversion when the
  format is "bopomofo" and skip it when it's "pinyin".)

Files:
  g2pw.mnn                                converted model
  vocab.txt                               bert-base-chinese WordPiece vocab
  POLYPHONIC_CHARS.txt                    char -> candidate reading(s), tab-separated,
                                           one (char, reading) pair per line
  MONOPHONIC_CHARS.txt                    char -> single fixed reading (skip the model
                                           entirely for these -- see g2pw/api.py's
                                           _prepare_data, which only calls the model for
                                           characters that appear in POLYPHONIC_CHARS.txt)
  char_bopomofo_dict.json                 fallback reading table for characters in
                                           neither of the above two lists
  bopomofo_to_pinyin_wo_tune_dict.json    Bopomofo component -> Pinyin component
                                           (tone digit is the reading's own last
                                           character, carried over unchanged)
  bert-base-chinese_s2t_dict.txt          simplified -> traditional character map.
                                           g2pW's lookup tables (POLYPHONIC_CHARS.txt,
                                           MONOPHONIC_CHARS.txt, char_bopomofo_dict.json)
                                           are keyed by TRADITIONAL characters. When
                                           looking up a simplified character, convert it
                                           to traditional with this table FIRST -- but
                                           still feed the ORIGINAL (simplified) sentence
                                           to the BERT tokenizer itself; bert-base-chinese's
                                           vocab already covers both scripts, only the
                                           g2pW-specific lookup tables need the conversion.

Model I/O (verified against g2pW's export_onnx_model.py):
  Inputs:
    input_ids       int64 [1, seq_len]    WordPiece token ids, [CLS] + tokens + [SEP]
    token_type_ids  int64 [1, seq_len]    all zeros
    attention_mask  int64 [1, seq_len]    all ones
    phoneme_mask    float [1, num_labels] 1.0 at this char's candidate reading indices
    char_ids        int64 [1]             index of the query char in the sorted global char list
    position_ids    int64 [1]             token index of the query char (CLS-offset)
  Output:
    probs           float [1, num_labels]

  num_labels and the global char list are both derived from
  POLYPHONIC_CHARS.txt (sorted distinct readings, and sorted distinct
  characters, respectively) -- NOT stored as separate files, since they're
  fully reconstructible from POLYPHONIC_CHARS.txt with the same sorting
  g2pW's own code uses (see g2pw/dataset.py's get_phoneme_labels /
  get_char_phoneme_labels).

  Note: pos_ids is NOT a graph input for this checkpoint -- no LTP POS tag
  mapping is needed on the Android side for this model.

This pack only covers the model + data conversion step. The native MNN
inference wrapper (tokenizer, mask/id construction, postprocessing) is a
separate follow-up.
""".strip() + "\n"

    (out_dir / "README.txt").write_text(text, encoding="utf-8")


def make_zip(zip_path, files):
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=f.name)


def main():
    parser = argparse.ArgumentParser(description="Build g2pW MNN model + aux data pack.")
    parser.add_argument("--out-dir", default="build/g2pw-model")
    parser.add_argument("--zip", default="build/g2pw-model.zip")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--keep-work", action="store_true")
    parser.add_argument("--g2pw-commit", default=G2PW_REPO_COMMIT, help="GitYCC/g2pW commit SHA to fetch aux files from")
    parser.add_argument("--weight-quant-bits", type=int, default=0, help="0 disables weight quantization (default: off, this model is already small)")
    parser.add_argument("--no-weight-quant-asymmetric", dest="weight_quant_asymmetric", action="store_false", default=True)

    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    zip_path = Path(args.zip).resolve()

    if out_dir.exists():
        log(f"Cleaning {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    work_dir = Path(args.work_dir).resolve() if args.work_dir else out_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    mnn_cmd = find_mnn_convert()

    manifest = {
        "purpose": "g2pW polyphone disambiguation model (ONNX->MNN) + aux data",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_zip_url": G2PW_MODEL_ZIP_URL,
        "g2pw_repo_commit": args.g2pw_commit,
        "weight_quant_bits": args.weight_quant_bits,
    }

    try:
        # 1. Official model zip (contains g2pw.onnx, POLYPHONIC_CHARS.txt,
        #    MONOPHONIC_CHARS.txt, config.py, version).
        zip_path_local = work_dir / "G2PWModel-v2-onnx.zip"
        download_g2pw_model_zip(zip_path_local)
        model_dir = extract_zip(zip_path_local, work_dir / "extracted")

        onnx_path = model_dir / "g2pw.onnx"
        if not onnx_path.exists():
            candidates = list(model_dir.rglob("*.onnx"))
            if not candidates:
                fail(f"No .onnx file found under {model_dir}")
            onnx_path = candidates[0]
            log(f"WARN: g2pw.onnx not found at expected path, using {onnx_path}")

        polyphonic_src = model_dir / "POLYPHONIC_CHARS.txt"
        monophonic_src = model_dir / "MONOPHONIC_CHARS.txt"
        if not polyphonic_src.exists():
            candidates = list(model_dir.rglob("POLYPHONIC_CHARS.txt"))
            if candidates:
                polyphonic_src = candidates[0]
        if not monophonic_src.exists():
            candidates = list(model_dir.rglob("MONOPHONIC_CHARS.txt"))
            if candidates:
                monophonic_src = candidates[0]

        if not polyphonic_src.exists():
            fail(f"POLYPHONIC_CHARS.txt not found under {model_dir}")

        polyphonic_format, samples = detect_polyphonic_format(polyphonic_src)
        manifest["polyphonic_format"] = polyphonic_format
        manifest["polyphonic_format_samples"] = samples[:10]
        log(f"Detected POLYPHONIC_CHARS.txt reading format: {polyphonic_format} (samples: {samples[:10]})")
        if polyphonic_format == "unknown":
            log("WARN: could not confidently detect reading format -- check samples above manually.")

        # 2. Small aux files from the g2pW python package (github raw).
        aux_files = download_g2pw_aux_files(work_dir, args.g2pw_commit)

        # 3. bert-base-chinese vocab.txt (tokenizer).
        vocab_path = download_bert_vocab(work_dir)

        # 4. Convert.
        mnn_path = out_dir / "g2pw.mnn"
        log("Converting g2pw.onnx -> g2pw.mnn ...")
        convert_onnx_to_mnn(
            mnn_cmd, onnx_path, mnn_path,
            weight_quant_bits=args.weight_quant_bits,
            weight_quant_asymmetric=args.weight_quant_asymmetric,
        )

        # 5. Assemble output dir.
        shutil.copy(polyphonic_src, out_dir / "POLYPHONIC_CHARS.txt")
        if monophonic_src.exists():
            shutil.copy(monophonic_src, out_dir / "MONOPHONIC_CHARS.txt")
        else:
            log("WARN: MONOPHONIC_CHARS.txt not found in model zip, skipping")

        config_src = model_dir / "config.py"
        if config_src.exists():
            shutil.copy(config_src, out_dir / "config.py")
        else:
            log("WARN: config.py not found in model zip -- use_char_phoneme/use_mask "
                "can't be confirmed, don't hand-build label/mask indexing without it")

        for fname, path in aux_files.items():
            shutil.copy(path, out_dir / fname)

        shutil.copy(vocab_path, out_dir / "vocab.txt")

    finally:
        if not args.keep_work and work_dir.exists():
            log(f"Removing work dir {work_dir}")
            shutil.rmtree(work_dir, ignore_errors=True)

    write_readme(out_dir, manifest)

    final_files = [out_dir / "g2pw.mnn", out_dir / "vocab.txt", out_dir / "POLYPHONIC_CHARS.txt", out_dir / "README.txt"]
    for optional in ("MONOPHONIC_CHARS.txt", "config.py", "char_bopomofo_dict.json",
                     "bopomofo_to_pinyin_wo_tune_dict.json", "bert-base-chinese_s2t_dict.txt"):
        p = out_dir / optional
        if p.exists():
            final_files.append(p)

    missing = [str(f) for f in final_files if not f.exists()]
    if missing:
        fail("Missing expected output files:\n  " + "\n  ".join(missing))

    manifest["files"] = [
        {"name": f.name, "bytes": f.stat().st_size, "sha256": sha256_file(f)} for f in final_files
    ]

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    zip_files = final_files + [manifest_path]
    make_zip(zip_path, zip_files)

    log("")
    log("Build complete.")
    log(f"Output dir: {out_dir}")
    log(f"Zip:        {zip_path}")
    log(f"Zip SHA256: {sha256_file(zip_path)}")


if __name__ == "__main__":
    main()
