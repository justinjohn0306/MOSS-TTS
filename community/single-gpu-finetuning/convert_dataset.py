r"""Convert a transcript file to MOSS-TTS-Local v1.5 raw JSONL (the input to prepare_data.py).

Accepts TWO metadata layouts (auto-detected per file):

  filelist   <audio_path>|<text>                 e.g.  wavs/1.wav|Hello there, how are you?
             audio_path is taken verbatim (resolved against --audio-root if relative).

  ljspeech   <ID>|<raw>|<normalized>             e.g.  LJ001-0001|There are 12 cats.|There are twelve cats.
             audio is <audio-root>/wavs/<ID>.wav ; text uses the normalized column by default.

Two OUTPUT modes match v1.5's two formats:
  regular     ->  {"audio","text","language"}
  single-ref  ->  {"audio","text","ref_audio","language"}   # voice cloning, one fixed reference

Examples:
  python convert_dataset.py --input /path/to/metadata.csv --output data/train_raw.jsonl
  python convert_dataset.py --input /path/to/metadata.csv --output data/train_ref.jsonl \
         --mode single-ref --ref-audio wavs/1.wav --language English
  python convert_dataset.py --ljspeech-dir /path/to/LJSpeech-1.1 --output data/lj_raw.jsonl
"""
import argparse
import json
import os
import sys

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".wma")


def looks_like_path(s):
    s = s.strip().lower()
    return ("/" in s) or ("\\" in s) or s.endswith(AUDIO_EXTS)


def resolve_ref(ref, audio_root):
    """Resolve --ref-audio given as a path, a filename, or a bare clip ID."""
    cands = [
        ref,
        os.path.join(audio_root, ref),
        os.path.join(audio_root, ref + ".wav"),
        os.path.join(audio_root, "wavs", ref),
        os.path.join(audio_root, "wavs", ref + ".wav"),
    ]
    for c in cands:
        if os.path.isfile(c):
            return os.path.abspath(c)
    return None


def main():
    ap = argparse.ArgumentParser(
        description="Convert a transcript file (filelist or LJSpeech) to MOSS-TTS v1.5 raw JSONL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", help="Transcript file (metadata.csv / filelist.txt).")
    ap.add_argument("--ljspeech-dir", help="Shortcut: <dir>/metadata.csv with audio in <dir>/wavs/.")
    ap.add_argument("--audio-root", help="Base dir for relative audio paths (default: the input file's folder).")
    ap.add_argument("--output", required=True, help="Output .jsonl path.")
    ap.add_argument("--format", choices=["auto", "filelist", "ljspeech"], default="auto",
                    help="Metadata layout (default: auto-detect from the first line).")
    ap.add_argument("--delimiter", default="|", help="Column delimiter (default: '|').")
    ap.add_argument("--mode", choices=["regular", "single-ref"], default="regular",
                    help="regular = {audio,text,language}; single-ref adds a fixed ref_audio (voice cloning).")
    ap.add_argument("--ref-audio", default=None,
                    help="single-ref only: reference wav path / filename / clip ID. Defaults to the first clip.")
    ap.add_argument("--language", default="English",
                    help="Language tag written to every record (full tag, e.g. English, Chinese, Spanish).")
    ap.add_argument("--text-field", choices=["normalized", "raw"], default="normalized",
                    help="LJSpeech only: which transcript column (default: normalized).")
    ap.add_argument("--limit", type=int, default=None, help="Only convert the first N valid clips.")
    ap.add_argument("--relative", action="store_true",
                    help="Write audio paths relative to the output dir instead of absolute.")
    args = ap.parse_args()

    if args.ljspeech_dir:
        root = os.path.abspath(args.ljspeech_dir)
        input_path = os.path.abspath(args.input) if args.input else os.path.join(root, "metadata.csv")
        audio_root = os.path.abspath(args.audio_root) if args.audio_root else root
    elif args.input:
        input_path = os.path.abspath(args.input)
        audio_root = os.path.abspath(args.audio_root) if args.audio_root else os.path.dirname(input_path)
    else:
        sys.exit("[error] provide --input <metadata file> (or --ljspeech-dir <dir>)")
    if not os.path.isfile(input_path):
        sys.exit(f"[error] input not found: {input_path}")

    out_path = os.path.abspath(args.output)
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    with open(input_path, "r", encoding="utf-8-sig") as f:
        lines = [ln.rstrip("\r\n") for ln in f if ln.strip()]
    if not lines:
        sys.exit(f"[error] no rows in {input_path}")

    fmt = args.format
    if fmt == "auto":
        first_col = lines[0].split(args.delimiter, 1)[0]
        fmt = "filelist" if looks_like_path(first_col) else "ljspeech"

    def fmt_path(abs_path):
        return os.path.relpath(abs_path, out_dir) if args.relative else abs_path

    records = []
    n_missing = n_empty = 0
    for line in lines:
        if fmt == "filelist":
            bits = line.split(args.delimiter, 1)
            audio_rel = bits[0].strip()
            text = bits[1].strip() if len(bits) > 1 else ""
            audio_abs = audio_rel if os.path.isabs(audio_rel) else os.path.join(audio_root, audio_rel)
        else:  # ljspeech
            bits = line.split(args.delimiter, 2)
            clip_id = bits[0].strip()
            raw = bits[1].strip() if len(bits) > 1 else ""
            norm = bits[2].strip() if len(bits) > 2 else raw
            text = norm if args.text_field == "normalized" else raw
            audio_abs = os.path.join(audio_root, "wavs", clip_id + ".wav")
        if not text:
            n_empty += 1
            continue
        audio_abs = os.path.abspath(audio_abs)
        if not os.path.isfile(audio_abs):
            n_missing += 1
            continue
        records.append((audio_abs, text))
        if args.limit and len(records) >= args.limit:
            break

    if not records:
        sys.exit(f"[error] no valid records (missing audio: {n_missing}, empty text: {n_empty}). "
                 f"Check --audio-root ({audio_root}) and --format.")

    ref_field = None
    if args.mode == "single-ref":
        if args.ref_audio:
            ref_abs = resolve_ref(args.ref_audio, audio_root)
            if ref_abs is None:
                sys.exit(f"[error] could not resolve --ref-audio {args.ref_audio!r}")
        else:
            ref_abs = records[0][0]
        ref_field = fmt_path(ref_abs)

    with open(out_path, "w", encoding="utf-8") as fout:
        for audio_abs, text in records:
            rec = {"audio": fmt_path(audio_abs), "text": text, "language": args.language}
            if ref_field is not None:
                rec["ref_audio"] = ref_field
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[convert] wrote {len(records)} records -> {out_path}")
    print(f"  format={fmt}  mode={args.mode}  language={args.language!r}  paths={'relative' if args.relative else 'absolute'}")
    if ref_field is not None:
        print(f"  ref_audio = {ref_field}")
    if n_missing:
        print(f"  [warn] skipped {n_missing} rows whose audio file was missing")
    if n_empty:
        print(f"  [warn] skipped {n_empty} rows with empty text")


if __name__ == "__main__":
    main()
