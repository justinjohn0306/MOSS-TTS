"""Synthesize speech from text with MOSS-TTS-Local v1.5 (base model or a finetuned checkpoint).

Usage:
    python infer.py "text to speak" [out.wav] [model_path] [--ref reference.wav]

- model_path defaults to the base HF model; pass a checkpoint dir
  (e.g. output/sft/checkpoint-last) to use your finetuned model.
- --ref <wav> conditions generation on a reference clip (voice cloning): the codec
  encodes it and the model clones that voice. Use a clean clip a few seconds long.
- Inference VRAM ~= model (~9 GB bf16) + codec (~8 GB) < 24 GB.
"""
import argparse
import os
import time

import torch
import torchaudio
from transformers import AutoModel, AutoProcessor

DEFAULT_MODEL = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
CODEC = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"

ap = argparse.ArgumentParser(description="MOSS-TTS-Local v1.5 inference (optional voice cloning via --ref).")
ap.add_argument("text", help="Text to speak.")
ap.add_argument("out", nargs="?", default="infer_out.wav", help="Output wav path (default: infer_out.wav).")
ap.add_argument("model", nargs="?", default=DEFAULT_MODEL, help="Base model or a finetuned checkpoint dir.")
ap.add_argument("--ref", default=None, help="Reference wav to clone the voice from (optional).")
ap.add_argument("--language", default="English", help="Language tag (default: English).")
ap.add_argument("--max-new-tokens", type=int, default=1024)
args = ap.parse_args()

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[infer] model={args.model}\n[infer] device={device}  ref={args.ref or '(none)'}", flush=True)

t = time.time()
processor = AutoProcessor.from_pretrained(
    args.model, trust_remote_code=True,
    codec_path=CODEC, codec_weight_dtype="fp32", codec_compute_dtype="bf16",
)
model = AutoModel.from_pretrained(
    args.model, trust_remote_code=True, dtype=torch.bfloat16,
).to(device)
model.eval()
# Put the codec (audio tokenizer) on the GPU too: it encodes the reference and decodes the
# generated codes -- otherwise both run on CPU (slow + the "flash_attn unavailable" fallback).
if getattr(processor, "audio_tokenizer", None) is not None:
    processor.audio_tokenizer = processor.audio_tokenizer.to(device)
    processor.audio_tokenizer.eval()
print(f"[infer] loaded in {time.time()-t:0.0f}s (model + codec on {device})", flush=True)

user_kwargs = {"text": args.text, "language": args.language}
if args.ref:
    user_kwargs["reference"] = processor.encode_audios_from_path([args.ref])  # encode the ref clip

conv = [[processor.build_user_message(**user_kwargs)]]
batch = processor(conv, mode="generation")
t = time.time()
with torch.inference_mode():
    gen = model.generate(input_ids=batch["input_ids"].to(device), max_new_tokens=args.max_new_tokens)
msg = processor.decode(gen)[0]
audio = msg.audio_codes_list[0].cpu()
sr = processor.model_config.sampling_rate
torchaudio.save(args.out, audio, sr)
print(f"[infer] generated in {time.time()-t:0.0f}s -> {args.out}  (shape={tuple(audio.shape)}, sr={sr})", flush=True)
