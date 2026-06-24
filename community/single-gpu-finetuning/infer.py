"""Synthesize speech from text with MOSS-TTS-Local v1.5 (base model or a finetuned checkpoint).

Usage:
    python infer.py "text to speak" [out.wav] [model_path]

- model_path defaults to the base HF model; pass a checkpoint dir
  (e.g. output/sft/checkpoint-last) to use your finetuned model.
- Inference VRAM ~= model (~9 GB bf16) + codec (~8 GB) < 24 GB.
"""
import os
import sys
import time

import torch
import torchaudio
from transformers import AutoModel, AutoProcessor

DEFAULT_MODEL = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
CODEC = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"

text = sys.argv[1] if len(sys.argv) > 1 else "Hello from MOSS text to speech."
out = sys.argv[2] if len(sys.argv) > 2 else "infer_out.wav"
model_path = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_MODEL
out_dir = os.path.dirname(os.path.abspath(out))
os.makedirs(out_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[infer] model={model_path}\n[infer] device={device}", flush=True)

t = time.time()
processor = AutoProcessor.from_pretrained(
    model_path, trust_remote_code=True,
    codec_path=CODEC, codec_weight_dtype="fp32", codec_compute_dtype="bf16",
)
model = AutoModel.from_pretrained(
    model_path, trust_remote_code=True, dtype=torch.bfloat16,
).to(device)
model.eval()
# Put the codec (audio tokenizer) on the GPU too, else the codes->waveform decode runs on CPU.
if getattr(processor, "audio_tokenizer", None) is not None:
    processor.audio_tokenizer = processor.audio_tokenizer.to(device)
    processor.audio_tokenizer.eval()
print(f"[infer] loaded in {time.time()-t:0.0f}s (model + codec on {device})", flush=True)

conv = [[processor.build_user_message(text=text, language="English")]]
batch = processor(conv, mode="generation")
t = time.time()
with torch.inference_mode():
    gen = model.generate(input_ids=batch["input_ids"].to(device), max_new_tokens=1024)
msg = processor.decode(gen)[0]
audio = msg.audio_codes_list[0].cpu()
sr = processor.model_config.sampling_rate
torchaudio.save(out, audio, sr)
print(f"[infer] generated in {time.time()-t:0.0f}s -> {out}  (shape={tuple(audio.shape)}, sr={sr})", flush=True)
