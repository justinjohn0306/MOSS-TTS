# Single-GPU Finetuning (Windows & Linux)

A community toolkit for **full finetuning** of MOSS-TTS-Local v1.5 (~4.55B) on a **single
24 GB GPU** using DeepSpeed ZeRO-3 + CPU offload. Runs on Linux and natively on Windows.

**Contributor:** [justinjohn0306](https://github.com/justinjohn0306)

## Overview

v1.5 has ~4.55B parameters, so plain bf16 AdamW needs ~36 GB (weights + grads + optimizer)
and will not fit on a 24 GB card. This toolkit uses **ZeRO-3 with CPU offload** to park the
params and optimizer state in system RAM, keeping GPU usage under 24 GB. It also bundles an
accelerate-config generator, a checkpoint post-processor, an inference script, and — for
Windows — the shims and build needed to run DeepSpeed without NCCL/libuv.

## Requirements

- A 24 GB+ NVIDIA GPU and **~64 GB system RAM** (the offload target).
- The MOSS-TTS repo installed per the root README, plus `accelerate` and `deepspeed`.
- **Linux:** `pip install deepspeed` (NCCL works out of the box).
- **Windows:** Visual Studio 2019+ (or Build Tools) with the C++ workload, a CUDA Toolkit
  matching your PyTorch CUDA, then build DeepSpeed from source (below).

## Installation

**Linux**

```bash
pip install deepspeed
```

**Windows** — DeepSpeed has no official Windows wheels, so build the `cpu_adam` op from
source (set `CUDA_HOME` to your CUDA Toolkit if `CUDA_PATH` isn't it):

```bat
build_deepspeed_windows.bat
```

This builds DeepSpeed (cpu_adam only) from the GitHub source with `DS_SKIP_CUDA_CHECK=1`
and installs it. See [Windows notes](#windows-notes) for what it handles and why.

## Data preparation

**1. Build a raw JSONL** — one object per line. Required fields are `audio` (path to the
target clip) and `text` (its exact transcript). `language` is recommended (a full tag such
as `English`); an optional `ref_audio` (a single reference clip) enables voice cloning.
Relative `audio` paths resolve against the JSONL's folder.

```jsonl
{"audio": "wavs/1.wav", "text": "Hello there, how are you?", "language": "English"}
{"audio": "wavs/2.wav", "text": "A cloned line.", "ref_audio": "wavs/ref.wav", "language": "English"}
```

**2. Encode audio to codes** with the repo's `prepare_data.py`:

```bash
python ../../moss_tts_local_v1.5/finetuning/prepare_data.py \
    --model-path OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
    --codec-path OpenMOSS-Team/MOSS-Audio-Tokenizer-v2 \
    --codec-weight-dtype fp32 --codec-compute-dtype bf16 --device auto \
    --input-jsonl data/train_raw.jsonl --output-jsonl data/train.jsonl
```

## Training

Activate your Python environment, then:

```bash
# Linux
bash run_train.sh data/train.jsonl output/sft 8 12
```

```bat
:: Windows (from cmd)
run_train.bat data\train.jsonl output\sft 8 12
```

Arguments: `<train.jsonl> <output_dir> [grad_accum_steps=8] [num_epochs=12]`. The launcher
regenerates the offload config to match `grad_accum_steps`, runs training, and post-processes
the checkpoints so they load directly.

### Example configuration (~30 min of single-speaker data)

| Parameter | Value |
|---|---|
| Strategy | ZeRO-3 + CPU offload (`configs/accelerate_zero3_offload_1gpu.yaml`) |
| Per-device batch size | 1 |
| Gradient accumulation | 8 (effective batch 8) |
| Learning rate | 2e-5 |
| Scheduler | cosine, 5% warmup |
| Epochs | 12 |
| Mixed precision | bf16 |
| Gradient checkpointing | on (`module_wrapper`) |
| Channelwise loss weight | 1,32 |

Throughput is dominated by offload (params/optimizer move over the PCIe bus each step), so a
30-minute set runs in a few hours on one 24 GB card. Save every 2 epochs and pick the best
checkpoint by ear; reduce epochs (or the learning rate) if it overfits.

## Inference

```bash
python infer.py "Text to speak." out.wav output/sft/checkpoint-last
```

Drop the last argument to use the base model for comparison. The script loads both the model
and the codec onto the GPU.

## Windows notes

Native Windows has no NCCL, and PyTorch's Windows build has no libuv, so DeepSpeed needs a
few shims. They load automatically via `pyfix/sitecustomize.py` on `PYTHONPATH` (the launcher
sets this) and are **inert on Linux**:

- Force `TCPStore(use_libuv=False)` — the Windows PyTorch build has no libuv.
- Force the `gloo` backend — Windows has no NCCL.
- Short-circuit collectives at `world_size == 1` — gloo can't do GPU `reduce_scatter`/
  `all_reduce`; for a single GPU they are no-ops. **Single-GPU only** — multi-GPU needs
  NCCL (Linux or WSL2).

`build_deepspeed_windows.bat` builds DeepSpeed from the GitHub source tag (the PyPI sdist
omits the Windows launcher scripts) with only `cpu_adam` enabled and `DS_SKIP_CUDA_CHECK=1`
(a minor toolkit-vs-torch CUDA mismatch is fine).

## Notes

- `module_wrapper` is the only ZeRO-3-stable gradient-checkpointing mode here, but it leaves a
  `._checkpoint_wrapped_module` prefix in saved keys; the launchers run `fix_checkpoint_keys.py`
  automatically so checkpoints load with `AutoModel.from_pretrained`.
- `zero3_init_flag` is `false` (the model's `__init__` is not safe under `zero.Init`).
- The deepspeed config's `gradient_accumulation_steps` must equal the CLI value; `make_ds_config.py`
  keeps them in sync.

## Files

- `make_ds_config.py` — writes the ZeRO-3 offload accelerate config with a chosen grad-accum.
- `run_train.sh` / `run_train.bat` — single-GPU offload training launchers (Linux / Windows).
- `fix_checkpoint_keys.py` — strips the gradient-checkpointing key prefix from saved checkpoints.
- `infer.py` — synthesize speech from text with a base model or a finetuned checkpoint.
- `build_deepspeed_windows.bat` — build + install DeepSpeed (cpu_adam) on Windows.
- `pyfix/sitecustomize.py` — Windows-only DeepSpeed shims (auto no-op elsewhere).
- `configs/accelerate_zero3_offload_1gpu.yaml` — ZeRO-3 + CPU offload config.
