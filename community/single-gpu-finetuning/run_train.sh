#!/usr/bin/env bash
# Single-GPU full finetuning of MOSS-TTS-Local v1.5 via DeepSpeed ZeRO-3 + CPU offload (Linux).
# Activate your Python env first (torch + accelerate + deepspeed installed).
#
# Usage: bash run_train.sh <train.jsonl> <output_dir> [grad_accum=8] [num_epochs=12]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
FT="$REPO_ROOT/moss_tts_local_v1.5/finetuning"
CFG="$HERE/configs/accelerate_zero3_offload_1gpu.yaml"

TRAIN="${1:-data/train.jsonl}"
OUT="${2:-output/sft}"
GAS="${3:-8}"
EPOCHS="${4:-12}"

# Harmless on Linux (the shims only activate on Windows); keeps one code path.
export PYTHONPATH="$HERE/pyfix:${PYTHONPATH:-}"

# Keep the deepspeed config's grad-accum in sync with the CLI value.
python "$HERE/make_ds_config.py" "$GAS" "$CFG"

echo "[train] data=$TRAIN output=$OUT grad_accum=$GAS epochs=$EPOCHS"
accelerate launch --config_file "$CFG" "$FT/sft.py" \
  --model-path OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
  --codec-path OpenMOSS-Team/MOSS-Audio-Tokenizer-v2 \
  --codec-weight-dtype fp32 \
  --codec-compute-dtype bf16 \
  --train-jsonl "$TRAIN" \
  --output-dir "$OUT" \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps "$GAS" \
  --learning-rate 2.0e-5 \
  --warmup-ratio 0.05 \
  --lr-scheduler-type cosine \
  --num-epochs "$EPOCHS" \
  --save-every-epochs 2 \
  --mixed-precision bf16 \
  --channelwise-loss-weight 1,32 \
  --gradient-checkpointing

# Strip the gradient-checkpointing key prefix so checkpoints load with from_pretrained.
python "$HERE/fix_checkpoint_keys.py" "$OUT"
