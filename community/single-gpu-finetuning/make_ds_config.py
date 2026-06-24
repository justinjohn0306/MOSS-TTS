"""Write the single-GPU ZeRO-3 + CPU-offload accelerate config with a chosen
gradient_accumulation_steps.

accelerate fills ACCELERATE_GRADIENT_ACCUMULATION_STEPS from the deepspeed config and that
OVERRIDES sft.py's --gradient-accumulation-steps, so the two must be identical. The launchers
regenerate this file from a single value so they can't drift. (`auto` is not allowed here:
accelerate int()'s this field directly.)

Usage: python make_ds_config.py <grad_accum_steps> <out_yaml>
"""
import os
import sys

gas = sys.argv[1] if len(sys.argv) > 1 else "8"
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "configs", "accelerate_zero3_offload_1gpu.yaml"
)

content = f"""# Auto-generated. Single-GPU DeepSpeed ZeRO-3 + CPU offload.
# gradient_accumulation_steps MUST match sft.py's --gradient-accumulation-steps.
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: DEEPSPEED
downcast_bf16: 'no'
deepspeed_config:
  gradient_accumulation_steps: {gas}
  offload_optimizer_device: cpu
  offload_param_device: cpu
  zero3_init_flag: false
  zero3_save_16bit_model: true
  zero_stage: 3
enable_cpu_affinity: false
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
use_cpu: false
"""

os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    f.write(content)
print(f"[make_ds_config] wrote {out} (gradient_accumulation_steps={gas})")
