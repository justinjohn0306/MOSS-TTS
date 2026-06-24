"""Strip the `._checkpoint_wrapped_module` prefix that `module_wrapper` gradient
checkpointing bakes into saved state-dict keys, so checkpoints load cleanly with
AutoModel.from_pretrained.

sft.py's default (and the ZeRO-3-stable) gradient_checkpointing_impl, `module_wrapper`,
wraps each decoder layer in torch's CheckpointWrapper, inserting `._checkpoint_wrapped_module`
into every transformer.layers.* key. Without this fix those weights load as random.
(The tied audio_lm_heads/text_lm_head are intentionally absent from the file — they re-tie to
the saved embeddings on load.)

Usage: python fix_checkpoint_keys.py <output_dir_or_checkpoint_dir>
"""
import glob
import os
import sys

from safetensors import safe_open
from safetensors.torch import save_file

PREFIX = "._checkpoint_wrapped_module"


def fix_one(st_path: str) -> None:
    with safe_open(st_path, framework="pt") as f:
        meta = f.metadata() or {}
        sd = {k: f.get_tensor(k) for k in f.keys()}
    renamed = 0
    new = {}
    for k, v in sd.items():
        nk = k.replace(PREFIX, "")
        if nk != k:
            renamed += 1
        new[nk] = v.contiguous()
    if renamed:
        if "format" not in meta:
            meta["format"] = "pt"
        save_file(new, st_path, metadata=meta)
        print(f"  fixed {renamed}/{len(sd)} keys: {st_path}")
    else:
        print(f"  already clean: {st_path}")


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    direct = os.path.join(target, "model.safetensors")
    paths = [direct] if os.path.isfile(direct) else glob.glob(
        os.path.join(target, "checkpoint-*", "model.safetensors")
    )
    if not paths:
        print(f"no model.safetensors found under {target}")
        return
    for p in paths:
        fix_one(p)
    print("done.")


if __name__ == "__main__":
    main()
