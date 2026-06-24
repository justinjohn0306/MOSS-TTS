"""Auto-loaded fixes (via PYTHONPATH) for single-GPU DeepSpeed ZeRO training on native
Windows. Inert on Linux/macOS. Windows lacks NCCL and PyTorch's Windows build lacks libuv,
so DeepSpeed's distributed paths break in three ways -- all patched here:

1) No libuv: torch.distributed's rendezvous creates TCPStore forcing use_libuv=True, but the
   Windows build has none -> DistStoreError. The USE_LIBUV=0 env var is ignored on some paths;
   we hard-force use_libuv=False.

2) No NCCL: accelerate picks backend "nccl" whenever CUDA is available; Windows has none.
   Force the distributed backend to "gloo".

3) gloo can't do GPU collectives: gloo's reduce_scatter / all_reduce on CUDA tensors crash.
   On a SINGLE process (world_size == 1) every collective is a no-op or a local copy, so we
   short-circuit them. Single-GPU only -- real multi-GPU needs NCCL (Linux/WSL), where these
   patches do nothing.
"""
import sys


def _force_no_libuv():
    import importlib
    for modname in (
        "torch.distributed.rendezvous",
        "torch.distributed.elastic.rendezvous.static_tcp_rendezvous",
        "torch.distributed.distributed_c10d",
    ):
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        ts = getattr(mod, "TCPStore", None)
        if ts is None or getattr(ts, "_no_libuv_wrapped", False):
            continue

        def _make(orig):
            def _wrapped(*args, **kwargs):
                kwargs["use_libuv"] = False
                return orig(*args, **kwargs)
            _wrapped._no_libuv_wrapped = True
            return _wrapped

        mod.TCPStore = _make(ts)


def _force_gloo_backend():
    import torch.distributed as dist
    orig = dist.init_process_group
    if getattr(orig, "_gloo_forced", False):
        return

    def _wrapped(*args, **kwargs):
        if args:
            args = ("gloo",) + tuple(args[1:])
        else:
            kwargs["backend"] = "gloo"
        return orig(*args, **kwargs)

    _wrapped._gloo_forced = True
    dist.init_process_group = _wrapped


class _DummyWork:
    def wait(self, *a, **k):
        return None
    def is_completed(self):
        return True
    def is_success(self):
        return True
    def get_future(self):
        import torch
        fut = torch.futures.Future()
        fut.set_result(None)
        return fut


def _shortcircuit_collectives():
    """Replace torch.distributed collectives with single-process no-ops when world_size == 1.
    DeepSpeed binds these symbols at backend init (after this runs at startup)."""
    import torch.distributed as dist

    def ws(group):
        try:
            return dist.get_world_size(group)
        except Exception:
            try:
                return dist.get_world_size()
            except Exception:
                return 1

    def ret(async_op):
        return _DummyWork() if async_op else None

    def _passthrough(name):
        orig = getattr(dist, name, None)
        if orig is None or getattr(orig, "_ws1_shortcut", False):
            return

        def _wrapped(*args, **kwargs):
            if ws(kwargs.get("group", None)) == 1:
                return ret(kwargs.get("async_op", False))
            return orig(*args, **kwargs)

        _wrapped._ws1_shortcut = True
        setattr(dist, name, _wrapped)

    for _n in ("all_reduce", "reduce", "broadcast", "barrier"):
        _passthrough(_n)

    def _patch_reduce_scatter_tensor():
        orig = dist.reduce_scatter_tensor
        if getattr(orig, "_ws1_shortcut", False):
            return

        def _wrapped(output, input, op=None, group=None, async_op=False):
            if ws(group) == 1:
                o = output.reshape(-1)
                o.copy_(input.reshape(-1)[: o.numel()])
                return ret(async_op)
            return orig(output, input, op=op, group=group, async_op=async_op) if op is not None \
                else orig(output, input, group=group, async_op=async_op)

        _wrapped._ws1_shortcut = True
        dist.reduce_scatter_tensor = _wrapped

    def _patch_all_gather_into_tensor():
        orig = getattr(dist, "all_gather_into_tensor", None)
        if orig is None or getattr(orig, "_ws1_shortcut", False):
            return

        def _wrapped(output_tensor, input_tensor, group=None, async_op=False):
            if ws(group) == 1:
                output_tensor.reshape(-1).copy_(input_tensor.reshape(-1))
                return ret(async_op)
            return orig(output_tensor, input_tensor, group=group, async_op=async_op)

        _wrapped._ws1_shortcut = True
        dist.all_gather_into_tensor = _wrapped

    def _patch_all_gather():
        orig = dist.all_gather
        if getattr(orig, "_ws1_shortcut", False):
            return

        def _wrapped(tensor_list, tensor, group=None, async_op=False):
            if ws(group) == 1:
                tensor_list[0].copy_(tensor)
                return ret(async_op)
            return orig(tensor_list, tensor, group=group, async_op=async_op)

        _wrapped._ws1_shortcut = True
        dist.all_gather = _wrapped

    def _patch_reduce_scatter():
        orig = dist.reduce_scatter
        if getattr(orig, "_ws1_shortcut", False):
            return

        def _wrapped(output, input_list, op=None, group=None, async_op=False):
            if ws(group) == 1:
                output.copy_(input_list[0])
                return ret(async_op)
            return orig(output, input_list, op=op, group=group, async_op=async_op) if op is not None \
                else orig(output, input_list, group=group, async_op=async_op)

        _wrapped._ws1_shortcut = True
        dist.reduce_scatter = _wrapped

    _patch_reduce_scatter_tensor()
    _patch_all_gather_into_tensor()
    _patch_all_gather()
    _patch_reduce_scatter()


if sys.platform == "win32":
    for _fn in (_force_no_libuv, _force_gloo_backend, _shortcircuit_collectives):
        try:
            _fn()
        except Exception as _e:  # pragma: no cover
            sys.stderr.write(f"[pyfix] {_fn.__name__} skipped: {_e}\n")
