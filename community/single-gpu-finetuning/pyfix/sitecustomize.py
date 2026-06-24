"""Auto-loaded fixes (via PYTHONPATH) for single-GPU DeepSpeed ZeRO training on native
Windows. Inert on Linux/macOS. Windows lacks NCCL and PyTorch's Windows build lacks libuv,
so DeepSpeed's distributed paths break in three ways -- all patched here -- plus we quiet a
handful of benign Windows-vs-Linux messages that look alarming but are harmless.

Distributed fixes (single GPU only; real multi-GPU needs NCCL on Linux/WSL):
1) No libuv: torch.distributed's rendezvous forces TCPStore(use_libuv=True) -> DistStoreError.
   Hard-force use_libuv=False.
2) No NCCL: accelerate picks backend "nccl" whenever CUDA is available. Force "gloo".
3) gloo can't do GPU collectives (reduce_scatter/all_reduce on CUDA crash). At world_size==1
   every collective is a no-op/local copy, so short-circuit them.

Cosmetic noise suppression (so the console isn't full of scary-looking non-errors):
4) DeepSpeed probes every op's is_compatible() at import; on Windows the async_io/GDS probes
   shell out to the MSVC toolchain to test-link aio.lib/cufile.lib and print
   "LINK : fatal error LNK1181: cannot open input file 'aio.lib'/'cufile.lib'" to stdout.
   Those ops don't exist on Windows and we don't use them -> stdout muted during that probe.
5) torch.distributed.elastic logs SIGHUP/SIGQUIT signal-handler tracebacks and a
   "Redirects are currently not supported" note on Windows -> filtered out.
6) c10d prints a "[W] socket ... failed to connect ... 10049" rendezvous warning ->
   TORCH_CPP_LOG_LEVEL=ERROR (set before torch imports).
"""
import sys


def _quiet_windows_noise():
    """(5) Drop the benign torch.distributed.elastic warnings (SIGHUP/SIGQUIT, redirects)."""
    import logging

    class _Drop(logging.Filter):
        _pats = (
            "Redirects are currently not supported",
            "Failed to register signal handler",
            "module 'signal' has no attribute",
        )

        def filter(self, record):
            try:
                msg = record.getMessage()
            except Exception:
                return True
            return not any(p in msg for p in self._pats)

    _drop = _Drop()
    for name in (
        "torch.distributed.elastic.multiprocessing.api",
        "torch.distributed.elastic.multiprocessing.redirects",
    ):
        logging.getLogger(name).addFilter(_drop)


def _run_with_stdout_muted(fn):
    """Run fn() with process stdout redirected to NUL at both the CRT-fd and the Win32-handle
    level -- the latter so child processes (cl.exe/link.exe) are muted too, since on Windows a
    subprocess inherits the parent's STD_OUTPUT_HANDLE, not the C runtime's fd 1. Restored after."""
    import os
    import ctypes
    import msvcrt

    STD_OUTPUT_HANDLE = 0xFFFFFFF5  # (DWORD)-11
    k32 = ctypes.windll.kernel32
    k32.GetStdHandle.restype = ctypes.c_void_p
    k32.SetStdHandle.argtypes = (ctypes.c_uint, ctypes.c_void_p)

    try:
        sys.stdout.flush()
    except Exception:
        pass

    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd = os.dup(1)
    saved_handle = k32.GetStdHandle(STD_OUTPUT_HANDLE)
    try:
        os.dup2(devnull_fd, 1)
        k32.SetStdHandle(STD_OUTPUT_HANDLE, msvcrt.get_osfhandle(devnull_fd))
        return fn()
    finally:
        try:
            sys.stdout.flush()
        except Exception:
            pass
        os.dup2(saved_fd, 1)
        if saved_handle:
            k32.SetStdHandle(STD_OUTPUT_HANDLE, saved_handle)
        os.close(saved_fd)
        os.close(devnull_fd)


def _silence_ds_op_probes():
    """(4) `deepspeed/git_version_info.py` calls is_compatible() on every op at import; the
    async_io/GDS probes test-link aio.lib/cufile.lib via MSVC, which prints LNK1181 "fatal
    error" lines to stdout (DeepSpeed only redirects stderr there, so they leak on Windows).
    The ops genuinely can't build on Windows and we don't use them.

    Rather than patch is_compatible per-builder (the probed class varies with the install
    layout -- `op_builder.*` for source installs, `deepspeed.ops.op_builder.*` otherwise), we
    install a one-shot meta-path finder that runs `git_version_info` with stdout muted. That
    swallows the probe's compile/link output (and the child processes') no matter which class
    is probed; the compatibility result is unchanged (False)."""
    import importlib.abc
    import importlib.machinery

    TARGET = "deepspeed.git_version_info"

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != TARGET:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or not getattr(spec, "loader", None) or not hasattr(spec.loader, "exec_module"):
                return spec
            try:
                sys.meta_path.remove(self)  # one-shot: only this module needs muting
            except ValueError:
                pass
            _orig_exec = spec.loader.exec_module

            def _exec(module):
                _run_with_stdout_muted(lambda: _orig_exec(module))

            spec.loader.exec_module = _exec
            return spec

    sys.meta_path.insert(0, _Finder())


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
    import os
    # (6) set before torch imports so c10d's C++ warnings (e.g. the socket 10049 note) are quiet
    os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
    for _fn in (
        _quiet_windows_noise,
        _silence_ds_op_probes,
        _force_no_libuv,
        _force_gloo_backend,
        _shortcircuit_collectives,
    ):
        try:
            _fn()
        except Exception as _e:  # pragma: no cover
            sys.stderr.write(f"[pyfix] {_fn.__name__} skipped: {_e}\n")
