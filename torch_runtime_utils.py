import os
from pathlib import Path

import torch


def configure_cuda_performance_flags():
    torch.backends.cudnn.benchmark = True

    # Keep TF32 configuration on the legacy path consistently. Mixing the
    # backend-specific fp32_precision API with the older matmul precision API
    # can trip torch.compile/Inductor in newer PyTorch releases.
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True

    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True


def is_torch_compile_failure(exc):
    needles = (
        "torch._dynamo",
        "torch._inductor",
        "torch.compile",
        "Dynamo failed",
        "TorchRuntimeError",
        "fake tensors",
        "BackendCompilerFailed",
        "InductorError",
        "LoweringException",
        "compile_fx",
    )

    current = exc
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = f"{type(current).__name__}: {current}"
        if any(needle in message for needle in needles):
            return True
        current = current.__cause__ or current.__context__
    return False


def mark_torch_compile_step_begin():
    if hasattr(torch, "compiler") and hasattr(
        torch.compiler, "cudagraph_mark_step_begin"
    ):
        torch.compiler.cudagraph_mark_step_begin()


def is_cuda_oom(exc):
    oom_types = tuple(
        t
        for t in (
            getattr(torch, "OutOfMemoryError", None),
            (
                getattr(torch.cuda, "OutOfMemoryError", None)
                if hasattr(torch, "cuda")
                else None
            ),
        )
        if t is not None
    )
    if oom_types and isinstance(exc, oom_types):
        return True
    return "out of memory" in str(exc).lower()


def is_cuda_invalid_argument(exc):
    accelerator_error = getattr(torch, "AcceleratorError", None)
    if accelerator_error is not None and isinstance(exc, accelerator_error):
        return "invalid argument" in str(exc).lower()
    return "cuda error: invalid argument" in str(exc).lower()


def configure_compile_cache(cache_dir):
    cache_path = Path(cache_dir).expanduser().resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    triton_cache_path = cache_path / "triton"
    triton_cache_path.mkdir(parents=True, exist_ok=True)

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_path)
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
    os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "1")
    os.environ.setdefault("TRITON_CACHE_DIR", str(triton_cache_path))
    return str(cache_path)


def load_compile_artifacts(artifact_path):
    load_fn = getattr(getattr(torch, "compiler", None), "load_cache_artifacts", None)
    if load_fn is None:
        return False

    artifact_path = Path(artifact_path)
    if not artifact_path.is_file():
        return False

    try:
        artifact_bytes = artifact_path.read_bytes()
    except OSError as exc:
        print(f"Failed to read torch.compile artifacts from {artifact_path}: {exc}")
        return False

    if not artifact_bytes:
        return False

    try:
        load_fn(artifact_bytes)
    except Exception as exc:
        print(f"Failed to load torch.compile artifacts from {artifact_path}: {exc}")
        return False

    print(f"Loaded torch.compile artifacts from {artifact_path}.")
    return True


def save_compile_artifacts(artifact_path):
    save_fn = getattr(getattr(torch, "compiler", None), "save_cache_artifacts", None)
    if save_fn is None:
        return False

    try:
        serialized = save_fn()
    except Exception as exc:
        print(f"Failed to serialize torch.compile artifacts: {exc}")
        return False

    if not serialized:
        return False

    artifact_bytes, _cache_info = serialized
    if not artifact_bytes:
        return False

    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
    try:
        temp_path.write_bytes(artifact_bytes)
        temp_path.replace(artifact_path)
    except OSError as exc:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        print(f"Failed to write torch.compile artifacts to {artifact_path}: {exc}")
        return False

    print(
        f"Saved torch.compile artifacts to {artifact_path} "
        f"({len(artifact_bytes) / 1024:.1f} KiB)."
    )
    return True
