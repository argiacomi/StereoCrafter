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
