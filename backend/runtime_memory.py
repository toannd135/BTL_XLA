"""Explicit device selection and CUDA memory snapshots for runtime reporting."""

import torch


def resolve_device(requested="auto", role="inference"):
    requested = str(requested).strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda:0", "CUDA detected; selected cuda:0"
        return "cpu", "CUDA unavailable; explicitly using CPU because device=auto"
    if requested == "cpu":
        return "cpu", "CPU explicitly selected"
    if requested == "cuda":
        requested = "cuda:0"
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"{role} requested {requested}, but CUDA is unavailable")
        try:
            index = torch.device(requested).index
            index = torch.cuda.current_device() if index is None else index
            if index >= torch.cuda.device_count():
                raise RuntimeError(f"{role} requested {requested}, but only {torch.cuda.device_count()} CUDA device(s) exist")
        except (ValueError, AssertionError, RuntimeError) as exc:
            raise RuntimeError(f"Invalid {role} device {requested}") from exc
        return requested, f"{requested} explicitly selected"
    raise ValueError(f"Unsupported {role} device {requested!r}; use auto, cpu, or cuda[:index]")


def reset_peak_memory(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(torch.device(device))


def memory_snapshot(device):
    """Return MB counters or None for CPU/unavailable CUDA; never invent zero usage."""
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return None
    d = torch.device(device)
    free, total = torch.cuda.mem_get_info(d)
    mb = 1024 * 1024
    return {
        "device": str(d),
        "total_mb": round(total / mb, 1),
        "free_mb": round(free / mb, 1),
        "allocated_mb": round(torch.cuda.memory_allocated(d) / mb, 1),
        "reserved_mb": round(torch.cuda.memory_reserved(d) / mb, 1),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated(d) / mb, 1),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved(d) / mb, 1),
    }
