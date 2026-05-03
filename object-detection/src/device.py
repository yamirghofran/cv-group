from __future__ import annotations


def resolve_device(requested: str) -> str | int:
    if requested != "auto":
        if requested == "cuda":
            return 0
        return requested

    import torch

    if torch.cuda.is_available():
        return 0
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
