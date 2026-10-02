"""Checkpoint I/O: save/load network + optimizer state, resumably.

A checkpoint is a single ``step_{step:06d}.pt`` file (``torch.save`` of a plain
dict). The filename encodes the training step, so :func:`latest` can pick the
newest without any sidecar index -- keeping resume-from-disk (the Pi requirement)
dead simple. ``net_arch`` is stored alongside the weights so :func:`load` can
reconstruct a :class:`CarcassonneNet` when no live instance is passed.

``format`` marks the payload layout. Iteration 3 (the dynamic-crop net) is a
clean break: a checkpoint without the current marker is refused with one clear
error instead of torch's wall of missing/unexpected state_dict keys.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from carcassonne.nn.model import ARCH_KEYS, CarcassonneNet

CHECKPOINT_FORMAT = 3  # bump when the payload or net layout breaks compatibility

CheckpointId = str  # a checkpoint's filename stem, e.g. "step_000123"

_PREFIX = "step_"
_SUFFIX = ".pt"


def pick_device() -> torch.device:
    """Best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _filename(step: int) -> str:
    return f"{_PREFIX}{step:06d}{_SUFFIX}"


def save(
    directory: Path,
    step: int,
    net: CarcassonneNet,
    optimizer: torch.optim.Optimizer | None = None,
    config: dict[str, Any] | None = None,
) -> Path:
    """Write a checkpoint to ``directory/step_{step:06d}.pt`` and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / _filename(step)
    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "step": step,
        "model_state": net.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "config": config,
        "net_arch": net.arch,
    }
    torch.save(payload, path)
    return path


def latest(directory: Path) -> Path | None:
    """Path of the highest-step checkpoint in ``directory``, or None if empty."""
    candidates = sorted(directory.glob(f"{_PREFIX}*{_SUFFIX}"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: int(p.stem[len(_PREFIX) :]))


def load(
    path: Path,
    device: torch.device,
    net: CarcassonneNet | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Restore a checkpoint.

    Raises :class:`ValueError` if the file isn't a checkpoint dict or its
    ``format`` isn't :data:`CHECKPOINT_FORMAT` (an old run). If ``net`` is None,
    a fresh :class:`CarcassonneNet` is built from the stored ``net_arch``, which
    must then carry every key in :data:`ARCH_KEYS` (ValueError naming the gap).
    Weights (and optimizer state, when both an optimizer and stored state are
    present) are loaded in place. Returns the payload dict augmented
    with the live ``net`` (and ``optimizer``).
    """
    payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError(
            f"checkpoint {path} is not a checkpoint payload (got {type(payload).__name__},"
            f" expected the dict written by save())"
        )
    fmt = payload.get("format")
    if fmt != CHECKPOINT_FORMAT:
        raise ValueError(
            f"checkpoint {path} is from a pre-iteration-3 run (format {fmt}); the "
            f"dynamic-crop net can't load it -- start a new run directory"
        )

    if net is None:
        net = CarcassonneNet.from_arch(_stored_arch(path, payload))
    net.load_state_dict(payload["model_state"])
    net.to(device)

    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])

    payload["net"] = net
    payload["optimizer"] = optimizer
    return payload


def _stored_arch(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """The payload's ``net_arch``, required complete. Unlike run-config creation
    (where :meth:`CarcassonneNet.from_arch` defaults fill gaps on purpose), a
    checkpoint must rebuild exactly the shape it was saved with."""
    arch = payload.get("net_arch")
    if not isinstance(arch, dict):
        raise ValueError(f"checkpoint {path} has no net_arch dict (got {arch!r})")
    for key in ARCH_KEYS:
        if key not in arch:
            raise ValueError(f"checkpoint {path} net_arch is missing {key!r}: {arch!r}")
    return arch
