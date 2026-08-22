"""Checkpoint I/O: save/load network + optimizer state, resumably.

A checkpoint is a single ``step_{step:06d}.pt`` file (``torch.save`` of a plain
dict). The filename encodes the training step, so :func:`latest` can pick the
newest without any sidecar index -- keeping resume-from-disk (the Pi requirement)
dead simple. ``net_arch`` is stored alongside the weights so :func:`load` can
reconstruct a :class:`CarcassonneNet` when no live instance is passed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from carcassonne.nn.actions import ACTION_SPACE
from carcassonne.nn.model import CarcassonneNet

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


def _net_arch(net: CarcassonneNet) -> dict[str, int]:
    return {
        "channels": net.channels,
        "n_blocks": net.n_blocks,
        "num_planes": net.num_planes,
        "action_space": ACTION_SPACE,
    }


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
        "step": step,
        "model_state": net.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "config": config,
        "net_arch": _net_arch(net),
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

    If ``net`` is None, a fresh :class:`CarcassonneNet` is built from the stored
    ``net_arch``. Weights (and optimizer state, when both an optimizer and stored
    state are present) are loaded in place. Returns the payload dict augmented
    with the live ``net`` (and ``optimizer``).
    """
    payload: dict[str, Any] = torch.load(path, map_location=device)

    if net is None:
        arch = payload["net_arch"]
        net = CarcassonneNet(
            num_planes=arch["num_planes"],
            channels=arch["channels"],
            n_blocks=arch["n_blocks"],
        )
    net.load_state_dict(payload["model_state"])
    net.to(device)

    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])

    payload["net"] = net
    payload["optimizer"] = optimizer
    return payload
