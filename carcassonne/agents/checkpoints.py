"""Checkpoint opponents: ``ckpt:<id>`` specs backed by a trained network.

The web app lets you play against any saved checkpoint. A spec ``ckpt:<id>``
resolves a ``.pt`` file from a configured directory, loads its network, and wraps
it in an :class:`~carcassonne.agents.mcts.MctsAgent` doing a modest interactive
search (``PLAY_SIMS`` simulations -- ~1-2s per move on CPU, fine for click-to-play
and click-to-hint).

The registry (:mod:`carcassonne.agents.base`) is a process-global; the checkpoints
directory is per-app. We reconcile this the same way sessions get their replays
dir: a module-level dir set once at app creation via :func:`set_checkpoints_dir`.
Single uvicorn worker, so there is no cross-app contention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from carcassonne.agents.base import Agent, register_prefix
from carcassonne.agents.mcts import MctsAgent, MctsConfig
from carcassonne.nn.checkpoint import latest, load, pick_device

_PREFIX = "ckpt:"

# Interactive search budget. Modest on purpose: MCTS at 100 sims costs ~1-2s per
# move on CPU, acceptable for a single-user click-to-play / click-to-hint UI.
# Tests monkeypatch this down (e.g. to 8) to stay fast.
PLAY_SIMS = 100

_checkpoints_dir: Path | None = None


def set_checkpoints_dir(directory: Path | None) -> None:
    """Point the ``ckpt:`` factory at a directory (or ``None`` to disable it).

    Called once by ``create_app``. Idempotent."""
    global _checkpoints_dir
    _checkpoints_dir = directory


def list_checkpoints() -> list[dict[str, Any]]:
    """Available checkpoints as ``[{id, step}]``, oldest first.

    Empty if no directory is configured or it holds no ``step_*.pt`` files. Only
    the ``step_{n:06d}.pt`` naming (see ``nn.checkpoint``) is listed; the factory
    accepts more id forms, but listing sticks to what training actually writes."""
    if _checkpoints_dir is None or not _checkpoints_dir.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(_checkpoints_dir.glob("step_*.pt")):
        stem = path.stem
        try:
            step = int(stem[len("step_") :])
        except ValueError:
            continue  # a stray step_*.pt with a non-numeric suffix: skip it
        out.append({"id": stem, "step": step})
    return out


def make_checkpoint_agent(spec: str) -> Agent:
    """Build an MctsAgent from a ``ckpt:<id>`` spec. ``ValueError`` if it can't.

    ``<id>`` is ``latest`` (newest by step), a bare filename stem
    (``step_000123``), or a step number (``123`` or ``000123``). The load error
    surfaces as ``ValueError`` so the web layer maps it to a 422."""
    if not spec.startswith(_PREFIX):
        raise ValueError(f"not a checkpoint spec: {spec!r}")
    ident = spec[len(_PREFIX) :]
    if _checkpoints_dir is None:
        raise ValueError(f"no checkpoints directory configured; cannot load {spec!r}")
    path = _resolve(_checkpoints_dir, ident)
    device = pick_device()
    try:
        payload = load(path, device)
    except Exception as e:  # noqa: BLE001 -- torch.load raises assorted errors
        raise ValueError(f"failed to load checkpoint {path.name!r}: {e}") from e
    net = payload["net"]
    agent = MctsAgent(
        net, device, config=MctsConfig(sims=PLAY_SIMS), self_play=False, name=path.stem
    )
    return agent


def _resolve(directory: Path, ident: str) -> Path:
    """Map a ckpt id to an existing file, or raise ``ValueError``.

    The id arrives unsanitised from the web ``opponent`` field, so reject anything
    with path separators or ``..`` — otherwise ``ckpt:../../x`` would load (and
    unpickle) an arbitrary file, an RCE vector under ``serve --host 0.0.0.0``.
    """
    if ident == "latest":
        path = latest(directory)
        if path is None:
            raise ValueError(f"no checkpoints found in {directory}")
        return path
    if "/" in ident or "\\" in ident or ".." in ident:
        raise ValueError(f"invalid checkpoint id {ident!r}")
    candidates = [directory / f"{ident}.pt", directory / f"step_{ident}.pt"]
    if ident.isdigit():
        candidates.append(directory / f"step_{int(ident):06d}.pt")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError(f"no such checkpoint {ident!r} in {directory}")


register_prefix(_PREFIX, make_checkpoint_agent)
