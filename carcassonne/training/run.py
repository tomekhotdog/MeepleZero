"""``TrainingRun``: a resumable training run *is* a directory on disk.

Everything a run needs to survive a crash or power-cut lives under one root:

    <root>/config.json      -- run hyper-parameters (the run's identity)
    <root>/checkpoints/     -- network/optimizer snapshots (nn.checkpoint)
    <root>/replays/         -- one JSONL replay per self-play game
    <root>/buffer.sqlite    -- the replay buffer (training targets)
    <root>/metrics.jsonl    -- per-step learner metrics

``create`` is the single point that lays this out (and refuses to clobber an
existing run); ``open`` reconstructs the handle from an existing directory. The
paths are plain properties so the self-play, buffer, and learner code never
hard-code layout -- this is the Pi "resume from disk" requirement made structural.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingRun:
    root: Path

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def replays_dir(self) -> Path:
        return self.root / "replays"

    @property
    def buffer_path(self) -> Path:
        return self.root / "buffer.sqlite"

    @property
    def metrics_path(self) -> Path:
        return self.root / "metrics.jsonl"

    @property
    def config_path(self) -> Path:
        return self.root / "config.json"

    @staticmethod
    def create(root: Path, config: dict[str, Any]) -> TrainingRun:
        """Lay out a fresh run under ``root`` and persist ``config``.

        Raises :class:`FileExistsError` if a ``config.json`` already exists, so a
        run is never silently overwritten (resume goes through :meth:`open`).
        """
        run = TrainingRun(root)
        if run.config_path.exists():
            raise FileExistsError(f"training run already exists at {root} (config.json present)")
        root.mkdir(parents=True, exist_ok=True)
        run.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        run.replays_dir.mkdir(parents=True, exist_ok=True)
        run.config_path.write_text(
            json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
        )
        return run

    @staticmethod
    def open(root: Path) -> TrainingRun:
        """Handle onto an existing run. Raises if ``root`` is not a training run."""
        run = TrainingRun(root)
        if not run.config_path.exists():
            raise FileNotFoundError(f"no training run at {root} (config.json missing)")
        return run

    def config(self) -> dict[str, Any]:
        parsed: dict[str, Any] = json.loads(self.config_path.read_text(encoding="utf-8"))
        return parsed
