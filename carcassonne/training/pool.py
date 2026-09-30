"""Multiprocess self-play: a spawn-based worker pool playing whole games.

Game-level parallelism only (design decision): each worker owns a private copy
of the best net (rebuilt from a state_dict at pool start), pins torch to one
intra-op thread (4 workers x default threads would oversubscribe the Pi), and
plays complete games via :func:`play_selfplay_game`. Workers write replay files
(distinct per game); the PARENT keeps sole ownership of the sqlite buffer.

Determinism: each game is fully determined by its seed, identical to sequential
mode; the parent supplies explicit order keys, so buffer contents do not depend
on which worker finished first. The pool must be recreated when the best net
changes (promotion) -- the orchestrator does.

Signals: workers ignore SIGINT/SIGTERM. Ctrl-C (and systemd's default stop)
signal the whole process group; the parent's stop flag finishes the current
iteration, so a worker must not die mid-game underneath it. The parent then
shuts the pool down. If the parent dies without doing so (SIGKILL, OOM killer)
a worker would block on its job queue forever, so each runs a watchdog thread
that exits the process once it is re-parented.

Spawn (not fork) context: fork + a threaded torch parent is unsafe, and spawn
behaves identically on macOS dev and the Pi.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import threading
import time
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import torch

from carcassonne.agents.mcts import MctsConfig
from carcassonne.nn.checkpoint import _net_arch
from carcassonne.nn.model import CarcassonneNet
from carcassonne.training.buffer import Example
from carcassonne.training.run import TrainingRun
from carcassonne.training.selfplay import play_selfplay_game

type Job = tuple[int, str, str]  # (seed, game_id, started_at)
type GameResult = tuple[str, str, list[Example]]  # (game_id, replay_path, examples)

# Per-worker-process state, set once by _init_worker (spawned process globals).
_worker_net: CarcassonneNet | None = None
_worker_cfg: MctsConfig | None = None
_worker_run: TrainingRun | None = None


def _init_worker(
    arch: dict[str, int], state_dict: dict[str, Any], mcts_cfg: MctsConfig, run_root: str
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    threading.Thread(target=_exit_when_orphaned, args=(os.getppid(),), daemon=True).start()
    torch.set_num_threads(1)
    global _worker_net, _worker_cfg, _worker_run
    net = CarcassonneNet.from_arch(arch)
    net.load_state_dict(state_dict)
    net.eval()
    _worker_net = net
    _worker_cfg = mcts_cfg
    _worker_run = TrainingRun(Path(run_root))


def _exit_when_orphaned(parent_pid: int) -> None:
    while os.getppid() == parent_pid:
        time.sleep(1.0)
    os._exit(1)


def _play_one(job: Job) -> GameResult:
    assert _worker_net is not None and _worker_cfg is not None and _worker_run is not None
    seed, game_id, started_at = job
    return play_selfplay_game(
        net=_worker_net,
        device=torch.device("cpu"),
        seed=seed,
        run=_worker_run,
        mcts_config=_worker_cfg,
        game_id=game_id,
        started_at=started_at,
    )


class SelfplayPool:
    """A fixed-net worker pool. Recreate on promotion (the net is baked in at start)."""

    def __init__(
        self, workers: int, net: CarcassonneNet, mcts_config: MctsConfig, run_root: Path
    ) -> None:
        state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        self._executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
            initializer=_init_worker,
            initargs=(_net_arch(net), state, mcts_config, str(run_root)),
        )

    def play(self, jobs: Iterable[Job]) -> Iterator[GameResult]:
        """Run all jobs across the workers; yield results in job (submission)
        order. A worker exception is re-raised here, in the parent."""
        yield from self._executor.map(_play_one, list(jobs))

    def close(self) -> None:
        """Shut the workers down (waits for them, cancelling unstarted jobs)."""
        self._executor.shutdown(wait=True, cancel_futures=True)
