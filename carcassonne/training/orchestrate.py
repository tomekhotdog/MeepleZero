"""The training loop: self-play, learning, and arena gating in one process.

A single process owns everything for a :class:`TrainingRun`, iterating three
phases:

1. **Self-play** with the current *best* net (strongest so far) -> replays + buffer.
2. **Learn** ``learn_steps_per_iter`` gradient steps on the *learning* net.
3. **Gate** (every ``gate_every`` iters): the learning net plays the best net; on
   a >55% win rate it is promoted (checkpointed and copied into the best net).
   A yardstick match vs greedy is always recorded as absolute progress.

**Two nets, deliberately separate** (AlphaGo-Zero-style gating): self-play always
uses the *best* net so training data comes from the strongest available play,
while the *learning* net is free to move around under SGD. Promotion is the only
bridge between them -- ``best.load_state_dict(learning.state_dict())`` on a win.

**Metrics tagging.** The learner appends its own per-step lines (``step``,
``loss``, ...) to ``metrics.jsonl`` unchanged. The orchestrator appends arena
lines tagged ``"kind": "gate"``. The dashboard (Task 21) distinguishes them by
that key: a ``"kind": "gate"`` line is an arena result; an untagged line is a
learner step.

**Resumability.** Everything reconstructs from the run directory: the buffer
sqlite persists, checkpoints persist, ``Learner.from_run`` restores the step, and
``train_state.json`` records the last *promoted* step and total games generated.
On resume the learning net is the latest checkpoint but the best net is the
promoted one from ``train_state.json`` -- so gating discipline survives restarts
(critical on the Pi, which resumes many times). A crash loses at most the current
iteration's un-checkpointed steps; a clean stop (SIGINT/SIGTERM or ``should_stop``)
finishes the current iteration, flushes a checkpoint, and returns.
``games_generated`` is persisted after *every* buffer insert, so a failure part-way
through self-play never makes a resume replay the seeds of games already stored.

**Self-play workers.** With ``workers == 1`` self-play runs in-process. With
``workers > 1`` it runs on a :class:`SelfplayPool` of spawned processes, each
holding a copy of the best net; the pool is created lazily, closed and dropped on
promotion (its net is baked in at start) so the next iteration rebuilds it from
the promoted net, and closed on exit. Seeds and game ids are derived identically
in both modes, and the parent alone inserts into the buffer, so the data does not
depend on the worker count (a run may resume on a machine with a different one)
-- exactly so at the same intra-op thread count (workers pin 1). A group-wide
signal can kill a worker still in its spawn bootstrap (before it ignores
signals); the resulting ``BrokenProcessPool`` with the stop flag set is the stop:
the games already inserted are kept, the partial iteration's learn/gate phases
are skipped (as a stop between iterations would), and the normal final
checkpoint runs. Without the stop flag it propagates.
"""

from __future__ import annotations

import json
import os
import random
import signal
import time
from collections.abc import Callable
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from datetime import UTC, datetime
from types import FrameType

import torch

from carcassonne.agents.mcts import MctsConfig
from carcassonne.nn import checkpoint
from carcassonne.nn.encode import NUM_PLANES
from carcassonne.nn.model import DEFAULT_CHANNELS, DEFAULT_N_BLOCKS, CarcassonneNet
from carcassonne.training.arena import gate, yardstick
from carcassonne.training.buffer import ReplayBuffer
from carcassonne.training.learner import Learner, LearnerConfig
from carcassonne.training.pool import Job, SelfplayPool
from carcassonne.training.run import TrainingRun
from carcassonne.training.selfplay import play_selfplay_game

# Seed namespaces kept far apart so self-play games and the two arena matches
# never share a seed (which would replay identical games).
_GATE_SEED_BASE = 10_000_000
_YARDSTICK_SEED_BASE = 20_000_000


@dataclass
class TrainConfig:
    iterations: int = 1_000_000  # effectively "until stopped"
    selfplay_games_per_iter: int = 20
    learn_steps_per_iter: int = 40
    gate_every: int = 5  # iterations between arena gates
    gate_games: int = 20
    window_games: int = 2000  # buffer eviction window
    mcts_sims: int = 100  # self-play + gating sims
    channels: int = DEFAULT_CHANNELS  # the model owns the arch defaults
    n_blocks: int = DEFAULT_N_BLOCKS
    seed: int = 0
    # learner knobs (passed through to LearnerConfig)
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    ckpt_every: int = 100
    workers: int = 1  # self-play processes; 1 = in-process (no pool)


def run_config_dict(cfg: TrainConfig) -> dict[str, object]:
    """The config persisted at ``run.config_path``. ``net_arch`` is what
    ``Learner.from_run`` reads to build a fresh net; ``train`` records the rest."""
    return {
        "net_arch": {"num_planes": NUM_PLANES, "channels": cfg.channels, "n_blocks": cfg.n_blocks},
        "train": {
            "iterations": cfg.iterations,
            "selfplay_games_per_iter": cfg.selfplay_games_per_iter,
            "learn_steps_per_iter": cfg.learn_steps_per_iter,
            "gate_every": cfg.gate_every,
            "gate_games": cfg.gate_games,
            "window_games": cfg.window_games,
            "mcts_sims": cfg.mcts_sims,
            "seed": cfg.seed,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "batch_size": cfg.batch_size,
            "ckpt_every": cfg.ckpt_every,
            "workers": cfg.workers,
        },
    }


class _StopSignal:
    """Context manager that flips a flag on SIGINT/SIGTERM and restores the prior
    handlers on exit. Skips installation off the main thread (e.g. under pytest),
    where ``signal.signal`` raises -- callers there use ``should_stop`` instead."""

    def __init__(self) -> None:
        self._stop = False
        self._prev: dict[int, object] = {}

    def __enter__(self) -> _StopSignal:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._prev[sig] = signal.signal(sig, self._handle)
            except ValueError:
                pass  # not the main thread; nothing to install or restore
        return self

    def _handle(self, signum: int, frame: FrameType | None) -> None:
        self._stop = True

    def __exit__(self, *exc: object) -> None:
        for sig, handler in self._prev.items():
            try:
                signal.signal(sig, handler)  # type: ignore[arg-type]
            except ValueError:
                pass

    @property
    def is_set(self) -> bool:
        return self._stop


def train(
    run: TrainingRun,
    cfg: TrainConfig,
    device: torch.device | None = None,
    max_iterations: int | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> None:
    """Run the AlphaZero training loop on ``run`` until stopped.

    Stops when the iteration limit (``cfg.iterations`` or the tighter
    ``max_iterations``) is reached, a SIGINT/SIGTERM arrives, or ``should_stop()``
    returns True. The stop is checked between iterations so the current one always
    finishes cleanly -- except when the stop signal itself broke the self-play pool,
    which ends self-play there and skips that iteration's learn/gate. A final
    checkpoint is always flushed before returning.
    """
    device = device or checkpoint.pick_device()
    limit = cfg.iterations if max_iterations is None else min(cfg.iterations, max_iterations)

    with _StopSignal() as stop:

        def stopping() -> bool:
            return stop.is_set or (should_stop is not None and should_stop())

        buffer = ReplayBuffer(run.buffer_path)
        pool: SelfplayPool | None = None
        try:
            learner, best_net = _establish_nets(run, cfg, buffer, device)
            gate_cfg = MctsConfig(sims=cfg.mcts_sims)
            # order_key must strictly increase across resumes so newer games survive
            # eviction; wall-clock ns gives that. game_counter is the total games ever
            # inserted (persisted after each insert), so self-play seeds never repeat
            # across resumes -- even after a failure part-way through an iteration.
            order_key = time.time_ns()
            state = _read_state(run)
            game_counter = state["games_generated"]

            for it in range(1, limit + 1):
                if stopping():
                    break

                # -- self-play (best net) -----------------------------------
                # Both paths derive seed / game_id / order_key identically, so a
                # game's content never depends on the worker count.
                if cfg.workers > 1:
                    if pool is None:
                        pool = SelfplayPool(cfg.workers, best_net, gate_cfg, run.root)
                    jobs: list[Job] = []
                    keys: dict[str, int] = {}
                    for i in range(cfg.selfplay_games_per_iter):
                        started_at = datetime.now(UTC).isoformat(timespec="seconds")
                        game_id = f"g{order_key + i:020d}"
                        jobs.append((cfg.seed + game_counter + i, game_id, started_at))
                        keys[game_id] = order_key + i
                    order_key += cfg.selfplay_games_per_iter
                    try:
                        # Results arrive in submission order, so after the k-th
                        # insert the persisted counter covers exactly jobs[:k].
                        for game_id, replay_path, examples in pool.play(jobs):
                            buffer.add_game(game_id, replay_path, examples, keys[game_id])
                            game_counter += 1
                            _write_state(run, games_generated=game_counter)
                    except BrokenProcessPool:
                        if not stopping():
                            raise
                        break  # a stop signal killed a still-booting worker: stop now
                else:
                    for _ in range(cfg.selfplay_games_per_iter):
                        started_at = datetime.now(UTC).isoformat(timespec="seconds")
                        game_id = f"g{order_key:020d}"
                        _, replay_path, examples = play_selfplay_game(
                            net=best_net,
                            device=device,
                            seed=cfg.seed + game_counter,
                            run=run,
                            mcts_config=gate_cfg,
                            game_id=game_id,
                            started_at=started_at,
                        )
                        buffer.add_game(game_id, replay_path, examples, order_key)
                        order_key += 1
                        game_counter += 1
                        _write_state(run, games_generated=game_counter)
                buffer.evict_to(cfg.window_games)

                # -- learn (learning net) -----------------------------------
                last: dict[str, float] = {}
                for _ in range(cfg.learn_steps_per_iter):
                    last = learner.train_step()

                # -- gate (learning vs best) --------------------------------
                gated = it % cfg.gate_every == 0
                promoted = gated and _gate_and_maybe_promote(
                    run, cfg, learner, best_net, device, gate_cfg, it
                )
                if promoted and pool is not None:
                    pool.close()
                    pool = None  # recreated next iteration with the promoted net

                _print_progress(it, buffer, last, gated)

            learner.checkpoint()  # clean-stop flush: always resumable
        finally:
            if pool is not None:
                pool.close()
            buffer.close()


def _establish_nets(
    run: TrainingRun, cfg: TrainConfig, buffer: ReplayBuffer, device: torch.device
) -> tuple[Learner, CarcassonneNet]:
    """Build the learning net (restored or fresh) and the best net.

    A fresh run has no checkpoint yet, so ``Learner.from_run`` builds a fresh net;
    we snapshot it at step 0 as the initial best (so on a fresh run learning and
    best start byte-identical). The best net is then loaded from the last
    *promoted* checkpoint recorded in ``train_state.json``.
    """
    learner_cfg = LearnerConfig(
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        batch_size=cfg.batch_size,
        ckpt_every=cfg.ckpt_every,
        device=device,
    )
    was_fresh = checkpoint.latest(run.checkpoints_dir) is None
    learner = Learner.from_run(run, buffer, learner_cfg, random.Random(cfg.seed))
    if was_fresh:
        learner.checkpoint()  # persist the initial (step 0) best
        _write_state(run, best_step=learner.step)
    best_net = _load_best_net(run, device)
    return learner, best_net


def _load_best_net(run: TrainingRun, device: torch.device) -> CarcassonneNet:
    """Load the last *promoted* net (from train_state.json), not merely the latest
    checkpoint -- otherwise a resume would seed self-play from an un-gated net.

    Raises :class:`FileNotFoundError` if the promoted checkpoint is gone: nothing
    prunes checkpoints and the step-0 best is written before ``best_step`` is, so
    a missing file means the run directory was tampered with. Falling back to the
    latest checkpoint would silently defeat the arena gate."""
    best_step = _read_state(run)["best_step"]
    path = run.checkpoints_dir / f"step_{best_step:06d}.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"training run at {run.root} records best_step={best_step} but its "
            f"checkpoint {path} is missing; refusing to seed self-play from an "
            f"un-gated checkpoint"
        )
    net: CarcassonneNet = checkpoint.load(path, device)["net"]
    net.eval()
    return net


def _read_state(run: TrainingRun) -> dict[str, int]:
    path = run.root / "train_state.json"
    if not path.is_file():
        return {"best_step": 0, "games_generated": 0}
    data: dict[str, int] = json.loads(path.read_text(encoding="utf-8"))
    return {
        "best_step": data.get("best_step", 0),
        "games_generated": data.get("games_generated", 0),
    }


def _write_state(run: TrainingRun, *, best_step: int | None = None,
                 games_generated: int | None = None) -> None:
    """Merge-update the persisted run state (last promoted step, total games).

    Written once per self-play game, so it is replaced atomically (temp file +
    ``os.replace``): a kill mid-write must not leave a truncated file."""
    state = _read_state(run)
    if best_step is not None:
        state["best_step"] = best_step
    if games_generated is not None:
        state["games_generated"] = games_generated
    path = run.root / "train_state.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)


def _gate_and_maybe_promote(
    run: TrainingRun,
    cfg: TrainConfig,
    learner: Learner,
    best_net: CarcassonneNet,
    device: torch.device,
    gate_cfg: MctsConfig,
    it: int,
) -> bool:
    """Run the gate + yardstick, promote on a win; return whether it promoted."""
    wr_best, promoted = gate(
        learner.net, best_net, device, gate_cfg, cfg.gate_games, _GATE_SEED_BASE + it
    )
    wr_greedy = yardstick(
        learner.net, device, gate_cfg, cfg.gate_games, _YARDSTICK_SEED_BASE + it
    )
    if promoted:
        learner.checkpoint()  # the new best is a real checkpoint
        best_net.load_state_dict(learner.net.state_dict())
        best_net.eval()
        _write_state(run, best_step=learner.step)  # survives resume as the true best
    _append_gate_line(
        run,
        {
            "kind": "gate",
            "iter": it,
            "candidate_step": learner.step,
            "promoted": promoted,
            "wr_best": round(wr_best, 4),
            "wr_greedy": round(wr_greedy, 4),
        },
    )
    print(
        f"  gate @ iter {it}: candidate_step={learner.step} "
        f"wr_best={wr_best:.1%} wr_greedy={wr_greedy:.1%} "
        f"{'PROMOTED' if promoted else 'kept best'}"
    )
    return promoted


def _append_gate_line(run: TrainingRun, line: dict[str, object]) -> None:
    with run.metrics_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(line, separators=(",", ":")) + "\n")


def _print_progress(it: int, buffer: ReplayBuffer, last: dict[str, float], gated: bool) -> None:
    loss = last.get("loss")
    loss_str = f"{loss:.4f}" if loss is not None else "n/a"
    print(
        f"iter {it}: buffer={len(buffer)} examples / {buffer.n_games()} games  "
        f"last_loss={loss_str}"
        + ("  [gated]" if gated else "")
    )
