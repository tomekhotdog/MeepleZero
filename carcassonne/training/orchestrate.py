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
sqlite persists, checkpoints persist, and ``Learner.from_run`` restores the step.
On resume the latest checkpoint is loaded as *both* the learning and best net (a
pragmatic simplification -- the true historical best is recoverable from the
gate lines if ever needed). A crash loses at most the current iteration's
un-checkpointed steps; a clean stop (SIGINT/SIGTERM or ``should_stop``) finishes
the current iteration, flushes a checkpoint, and returns.

Self-play is sequential here. Multiprocessing workers (design note) are a future
optimisation deliberately not built now -- sequential keeps the loop, and the
smoke test, simple and fast.
"""

from __future__ import annotations

import json
import random
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from types import FrameType

import torch

from carcassonne.agents.mcts import MctsConfig
from carcassonne.nn import checkpoint
from carcassonne.nn.encode import NUM_PLANES
from carcassonne.nn.model import CarcassonneNet
from carcassonne.training.arena import gate, yardstick
from carcassonne.training.buffer import ReplayBuffer
from carcassonne.training.learner import Learner, LearnerConfig
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
    channels: int = 64
    n_blocks: int = 5
    seed: int = 0
    # learner knobs (passed through to LearnerConfig)
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    ckpt_every: int = 100


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
    finishes cleanly; a final checkpoint is always flushed before returning.
    """
    device = device or checkpoint.pick_device()
    limit = cfg.iterations if max_iterations is None else min(cfg.iterations, max_iterations)

    with _StopSignal() as stop:
        buffer = ReplayBuffer(run.buffer_path)
        try:
            learner, best_net = _establish_nets(run, cfg, buffer, device)
            gate_cfg = MctsConfig(sims=cfg.mcts_sims)
            # order_key must strictly increase across resumes so newer games survive
            # eviction; wall-clock ns gives that. game_counter varies self-play seeds.
            order_key = time.time_ns()
            game_counter = buffer.n_games()

            for it in range(1, limit + 1):
                if stop.is_set or (should_stop is not None and should_stop()):
                    break

                # -- self-play (best net) -----------------------------------
                for _ in range(cfg.selfplay_games_per_iter):
                    started_at = datetime.now(UTC).isoformat(timespec="seconds")
                    play_selfplay_game(
                        net=best_net,
                        device=device,
                        seed=cfg.seed + game_counter,
                        run=run,
                        mcts_config=gate_cfg,
                        game_id=f"g{order_key:020d}",
                        started_at=started_at,
                        order_key=order_key,
                        buffer=buffer,
                    )
                    order_key += 1
                    game_counter += 1
                buffer.evict_to(cfg.window_games)

                # -- learn (learning net) -----------------------------------
                last: dict[str, float] = {}
                for _ in range(cfg.learn_steps_per_iter):
                    last = learner.train_step()

                # -- gate (learning vs best) --------------------------------
                gated = it % cfg.gate_every == 0
                if gated:
                    _gate_and_maybe_promote(run, cfg, learner, best_net, device, gate_cfg, it)

                _print_progress(it, buffer, last, gated)

            learner.checkpoint()  # clean-stop flush: always resumable
        finally:
            buffer.close()


def _establish_nets(
    run: TrainingRun, cfg: TrainConfig, buffer: ReplayBuffer, device: torch.device
) -> tuple[Learner, CarcassonneNet]:
    """Build the learning net (restored or fresh) and the best net.

    A fresh run has no checkpoint yet, so ``Learner.from_run`` builds a fresh net;
    we snapshot it at step 0 as the initial best. The best net is then loaded from
    the latest checkpoint (step 0 on a fresh run, the resumed step otherwise), so
    learning and best start byte-identical.
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
    best_net = _load_best_net(run, device)
    return learner, best_net


def _load_best_net(run: TrainingRun, device: torch.device) -> CarcassonneNet:
    latest = checkpoint.latest(run.checkpoints_dir)
    assert latest is not None  # _establish_nets guarantees a step-0 checkpoint exists
    net: CarcassonneNet = checkpoint.load(latest, device)["net"]
    net.eval()
    return net


def _gate_and_maybe_promote(
    run: TrainingRun,
    cfg: TrainConfig,
    learner: Learner,
    best_net: CarcassonneNet,
    device: torch.device,
    gate_cfg: MctsConfig,
    it: int,
) -> None:
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
