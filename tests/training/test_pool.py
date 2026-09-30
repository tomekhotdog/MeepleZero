"""Spawn-pool self-play: format-identical output, parent-owned buffer.

The pool is only an execution strategy: a game is fully determined by its seed, so
a pooled game must equal the same seed played sequentially -- that is what lets a
``workers=1`` run and a ``workers=N`` run (or a resume on another machine) share a
buffer without the data depending on how it was generated.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import random
from pathlib import Path

import pytest
import torch

from carcassonne.agents.mcts import MctsConfig
from carcassonne.nn.model import CarcassonneNet
from carcassonne.training import orchestrate
from carcassonne.training.buffer import ReplayBuffer
from carcassonne.training.orchestrate import TrainConfig, run_config_dict, train
from carcassonne.training.pool import SelfplayPool
from carcassonne.training.run import TrainingRun
from carcassonne.training.selfplay import play_selfplay_game


def _tiny_net() -> CarcassonneNet:
    torch.manual_seed(0)
    net = CarcassonneNet(channels=16, n_blocks=1)
    net.eval()
    return net


def _move_lines(replay_path: str) -> list[object]:
    """The move-bearing content of a replay: the header (run-specific started_at)
    and each annotation's wall-clock ``think_ms`` are dropped; everything else --
    moves, scores, search values, priors, visit counts -- must match exactly."""
    lines = [json.loads(line) for line in Path(replay_path).read_text().splitlines() if line]
    for line in lines[1:]:
        if isinstance(line.get("annot"), dict):
            line["annot"].pop("think_ms", None)
    return lines[1:]


def test_pool_plays_games_and_parent_inserts(tmp_path: Path) -> None:
    run = TrainingRun.create(tmp_path / "run", {"net_arch": {}})
    net = _tiny_net()
    cfg = MctsConfig(sims=2)
    buffer = ReplayBuffer(run.buffer_path)
    results = {}
    pool = SelfplayPool(workers=2, net=net, mcts_config=cfg, run_root=run.root)
    try:
        jobs = [(seed, f"g{seed:04d}", "t0") for seed in (0, 1, 2)]
        for game_id, replay_path, examples in pool.play(jobs):
            buffer.add_game(game_id, replay_path, examples, order_key=int(game_id[1:]))
            results[game_id] = (replay_path, examples)
    finally:
        pool.close()
    assert buffer.n_games() == 3
    assert len(buffer) > 0
    # states re-derivable: buffer + replays are format-identical to sequential mode
    ex = buffer.sample(1, random.Random(0))[0]
    assert buffer.state_for(ex.game_id, ex.move_n) is not None
    buffer.close()

    # Determinism by seed: the pooled game equals the same seed played sequentially.
    seq_run = TrainingRun.create(tmp_path / "seq", {"net_arch": {}})
    _, seq_replay, seq_examples = play_selfplay_game(
        net=net,
        device=torch.device("cpu"),
        seed=1,
        run=seq_run,
        mcts_config=cfg,
        game_id="g0001",
        started_at="t0",
    )
    pool_replay, pool_examples = results["g0001"]
    assert pool_examples == seq_examples
    assert _move_lines(pool_replay) == _move_lines(seq_replay)


def test_train_with_workers_recreates_pool_on_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Promotion is forced (the gate itself is covered by the smoke test) so the
    # close-and-recreate path runs deterministically; a pool baked with a stale
    # net after promotion would silently generate data from the wrong player.
    events: list[str] = []

    class SpyPool(SelfplayPool):
        def __init__(self, *args: object, **kwargs: object) -> None:
            events.append("open")
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]

        def close(self) -> None:
            events.append("close")
            super().close()

    promotions = iter([True, False])
    monkeypatch.setattr(orchestrate, "SelfplayPool", SpyPool)
    monkeypatch.setattr(
        orchestrate, "_gate_and_maybe_promote", lambda *a, **k: next(promotions)
    )

    cfg = TrainConfig(
        channels=8, n_blocks=1, mcts_sims=2, selfplay_games_per_iter=2,
        learn_steps_per_iter=1, gate_every=1, gate_games=2, window_games=50, workers=2,
    )
    run = TrainingRun.create(tmp_path / "run", run_config_dict(cfg))
    train(run, cfg, device=torch.device("cpu"), max_iterations=2)

    # iter 1: open, promote -> close; iter 2: reopen, no promote; exit -> close.
    assert events == ["open", "close", "open", "close"]
    assert mp.active_children() == []
    buf = ReplayBuffer(run.buffer_path)
    try:
        assert buf.n_games() == 4  # parent inserted every pooled game
    finally:
        buf.close()
    assert json.loads((run.root / "train_state.json").read_text())["games_generated"] == 4
