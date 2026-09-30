"""Spawn-pool self-play: format-identical output, parent-owned buffer.

The pool is only an execution strategy: a game is fully determined by its seed, so
a pooled game must equal the same seed played sequentially -- that is what lets a
``workers=1`` run and a ``workers=N`` run (or a resume on another machine) share a
buffer without the data depending on how it was generated.
"""

from __future__ import annotations

import copy
import json
import multiprocessing as mp
import os
import random
from collections.abc import Iterable, Iterator
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import pytest
import torch

from carcassonne.agents.mcts import MctsConfig
from carcassonne.nn.model import CarcassonneNet
from carcassonne.training import orchestrate
from carcassonne.training.buffer import ReplayBuffer
from carcassonne.training.orchestrate import TrainConfig, run_config_dict, train
from carcassonne.training.pool import GameResult, Job, SelfplayPool
from carcassonne.training.run import TrainingRun
from carcassonne.training.selfplay import play_selfplay_game


def _tiny_net(seed: int = 0) -> CarcassonneNet:
    torch.manual_seed(seed)
    net = CarcassonneNet(channels=16, n_blocks=1)
    net.eval()
    return net


@pytest.fixture
def one_thread() -> Iterator[None]:
    """Pin the parent to the workers' intra-op thread count for pool-vs-sequential
    comparisons: a different count can reorder float reductions and flip a
    near-tied visit count, which is not a pool bug."""
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(prev)


def _play_seq(net: CarcassonneNet, run: TrainingRun, seed: int, cfg: MctsConfig) -> GameResult:
    return play_selfplay_game(
        net=net, device=torch.device("cpu"), seed=seed, run=run, mcts_config=cfg,
        game_id=f"g{seed:04d}", started_at="t0",
    )


def _move_lines(replay_path: str) -> list[object]:
    """The move-bearing content of a replay: the header (run-specific started_at)
    and each annotation's wall-clock ``think_ms`` are dropped; everything else --
    moves, scores, search values, priors, visit counts -- must match exactly."""
    lines = [json.loads(line) for line in Path(replay_path).read_text().splitlines() if line]
    for line in lines[1:]:
        if isinstance(line.get("annot"), dict):
            line["annot"].pop("think_ms", None)
    return lines[1:]


@pytest.mark.usefixtures("one_thread")
def test_pool_plays_games_and_parent_inserts(tmp_path: Path) -> None:
    run = TrainingRun.create(tmp_path / "run", {"net_arch": {}})
    net = _tiny_net()
    cfg = MctsConfig(sims=2)
    buffer = ReplayBuffer(run.buffer_path)
    results = {}
    env_vars = ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
    parent_env = {var: os.environ.get(var) for var in env_vars}
    pool = SelfplayPool(workers=2, net=net, mcts_config=cfg, run_root=run.root)
    try:
        jobs = [(seed, f"g{seed:04d}", "t0") for seed in (0, 1, 2)]
        for game_id, replay_path, examples in pool.play(jobs):
            buffer.add_game(game_id, replay_path, examples, order_key=int(game_id[1:]))
            results[game_id] = (replay_path, examples)
        # Workers were spawned inside play() with BLAS/OpenMP pinned to one thread;
        # the parent's own environment is left as it was.
        for var in env_vars:
            assert pool._executor.submit(os.getenv, var).result() == "1"
        assert {var: os.environ.get(var) for var in env_vars} == parent_env
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


@pytest.mark.usefixtures("one_thread")
def test_pool_plays_with_the_net_as_of_construction(tmp_path: Path) -> None:
    # The orchestrator promotes by mutating the best net IN PLACE and rebuilding
    # the pool. A pool must play with the weights it was built from: not a stale
    # copy (pool2 must see the new net) and not a live alias (pool1, whose workers
    # only spawn -- and pickle their weights -- on first play, after the mutation,
    # must still see the old one). Either bug silently trains on the wrong player.
    run = TrainingRun.create(tmp_path / "run", {"net_arch": {}})
    cfg = MctsConfig(sims=4)
    seed = 3
    net = _tiny_net(0)
    net_a = copy.deepcopy(net)
    pool1 = SelfplayPool(workers=1, net=net, mcts_config=cfg, run_root=run.root)
    net.load_state_dict(_tiny_net(1).state_dict())  # "promotion": mutate in place
    pool2 = SelfplayPool(workers=1, net=net, mcts_config=cfg, run_root=run.root)
    try:
        [(_, _, ex1)] = pool1.play([(seed, "p1", "t0")])
        [(_, _, ex2)] = pool2.play([(seed, "p2", "t0")])
    finally:
        pool1.close()
        pool2.close()

    seq_run = TrainingRun.create(tmp_path / "seq", {"net_arch": {}})
    assert ex1 != ex2  # the mutation changes the game, so the checks below bite
    assert ex1 == _play_seq(net_a, seq_run, seed, cfg)[2]
    assert ex2 == _play_seq(net, seq_run, seed, cfg)[2]


class _FakePool:
    """Stands in for SelfplayPool in orchestrator tests: plays jobs in-process and,
    optionally, raises BrokenProcessPool after ``fail_after`` results -- what the
    parent sees when a worker dies (e.g. to a signal during its spawn bootstrap).
    Tests the orchestrator's handling without real process death."""

    fail_after: int | None = None
    on_break: object = None  # callable run just before raising (e.g. raise the stop flag)
    seeds: list[int] = []
    closed = 0

    def __init__(
        self, workers: int, net: CarcassonneNet, mcts_config: MctsConfig, run_root: Path
    ) -> None:
        self._net, self._cfg, self._run = net, mcts_config, TrainingRun(run_root)

    def play(self, jobs: Iterable[Job]) -> Iterator[GameResult]:
        for i, (seed, game_id, started_at) in enumerate(jobs):
            if i == _FakePool.fail_after:
                if callable(_FakePool.on_break):
                    _FakePool.on_break()
                raise BrokenProcessPool("worker died")
            _FakePool.seeds.append(seed)
            yield play_selfplay_game(
                net=self._net, device=torch.device("cpu"), seed=seed, run=self._run,
                mcts_config=self._cfg, game_id=game_id, started_at=started_at,
            )

    def close(self) -> None:
        _FakePool.closed += 1


@pytest.fixture
def fake_pool(monkeypatch: pytest.MonkeyPatch) -> type[_FakePool]:
    monkeypatch.setattr(orchestrate, "SelfplayPool", _FakePool)
    monkeypatch.setattr(_FakePool, "fail_after", None)
    monkeypatch.setattr(_FakePool, "on_break", None)
    monkeypatch.setattr(_FakePool, "seeds", [])
    monkeypatch.setattr(_FakePool, "closed", 0)
    return _FakePool


def _pool_cfg(**overrides: int) -> TrainConfig:
    base = dict(
        channels=8, n_blocks=1, mcts_sims=2, selfplay_games_per_iter=4,
        learn_steps_per_iter=1, gate_every=1000, gate_games=2, window_games=50, workers=2,
    )
    base.update(overrides)
    return TrainConfig(**base)  # type: ignore[arg-type]


def _games_generated(run: TrainingRun) -> int:
    return int(json.loads((run.root / "train_state.json").read_text())["games_generated"])


def test_broken_pool_under_stop_signal_is_a_clean_stop(
    tmp_path: Path, fake_pool: type[_FakePool]
) -> None:
    # Ctrl-C / systemd stop signals the whole group; a worker still importing
    # torch dies before it can ignore the signal. That is the stop the user asked
    # for, not a crash: keep the games already made, skip the partial iteration's
    # learning, flush the normal final checkpoint.
    stop = {"set": False}
    fake_pool.fail_after = 1
    fake_pool.on_break = lambda: stop.update(set=True)
    cfg = _pool_cfg()
    run = TrainingRun.create(tmp_path / "run", run_config_dict(cfg))

    train(run, cfg, device=torch.device("cpu"), max_iterations=3,
          should_stop=lambda: stop["set"])

    assert fake_pool.closed == 1
    assert mp.active_children() == []
    buf = ReplayBuffer(run.buffer_path)
    try:
        assert buf.n_games() == 1  # the game inserted before the break survives
    finally:
        buf.close()
    assert _games_generated(run) == 1
    latest = sorted(run.checkpoints_dir.glob("step_*.pt"))[-1]
    assert latest.name == "step_000000.pt"  # no learning on the partial iteration


def test_broken_pool_without_stop_propagates_and_resume_skips_inserted_seeds(
    tmp_path: Path, fake_pool: type[_FakePool]
) -> None:
    # A worker death nobody asked for is a real failure: fail loud. The games
    # inserted before it are in the buffer, so the persisted counter must cover
    # them -- otherwise a resume replays their seeds and duplicates the games.
    fake_pool.fail_after = 2
    cfg = _pool_cfg()
    run = TrainingRun.create(tmp_path / "run", run_config_dict(cfg))

    with pytest.raises(BrokenProcessPool):
        train(run, cfg, device=torch.device("cpu"), max_iterations=1)
    assert _games_generated(run) == 2

    fake_pool.fail_after = None
    train(TrainingRun.open(run.root), cfg, device=torch.device("cpu"), max_iterations=1)

    # seeds 0,1 before the failure; the resume continues at 2 -- none replayed
    assert fake_pool.seeds == [cfg.seed + i for i in range(6)]
    assert _games_generated(run) == 6
    buf = ReplayBuffer(run.buffer_path)
    try:
        assert buf.n_games() == 6
    finally:
        buf.close()


def test_sequential_failure_mid_iteration_persists_inserted_games(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Same resume guarantee without a pool: games inserted before a failure are
    # counted, so their seeds are never replayed.
    calls = {"n": 0}

    def flaky(**kwargs: object) -> GameResult:
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("game 3 failed")
        return play_selfplay_game(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(orchestrate, "play_selfplay_game", flaky)
    cfg = _pool_cfg(workers=1)
    run = TrainingRun.create(tmp_path / "run", run_config_dict(cfg))
    with pytest.raises(RuntimeError, match="game 3 failed"):
        train(run, cfg, device=torch.device("cpu"), max_iterations=1)
    assert _games_generated(run) == 2
