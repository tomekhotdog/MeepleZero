"""End-to-end gate for the WHOLE training system, kept deliberately TINY.

Per the perf guardrail everything here uses channels=8, n_blocks=1, sims=4 and a
couple of games/steps, bounded by ``max_iterations`` -- never an unbounded loop.
This is the only place self-play, the learner, and arena gating run together, so
it is the system smoke test; it must stay fast (target < ~60s).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from carcassonne.agents.mcts import MctsConfig
from carcassonne.nn import checkpoint
from carcassonne.nn.model import CarcassonneNet
from carcassonne.training.arena import gate, yardstick
from carcassonne.training.buffer import ReplayBuffer
from carcassonne.training.orchestrate import TrainConfig, run_config_dict, train
from carcassonne.training.run import TrainingRun

DEVICE = torch.device("cpu")


def _tiny_cfg(**overrides: int) -> TrainConfig:
    base: dict[str, int] = dict(
        channels=8,
        n_blocks=1,
        mcts_sims=4,
        selfplay_games_per_iter=2,
        learn_steps_per_iter=2,
        gate_every=1,
        gate_games=2,
        window_games=50,
    )
    base.update(overrides)
    return TrainConfig(**base)


def _read_metrics(run: TrainingRun) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in run.metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _tiny_net() -> CarcassonneNet:
    torch.manual_seed(0)
    return CarcassonneNet(channels=8, n_blocks=1)


def test_train_smoke(tmp_path: Path) -> None:
    cfg = _tiny_cfg()
    run = TrainingRun.create(tmp_path / "run", run_config_dict(cfg))

    train(run, cfg, device=DEVICE, max_iterations=2)

    # A checkpoint exists and re-loads into a valid net.
    latest = checkpoint.latest(run.checkpoints_dir)
    assert latest is not None
    payload = checkpoint.load(latest, DEVICE)
    assert isinstance(payload["net"], CarcassonneNet)

    # Metrics hold BOTH learner step lines (untagged) and >=1 gate line.
    metrics = _read_metrics(run)
    learner_lines = [m for m in metrics if m.get("kind") != "gate" and "step" in m]
    gate_lines = [m for m in metrics if m.get("kind") == "gate"]
    assert learner_lines, "expected learner step lines"
    assert gate_lines, "expected at least one gate line"
    for g in gate_lines:
        assert 0.0 <= g["wr_best"] <= 1.0  # type: ignore[operator]
        assert 0.0 <= g["wr_greedy"] <= 1.0  # type: ignore[operator]
        assert isinstance(g["promoted"], bool)

    # The buffer was populated by self-play.
    buf = ReplayBuffer(run.buffer_path)
    try:
        assert buf.n_games() > 0
        assert len(buf) > 0
    finally:
        buf.close()


def test_train_resume_continues_steps(tmp_path: Path) -> None:
    # Resume is orthogonal to gating; gate_every high + 1 self-play game keeps this
    # fast (no arena) while still exercising the from-disk restore path.
    cfg = _tiny_cfg(gate_every=100, selfplay_games_per_iter=1)
    run = TrainingRun.create(tmp_path / "run", run_config_dict(cfg))
    train(run, cfg, device=DEVICE, max_iterations=1)

    step_after_first = int(checkpoint.load(_latest(run), DEVICE)["step"])
    metrics_before = len(_read_metrics(run))

    # Resume: re-open the same run and run one more iteration.
    reopened = TrainingRun.open(run.root)
    train(reopened, cfg, device=DEVICE, max_iterations=1)

    step_after_resume = int(checkpoint.load(_latest(reopened), DEVICE)["step"])
    assert step_after_resume > step_after_first  # step counter carried over and advanced
    assert len(_read_metrics(reopened)) > metrics_before  # new lines appended, not clobbered


def test_gate_and_yardstick_contract(tmp_path: Path) -> None:
    net_a = _tiny_net()
    net_b = _tiny_net()
    net_b.load_state_dict(net_a.state_dict())  # identical weights -> balanced play
    mcts = MctsConfig(sims=4)

    win_rate, promote = gate(net_a, net_b, DEVICE, mcts, n_games=2, base_seed=0)
    assert 0.0 <= win_rate <= 1.0
    assert isinstance(promote, bool)
    assert promote == (win_rate > 0.55)

    wr_greedy = yardstick(net_a, DEVICE, mcts, n_games=2, base_seed=0)
    assert 0.0 <= wr_greedy <= 1.0


def test_stop_flag_finishes_current_iteration_cleanly(tmp_path: Path) -> None:
    # gate_every huge so no arena runs; should_stop trips after the first iteration.
    cfg = _tiny_cfg(gate_every=100, selfplay_games_per_iter=1)
    run = TrainingRun.create(tmp_path / "run", run_config_dict(cfg))

    calls = {"n": 0}

    def should_stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 1  # False on the 1st check, True from the 2nd on

    train(run, cfg, device=DEVICE, max_iterations=10, should_stop=should_stop)

    # Exactly one iteration ran (2 learn steps), then a clean flush checkpoint.
    latest = _latest(run)
    assert int(checkpoint.load(latest, DEVICE)["step"]) == cfg.learn_steps_per_iter
    buf = ReplayBuffer(run.buffer_path)
    try:
        assert buf.n_games() == cfg.selfplay_games_per_iter  # one iteration's self-play only
    finally:
        buf.close()


def _latest(run: TrainingRun) -> Path:
    latest = checkpoint.latest(run.checkpoints_dir)
    assert latest is not None
    return latest
