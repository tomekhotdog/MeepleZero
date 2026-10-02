"""A fresh `carcassonne train` must build the iteration-3 net (<=250k params):
the Pi throughput budget depends on it, and the CLI / TrainConfig defaults are
the only thing a user who passes no arch flags gets. Those defaults must apply
only when a run is *created*: a resumed run keeps the arch its checkpoints were
trained with, or every default change would break resuming existing runs."""
from pathlib import Path

import torch

from carcassonne.cli.main import _build_parser, main
from carcassonne.nn import checkpoint
from carcassonne.nn.model import CarcassonneNet, param_count
from carcassonne.training.buffer import ReplayBuffer
from carcassonne.training.orchestrate import TrainConfig, _establish_nets, run_config_dict
from carcassonne.training.run import TrainingRun

DEVICE = torch.device("cpu")


def test_default_train_run_builds_the_budget_net() -> None:
    args = _build_parser().parse_args(["train", "--run", "unused"])
    cfg = TrainConfig()
    assert (args.channels, args.n_blocks) == (cfg.channels, cfg.n_blocks)
    arch = run_config_dict(cfg)["net_arch"]
    assert isinstance(arch, dict)
    net = CarcassonneNet.from_arch(arch)
    assert param_count(net) <= 250_000


def test_bench_measures_the_net_train_builds_by_default() -> None:
    """The throughput yardstick is only meaningful for the net training uses."""
    bench = _build_parser().parse_args(["bench"])
    cfg = TrainConfig()
    assert (bench.channels, bench.n_blocks) == (cfg.channels, cfg.n_blocks)


def _bootstrap_tiny_run(root: Path) -> Path:
    """The documented Pi bootstrap: create the run with --iterations 0."""
    run_dir = root / "run"
    rc = main(
        ["train", "--run", str(run_dir), "--channels", "8", "--n-blocks", "1",
         "--workers", "1", "--device", "cpu", "--iterations", "0"]
    )
    assert rc == 0
    return run_dir


def test_iterations_zero_creates_a_resumable_run_without_playing(tmp_path: Path) -> None:
    run_dir = _bootstrap_tiny_run(tmp_path)
    assert (run_dir / "config.json").is_file()
    latest = checkpoint.latest(run_dir / "checkpoints")
    assert latest is not None and latest.name == "step_000000.pt"
    assert not list((run_dir / "replays").glob("*.jsonl"))  # no self-play happened


def test_resume_keeps_the_checkpoint_arch_not_the_defaults(tmp_path: Path) -> None:
    run = TrainingRun.open(_bootstrap_tiny_run(tmp_path))
    default_cfg = TrainConfig()  # 48x4: what a resume without arch flags passes
    assert (default_cfg.channels, default_cfg.n_blocks) != (8, 1)
    buffer = ReplayBuffer(run.buffer_path)
    try:
        learner, best = _establish_nets(run, default_cfg, buffer, DEVICE)
    finally:
        buffer.close()
    for net in (learner.net, best):
        assert (net.channels, net.n_blocks) == (8, 1)
