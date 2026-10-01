"""A fresh `carcassonne train` must build the iteration-3 net (<=250k params):
the Pi throughput budget depends on it, and the CLI / TrainConfig defaults are
the only thing a user who passes no arch flags gets."""
from carcassonne.cli.main import _build_parser
from carcassonne.nn.model import CarcassonneNet, param_count
from carcassonne.training.orchestrate import TrainConfig, run_config_dict


def test_default_train_run_builds_the_budget_net() -> None:
    args = _build_parser().parse_args(["train", "--run", "unused"])
    cfg = TrainConfig()
    assert (args.channels, args.n_blocks) == (cfg.channels, cfg.n_blocks)
    arch = run_config_dict(cfg)["net_arch"]
    assert isinstance(arch, dict)
    net = CarcassonneNet.from_arch(arch)
    assert param_count(net) <= 250_000
    # bench measures the same net train builds by default
    bench = _build_parser().parse_args(["bench"])
    assert (bench.channels, bench.n_blocks) == (cfg.channels, cfg.n_blocks)
