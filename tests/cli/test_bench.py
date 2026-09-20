"""`carcassonne bench` times self-play games — the before/after throughput yardstick."""
from carcassonne.cli.main import main


def test_bench_runs_and_reports(capsys) -> None:
    rc = main(["bench", "--games", "1", "--sims", "2", "--channels", "16", "--n-blocks", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "games/hour=" in out
    assert "s/game=" in out
    assert "params=" in out
