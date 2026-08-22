"""`carcassonne evaluate` end-to-end: result table, determinism, seat-swap wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from carcassonne.cli.main import main
from carcassonne.game.replay import load_replay

ARGS = ["evaluate", "--p0", "greedy", "--p1", "random", "--games", "4", "--seed", "1"]


def test_evaluate_prints_result_table_and_is_deterministic(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(ARGS)
    out1 = capsys.readouterr().out
    assert rc == 0
    for i in range(4):
        assert f"game {i}: seed={1 + i}" in out1
    assert "result: p0[greedy] vs p1[random] over 4 games" in out1
    assert "wins: p0=" in out1
    assert "draws=" in out1
    assert "mean scores: p0=" in out1
    assert "p0 win rate:" in out1

    # Same invocation -> identical output (game seeds and seat mapping are pinned).
    rc2 = main(ARGS)
    out2 = capsys.readouterr().out
    assert rc2 == 0
    assert out2 == out1


def test_evaluate_no_swap_keeps_p0_in_seat_0(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(
        [
            "evaluate",
            "--p0",
            "greedy",
            "--p1",
            "random",
            "--games",
            "2",
            "--seed",
            "9",
            "--no-swap",
            "--out",
            str(tmp_path),
        ]
    )
    assert rc == 0
    headers = [load_replay(p).header for p in sorted(tmp_path.glob("*.jsonl"))]
    assert len(headers) == 2
    assert all(h.agents == ("greedy", "random") for h in headers)
    assert "replay=" in capsys.readouterr().out


def test_evaluate_unknown_agent_fails_without_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["evaluate", "--p0", "skynet", "--p1", "random", "--games", "1", "--seed", "1"])
    assert rc != 0
    err = capsys.readouterr().err
    assert "skynet" in err
    assert "greedy" in err  # helpful: lists known specs
