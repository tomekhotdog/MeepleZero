"""`carcassonne simulate` end-to-end: replay files verify, errors exit non-zero."""

from __future__ import annotations

from pathlib import Path

import pytest

from carcassonne.cli.main import main
from carcassonne.game.replay import load_replay, replay_states


def test_simulate_two_games_writes_verified_replays(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(
        [
            "simulate",
            "--p0",
            "random",
            "--p1",
            "random",
            "--seed",
            "7",
            "--out",
            str(tmp_path),
            "--games",
            "2",
        ]
    )
    assert rc == 0

    files = sorted(tmp_path.glob("*.jsonl"))
    assert len(files) == 2
    seeds: set[int] = set()
    for path in files:
        replay = load_replay(path)
        seeds.add(replay.header.seed)
        assert replay.header.agents == ("random", "random")
        list(replay_states(replay))  # full engine-verified replay; must not raise
    assert seeds == {7, 8}  # games use seed, seed+1, ...

    out = capsys.readouterr().out
    assert "game 0: seed=7" in out
    assert "game 1: seed=8" in out
    assert "summary:" in out


def test_simulate_no_replay_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(
        [
            "simulate",
            "--p0",
            "random",
            "--p1",
            "random",
            "--seed",
            "3",
            "--out",
            str(tmp_path),
            "--no-replay",
        ]
    )
    assert rc == 0
    assert list(tmp_path.glob("*.jsonl")) == []
    out = capsys.readouterr().out
    assert "summary:" in out
    assert "replay=" not in out


def test_unknown_agent_spec_fails_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(
        [
            "simulate",
            "--p0",
            "skynet",
            "--p1",
            "random",
            "--seed",
            "1",
            "--out",
            str(tmp_path),
        ]
    )
    assert rc != 0
    err = capsys.readouterr().err
    assert "skynet" in err
    assert "random" in err  # helpful: lists known specs
