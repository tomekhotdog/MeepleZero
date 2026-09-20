"""`carcassonne simulate` end-to-end: replay files verify, errors exit non-zero."""

from __future__ import annotations

import re
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


def test_simulate_four_players_with_agents_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(
        [
            "simulate",
            "--agents",
            "greedy,random,random,random",
            "--seed",
            "5",
            "--out",
            str(tmp_path),
        ]
    )
    assert rc == 0
    (path,) = tmp_path.glob("*.jsonl")
    replay = load_replay(path)
    assert replay.header.agents == ("greedy", "random", "random", "random")
    assert replay.header.config.num_players == 4
    list(replay_states(replay))
    out = capsys.readouterr().out
    assert "winners=" in out and "summary:" in out
    # Tally conservation: each game is classified exactly once — as one seat's
    # win or as shared — so a tie double-counted as a win cannot slip through.
    wins = [int(x) for x in re.findall(r"wins=(\d+)", out)]
    shared_m = re.search(r"shared=(\d+)", out)
    assert shared_m is not None and len(wins) == 4
    assert sum(wins) + int(shared_m.group(1)) == 1  # exactly one game played
    winners = replay.winners
    if len(winners) == 1:
        assert wins[winners[0]] == 1  # the replay's sole winner got the tally
    else:
        assert shared_m.group(1) == "1"


def test_simulate_rejects_both_agent_styles(tmp_path: Path) -> None:
    rc = main(
        [
            "simulate",
            "--agents",
            "random,random",
            "--p0",
            "random",
            "--p1",
            "random",
            "--seed",
            "1",
            "--no-replay",
            "--out",
            str(tmp_path),
        ]
    )
    assert rc != 0


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
