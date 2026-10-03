"""Every boundary that validates a player count agrees with core's constants.

Why: the supported table size is one rule (``MIN_PLAYERS..MAX_PLAYERS``) checked
at four boundaries -- ``core.new_game``, replay headers, the web session store
and the CLI. If one drifts (say the web accepts 6 seats but core rejects them),
a request that passes the outer check fails deep inside, or a replay that core
would play is refused on load. Each site keeps its own error type; this test
pins only that they all accept ``MAX_PLAYERS`` and reject ``MAX_PLAYERS + 1``
(and ``MIN_PLAYERS - 1``), so changing the constants moves all four together.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from carcassonne.cli.main import main
from carcassonne.core import MAX_PLAYERS, MIN_PLAYERS, GameConfig, RulesError, new_game
from carcassonne.game.replay import REPLAY_VERSION, load_replay
from carcassonne.web.app import create_app

OUT_OF_RANGE = [MIN_PLAYERS - 1, MAX_PLAYERS + 1]
IN_RANGE = [MIN_PLAYERS, MAX_PLAYERS]


@pytest.mark.parametrize("n", IN_RANGE)
def test_core_accepts_bounds(n: int) -> None:
    assert len(new_game(0, GameConfig(num_players=n)).scores) == n


@pytest.mark.parametrize("n", OUT_OF_RANGE)
def test_core_rejects_outside_bounds(n: int) -> None:
    with pytest.raises(RulesError, match=rf"{MIN_PLAYERS}\.\.{MAX_PLAYERS}, got {n}"):
        new_game(0, GameConfig(num_players=n))


def _replay_file(tmp_path: Path, n: int) -> Path:
    header = {
        "v": REPLAY_VERSION,
        "seed": 0,
        "config": {"meeples_per_player": 7, "num_players": n},
        "agents": ["random"] * n,
        "checkpoint": None,
        "started_at": "2026-01-01T00:00:00+00:00",
    }
    end = {"end": True, "final_scores": [0] * n, "winners": list(range(n))}
    path = tmp_path / f"np{n}.jsonl"
    path.write_text(json.dumps(header) + "\n" + json.dumps(end) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("n", IN_RANGE)
def test_replay_header_accepts_bounds(tmp_path: Path, n: int) -> None:
    assert load_replay(_replay_file(tmp_path, n)).header.config.num_players == n


@pytest.mark.parametrize("n", OUT_OF_RANGE)
def test_replay_header_rejects_outside_bounds(tmp_path: Path, n: int) -> None:
    with pytest.raises(RulesError, match=rf"{MIN_PLAYERS}\.\.{MAX_PLAYERS}, got {n}"):
        load_replay(_replay_file(tmp_path, n))


@pytest.mark.parametrize("n", IN_RANGE)
def test_web_create_accepts_bounds(tmp_path: Path, n: int) -> None:
    client = TestClient(create_app(tmp_path / "replays"))
    r = client.post("/api/games", json={"opponents": ["random"] * (n - 1), "num_players": n})
    assert r.status_code == 200, r.json()


@pytest.mark.parametrize("n", OUT_OF_RANGE)
def test_web_create_rejects_outside_bounds(tmp_path: Path, n: int) -> None:
    client = TestClient(create_app(tmp_path / "replays"))
    r = client.post("/api/games", json={"opponents": ["random"] * (n - 1), "num_players": n})
    assert r.status_code == 422
    assert r.json()["error"] == "invalid_game_config"
    assert f"{MIN_PLAYERS}..{MAX_PLAYERS} players, got {n}" in r.json()["explanation"]


def _simulate(n: int) -> int:
    return main(["simulate", "--agents", ",".join(["random"] * n), "--seed", "0", "--no-replay"])


@pytest.mark.parametrize("n", IN_RANGE)
def test_cli_accepts_bounds(n: int) -> None:
    assert _simulate(n) == 0


@pytest.mark.parametrize("n", OUT_OF_RANGE)
def test_cli_rejects_outside_bounds(n: int, capsys: pytest.CaptureFixture[str]) -> None:
    assert _simulate(n) == 2
    assert f"need {MIN_PLAYERS}-{MAX_PLAYERS} agents, got {n}" in capsys.readouterr().err
