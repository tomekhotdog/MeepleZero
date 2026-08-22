"""Checkpoint opponents + live MCTS hints -- the RL microscope wiring.

A tiny untrained net is saved to a tmp checkpoints dir; the web app is built
around it. MCTS sims are monkeypatched down (``PLAY_SIMS`` = 8) so the search is
real but fast. We assert the *shape and provenance* of the data (real visit
counts, a bounded value, a normalised policy), not any learned quality -- the net
is random.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from carcassonne.agents import checkpoints
from carcassonne.nn import checkpoint as ckpt_io
from carcassonne.nn.model import CarcassonneNet
from carcassonne.web.app import create_app

_TEST_SIMS = 8


@pytest.fixture()
def checkpoints_dir(tmp_path: Path) -> Path:
    """A checkpoints dir holding one tiny untrained checkpoint (step 0)."""
    directory = tmp_path / "ckpts"
    net = CarcassonneNet(channels=8, n_blocks=1)  # tiny: fast to build + search
    ckpt_io.save(directory, step=0, net=net)
    return directory


@pytest.fixture()
def client(tmp_path: Path, checkpoints_dir: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Shrink the search so the interactive path stays fast in tests. Patched before
    # any game is created, since the agent reads PLAY_SIMS at construction time.
    monkeypatch.setattr(checkpoints, "PLAY_SIMS", _TEST_SIMS)
    return TestClient(create_app(tmp_path / "replays", checkpoints_dir))


def test_checkpoints_endpoint_lists_saved_checkpoint(client: TestClient) -> None:
    r = client.get("/api/checkpoints")
    assert r.status_code == 200
    listed = r.json()["checkpoints"]
    assert listed == [{"id": "step_000000", "step": 0}]


def test_checkpoints_endpoint_empty_without_dir(tmp_path: Path) -> None:
    empty = TestClient(create_app(tmp_path / "replays"))  # no checkpoints_dir
    assert empty.get("/api/checkpoints").json() == {"checkpoints": []}


def test_play_vs_checkpoint_records_real_mcts_annot(client: TestClient) -> None:
    r = client.post("/api/games", json={"opponent": "ckpt:latest", "human_player": 0, "seed": 7})
    assert r.status_code == 200
    game_id = r.json()["game_id"]

    saw_ai_reply = False
    for _ in range(3):  # a few moves is enough to observe a real AI reply
        state = client.get(f"/api/games/{game_id}").json()["state"]
        if state["terminal"]:
            break
        moved = client.post(f"/api/games/{game_id}/move", json={"idx": 0})
        assert moved.status_code == 200
        ai_move = moved.json()["ai_move"]
        if ai_move is None:
            continue  # human's move ended the game before the AI replied
        annot = ai_move["annot"]
        assert annot is not None, "MctsAgent must record an annot"
        assert annot["sims"] == _TEST_SIMS  # real search budget, not greedy's 0
        assert -1.0 <= annot["value"] <= 1.0
        top = annot["top"]
        assert top, "search must report top candidates"
        # Real MCTS visit counts: at least one candidate was visited, and total
        # visits cannot exceed the simulation budget.
        assert max(entry["visits"] for entry in top) > 0
        assert sum(entry["visits"] for entry in top) <= _TEST_SIMS
        saw_ai_reply = True
    assert saw_ai_reply, "expected at least one AI reply in the first few moves"


def test_hint_on_checkpoint_game_is_real_search(client: TestClient) -> None:
    r = client.post("/api/games", json={"opponent": "ckpt:latest", "human_player": 0, "seed": 7})
    game_id = r.json()["game_id"]

    body = client.get(f"/api/games/{game_id}/hint").json()
    assert -1.0 <= body["value"] <= 1.0
    policy = body["policy"]
    assert abs(sum(p["prob"] for p in policy) - 1.0) < 1e-9
    top = body["top"]
    assert top, "an MCTS hint must carry top candidates"
    assert max(entry["visits"] for entry in top) > 0  # real search, not priors only


def test_repeated_hints_do_not_perturb_checkpoint_game(client: TestClient) -> None:
    """Hints use a throwaway rng, so spamming them cannot change the trajectory.

    A hinted game and a clean game on the same seed must play identically."""

    def create() -> str:
        r = client.post(
            "/api/games", json={"opponent": "ckpt:latest", "human_player": 0, "seed": 42}
        )
        assert r.status_code == 200
        return str(r.json()["game_id"])

    hinted, clean = create(), create()
    for _ in range(3):
        for _ in range(2):
            assert client.get(f"/api/games/{hinted}/hint").status_code == 200
        moved_h = client.post(f"/api/games/{hinted}/move", json={"idx": 0}).json()
        moved_c = client.post(f"/api/games/{clean}/move", json={"idx": 0}).json()
        assert (moved_h["ai_move"] or {}).get("move") == (moved_c["ai_move"] or {}).get("move")
        assert moved_h["state"]["scores"] == moved_c["state"]["scores"]
        if moved_h["state"]["terminal"]:
            break


def test_unknown_checkpoint_is_422_with_explanation(client: TestClient) -> None:
    r = client.post(
        "/api/games", json={"opponent": "ckpt:step_999999", "human_player": 0, "seed": 1}
    )
    assert r.status_code == 422
    body: dict[str, Any] = r.json()
    assert body["error"] == "unknown_opponent"
    assert "step_999999" in body["explanation"]


def test_checkpoint_id_path_traversal_is_rejected(client: TestClient) -> None:
    # ckpt:../.. must never reach the filesystem (would unpickle an arbitrary file)
    for evil in ("ckpt:../../etc/passwd", "ckpt:../secret", r"ckpt:..\\secret"):
        r = client.post("/api/games", json={"opponent": evil, "human_player": 0, "seed": 1})
        assert r.status_code == 422, evil
