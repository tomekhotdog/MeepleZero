"""Web API end-to-end tests via FastAPI's TestClient (httpx transport).

Each test builds its own app around a tmp replays dir; full games are played
vs the random agent (fast) by always taking legal move idx 0.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from carcassonne.core.tiles import TILE_TYPES
from carcassonne.game.replay import load_replay, replay_states
from carcassonne.web.app import create_app


@pytest.fixture()
def replays_dir(tmp_path: Path) -> Path:
    return tmp_path / "replays"


@pytest.fixture()
def client(replays_dir: Path) -> TestClient:
    return TestClient(create_app(replays_dir))


def _play_full_game(
    client: TestClient, opponent: str = "random", seed: int = 11
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Create a game and play legal move idx 0 until terminal.

    Returns (game_id, final state view, final /move response body)."""
    r = client.post(
        "/api/games", json={"opponent": opponent, "human_player": 0, "seed": seed}
    )
    assert r.status_code == 200
    game_id: str = r.json()["game_id"]
    state: dict[str, Any] = r.json()["state"]
    last: dict[str, Any] = {}
    while not state["terminal"]:
        legal = client.get(f"/api/games/{game_id}/legal").json()["moves"]
        assert legal, "non-terminal state must have legal moves"
        moved = client.post(f"/api/games/{game_id}/move", json={"idx": 0})
        assert moved.status_code == 200
        last = moved.json()
        state = last["state"]
    return game_id, state, last


# --- 1. health + tiledefs -------------------------------------------------


def test_health(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_tiledefs_covers_all_tile_types_and_d_is_correct(client: TestClient) -> None:
    r = client.get("/api/tiledefs")
    assert r.status_code == 200
    tiles = r.json()["tiles"]
    assert set(tiles) == set(TILE_TYPES)
    d = tiles["D"]  # city N, road E-W: the start tile type
    assert d["edges"] == ["city", "road", "field", "road"]
    assert d["features"] == [
        {"kind": "city", "edges": [0], "shield": False},
        {"kind": "road", "edges": [1, 3], "shield": False},
    ]


# --- 2. full scripted game vs random --------------------------------------


def test_full_game_vs_random_writes_verified_replay(
    client: TestClient, replays_dir: Path
) -> None:
    game_id, state, last = _play_full_game(client, seed=5)

    assert state["terminal"] is True
    assert len(state["final_scores"]) == 2
    assert last["ai_move"] is None or last["ai_move"]["move"] is not None

    # game end announces the replay file, which loads and engine-verifies
    name = last["replay"]
    path = replays_dir / name
    assert path.exists()
    replay = load_replay(path)
    assert replay.header.agents == ("human", "random")
    states = list(replay_states(replay))  # engine-verified; must not raise
    assert len(states) == len(replay.moves) + 1
    assert replay.final_scores == tuple(state["final_scores"])

    # GET state agrees; hint on a finished game is a 404
    g = client.get(f"/api/games/{game_id}")
    assert g.json()["state"] == state
    assert client.get(f"/api/games/{game_id}/hint").status_code == 404


# --- 3. AI moves first when human is player 1 ------------------------------


def test_ai_moves_first_when_human_is_player_1(client: TestClient) -> None:
    r = client.post(
        "/api/games", json={"opponent": "greedy", "human_player": 1, "seed": 3}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["state"]["turn"] == 1
    assert body["state"]["current_player"] == 1
    g = client.get(f"/api/games/{body['game_id']}")
    assert g.status_code == 200
    assert g.json()["state"]["turn"] == 1
    assert len(g.json()["state"]["tiles"]) == 2  # start tile + the AI's placement


# --- 4. error mapping -------------------------------------------------------


def test_stale_or_invalid_idx_is_409_with_explanation(client: TestClient) -> None:
    r = client.post("/api/games", json={"opponent": "random", "human_player": 0, "seed": 1})
    game_id = r.json()["game_id"]
    n = len(client.get(f"/api/games/{game_id}/legal").json()["moves"])
    bad = client.post(f"/api/games/{game_id}/move", json={"idx": n})
    assert bad.status_code == 409
    body = bad.json()
    assert body["error"]
    assert body["explanation"]


def test_unknown_game_is_404(client: TestClient) -> None:
    r = client.get("/api/games/deadbeef")
    assert r.status_code == 404
    assert r.json()["error"]
    assert r.json()["explanation"]


def test_unknown_opponent_is_422(client: TestClient) -> None:
    r = client.post("/api/games", json={"opponent": "skynet", "human_player": 0})
    assert r.status_code == 422
    assert "skynet" in r.json()["explanation"]


# --- 5. hint ---------------------------------------------------------------


def test_hint_greedy_returns_normalised_policy_over_legal_moves(
    client: TestClient,
) -> None:
    r = client.post("/api/games", json={"opponent": "greedy", "human_player": 0, "seed": 9})
    game_id = r.json()["game_id"]
    hint = client.get(f"/api/games/{game_id}/hint")
    assert hint.status_code == 200
    body = hint.json()
    assert -1.0 <= body["value"] <= 1.0
    policy = body["policy"]
    assert policy
    assert abs(sum(p["prob"] for p in policy) - 1.0) < 1e-9
    legal = client.get(f"/api/games/{game_id}/legal").json()["moves"]
    placements = {(m["x"], m["y"], m["rot"]) for m in legal}
    assert all((p["x"], p["y"], p["rot"]) in placements for p in policy)


def test_hint_random_is_normalised_too(client: TestClient) -> None:
    r = client.post("/api/games", json={"opponent": "random", "human_player": 0, "seed": 9})
    game_id = r.json()["game_id"]
    body = client.get(f"/api/games/{game_id}/hint").json()
    assert abs(sum(p["prob"] for p in body["policy"]) - 1.0) < 1e-9
    assert body["value"] == 0.0


# --- 6. replays ------------------------------------------------------------


def test_replays_list_and_detail(client: TestClient, replays_dir: Path) -> None:
    _, state, last = _play_full_game(client, seed=13)
    name = last["replay"]

    # an unfinished file (no end line) is skipped by the listing
    (replays_dir / "unfinished.jsonl").write_text(
        json.dumps({"v": 1, "seed": 0, "config": {"meeples_per_player": 7},
                    "agents": ["a", "b"], "checkpoint": None, "started_at": "t"}) + "\n"
    )

    listed = client.get("/api/replays").json()["replays"]
    assert all(e["name"] != "unfinished.jsonl" for e in listed)
    entry = next(e for e in listed if e["name"] == name)
    assert entry["agents"] == ["human", "random"]
    assert entry["final_scores"] == state["final_scores"]
    assert entry["turns"] == state["turn"]
    assert entry["started_at"]

    detail = client.get(f"/api/replays/{name}")
    assert detail.status_code == 200
    d = detail.json()
    assert d["header"]["agents"] == ["human", "random"]
    assert d["final_scores"] == state["final_scores"]
    assert len(d["states"]) == entry["turns"] + 1
    assert d["states"][-1]["terminal"] is True
    assert d["moves"][0]["n"] == 0
    assert len(d["moves"]) == entry["turns"]


def test_tampered_replay_is_400_with_explanation(
    client: TestClient, replays_dir: Path
) -> None:
    _, _, last = _play_full_game(client, seed=17)
    lines = (replays_dir / last["replay"]).read_text().splitlines()
    rec = json.loads(lines[5])
    rec["score_after"][0] += 5
    lines[5] = json.dumps(rec)
    (replays_dir / "tampered.jsonl").write_text("\n".join(lines) + "\n")

    r = client.get("/api/replays/tampered.jsonl")
    assert r.status_code == 400
    assert "score_after" in r.json()["explanation"]


def test_unknown_replay_is_404(client: TestClient) -> None:
    r = client.get("/api/replays/nope.jsonl")
    assert r.status_code == 404
    assert r.json()["error"]


# --- review-fix regression tests -------------------------------------------


def test_hints_do_not_perturb_the_game_trajectory(client: TestClient) -> None:
    """Peeking at hints must never change how the AI subsequently plays.

    Two same-seed games vs greedy, one spammed with hints, must stay identical."""

    def create(seed: int) -> str:
        r = client.post("/api/games", json={"opponent": "greedy", "human_player": 0, "seed": seed})
        assert r.status_code == 200
        game_id: str = r.json()["game_id"]
        return game_id

    hinted, clean = create(99), create(99)
    for _ in range(12):  # enough turns for divergence to show if hints leaked rng
        for _ in range(3):
            assert client.get(f"/api/games/{hinted}/hint").status_code == 200
        moved_h = client.post(f"/api/games/{hinted}/move", json={"idx": 0}).json()
        moved_c = client.post(f"/api/games/{clean}/move", json={"idx": 0}).json()
        # compare the moves themselves, not annots (think_ms is wall-clock)
        assert (moved_h["ai_move"] or {}).get("move") == (moved_c["ai_move"] or {}).get("move")
        assert moved_h["state"]["scores"] == moved_c["state"]["scores"]
        if moved_h["state"]["terminal"]:
            break


def test_meeple_appears_in_state_view(client: TestClient) -> None:
    r = client.post("/api/games", json={"opponent": "random", "human_player": 0, "seed": 3})
    game_id = r.json()["game_id"]
    legal = client.get(f"/api/games/{game_id}/legal").json()["moves"]
    idx = next(i for i, m in enumerate(legal) if (m["action"] or {}).get("type") == "meeple")
    chosen = legal[idx]
    state = client.post(f"/api/games/{game_id}/move", json={"idx": idx}).json()["state"]
    placed = next(t for t in state["tiles"] if t["x"] == chosen["x"] and t["y"] == chosen["y"])
    meeple = next(
        m
        for m in placed["meeples"]
        if m["player"] == 0
        and m["kind"] == "meeple"
        and m["feature"] == chosen["action"]["feature"]
    )
    # the meeple carries its feature's scoring report for hover breakdown
    assert meeple["score_now"] >= 0
    assert meeple["score_potential"] >= meeple["score_now"]  # completing never loses points
    assert isinstance(meeple["complete"], bool)
    assert meeple["feature_kind"] in {"city", "road", "monastery", "garden"}
    tiles = meeple["feature_tiles"]
    assert tiles  # non-empty footprint
    assert [chosen["x"], chosen["y"]] in tiles  # includes the meeple's own tile


def test_session_store_evicts_and_closes_abandoned_writers(tmp_path: Path) -> None:
    from carcassonne.web.sessions import _MAX_SESSIONS, SessionStore

    store = SessionStore(replays_dir=tmp_path)
    ids = [store.create("random", 0, seed).id for seed in range(_MAX_SESSIONS + 3)]
    assert len(store._sessions) <= _MAX_SESSIONS
    # evicted sessions are gone and their replay files exist as crash artifacts
    evicted = [gid for gid in ids if gid not in store._sessions]
    assert evicted
    artifacts = list(tmp_path.glob("*.jsonl"))
    assert len(artifacts) == len(ids)  # every session wrote its header line
