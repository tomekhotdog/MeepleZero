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


def test_tiledefs_carries_counts_summing_to_72(client: TestClient) -> None:
    tiles = client.get("/api/tiledefs").json()["tiles"]
    assert all(t["count"] >= 1 for t in tiles.values())
    assert sum(t["count"] for t in tiles.values()) == 72


# --- 2. full scripted game vs random --------------------------------------


def test_full_game_vs_random_writes_verified_replay(
    client: TestClient, replays_dir: Path
) -> None:
    game_id, state, last = _play_full_game(client, seed=5)

    assert state["terminal"] is True
    assert len(state["final_scores"]) == 2
    assert len(last["ai_moves"]) <= 1  # 2-player: at most the one AI reply
    assert all(m["seat"] == 1 and m["move"] is not None for m in last["ai_moves"])

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
        assert [m["move"] for m in moved_h["ai_moves"]] == [m["move"] for m in moved_c["ai_moves"]]
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
    ids = [store.create(["random"], 0, seed).id for seed in range(_MAX_SESSIONS + 3)]
    assert len(store._sessions) <= _MAX_SESSIONS
    # evicted sessions are gone and their replay files exist as crash artifacts
    evicted = [gid for gid in ids if gid not in store._sessions]
    assert evicted
    artifacts = list(tmp_path.glob("*.jsonl"))
    assert len(artifacts) == len(ids)  # every session wrote its header line


# --- N-seat sessions --------------------------------------------------------


def test_three_player_game_end_to_end(client: TestClient, replays_dir: Path) -> None:
    r = client.post(
        "/api/games",
        json={"opponents": ["random", "random"], "num_players": 3, "human_player": 1, "seed": 7},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["seats"] == ["random", "human", "random"]
    # AI seat 0 has already moved; it is now the human's turn (seat 1)
    assert body["state"]["current_player"] == 1
    assert [m["seat"] for m in body["ai_moves"]] == [0]
    game_id = body["game_id"]
    state = body["state"]
    last: dict[str, Any] = {}
    while not state["terminal"]:
        moved = client.post(f"/api/games/{game_id}/move", json={"idx": 0})
        assert moved.status_code == 200
        last = moved.json()
        state = last["state"]
    assert len(state["final_scores"]) == 3
    (path,) = replays_dir.glob("*.jsonl")
    replay = load_replay(path)
    assert replay.header.config.num_players == 3
    list(replay_states(replay))
    # the replay names every seat in seat order and ends on the scores the API reported
    detail = client.get(f"/api/replays/{last['replay']}").json()
    assert detail["header"]["agents"] == ["random", "human", "random"]
    assert detail["final_scores"] == state["final_scores"]
    assert list(replay.final_scores) == state["final_scores"]


def _create(client: TestClient, **body: Any) -> Any:
    r = client.post("/api/games", json=body)
    assert r.status_code == 200, r.json()
    return r.json()


def test_human_in_last_seat_waits_for_every_ai_seat_in_order(client: TestClient) -> None:
    body = _create(
        client, opponents=["random"] * 3, num_players=4, human_player=3, seed=21
    )
    assert body["seats"] == ["random", "random", "random", "human"]
    assert [m["seat"] for m in body["ai_moves"]] == [0, 1, 2]
    assert body["state"]["current_player"] == 3
    moved = client.post(f"/api/games/{body['game_id']}/move", json={"idx": 0}).json()
    assert not moved["state"]["terminal"]  # one move into a 72-tile game
    # turn order wraps around: after seat 3 come seats 0, 1, 2
    assert [m["seat"] for m in moved["ai_moves"]] == [0, 1, 2]
    assert moved["state"]["current_player"] == 3


def test_human_in_first_seat_of_four_is_followed_by_seats_1_2_3(client: TestClient) -> None:
    body = _create(
        client, opponents=["random"] * 3, num_players=4, human_player=0, seed=22
    )
    assert body["ai_moves"] == []
    moved = client.post(f"/api/games/{body['game_id']}/move", json={"idx": 0}).json()
    assert [m["seat"] for m in moved["ai_moves"]] == [1, 2, 3]
    assert moved["state"]["current_player"] == 0


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"opponent": "random", "opponents": ["random"]}, id="both"),
        pytest.param({}, id="neither"),
        pytest.param({"opponents": ["random", "random"], "num_players": 4}, id="mismatch"),
        pytest.param({"opponents": ["random"] * 5, "num_players": 6}, id="six_players"),
        pytest.param({"opponents": [], "num_players": 1}, id="one_player"),
        pytest.param({"opponents": ["random", "random"], "num_players": 3, "human_player": 3},
                     id="human_seat_n"),
        pytest.param({"opponent": "random", "human_player": -1}, id="human_seat_negative"),
    ],
)
def test_invalid_game_config_is_a_uniform_422(client: TestClient, body: dict[str, Any]) -> None:
    r = client.post("/api/games", json=body)
    assert r.status_code == 422
    payload = r.json()
    assert payload["error"] == "invalid_game_config"  # our shape, not pydantic's {"detail"}
    assert payload["explanation"]


def test_unknown_opponent_spec_keeps_its_own_error_code(client: TestClient) -> None:
    r = client.post("/api/games", json={"opponents": ["random", "skynet"], "num_players": 3})
    assert r.status_code == 422
    assert r.json()["error"] == "unknown_opponent"


def test_store_rejects_out_of_range_human_seat(tmp_path: Path) -> None:
    """Seat validation lives in the store, so non-HTTP callers get it too."""
    from carcassonne.web.sessions import InvalidGameConfig, SessionStore

    with pytest.raises(InvalidGameConfig):
        SessionStore(replays_dir=tmp_path).create(["random"], human_player=2, seed=0)


def test_rejected_create_never_evicts_a_session(tmp_path: Path) -> None:
    from carcassonne.web.sessions import (
        _MAX_SESSIONS,
        InvalidGameConfig,
        SessionStore,
        UnknownOpponent,
    )

    store = SessionStore(replays_dir=tmp_path)
    oldest = store.create(["random"], 0, 0).id
    for seed in range(1, _MAX_SESSIONS):
        store.create(["random"], 0, seed)
    assert len(store._sessions) == _MAX_SESSIONS  # at capacity: the next create evicts
    with pytest.raises(UnknownOpponent):
        store.create(["skynet"], 0, 99)
    with pytest.raises(InvalidGameConfig):
        store.create(["random"], 5, 99)
    assert store.get(oldest).id == oldest
    assert len(store._sessions) == _MAX_SESSIONS


def test_checkpoint_opponents_rejected_beyond_two_players(client: TestClient) -> None:
    r = client.post(
        "/api/games",
        json={"opponents": ["random", "ckpt:step_000100"], "num_players": 3},
    )
    assert r.status_code == 422
    # the guard's own message, not make_agent's "no checkpoints directory" 422
    assert "2-player only" in r.json()["explanation"]


def test_hint_in_three_player_game_is_an_explicit_409(client: TestClient) -> None:
    """No single opponent to ask: an explicit refusal, never a fabricated uniform hint."""
    r = client.post(
        "/api/games",
        json={"opponents": ["random", "random"], "num_players": 3, "human_player": 0, "seed": 5},
    )
    hint = client.get(f"/api/games/{r.json()['game_id']}/hint")
    assert hint.status_code == 409
    assert hint.json() == {
        "error": "hint_unavailable",
        "explanation": "hints are a 2-player feature (this game has 3 players)",
    }


def test_legacy_two_player_body_still_works(client: TestClient) -> None:
    r = client.post("/api/games", json={"opponent": "random", "human_player": 0, "seed": 3})
    assert r.status_code == 200
    assert r.json()["seats"] == ["human", "random"]


def test_opening_ai_moves_report_their_score_events(client: TestClient) -> None:
    # Why: when AI seats move before the human, their scoring must reach the
    # client's score log -- otherwise points appear on the cards with no event
    # explaining them. Seed 21 (greedy x4, human last) has two different AI
    # seats each completing a 2-tile city during the opening.
    body = _create(client, opponents=["greedy"] * 4, num_players=5, human_player=4, seed=21)
    events = body["events"]
    assert events, "seed 21 no longer scores in the opening; pick another seed"
    scores = body["state"]["scores"]
    # Mid-game scores move only via score events, so the events must account
    # for every opening point, per seat.
    for seat, total in enumerate(scores):
        assert sum(e["points"] for e in events if e["player"] == seat) == total
    # Play order: events come seat by seat, in the same order the AI moves ran.
    order = [m["seat"] for m in body["ai_moves"]]
    assert [order.index(e["player"]) for e in events] == sorted(
        order.index(e["player"]) for e in events
    )
    assert len({e["player"] for e in events}) >= 2
    for e in events:
        assert set(e) == {"player", "points", "kind", "tiles"}


def test_opening_events_empty_when_human_moves_first(client: TestClient) -> None:
    body = _create(client, opponents=["greedy"] * 4, num_players=5, human_player=0, seed=21)
    assert body["ai_moves"] == [] and body["events"] == []
