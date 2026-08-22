"""Replay format: writer round-trip, verifying loader, serde codecs."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from carcassonne.core import (
    GameConfig,
    RulesError,
    apply,
    final_scores,
    is_terminal,
    legal_moves,
    new_game,
)
from carcassonne.core.types import MeepleKind, Move, PlaceMeeple, Pos, RetrieveAbbot, Rotation
from carcassonne.game.replay import (
    MoveRecord,
    ReplayHeader,
    ReplayWriter,
    load_replay,
    replay_states,
)
from carcassonne.game.serde import (
    Annot,
    annot_from_json,
    annot_to_json,
    move_from_json,
    move_to_json,
)

SEED = 42
STARTED_AT = "2026-08-21T12:00:00+00:00"


def _header(seed: int = SEED) -> ReplayHeader:
    return ReplayHeader(
        v=1,
        seed=seed,
        config=GameConfig(),
        agents=("random", "random"),
        checkpoint=None,
        started_at=STARTED_AT,
    )


def _write_full_game(
    path: Path, seed: int = SEED
) -> tuple[list[MoveRecord], tuple[int, int]]:
    """Play a full seeded random game via the core API, recording every move."""
    rng = random.Random(seed)
    state = new_game(seed)
    records: list[MoveRecord] = []
    with ReplayWriter(path, _header(seed)) as w:
        n = 0
        while not is_terminal(state):
            player, tile = state.current_player, state.current_tile
            assert tile is not None
            moves = legal_moves(state)
            move = rng.choice(moves)
            nxt = apply(state, move)
            annot: Annot | None = None
            if n % 10 == 0:  # a real Annot with top-moves on a few turns
                annot = Annot(
                    value=0.125,
                    sims=200,
                    think_ms=31,
                    top=((move, 0.5, 88), (moves[0], 0.25, 44)),
                )
            w.record(n, player, tile, move, nxt.scores, annot)
            records.append(MoveRecord(n, player, tile, move, nxt.scores, annot))
            state = nxt
            n += 1
        finals = final_scores(state)
        w.finish(finals)
    return records, finals


@pytest.fixture(scope="module")
def full_game(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, list[MoveRecord], tuple[int, int]]:
    path = tmp_path_factory.mktemp("replays") / "game.jsonl"
    records, finals = _write_full_game(path)
    return path, records, finals


def test_round_trip(full_game: tuple[Path, list[MoveRecord], tuple[int, int]]) -> None:
    path, records, finals = full_game
    replay = load_replay(path)

    assert replay.header == _header()
    assert replay.moves == tuple(records)
    assert replay.final_scores == finals
    expected_winner = 0 if finals[0] > finals[1] else 1 if finals[1] > finals[0] else None
    assert replay.winner == expected_winner

    states = list(replay_states(replay))  # verifies every step; must not raise
    assert len(states) == len(records) + 1
    assert is_terminal(states[-1])
    assert final_scores(states[-1]) == finals


def test_tampered_score_raises_with_move_index(
    full_game: tuple[Path, list[MoveRecord], tuple[int, int]], tmp_path: Path
) -> None:
    src, _records, _finals = full_game
    lines = src.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[6])  # move record with n == 5
    assert row["n"] == 5
    row["score_after"] = [999, 999]
    lines[6] = json.dumps(row, separators=(",", ":"))
    tampered = tmp_path / "tampered.jsonl"
    tampered.write_text("\n".join(lines) + "\n", encoding="utf-8")

    replay = load_replay(tampered)  # structurally valid: parsing succeeds
    with pytest.raises(RulesError, match="move 5"):
        list(replay_states(replay))


def test_malformed_json_line_raises(
    full_game: tuple[Path, list[MoveRecord], tuple[int, int]], tmp_path: Path
) -> None:
    src, _records, _finals = full_game
    lines = src.read_text(encoding="utf-8").splitlines()
    lines[3] = "this is not json{{{"
    bad = tmp_path / "malformed.jsonl"
    bad.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(RulesError):
        load_replay(bad)


def test_missing_end_line_raises(tmp_path: Path) -> None:
    # A writer closed via context manager without finish() is a crash artifact:
    # it must not load.
    path = tmp_path / "incomplete.jsonl"
    state = new_game(SEED)
    with ReplayWriter(path, _header()) as w:
        tile = state.current_tile
        assert tile is not None
        move = legal_moves(state)[0]
        nxt = apply(state, move)
        w.record(0, state.current_player, tile, move, nxt.scores)
        # no finish()

    with pytest.raises(RulesError):
        load_replay(path)


# --- serde ---

ALL_ACTION_MOVES = (
    Move(Pos(1, 0), Rotation.R0, None),
    Move(Pos(0, -1), Rotation.R90, PlaceMeeple(0, MeepleKind.MEEPLE)),
    Move(Pos(-2, 3), Rotation.R180, PlaceMeeple(2, MeepleKind.ABBOT)),
    Move(Pos(4, 4), Rotation.R270, RetrieveAbbot()),
)


def test_move_serde_round_trip_all_action_shapes() -> None:
    for move in ALL_ACTION_MOVES:
        d = move_to_json(move)
        # must survive an actual JSON encode/decode, not just dict identity
        assert move_from_json(json.loads(json.dumps(d))) == move


def test_annot_serde_round_trip() -> None:
    annot = Annot(
        value=-0.25,
        sims=800,
        think_ms=1234,
        top=(
            (ALL_ACTION_MOVES[0], 0.5, 400),
            (ALL_ACTION_MOVES[1], 0.375, 300),
            (ALL_ACTION_MOVES[3], 0.125, 100),
        ),
    )
    d = annot_to_json(annot)
    assert annot_from_json(json.loads(json.dumps(d))) == annot


def test_serde_rejects_malformed() -> None:
    with pytest.raises(RulesError):
        move_from_json("not a dict")
    with pytest.raises(RulesError):
        move_from_json({"x": 0, "y": 0, "rot": 7, "action": None})  # bad rotation
    with pytest.raises(RulesError):
        move_from_json({"x": 0, "y": 0, "rot": 0})  # missing action key
    with pytest.raises(RulesError):
        move_from_json(
            {"x": 0, "y": 0, "rot": 0, "action": {"type": "teleport"}}
        )  # unknown action
    with pytest.raises(RulesError):
        move_from_json(
            {"x": 0, "y": 0, "rot": 0, "action": {"type": "meeple", "feature": 0, "kind": "king"}}
        )  # unknown meeple kind
    with pytest.raises(RulesError):
        move_from_json({"x": True, "y": 0, "rot": 0, "action": None})  # bool is not an int
    with pytest.raises(RulesError):
        annot_from_json({"value": 0.5, "sims": 10})  # missing fields
