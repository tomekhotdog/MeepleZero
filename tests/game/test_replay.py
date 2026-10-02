"""Replay format: writer round-trip, verifying loader, serde codecs."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from carcassonne.core import (
    GameConfig,
    MeepleKind,
    Move,
    PlaceMeeple,
    Pos,
    RetrieveAbbot,
    Rotation,
    RulesError,
    apply,
    final_scores,
    is_terminal,
    legal_moves,
    new_game,
)
from carcassonne.game.replay import (
    REPLAY_VERSION,
    MoveRecord,
    ReplayHeader,
    ReplayWriter,
    load_replay,
    replay_states,
    winners_of,
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
        v=REPLAY_VERSION,
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
    expected_winners = (0,) if finals[0] > finals[1] else (1,) if finals[1] > finals[0] else (0, 1)
    assert replay.winners == expected_winners

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


# --- multiplayer (v2) + v1 back-compat ---


def _write_game(tmp_path: Path, config: GameConfig, agents: tuple[str, ...], seed: int) -> Path:
    """Write a full seeded random game for `config` through ReplayWriter."""
    state = new_game(seed, config)
    header = ReplayHeader(
        v=REPLAY_VERSION,
        seed=seed,
        config=config,
        agents=agents,
        checkpoint=None,
        started_at=STARTED_AT,
    )
    path = tmp_path / "g.jsonl"
    rng = random.Random(seed)
    with ReplayWriter(path, header) as w:
        n = 0
        while not is_terminal(state):
            player, tile = state.current_player, state.current_tile
            assert tile is not None
            move = rng.choice(legal_moves(state))
            state = apply(state, move)
            w.record(n, player, tile, move, state.scores)
            n += 1
        w.finish(final_scores(state))
    return path


def test_winners_of_tie_sets() -> None:
    assert winners_of((5, 3)) == (0,)
    assert winners_of((3, 3)) == (0, 1)
    assert winners_of((1, 7, 7, 2)) == (1, 2)
    assert winners_of((0, 0, 0, 0, 0)) == (0, 1, 2, 3, 4)


def test_three_player_replay_round_trips(tmp_path: Path) -> None:
    """Write a full random 3-player game through ReplayWriter; load + verify."""
    config = GameConfig(num_players=3)
    path = _write_game(tmp_path, config, ("random", "random", "random"), seed=11)
    replay = load_replay(path)
    assert replay.header.config.num_players == 3
    assert len(replay.final_scores) == 3
    assert replay.winners == winners_of(replay.final_scores)
    list(replay_states(replay))  # engine-verified; must not raise


def test_v1_two_player_replay_still_loads(tmp_path: Path) -> None:
    """Take a real v2 file and rewrite header/end lines to the v1 schema."""
    path = _write_game(tmp_path, GameConfig(), ("random", "random"), seed=5)
    lines = path.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    header["v"] = 1
    header["config"] = {"meeples_per_player": header["config"]["meeples_per_player"]}
    end = json.loads(lines[-1])
    ws = end.pop("winners")
    end["winner"] = ws[0] if len(ws) == 1 else None
    v1 = tmp_path / "v1.jsonl"
    v1.write_text(
        "\n".join([json.dumps(header), *lines[1:-1], json.dumps(end)]) + "\n",
        encoding="utf-8",
    )
    replay = load_replay(v1)
    assert replay.header.v == 1
    assert replay.header.config.num_players == 2
    assert replay.winners == tuple(ws)
    list(replay_states(replay))  # engine-verified; must not raise


def _write_header_and_end(tmp_path: Path, num_players: int) -> Path:
    """A structurally complete zero-move v2 file whose seat count is `num_players`."""
    header = {
        "v": REPLAY_VERSION,
        "seed": SEED,
        "config": {"meeples_per_player": 7, "num_players": num_players},
        "agents": ["random"] * num_players,
        "checkpoint": None,
        "started_at": STARTED_AT,
    }
    end = {"end": True, "final_scores": [0] * num_players, "winners": list(range(num_players))}
    path = tmp_path / f"np{num_players}.jsonl"
    path.write_text(json.dumps(header) + "\n" + json.dumps(end) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("num_players", [0, 1, 6])
def test_header_rejects_num_players_outside_2_to_5(tmp_path: Path, num_players: int) -> None:
    # Without a header check such a file loads as a Replay and only fails much
    # later (new_game during replay_states) -- or never, for consumers that only
    # read the header/scores. Reject it where the defect is: the header.
    path = _write_header_and_end(tmp_path, num_players)
    with pytest.raises(RulesError, match=rf"num_players must be 2\.\.5, got {num_players}"):
        load_replay(path)


@pytest.mark.parametrize("num_players", [2, 3, 4, 5])
def test_header_accepts_num_players_2_to_5(tmp_path: Path, num_players: int) -> None:
    replay = load_replay(_write_header_and_end(tmp_path, num_players))
    assert replay.header.config.num_players == num_players


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
