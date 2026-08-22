import random

from carcassonne.core.engine import GameState, apply, legal_moves
from carcassonne.core.features import FeatureIndex, Meeple, NodeId
from carcassonne.core.game import final_scores, is_terminal, new_game
from carcassonne.core.state import Board, PlacedTile
from carcassonne.core.types import MeepleKind, Pos, Rotation


def _build(*tiles: tuple[Pos, str, Rotation]) -> tuple[Board, FeatureIndex]:
    board: Board = {}
    idx = FeatureIndex.empty()
    for pos, tid, rot in tiles:
        board[pos] = PlacedTile(tid, rot)
        idx = idx.with_tile(board, pos)
    return board, idx


def _state(board: Board, features: FeatureIndex, *, scores: tuple[int, int] = (0, 0)) -> GameState:
    return GameState(
        board=board,
        features=features,
        deck=(),
        current_tile=None,
        current_player=0,
        scores=scores,
        meeples=(7, 7),
        abbots=(True, True),
        abbot_at=(None, None),
        discarded=(),
        last_events=(),
        turn=0,
    )


def test_new_game_is_deterministic_per_seed() -> None:
    a, b = new_game(7), new_game(7)
    assert a.deck == b.deck
    assert a.current_tile == b.current_tile
    assert a.discarded == b.discarded
    assert new_game(7).deck != new_game(8).deck


def test_new_game_initial_state() -> None:
    state = new_game(0)
    assert state.board == {Pos(0, 0): PlacedTile("D", Rotation.R0)}
    # 71 non-start tiles live outside the board: deck + drawn current (+ any
    # unplaceable first draws set aside).
    outside = len(state.deck) + (state.current_tile is not None) + len(state.discarded)
    assert outside == 71
    assert state.current_tile is not None  # something always fits next to a lone D
    assert state.current_player == 0 and state.turn == 0
    assert state.scores == (0, 0)
    assert state.meeples == (7, 7)
    assert state.abbots == (True, True) and state.abbot_at == (None, None)
    assert state.features.root_data((Pos(0, 0), 0)).meeples == ()


def test_full_game_seed42_snapshot() -> None:
    # PINNED snapshot: guards refactors of the engine, deck order, and move
    # ordering. If this changes, the rules or determinism changed — investigate
    # before repinning.
    state = new_game(42)
    rng = random.Random(42)
    while not is_terminal(state):
        state = apply(state, rng.choice(legal_moves(state)))
    assert state.turn == 71  # no tile was ever unplaceable for this seed
    assert final_scores(state) == (42, 50)


def test_final_scores_end_game_values_and_purity() -> None:
    # Board: D at origin; F extends its city north (2 tiles + shield, open at
    # F's north -> incomplete); U + U_G extend its road east (3 tiles, open both
    # ends); B monastery south of D.
    board, idx = _build(
        (Pos(0, 0), "D", Rotation.R0),
        (Pos(0, 1), "F", Rotation.R90),  # city N-S, shield
        (Pos(1, 0), "U", Rotation.R90),  # road E-W
        (Pos(2, 0), "U_G", Rotation.R90),  # road E-W + garden
        (Pos(0, -1), "B", Rotation.R0),  # monastery
    )
    city: NodeId = (Pos(0, 0), 0)
    road: NodeId = (Pos(0, 0), 1)
    monastery: NodeId = (Pos(0, -1), 0)
    garden: NodeId = (Pos(2, 0), 1)
    cases: list[tuple[NodeId, Meeple, tuple[int, int]]] = [
        # incomplete city: 1*2 tiles + 1*1 shield = 3, to the meeple's owner
        (city, Meeple(0, MeepleKind.MEEPLE, city), (3, 0)),
        # incomplete road: 1*3 tiles = 3
        (road, Meeple(1, MeepleKind.MEEPLE, road), (0, 3)),
        # monastery: 1 + occupied 8-neighbours of (0,-1) = {(0,0),(1,0)} -> 3
        (monastery, Meeple(0, MeepleKind.MEEPLE, monastery), (3, 0)),
        # garden with abbot: 1 + occupied 8-neighbours of (2,0) = {(1,0)} -> 2
        (garden, Meeple(1, MeepleKind.ABBOT, garden), (0, 2)),
    ]
    for node, piece, (d0, d1) in cases:
        state = _state(board, idx.with_meeple(node, piece), scores=(10, 20))
        assert final_scores(state) == (10 + d0, 20 + d1), f"contribution of {node}"

    # All four at once: contributions sum; live scores are the baseline.
    all_idx = idx
    for node, piece, _ in cases:
        all_idx = all_idx.with_meeple(node, piece)
    state = _state(board, all_idx, scores=(10, 20))
    snapshot = dict(state.features.data)
    assert final_scores(state) == (16, 25)
    # Purity: final_scores mutates nothing and emits no events.
    assert state.scores == (10, 20)
    assert state.features.data == snapshot
    assert state.last_events == ()


def test_final_scores_without_meeples_is_identity() -> None:
    board, idx = _build((Pos(0, 0), "D", Rotation.R0))
    state = _state(board, idx, scores=(5, 9))
    assert final_scores(state) == (5, 9)
