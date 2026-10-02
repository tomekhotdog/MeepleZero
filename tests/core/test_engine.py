import dataclasses
import pickle
import random

import pytest

import carcassonne.core.engine as engine
from carcassonne.core import GameConfig, is_terminal, new_game
from carcassonne.core.engine import GameState, ScoreEvent, apply, legal_moves
from carcassonne.core.features import FeatureIndex, Meeple, NodeId
from carcassonne.core.placement import placements_for_tile
from carcassonne.core.state import Board, PlacedTile
from carcassonne.core.tiles import TILE_TYPES
from carcassonne.core.types import (
    FeatureKind,
    IllegalMove,
    MeepleKind,
    Move,
    PlaceMeeple,
    Player,
    Pos,
    RetrieveAbbot,
    Rotation,
)


def _build(*tiles: tuple[Pos, str, Rotation]) -> tuple[Board, FeatureIndex]:
    board: Board = {}
    idx = FeatureIndex.empty()
    for pos, tid, rot in tiles:
        board[pos] = PlacedTile(tid, rot)
        idx = idx.with_tile(board, pos)
    return board, idx


def _state(
    board: Board,
    features: FeatureIndex,
    *,
    deck: tuple[str, ...] = (),
    current_tile: str | None = None,
    current_player: Player = 0,
    scores: tuple[int, int] = (0, 0),
    meeples: tuple[int, int] = (7, 7),
    abbots: tuple[bool, bool] = (True, True),
    abbot_at: tuple[NodeId | None, NodeId | None] = (None, None),
) -> GameState:
    return GameState(
        board=board,
        features=features,
        deck=deck,
        current_tile=current_tile,
        current_player=current_player,
        scores=scores,
        meeples=meeples,
        abbots=abbots,
        abbot_at=abbot_at,
        discarded=(),
        last_events=(),
        turn=0,
    )


def test_city_completion_scores_majority_owner() -> None:
    # Smallest real shielded city: E-cap + F (city E-W, shield) + E-cap = 3 tiles,
    # 1 shield -> 2*3 + 2*1 = 8 pts. (A 2-tile shielded city is unbuildable: no
    # base-game shield tile has a single city edge.) Majority 2v1 for player 0 —
    # a merge history put two of their meeples on the blob; only player 0 scores.
    board, idx = _build(
        (Pos(-1, 0), "E", Rotation.R90),  # city faces E
        (Pos(0, 0), "F", Rotation.R0),  # city E-W, shield
    )
    idx = idx.with_meeple((Pos(-1, 0), 0), Meeple(0, MeepleKind.MEEPLE, (Pos(-1, 0), 0)))
    idx = idx.with_meeple((Pos(0, 0), 0), Meeple(0, MeepleKind.MEEPLE, (Pos(0, 0), 0)))
    idx = idx.with_meeple((Pos(0, 0), 0), Meeple(1, MeepleKind.MEEPLE, (Pos(0, 0), 0)))
    state = _state(board, idx, current_tile="E", current_player=1, meeples=(5, 6))

    out = apply(state, Move(Pos(1, 0), Rotation.R270, None))  # city faces W, completes

    tiles = frozenset({Pos(-1, 0), Pos(0, 0), Pos(1, 0)})
    assert out.last_events == (ScoreEvent(0, 8, FeatureKind.CITY, tiles),)
    assert out.scores == (8, 0)  # minority player 1 scores nothing
    assert out.meeples == (7, 7)  # all three meeples returned
    assert out.features.root_data((Pos(0, 0), 0)).meeples == ()


def test_road_junction_splits_roads() -> None:
    # W tile has three separate one-edge road features; only its E stub joins D's road.
    board, idx = _build((Pos(0, 0), "D", Rotation.R0))  # city N, road E-W
    state = _state(board, idx, current_tile="W")

    out = apply(state, Move(Pos(-1, 0), Rotation.R0, None))

    w = Pos(-1, 0)
    roots = {out.features.find((w, i)) for i in range(3)}
    assert len(roots) == 3  # the three stubs stay three separate features
    assert out.features.find((w, 0)) == out.features.find((Pos(0, 0), 1))
    assert out.features.root_data((w, 1)).open_edges == 1
    assert out.last_events == ()


def test_cannot_place_meeple_on_occupied_merged_city() -> None:
    board, idx = _build((Pos(0, 0), "D", Rotation.R0))
    idx = idx.with_meeple((Pos(0, 0), 0), Meeple(1, MeepleKind.MEEPLE, (Pos(0, 0), 0)))
    state = _state(board, idx, current_tile="E", meeples=(7, 6))

    bad = Move(Pos(0, 1), Rotation.R180, PlaceMeeple(0, MeepleKind.MEEPLE))
    moves = legal_moves(state)
    assert Move(Pos(0, 1), Rotation.R180, None) in moves
    assert bad not in moves
    with pytest.raises(IllegalMove):
        apply(state, bad)


def test_tie_majority_both_score() -> None:
    board, idx = _build(
        (Pos(0, 0), "D", Rotation.R0),  # city N
        (Pos(0, 2), "E", Rotation.R180),  # city S, facing back down
    )
    idx = idx.with_meeple((Pos(0, 0), 0), Meeple(0, MeepleKind.MEEPLE, (Pos(0, 0), 0)))
    idx = idx.with_meeple((Pos(0, 2), 0), Meeple(1, MeepleKind.MEEPLE, (Pos(0, 2), 0)))
    state = _state(board, idx, current_tile="G", meeples=(6, 6))

    out = apply(state, Move(Pos(0, 1), Rotation.R0, None))  # G bridges and completes

    tiles = frozenset({Pos(0, 0), Pos(0, 1), Pos(0, 2)})
    assert out.last_events == (
        ScoreEvent(0, 6, FeatureKind.CITY, tiles),
        ScoreEvent(1, 6, FeatureKind.CITY, tiles),
    )
    assert out.scores == (6, 6)
    assert out.meeples == (7, 7)


def test_monastery_scores_9_when_surrounded() -> None:
    ring = [Pos(x, y) for x in (-1, 0, 1) for y in (-1, 0, 1) if Pos(x, y) != Pos(1, 1)]
    board, idx = _build(*((p, "B", Rotation.R0) for p in ring))
    idx = idx.with_meeple((Pos(0, 0), 0), Meeple(0, MeepleKind.MEEPLE, (Pos(0, 0), 0)))
    state = _state(board, idx, current_tile="B", meeples=(6, 7))

    out = apply(state, Move(Pos(1, 1), Rotation.R0, None))  # 8th neighbour

    assert out.last_events == (ScoreEvent(0, 9, FeatureKind.MONASTERY, frozenset({Pos(0, 0)})),)
    assert out.scores == (9, 0)
    assert out.meeples == (7, 7)
    assert out.features.root_data((Pos(0, 0), 0)).meeples == ()


def test_monastery_placed_into_hole_with_abbot_completes_immediately() -> None:
    ring = [Pos(x, y) for x in (-1, 0, 1) for y in (-1, 0, 1) if Pos(x, y) != Pos(0, 0)]
    board, idx = _build(*((p, "B", Rotation.R0) for p in ring))
    state = _state(board, idx, current_tile="B")

    out = apply(state, Move(Pos(0, 0), Rotation.R0, PlaceMeeple(0, MeepleKind.ABBOT)))

    assert out.last_events == (ScoreEvent(0, 9, FeatureKind.MONASTERY, frozenset({Pos(0, 0)})),)
    assert out.scores == (9, 0)
    assert out.abbots == (True, True)  # placed and immediately returned
    assert out.abbot_at == (None, None)
    assert out.features.root_data((Pos(0, 0), 0)).meeples == ()


def test_abbot_on_garden_and_retrieve_scores_current_value() -> None:
    board, idx = _build((Pos(0, 0), "D", Rotation.R0))
    state = _state(board, idx, current_tile="U_G", deck=("U", "U"))

    # P0 places U_G (road E-W after R90) east of D and puts the abbot on its garden.
    s1 = apply(state, Move(Pos(1, 0), Rotation.R90, PlaceMeeple(1, MeepleKind.ABBOT)))
    assert s1.abbots == (False, True)
    assert s1.abbot_at == ((Pos(1, 0), 1), None)
    assert s1.meeples == (7, 7)  # abbot is not a plain meeple
    assert s1.features.root_data((Pos(1, 0), 1)).meeples == (
        Meeple(0, MeepleKind.ABBOT, (Pos(1, 0), 1)),
    )
    assert s1.current_player == 1 and s1.current_tile == "U"

    s2 = apply(s1, Move(Pos(2, 0), Rotation.R90, None))  # P1 extends the road
    assert s2.current_player == 0 and s2.current_tile == "U"

    # P0 retrieves: abbot tile (1,0) has 2 occupied 8-neighbours -> 1 + 2 = 3 pts.
    s3 = apply(s2, Move(Pos(3, 0), Rotation.R90, RetrieveAbbot()))
    assert s3.last_events == (ScoreEvent(0, 3, FeatureKind.GARDEN, frozenset({Pos(1, 0)})),)
    assert s3.scores == (3, 0)
    assert s3.abbots == (True, True)
    assert s3.abbot_at == (None, None)
    assert s3.features.root_data((Pos(1, 0), 1)).meeples == ()


def test_meeple_returns_to_supply_after_scoring() -> None:
    board, idx = _build((Pos(0, 0), "E", Rotation.R0))  # city N
    idx = idx.with_meeple((Pos(0, 0), 0), Meeple(0, MeepleKind.MEEPLE, (Pos(0, 0), 0)))
    state = _state(board, idx, current_tile="E", meeples=(6, 7))

    out = apply(state, Move(Pos(0, 1), Rotation.R180, None))  # 2-tile city, 4 pts

    assert out.scores == (4, 0)
    assert out.meeples == (7, 7)
    assert out.current_tile is None  # deck empty -> terminal
    assert legal_moves(out) == ()


def test_place_into_completion_scores_immediately() -> None:
    board, idx = _build((Pos(0, 0), "D", Rotation.R0))
    state = _state(board, idx, current_tile="E")

    move = Move(Pos(0, 1), Rotation.R180, PlaceMeeple(0, MeepleKind.MEEPLE))
    assert move in legal_moves(state)  # completing feature is unoccupied -> legal
    out = apply(state, move)

    tiles = frozenset({Pos(0, 0), Pos(0, 1)})
    assert out.last_events == (ScoreEvent(0, 4, FeatureKind.CITY, tiles),)
    assert out.scores == (4, 0)
    assert out.meeples == (7, 7)  # placed and immediately returned
    assert out.features.root_data((Pos(0, 1), 0)).meeples == ()


def test_unplaceable_tile_discarded_and_next_drawn() -> None:
    board, idx = _build((Pos(0, 0), "B", Rotation.R0))  # all-field edges
    state = _state(board, idx, current_tile="B", deck=("C", "U"))

    out = apply(state, Move(Pos(0, 1), Rotation.R0, None))

    # Verify the construction: C (city on all four edges) really has no placement.
    assert placements_for_tile(out.board, TILE_TYPES["C"]) == []
    assert out.discarded == ("C",)
    assert out.current_tile == "U"
    assert out.deck == ()


def test_apply_rejects_illegal_move() -> None:
    board, idx = _build((Pos(0, 0), "D", Rotation.R0))
    state = _state(board, idx, current_tile="E")

    with pytest.raises(IllegalMove):  # occupied position
        apply(state, Move(Pos(0, 0), Rotation.R0, None))
    with pytest.raises(IllegalMove):  # edge mismatch (field S vs city N)
        apply(state, Move(Pos(0, 1), Rotation.R0, None))
    with pytest.raises(IllegalMove):  # feature index out of range
        apply(state, Move(Pos(0, 1), Rotation.R180, PlaceMeeple(5, MeepleKind.MEEPLE)))
    with pytest.raises(IllegalMove):  # abbot may not go on a city
        apply(state, Move(Pos(0, 1), Rotation.R180, PlaceMeeple(0, MeepleKind.ABBOT)))
    with pytest.raises(IllegalMove):  # no abbot on the board to retrieve
        apply(state, Move(Pos(0, 1), Rotation.R180, RetrieveAbbot()))

    broke = _state(board, idx, current_tile="E", meeples=(0, 7))
    with pytest.raises(IllegalMove):  # no meeple left in supply
        apply(broke, Move(Pos(0, 1), Rotation.R180, PlaceMeeple(0, MeepleKind.MEEPLE)))

    garden = _state(board, idx, current_tile="U_G")
    with pytest.raises(IllegalMove):  # plain meeple may not go on a garden
        apply(garden, Move(Pos(1, 0), Rotation.R90, PlaceMeeple(1, MeepleKind.MEEPLE)))

    terminal = _state(board, idx, current_tile=None)
    with pytest.raises(IllegalMove):
        apply(terminal, Move(Pos(0, 1), Rotation.R180, None))


def test_legal_moves_action_order_is_deterministic() -> None:
    # P0's abbot already sits on a garden, so RetrieveAbbot is available and the
    # abbot is NOT in supply. Current tile A = (monastery, road S).
    board, idx = _build(
        (Pos(0, 0), "D", Rotation.R0),
        (Pos(1, 0), "U_G", Rotation.R90),
    )
    garden: NodeId = (Pos(1, 0), 1)
    idx = idx.with_meeple(garden, Meeple(0, MeepleKind.ABBOT, garden))
    state = _state(board, idx, current_tile="A", abbots=(False, True), abbot_at=(garden, None))

    moves = legal_moves(state)
    acts = tuple(m.action for m in moves if m.pos == Pos(-1, 0) and m.rotation == Rotation.R270)
    assert acts == (
        None,
        PlaceMeeple(0, MeepleKind.MEEPLE),  # monastery
        PlaceMeeple(1, MeepleKind.MEEPLE),  # road stub, unoccupied after merge
        RetrieveAbbot(),
    )
    # Placements come out position-sorted then rotation-ascending.
    placements = [(m.pos, m.rotation) for m in moves]
    assert placements == sorted(placements, key=lambda pr: (pr[0], pr[1]))


def _reference_legal_moves(state):
    """The pre-M18b legal_moves: integrates every placement and reads the merged root."""
    if state.current_tile is None:
        return ()
    tt = TILE_TYPES[state.current_tile]
    p = state.current_player
    any_piece = state.meeples[p] > 0 or state.abbots[p]
    can_retrieve = state.abbot_at[p] is not None
    moves = []
    for pos, rot in engine.placements_for_tile(state.board, tt):
        moves.append(Move(pos, rot, None))
        if any_piece:
            _, idx = engine._integrated(state.board, state.features, state.current_tile, pos, rot)
            for i in range(len(tt.features)):
                for kind in (MeepleKind.MEEPLE, MeepleKind.ABBOT):
                    if engine._meeple_legal(state, idx, tt, pos, i, kind):
                        moves.append(Move(pos, rot, PlaceMeeple(i, kind)))
        if can_retrieve:
            moves.append(Move(pos, rot, RetrieveAbbot()))
    return tuple(moves)


def _withheld_city_road_options(state, moves) -> int:
    """Placements where a city/road meeple option is absent although the player
    has meeples -- i.e. the feature would join one that is already claimed."""
    if state.current_tile is None or state.meeples[state.current_player] == 0:
        return 0
    tt = TILE_TYPES[state.current_tile]
    present = set(moves)
    return sum(
        1
        for m in moves
        if m.action is None
        for i, f in enumerate(tt.features)
        if f.kind in (FeatureKind.CITY, FeatureKind.ROAD)
        and Move(m.pos, m.rotation, PlaceMeeple(i, MeepleKind.MEEPLE)) not in present
    )


def test_legal_moves_match_reference_incl_claimed_features() -> None:
    withheld = 0
    for seed in range(6):
        for n in (2, 3):
            rng = random.Random(seed)
            state = new_game(seed, GameConfig(num_players=n))
            while not is_terminal(state):
                got, want = legal_moves(state), _reference_legal_moves(state)
                assert got == want
                withheld += _withheld_city_road_options(state, want)
                state = apply(state, rng.choice(got))
    # The fixture must exercise the claimed-feature path, or equality proves nothing.
    assert withheld > 50, f"only {withheld} claimed-feature cases; add seeds"


def test_legal_moves_never_copies_the_board(monkeypatch) -> None:
    rng = random.Random(0)
    state = new_game(0, GameConfig())
    for _ in range(15):  # mid-game board with claimed features; apply needs _integrated
        state = apply(state, rng.choice(legal_moves(state)))
    assert any(d.meeples for d in state.features.data.values())
    monkeypatch.setattr(
        engine, "_integrated", lambda *a: (_ for _ in ()).throw(AssertionError("copied"))
    )
    assert legal_moves(new_game(0, GameConfig()))
    assert legal_moves(state)


def test_no_tile_can_bridge_neighbour_features_through_a_sibling() -> None:
    """_claimed_on_placement only inspects the neighbours of the feature itself.
    That is exact only while a tile with 2+ CITY (or ROAD) features gives each of
    them a single edge: a multi-edge sibling could join the feature's neighbour
    root AND a second, otherwise unreachable root on placement."""
    for tt in TILE_TYPES.values():
        for kind in (FeatureKind.CITY, FeatureKind.ROAD):
            same = [f for f in tt.features if f.kind is kind]
            if len(same) >= 2:
                assert all(len(f.edges) == 1 for f in same), tt.id


def test_each_tile_edge_belongs_to_at_most_one_feature() -> None:
    # _claimed_on_placement and FeatureIndex.with_tile stop at the first feature
    # owning a side; a side shared by two features would silently drop a merge.
    for tt in TILE_TYPES.values():
        seen: set[object] = set()
        for f in tt.features:
            assert not (seen & f.edges), tt.id
            seen |= f.edges


def _counting_placements(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    orig = engine.placements_for_tile
    monkeypatch.setattr(
        engine, "placements_for_tile", lambda b, t: calls.append(1) or orig(b, t)
    )
    return calls


def test_legal_moves_reuses_placements_computed_while_drawing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # MCTS expands each child once: the draw already enumerated the tile's
    # placements, so legal_moves must not pay for that enumeration a second time.
    s0 = new_game(0, GameConfig())
    s1 = apply(s0, legal_moves(s0)[0])
    calls = _counting_placements(monkeypatch)
    legal_moves(s1)
    legal_moves(s0)  # new_game also carries them
    assert calls == []


def test_replaced_board_invalidates_the_carried_placements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # learner.rotate_state_90 replace()s the board: carried placements for the
    # old board would be wrong there, so a new board object must recompute.
    s0 = new_game(0, GameConfig())
    s1 = apply(s0, legal_moves(s0)[0])
    calls = _counting_placements(monkeypatch)
    s2 = dataclasses.replace(s1, board=dict(s1.board))  # new board object, same content
    assert legal_moves(s2) == legal_moves(s1)
    assert len(calls) == 1  # s2 recomputed; s1 reused


def test_replaced_tile_invalidates_the_carried_placements() -> None:
    # Same board, different tile: the carried placements belong to the old tile.
    s0 = new_game(0, GameConfig())
    s1 = apply(s0, legal_moves(s0)[0])
    assert s1.current_tile is not None
    other = next(
        tid
        for tid, tt in TILE_TYPES.items()
        if tid != s1.current_tile
        and placements_for_tile(s1.board, tt)
        and placements_for_tile(s1.board, tt)
        != placements_for_tile(s1.board, TILE_TYPES[s1.current_tile])
    )
    s2 = dataclasses.replace(s1, current_tile=other)
    fresh = dataclasses.replace(s2, placements_cache=None)
    assert legal_moves(s2) == legal_moves(fresh)
    assert legal_moves(s2) != legal_moves(s1)


def test_carried_placements_never_affect_equality() -> None:
    # MCTS subtree reuse matches states with ==; a cache must not split them.
    s0 = new_game(0, GameConfig())
    assert s0 == dataclasses.replace(s0, placements_cache=None)


def test_carried_placements_survive_pickling(monkeypatch: pytest.MonkeyPatch) -> None:
    # pickle memoises the shared board dict, so the identity check still holds.
    s0 = new_game(0, GameConfig())
    s1 = apply(s0, legal_moves(s0)[0])
    s = pickle.loads(pickle.dumps(s1))
    calls = _counting_placements(monkeypatch)
    assert legal_moves(s) == legal_moves(s1)
    assert calls == []
