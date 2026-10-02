import random

from carcassonne.core import GameConfig, apply, is_terminal, legal_moves, new_game
from carcassonne.core.placement import fits, frontier, placements_for_tile
from carcassonne.core.state import PlacedTile, empty_board_with_start
from carcassonne.core.tiles import TILE_TYPES
from carcassonne.core.types import Pos, Rotation


def test_start_board_and_first_tile_placements() -> None:
    board = empty_board_with_start()  # D at (0,0), R0: city N, road E-W
    # E's city is on its N side; at R180 the city faces S (world), so at (0,1)
    # it touches the start tile's N edge (CITY) — a match.
    placs = placements_for_tile(board, TILE_TYPES["E"])
    assert (Pos(0, 1), Rotation.R180) in placs  # city faces down onto start's city
    assert (Pos(0, 1), Rotation.R0) not in placs  # E's S edge is FIELD vs start's N CITY
    assert all(p != Pos(0, 0) for p, _ in placs)  # occupied position never offered


def test_fits_rejects_occupied_position() -> None:
    board = empty_board_with_start()
    assert not fits(board, TILE_TYPES["B"], Pos(0, 0), Rotation.R0)


def test_multi_neighbor_placement_requires_all_edges_to_match() -> None:
    # hand-built board: D at origin (E edge = ROAD), E at (1,1) rotated so its
    # S edge (facing (1,0)) is FIELD. A tile at (1,0) touches BOTH.
    board = empty_board_with_start()
    board[Pos(1, 1)] = PlacedTile("E", Rotation.R270)
    v = TILE_TYPES["V"]  # road on S+W
    assert fits(board, v, Pos(1, 0), Rotation.R0)  # W=ROAD ok, N=FIELD ok
    assert not fits(board, v, Pos(1, 0), Rotation.R90)  # W=ROAD ok, N=ROAD mismatch
    assert not fits(board, v, Pos(1, 0), Rotation.R270)  # N=FIELD ok, W=FIELD mismatch


def test_placements_are_deterministically_sorted() -> None:
    board = empty_board_with_start()
    board[Pos(1, 1)] = PlacedTile("E", Rotation.R270)
    placs = placements_for_tile(board, TILE_TYPES["V"])
    assert len(placs) > 2  # sort is actually exercised across positions
    assert placs == sorted(placs)


def test_no_floating_placements() -> None:
    board = empty_board_with_start()
    placs = placements_for_tile(board, TILE_TYPES["B"])
    assert all(abs(p.x) + abs(p.y) == 1 for p, _ in placs)  # only adjacent to start


def _reference_placements(board, tt):
    return [(p, r) for p in sorted(frontier(board)) for r in Rotation if fits(board, tt, p, r)]


def test_placements_match_reference_for_every_tile_type_on_real_boards() -> None:
    for seed in range(4):
        rng = random.Random(seed)
        state = new_game(seed, GameConfig())
        n = 0
        while not is_terminal(state):
            if n % 6 == 0:  # sample boards across the whole game
                for tt in TILE_TYPES.values():
                    expected = _reference_placements(state.board, tt)
                    assert placements_for_tile(state.board, tt) == expected
            state = apply(state, rng.choice(legal_moves(state)))
            n += 1


def test_placement_table_is_used(monkeypatch) -> None:
    """The hot path must not fall back to per-rotation fits() calls."""
    import carcassonne.core.placement as pl

    monkeypatch.setattr(pl, "fits", lambda *a: (_ for _ in ()).throw(AssertionError("fits called")))
    state = new_game(0, GameConfig())
    assert placements_for_tile(state.board, TILE_TYPES[state.current_tile])
