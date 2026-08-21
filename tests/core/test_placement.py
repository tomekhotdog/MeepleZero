from carcassonne.core.placement import placements_for_tile
from carcassonne.core.state import empty_board_with_start
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


def test_no_floating_placements() -> None:
    board = empty_board_with_start()
    placs = placements_for_tile(board, TILE_TYPES["B"])
    assert all(abs(p.x) + abs(p.y) == 1 for p, _ in placs)  # only adjacent to start
