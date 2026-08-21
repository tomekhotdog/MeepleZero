from __future__ import annotations

from carcassonne.core.state import Board
from carcassonne.core.tiles import TILE_TYPES, TileType, world_edge
from carcassonne.core.types import Pos, Rotation, Side


def frontier(board: Board) -> set[Pos]:
    """Empty positions adjacent to at least one placed tile."""
    out: set[Pos] = set()
    for pos in board:
        for side in Side:
            n = pos.neighbor(side)
            if n not in board:
                out.add(n)
    return out


def fits(board: Board, tt: TileType, pos: Pos, rot: Rotation) -> bool:
    touching = False
    for side in Side:
        n = pos.neighbor(side)
        placed = board.get(n)
        if placed is None:
            continue
        touching = True
        mine = world_edge(tt, rot, side)
        theirs = world_edge(TILE_TYPES[placed.type_id], placed.rotation, side.opposite)
        if mine is not theirs:
            return False
    return touching


def placements_for_tile(board: Board, tt: TileType) -> list[tuple[Pos, Rotation]]:
    return [(p, r) for p in sorted(frontier(board)) for r in Rotation if fits(board, tt, p, r)]
