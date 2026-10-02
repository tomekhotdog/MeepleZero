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
    """True iff tt at (pos, rot) touches >=1 tile and every touching edge matches."""
    if pos in board:
        return False
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


_DELTAS = ((0, 1), (1, 0), (0, -1), (-1, 0))  # indexed by Side; must match Pos.neighbor
_ROTATIONS = tuple(Rotation)

# _EDGES[type_id][rotation][world_side] -> EdgeKind: world_edge() for every tile
# type and rotation, precomputed once (placement enumeration is the self-play
# hot path -- it runs on every MCTS expansion).
_EDGES = {
    tid: tuple(tuple(world_edge(tt, r, s) for s in Side) for r in Rotation)
    for tid, tt in TILE_TYPES.items()
}


def placements_for_tile(board: Board, tt: TileType) -> list[tuple[Pos, Rotation]]:
    """All legal placements, deterministically ordered by (Pos, Rotation).

    Same result as filtering frontier() x Rotation through fits(), but with one
    neighbour lookup per frontier cell and side, and edges read from _EDGES."""
    occupied = {(p.x, p.y): placed for p, placed in board.items()}
    front = {
        (x + dx, y + dy)
        for (x, y) in occupied
        for dx, dy in _DELTAS
        if (x + dx, y + dy) not in occupied
    }
    mine = _EDGES[tt.id]
    out: list[tuple[Pos, Rotation]] = []
    for x, y in sorted(front):  # (x, y) tuple order == Pos order (order=True, fields x, y)
        need = []
        for side, (dx, dy) in enumerate(_DELTAS):
            placed = occupied.get((x + dx, y + dy))
            if placed is not None:
                need.append((side, _EDGES[placed.type_id][placed.rotation][(side + 2) % 4]))
        pos = Pos(x, y)
        for r in _ROTATIONS:
            edges = mine[r]
            if all(edges[s] is kind for s, kind in need):
                out.append((pos, r))
    return out
