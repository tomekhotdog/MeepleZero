from __future__ import annotations

import dataclasses
import random

import numpy as np
from numpy.typing import NDArray

from carcassonne.core import apply, is_terminal, legal_moves, new_game
from carcassonne.core.engine import GameState
from carcassonne.core.features import FeatureIndex, Meeple
from carcassonne.core.state import Board, PlacedTile
from carcassonne.core.types import Move, PlaceMeeple, Pos, Rotation
from carcassonne.nn.actions import WINDOW, window_origin
from carcassonne.nn.encode import NUM_PLANES, PLANES, encode_state

FloatArray = NDArray[np.float32]

# ---------------------------------------------------------------------------
# Basic shape / manifest


def test_shape_and_manifest() -> None:
    state = new_game(0)
    planes = encode_state(state)
    assert planes.shape == (NUM_PLANES, WINDOW, WINDOW)
    assert planes.dtype == np.float32
    assert NUM_PLANES == 62
    assert len(PLANES) == 62
    assert len(set(PLANES)) == 62  # no duplicate names


def test_occupied_and_current_tile_one_hot() -> None:
    state = new_game(1)
    for _ in range(10):
        planes = encode_state(state)
        assert int(planes[PLANES.index("occupied")].sum()) == len(state.board)
        # current-tile planes are broadcast constants: exactly one is set (all-ones).
        assert _n_current_tile_planes_set(planes) == 1
        state = apply(state, legal_moves(state)[0])

    terminal = _play_to_end(2)
    assert is_terminal(terminal)
    planes = encode_state(terminal)
    assert _n_current_tile_planes_set(planes) == 0  # no current tile when terminal


def test_perspective_swaps_on_apply() -> None:
    state = _first_state_with_unequal_scores(seed=3)
    me = state.current_player
    planes = encode_state(state)
    assert planes[PLANES.index("s_my_score")][0, 0] == np.float32(state.scores[me] / 50.0)
    assert planes[PLANES.index("s_opp_score")][0, 0] == np.float32(state.scores[1 - me] / 50.0)

    nxt = apply(state, legal_moves(state)[0])
    assert nxt.current_player == 1 - me  # perspective flips
    planes2 = encode_state(nxt)
    assert planes2[PLANES.index("s_my_score")][0, 0] == np.float32(
        nxt.scores[nxt.current_player] / 50.0
    )
    assert planes2[PLANES.index("s_opp_score")][0, 0] == np.float32(
        nxt.scores[1 - nxt.current_player] / 50.0
    )


# ---------------------------------------------------------------------------
# D4 equivariance: a 90-degree clockwise board rotation == np.rot90 of the
# spatial planes, with the four edge sides cyclically permuted (N<-W, E<-N,
# S<-E, W<-S). Verified on a genuine mid-game state carrying tiles, edges,
# meeples and completed features.


def _rot_pos(p: Pos) -> Pos:
    """World 90-degree clockwise rotation: (x, y) -> (y, -x)."""
    return Pos(p.y, -p.x)


def _rot_rotation(r: Rotation) -> Rotation:
    return Rotation((int(r) + 1) % 4)


def rotate_state_90(state: GameState) -> GameState:
    """A genuine 90-degree-clockwise rotation of the whole game state.

    Rebuilds the feature index from the rotated board (union-find is order
    independent), then re-places every meeple on its rotated node, so the result
    is a fully valid GameState equivalent to the original under a board rotation.
    """
    rotated = {_rot_pos(p): PlacedTile(pt.type_id, _rot_rotation(pt.rotation))
               for p, pt in state.board.items()}
    board: Board = {}
    idx = FeatureIndex.empty()
    for pos in sorted(rotated):
        board[pos] = rotated[pos]
        idx = idx.with_tile(board, pos)
    for d in state.features.data.values():
        for m in d.meeples:
            mpos, feat = m.node
            node = (_rot_pos(mpos), feat)
            idx = idx.with_meeple(node, Meeple(m.player, m.kind, node))
    def rot_node(n: tuple[Pos, int] | None) -> tuple[Pos, int] | None:
        return (_rot_pos(n[0]), n[1]) if n is not None else None

    abbot_at = (rot_node(state.abbot_at[0]), rot_node(state.abbot_at[1]))
    return dataclasses.replace(state, board=board, features=idx, abbot_at=abbot_at)


def _crop(planes: FloatArray, state: GameState) -> FloatArray:
    """Crop spatial planes to the occupied bounding box (removes centring)."""
    ox, oy = window_origin(state)
    xs = [p.x for p in state.board]
    ys = [p.y for p in state.board]
    x0, x1 = min(xs) - ox, max(xs) - ox
    y0, y1 = min(ys) - oy, max(ys) - oy
    return planes[:, y0 : y1 + 1, x0 : x1 + 1]


def test_d4_rotation_equivariance() -> None:
    state = _meeple_biased_game(seed=5, moves=22)
    # Sanity: the state we test is non-trivial (real content to transform).
    e = encode_state(state)
    assert int(e[PLANES.index("occupied")].sum()) == len(state.board)
    meeple_total = int(
        e[PLANES.index("meeple_me")].sum() + e[PLANES.index("meeple_opp")].sum()
    )
    assert meeple_total > 0  # meeple planes are genuinely exercised

    er = encode_state(rotate_state_90(state))
    co = _crop(e, state)
    cr = _crop(er, rotate_state_90(state))

    prev_side = {"N": "W", "E": "N", "S": "E", "W": "S"}
    edge_planes = set()
    for side in ("N", "E", "S", "W"):
        for kind in ("city", "road", "field"):
            i = PLANES.index(f"edge_{side}_{kind}")
            src = PLANES.index(f"edge_{prev_side[side]}_{kind}")
            edge_planes.add(i)
            assert np.array_equal(cr[i], np.rot90(co[src], 1)), f"edge {side}/{kind}"

    # Every other spatial plane transforms by a plain np.rot90 (no side perm).
    for i in range(24):
        if i in edge_planes:
            continue
        assert np.array_equal(cr[i], np.rot90(co[i], 1)), PLANES[i]


# ---------------------------------------------------------------------------
# helpers


def _n_current_tile_planes_set(planes: FloatArray) -> int:
    tile_idx = [PLANES.index(name) for name in PLANES if name.startswith("tile_")]
    return int(sum(bool(planes[i].any()) for i in tile_idx))


def _play_to_end(seed: int) -> GameState:
    rng = random.Random(seed)
    state = new_game(seed)
    while not is_terminal(state):
        state = apply(state, rng.choice(legal_moves(state)))
    return state


def _first_state_with_unequal_scores(seed: int) -> GameState:
    rng = random.Random(seed)
    state = new_game(seed)
    while not is_terminal(state):
        if state.scores[0] != state.scores[1]:
            return state
        state = apply(state, rng.choice(legal_moves(state)))
    raise AssertionError("no state with unequal scores found")


def _meeple_biased_game(seed: int, moves: int) -> GameState:
    """Play `moves` random moves, preferring ones that place a meeple."""
    rng = random.Random(seed)
    state = new_game(seed)
    for _ in range(moves):
        if is_terminal(state):
            break
        legal = legal_moves(state)
        preferred = [m for m in legal if _places_meeple(m)]
        state = apply(state, rng.choice(preferred or list(legal)))
    return state


def _places_meeple(move: Move) -> bool:
    return isinstance(move.action, PlaceMeeple)
