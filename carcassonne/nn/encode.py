"""``GameState`` -> ``float32`` planes for the residual CNN.

Every plane is a ``WINDOW x WINDOW`` grid indexed ``planes[c, wy, wx]`` (axis 1 is
window-y, axis 2 is window-x), so a 90-degree board rotation corresponds to
:func:`numpy.rot90` on the spatial axes (with the four edge sides cyclically
permuted). The encoding is from the *current player's* perspective: "me" is
``state.current_player``, "opp" the other player.

Plane layout (see :data:`PLANES` for the exact names, in order):

Spatial (24), painted per occupied window cell:
  * ``occupied`` (1)
  * edge-kind one-hot per world side N,E,S,W x {city, road, field} (12)
  * per-feature-kind presence on the tile: city, road, monastery, garden (4)
  * ``shield`` present on the tile (1)
  * ``completed``: the cell's tile carries a completed city/road feature (1)
  * meeple owner one-hot: me, opp (2)
  * abbot owner one-hot: me, opp (2)
  * ``monfill``: for monastery/garden cells, occupied-8-neighbours / 8 (1)

Broadcast scalar (38), constant across the grid:
  * my/opp score /50, my/opp meeples /7, my/opp abbot-in-supply (0/1),
    deck-remaining /71, turn /72 (8)
  * current-tile one-hot over the 30 tile-type ids, sorted (30)

Total channels ``NUM_PLANES == 62``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from carcassonne.core.engine import GameState, neighbours8
from carcassonne.core.tiles import TILE_TYPES, world_edge
from carcassonne.core.types import EdgeKind, FeatureKind, MeepleKind, RulesError, Side
from carcassonne.nn.actions import WINDOW, window_origin

_SIDES = (Side.N, Side.E, Side.S, Side.W)
_EDGE_KINDS = (EdgeKind.CITY, EdgeKind.ROAD, EdgeKind.FIELD)
_FEATURE_KINDS = (FeatureKind.CITY, FeatureKind.ROAD, FeatureKind.MONASTERY, FeatureKind.GARDEN)
_TILE_IDS: tuple[str, ...] = tuple(sorted(TILE_TYPES))

_SPATIAL_PLANES: list[str] = (
    ["occupied"]
    + [f"edge_{s.name}_{k.value}" for s in _SIDES for k in _EDGE_KINDS]
    + [f"feat_{k.value}" for k in _FEATURE_KINDS]
    + ["shield", "completed", "meeple_me", "meeple_opp", "abbot_me", "abbot_opp", "monfill"]
)
_SCALAR_PLANES: list[str] = [
    "s_my_score",
    "s_opp_score",
    "s_my_meeples",
    "s_opp_meeples",
    "s_my_abbot",
    "s_opp_abbot",
    "s_deck_remaining",
    "s_turn",
] + [f"tile_{tid}" for tid in _TILE_IDS]

PLANES: list[str] = _SPATIAL_PLANES + _SCALAR_PLANES
NUM_PLANES = len(PLANES)

assert len(_SPATIAL_PLANES) == 24, len(_SPATIAL_PLANES)
assert len(_SCALAR_PLANES) == 38, len(_SCALAR_PLANES)
assert NUM_PLANES == 62, NUM_PLANES

# Fixed plane offsets (kept in sync with PLANES by the asserts above).
_OCC = 0
_EDGE0 = 1  # 12 planes: side * 3 + kind
_FEAT0 = 13  # 4 planes
_SHIELD = 17
_COMPLETED = 18
_MEEPLE_ME = 19
_MEEPLE_OPP = 20
_ABBOT_ME = 21
_ABBOT_OPP = 22
_MONFILL = 23
_SCALAR0 = 24  # 8 scalar planes
_TILE0 = 32  # 30 current-tile one-hot planes


def encode_state(state: GameState) -> NDArray[np.float32]:
    """Encode ``state`` as a ``(NUM_PLANES, WINDOW, WINDOW)`` float32 tensor."""
    b = WINDOW
    planes = np.zeros((NUM_PLANES, b, b), dtype=np.float32)
    ox, oy = window_origin(state)
    me = state.current_player
    opp = 1 - me
    board = state.board

    for pos, placed in board.items():
        wx, wy = pos.x - ox, pos.y - oy
        if not (0 <= wx < b and 0 <= wy < b):
            raise RulesError(f"tile at {pos} maps outside the {b}x{b} window")
        tt = TILE_TYPES[placed.type_id]
        planes[_OCC, wy, wx] = 1.0
        for si, side in enumerate(_SIDES):
            ki = _EDGE_KINDS.index(world_edge(tt, placed.rotation, side))
            planes[_EDGE0 + si * 3 + ki, wy, wx] = 1.0
        has_monastic = False
        for f in tt.features:
            planes[_FEAT0 + _FEATURE_KINDS.index(f.kind), wy, wx] = 1.0
            if f.shield:
                planes[_SHIELD, wy, wx] = 1.0
            if f.kind in (FeatureKind.MONASTERY, FeatureKind.GARDEN):
                has_monastic = True
        if has_monastic:
            occupied_n = sum(1 for q in neighbours8(pos) if q in board)
            planes[_MONFILL, wy, wx] = occupied_n / 8.0

    # Every position here belongs to a placed tile, already proven in-window by the
    # loop above, so out-of-window here is a real bug — assert rather than skip.
    for d in state.features.data.values():
        if d.kind in (FeatureKind.CITY, FeatureKind.ROAD) and d.open_edges == 0:
            for tp in d.tiles:
                wx, wy = tp.x - ox, tp.y - oy
                assert 0 <= wx < b and 0 <= wy < b
                planes[_COMPLETED, wy, wx] = 1.0
        for m in d.meeples:
            mpos, _feat = m.node
            wx, wy = mpos.x - ox, mpos.y - oy
            assert 0 <= wx < b and 0 <= wy < b
            if m.kind is MeepleKind.MEEPLE:
                plane = _MEEPLE_ME if m.player == me else _MEEPLE_OPP
            else:
                plane = _ABBOT_ME if m.player == me else _ABBOT_OPP
            planes[plane, wy, wx] = 1.0

    planes[_SCALAR0 + 0] = state.scores[me] / 50.0
    planes[_SCALAR0 + 1] = state.scores[opp] / 50.0
    planes[_SCALAR0 + 2] = state.meeples[me] / 7.0
    planes[_SCALAR0 + 3] = state.meeples[opp] / 7.0
    planes[_SCALAR0 + 4] = 1.0 if state.abbots[me] else 0.0
    planes[_SCALAR0 + 5] = 1.0 if state.abbots[opp] else 0.0
    planes[_SCALAR0 + 6] = len(state.deck) / 71.0
    planes[_SCALAR0 + 7] = state.turn / 72.0
    if state.current_tile is not None:
        planes[_TILE0 + _TILE_IDS.index(state.current_tile)] = 1.0

    return planes
