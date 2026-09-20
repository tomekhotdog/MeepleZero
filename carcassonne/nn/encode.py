"""``GameState`` -> ``float32`` planes over the dynamic :class:`Crop`.

Output shape is ``(NUM_PLANES, H, W)`` where ``H x W`` is the per-state crop
(or ``pad_hw`` when the learner pads a batch; padding rows/cols are all-zero,
including broadcast planes -- the model relies on that for its padded==unpadded
invariant). Indexed ``planes[c, gy, gx]``. Perspective: "me" is
``state.current_player``, "opp" the other seat -- the encoding is strictly
2-player (me/opp planes); :func:`encode_state` raises :class:`RulesError` on
any other player count.

Layout (52 planes; see PLANES for names in order):

Spatial (21): occupied; 12 edge-kind one-hots (world side x {city,road,field});
monastery/garden presence; shield; meeple/abbot owner (me/opp); monfill.

FeaturePlanes (8): per-cell max over the {city, road} features touching the
tile, from the engine's FeatureIndex: normalised size, normalised open-edge
count, my/opp meeple count. Hands the net long-range connectivity that convs
cannot infer across snaky features. (Subsumes the old ``completed`` plane:
completed <=> open == 0.) The max is per component: on a tile carrying two
same-kind features (H, H_G, I have two cities), a cell can read as a composite
matching neither real feature (city A's size with city B's meeple count) -- a
deliberate, accepted approximation.

Current tile (15, broadcast over the crop): the drawn tile's R0 edge kinds
(12) + monastery/garden/shield flags -- structural properties, not a tile-ID
one-hot, so structurally identical tiles share representation.

Scalars (8, broadcast): scores/50, meeples/7, abbot flags, deck/71, turn/72.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from carcassonne.core.engine import GameState, neighbours8
from carcassonne.core.tiles import TILE_TYPES, derived_edges, world_edge
from carcassonne.core.types import EdgeKind, FeatureKind, MeepleKind, RulesError, Side
from carcassonne.nn.actions import Crop

_SIDES = (Side.N, Side.E, Side.S, Side.W)
_EDGE_KINDS = (EdgeKind.CITY, EdgeKind.ROAD, EdgeKind.FIELD)
_FP_KINDS = (FeatureKind.CITY, FeatureKind.ROAD)

_SPATIAL_PLANES: list[str] = (
    ["occupied"]
    + [f"edge_{s.name}_{k.value}" for s in _SIDES for k in _EDGE_KINDS]
    + ["feat_monastery", "feat_garden", "shield"]
    + ["meeple_me", "meeple_opp", "abbot_me", "abbot_opp", "monfill"]
)
_FP_PLANES: list[str] = [
    f"fp_{k.value}_{q}" for k in _FP_KINDS for q in ("size", "open", "mine", "theirs")
]
_CUR_PLANES: list[str] = [f"cur_edge_{s.name}_{k.value}" for s in _SIDES for k in _EDGE_KINDS] + [
    "cur_monastery",
    "cur_garden",
    "cur_shield",
]
_SCALAR_PLANES: list[str] = [
    "s_my_score",
    "s_opp_score",
    "s_my_meeples",
    "s_opp_meeples",
    "s_my_abbot",
    "s_opp_abbot",
    "s_deck_remaining",
    "s_turn",
]

PLANES: list[str] = _SPATIAL_PLANES + _FP_PLANES + _CUR_PLANES + _SCALAR_PLANES
NUM_PLANES = len(PLANES)

assert len(_SPATIAL_PLANES) == 21, len(_SPATIAL_PLANES)
assert len(_FP_PLANES) == 8 and len(_CUR_PLANES) == 15, (len(_FP_PLANES), len(_CUR_PLANES))
assert NUM_PLANES == 52, NUM_PLANES

# Fixed plane offsets (kept in sync with PLANES by the asserts above).
_OCC = 0
_EDGE0 = 1  # 12 planes: side * 3 + kind
_MONASTERY = 13
_GARDEN = 14
_SHIELD = 15
_MEEPLE_ME = 16
_MEEPLE_OPP = 17
_ABBOT_ME = 18
_ABBOT_OPP = 19
_MONFILL = 20
_FP0 = 21  # 8 planes: kind * 4 + {size, open, mine, theirs}
_CUR0 = 29  # 12 current-tile edge planes + 3 flags
_CUR_FLAGS = 41
_SCALAR0 = 44

# FeaturePlane normalisation caps (clipped): sizes rarely exceed 10 tiles,
# open edges 6, meeples-per-player-per-feature 3 in real base-game play.
_SIZE_CAP = 10.0
_OPEN_CAP = 6.0
_MEEPLE_CAP = 3.0


def valid_mask(crop: Crop, pad_hw: tuple[int, int] | None = None) -> NDArray[np.float32]:
    """``(ph, pw)`` float mask: 1 inside the crop, 0 in padding. The model's
    MaskedGroupNorm/value-pool consume this so padding never contaminates stats."""
    ph, pw = pad_hw or (crop.h, crop.w)
    m = np.zeros((ph, pw), dtype=np.float32)
    m[: crop.h, : crop.w] = 1.0
    return m


def encode_state(
    state: GameState, crop: Crop | None = None, pad_hw: tuple[int, int] | None = None
) -> NDArray[np.float32]:
    """Encode ``state`` as ``(NUM_PLANES, ph, pw)`` float32 planes.

    Every legal move's cell is inside the crop by construction (bbox + 1), so
    nothing is ever silently dropped. Padding (beyond ``crop.h/w``) is all-zero.
    """
    if len(state.scores) != 2:
        raise RulesError(
            f"the encoder is 2-player only (me/opp planes); got a {len(state.scores)}-player state"
        )
    c = crop or Crop.of(state)
    ph, pw = pad_hw or (c.h, c.w)
    if not (ph >= c.h and pw >= c.w):
        # Too-small padding would silently truncate cells — fail loud, and never
        # stripped by ``python -O`` (matches legal_mask's guard in actions.py).
        raise RulesError(f"padding {(ph, pw)} smaller than crop {(c.h, c.w)}")
    planes = np.zeros((NUM_PLANES, ph, pw), dtype=np.float32)
    me = state.current_player
    opp = 1 - me
    board = state.board

    for pos, placed in board.items():
        gx, gy = pos.x - c.ox, pos.y - c.oy
        tt = TILE_TYPES[placed.type_id]
        planes[_OCC, gy, gx] = 1.0
        for si, side in enumerate(_SIDES):
            ki = _EDGE_KINDS.index(world_edge(tt, placed.rotation, side))
            planes[_EDGE0 + si * 3 + ki, gy, gx] = 1.0
        has_monastic = False
        for f in tt.features:
            if f.shield:
                planes[_SHIELD, gy, gx] = 1.0
            if f.kind is FeatureKind.MONASTERY:
                planes[_MONASTERY, gy, gx] = 1.0
                has_monastic = True
            elif f.kind is FeatureKind.GARDEN:
                planes[_GARDEN, gy, gx] = 1.0
                has_monastic = True
        if has_monastic:
            occupied_n = sum(1 for q in neighbours8(pos) if q in board)
            planes[_MONFILL, gy, gx] = occupied_n / 8.0

    # One pass over feature roots (no per-tile union-find walks -- encode runs
    # at every MCTS leaf): FeaturePlanes max-painted onto every tile the root
    # spans, then meeple-owner planes. The per-cell max over same-kind features
    # is unchanged vs painting per tile-feature slot: a tile carrying two
    # same-kind features appears in both roots' tile sets. See the module
    # docstring for the accepted composite-value approximation.
    for d in state.features.data.values():
        if d.kind in _FP_KINDS:
            base = _FP0 + _FP_KINDS.index(d.kind) * 4
            mine = sum(1 for m in d.meeples if m.player == me)
            theirs = sum(1 for m in d.meeples if m.player == opp)
            vals = (
                min(len(d.tiles), _SIZE_CAP) / _SIZE_CAP,
                min(d.open_edges, _OPEN_CAP) / _OPEN_CAP,
                min(mine, _MEEPLE_CAP) / _MEEPLE_CAP,
                min(theirs, _MEEPLE_CAP) / _MEEPLE_CAP,
            )
            for tp in d.tiles:
                gx, gy = tp.x - c.ox, tp.y - c.oy
                for j, v in enumerate(vals):
                    planes[base + j, gy, gx] = max(planes[base + j, gy, gx], v)
        for m in d.meeples:
            mpos, _feat = m.node
            gx, gy = mpos.x - c.ox, mpos.y - c.oy
            if m.kind is MeepleKind.MEEPLE:
                plane = _MEEPLE_ME if m.player == me else _MEEPLE_OPP
            else:
                plane = _ABBOT_ME if m.player == me else _ABBOT_OPP
            planes[plane, gy, gx] = 1.0

    # Broadcast planes fill ONLY the crop region: padding must stay exactly zero.
    region = (slice(None, c.h), slice(None, c.w))
    if state.current_tile is not None:
        tt = TILE_TYPES[state.current_tile]
        for si, kind in enumerate(derived_edges(tt)):
            planes[(_CUR0 + si * 3 + _EDGE_KINDS.index(kind), *region)] = 1.0
        for f in tt.features:
            if f.kind is FeatureKind.MONASTERY:
                planes[(_CUR_FLAGS, *region)] = 1.0
            elif f.kind is FeatureKind.GARDEN:
                planes[(_CUR_FLAGS + 1, *region)] = 1.0
            if f.shield:
                planes[(_CUR_FLAGS + 2, *region)] = 1.0

    scalars = (
        state.scores[me] / 50.0,
        state.scores[opp] / 50.0,
        state.meeples[me] / 7.0,
        state.meeples[opp] / 7.0,
        1.0 if state.abbots[me] else 0.0,
        1.0 if state.abbots[opp] else 0.0,
        len(state.deck) / 71.0,
        state.turn / 72.0,
    )
    for j, v in enumerate(scalars):
        planes[(_SCALAR0 + j, *region)] = v

    return planes
