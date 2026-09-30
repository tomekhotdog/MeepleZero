"""World-coordinate ``Move`` codec + the per-state dynamic **Crop**.

Iteration 3 removed the fixed 31x31 window. The board is encoded over a
per-state Crop: the bounding box of placed tiles plus a 1-cell border. A new
tile can only be placed adjacent to an existing one, so the border provably
contains every legal placement -- every legal move is always representable
(the old "silently dropped out-of-window move" case is impossible).

Two vocabularies:

* **Key** ``(x, y, rot, a)`` -- world coordinates, the engine's own vocabulary.
  Stable under any crop; this is what the replay buffer stores.
* **flat index** ``((gy*W + gx) * 4 + rot) * 14 + a`` with ``gx = x - crop.ox``,
  ``gy = y - crop.oy`` -- the policy head's per-state layout. ``W`` is the
  *stride width*: the crop's own width, or the padded width in a learner batch.

``a`` selects the meeple decision on the freshly placed tile::

    a == 0            -> action None (place tile, no piece)
    1 <= a <= 6       -> PlaceMeeple(feature = a - 1, MEEPLE)   feature 0..5
    7 <= a <= 12      -> PlaceMeeple(feature = a - 7, ABBOT)    feature 0..5
    a == 13           -> RetrieveAbbot()
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from carcassonne.core.engine import GameState, legal_moves
from carcassonne.core.types import (
    MeepleKind,
    Move,
    PlaceMeeple,
    Pos,
    RetrieveAbbot,
    Rotation,
    RulesError,
    TurnAction,
)

ROTS = 4
ACTIONS = 14  # None + 6 meeple slots + 6 abbot slots + retrieve
ACTIONS_PER_CELL = ROTS * ACTIONS  # 56 policy channels per cell
_MAX_FEATURE = 5  # feature index must fit in the 6 slots reserved per meeple kind

type Key = tuple[int, int, int, int]  # (x, y, rot, a) in world coordinates


class Crop(NamedTuple):
    """Per-state board window: bounding box of placed tiles + 1-cell border."""

    ox: int  # world x mapped to grid column 0
    oy: int  # world y mapped to grid row 0
    h: int
    w: int

    @staticmethod
    def of(state: GameState) -> Crop:
        xs = [p.x for p in state.board]
        ys = [p.y for p in state.board]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        return Crop(minx - 1, miny - 1, (maxy - miny) + 3, (maxx - minx) + 3)

    @property
    def n_actions(self) -> int:
        return self.h * self.w * ACTIONS_PER_CELL


def _action_index(action: TurnAction) -> int:
    if action is None:
        return 0
    if isinstance(action, PlaceMeeple):
        if not 0 <= action.feature <= _MAX_FEATURE:
            raise RulesError(f"feature index {action.feature} out of action range")
        return (1 if action.kind is MeepleKind.MEEPLE else 7) + action.feature
    if isinstance(action, RetrieveAbbot):
        return 13
    raise RulesError(f"unknown action {action!r}")


def _decode_action(a: int) -> TurnAction:
    if a == 0:
        return None
    if 1 <= a <= 6:
        return PlaceMeeple(a - 1, MeepleKind.MEEPLE)
    if 7 <= a <= 12:
        return PlaceMeeple(a - 7, MeepleKind.ABBOT)
    if a == 13:
        return RetrieveAbbot()
    raise RulesError(f"action sub-index {a} out of range")


def move_key(move: Move) -> Key:
    """``Move`` -> world-coordinate Key (the buffer's storage format)."""
    return (move.pos.x, move.pos.y, int(move.rotation), _action_index(move.action))


def key_move(key: Key) -> Move:
    """Key -> ``Move`` (inverse of :func:`move_key`)."""
    x, y, rot, a = key
    return Move(Pos(x, y), Rotation(rot), _decode_action(a))


def rotate_key(key: Key, k: int = 1) -> Key:
    """Rotate a Key 90 degrees clockwise in world space, ``k`` times.

    Matches ``learner.rotate_state_90``: cell ``(x, y) -> (y, -x)`` and the tile
    rotation advances one quarter-turn. The sub-action ``a`` names a tile-local
    feature index, invariant under board rotation.
    """
    x, y, rot, a = key
    for _ in range(k % 4):
        x, y, rot = y, -x, (rot + 1) % 4
    return (x, y, rot, a)


def flat_index(crop: Crop, key: Key, stride_w: int | None = None) -> int:
    """Key -> flat policy index for ``crop``, using ``stride_w`` as row stride.

    ``stride_w`` defaults to ``crop.w``; the learner passes the padded batch
    width. Raises :class:`RulesError` if the key's cell is outside the crop --
    impossible for a legal move, so this failing is loud, never silent.
    """
    x, y, rot, a = key
    gx, gy = x - crop.ox, y - crop.oy
    if not (0 <= gx < crop.w and 0 <= gy < crop.h):
        raise RulesError(f"cell ({x}, {y}) outside crop {crop}")
    w = crop.w if stride_w is None else stride_w
    return ((gy * w + gx) * ROTS + rot) * ACTIONS + a


def encode_move(state: GameState, move: Move) -> int:
    """``Move`` -> flat index in ``[0, Crop.of(state).n_actions)``."""
    return flat_index(Crop.of(state), move_key(move))


def decode_move(state: GameState, idx: int) -> Move:
    """Flat index -> ``Move`` (inverse of :func:`encode_move`)."""
    crop = Crop.of(state)
    if not 0 <= idx < crop.n_actions:
        raise RulesError(f"action id {idx} out of range for crop {crop}")
    a = idx % ACTIONS
    rest = idx // ACTIONS
    rot = rest % ROTS
    rest //= ROTS
    gx = rest % crop.w
    gy = rest // crop.w
    return Move(Pos(gx + crop.ox, gy + crop.oy), Rotation(rot), _decode_action(a))


def legal_mask(
    state: GameState, crop: Crop | None = None, pad_hw: tuple[int, int] | None = None
) -> NDArray[np.bool_]:
    """Boolean mask, True at exactly the flat indices of legal moves.

    Shape ``(ph * pw * 56,)`` where ``(ph, pw)`` is ``pad_hw`` (the learner's
    padded batch geometry) or the crop's own ``(h, w)``. Never all-False for a
    non-terminal state: the crop contains every legal move by construction.
    """
    c = crop or Crop.of(state)
    ph, pw = pad_hw or (c.h, c.w)
    if not (ph >= c.h and pw >= c.w):
        # A stride narrower than the crop silently aliases distinct cells onto
        # the same flat index — fail loud, never silent (and never stripped by -O).
        raise RulesError(f"padding {(ph, pw)} smaller than crop {(c.h, c.w)}")
    mask = np.zeros(ph * pw * ACTIONS_PER_CELL, dtype=np.bool_)
    mask[move_indices(c, legal_moves(state), stride_w=pw)] = True
    return mask


def move_indices(
    crop: Crop, moves: Sequence[Move], stride_w: int | None = None
) -> NDArray[np.int64]:
    """Flat policy index of each move, aligned with ``moves`` (see :func:`flat_index`)."""
    return np.fromiter(
        (flat_index(crop, move_key(m), stride_w) for m in moves), dtype=np.int64, count=len(moves)
    )
