"""Deterministic ``Move`` <-> ``int`` mapping for the policy head.

The action space is a fixed ``WINDOW x WINDOW`` grid of board cells, times four
rotations, times 14 per-placement actions::

    action_id = ((wy * B + wx) * 4 + rot) * 14 + a

where ``a`` selects the meeple decision on the freshly placed tile:

    a == 0            -> action None (place tile, no piece)
    1 <= a <= 6       -> PlaceMeeple(feature = a - 1, MEEPLE)   feature 0..5
    7 <= a <= 12      -> PlaceMeeple(feature = a - 7, ABBOT)    feature 0..5
    a == 13           -> RetrieveAbbot()

Board cells are mapped into the window by centring the board's occupied bounding
box in the ``BxB`` grid (see :func:`window_origin`). The mapping is recomputed
from the *current* state on every call, so it is stable within a turn and always
keeps the live board centred.
"""

from __future__ import annotations

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

WINDOW = 31
_ROTS = 4
_ACTIONS = 14  # None + 6 meeple slots + 6 abbot slots + retrieve
_MAX_FEATURE = 5  # feature index must fit in the 6 slots reserved per meeple kind

ACTION_SPACE = WINDOW * WINDOW * _ROTS * _ACTIONS
assert ACTION_SPACE == 53816, ACTION_SPACE


def window_origin(state: GameState, window: int = WINDOW) -> tuple[int, int]:
    """World ``Pos`` that maps to window cell ``(0, 0)``.

    The board's occupied bounding box is centred in the ``window x window`` grid,
    so a fresh game (only the start tile at the origin) lands the start tile at
    the window centre.
    """
    xs = [p.x for p in state.board]
    ys = [p.y for p in state.board]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    origin_x = minx - (window - (maxx - minx + 1)) // 2
    origin_y = miny - (window - (maxy - miny + 1)) // 2
    return origin_x, origin_y


def to_window(state: GameState, pos: Pos, window: int = WINDOW) -> tuple[int, int]:
    """World ``Pos`` -> ``(wx, wy)`` window coordinates."""
    ox, oy = window_origin(state, window)
    return pos.x - ox, pos.y - oy


def from_window(state: GameState, wx: int, wy: int, window: int = WINDOW) -> Pos:
    """``(wx, wy)`` window coordinates -> world ``Pos``."""
    ox, oy = window_origin(state, window)
    return Pos(wx + ox, wy + oy)


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


def _encode_with_origin(move: Move, origin: tuple[int, int]) -> int:
    wx, wy = move.pos.x - origin[0], move.pos.y - origin[1]
    if not (0 <= wx < WINDOW and 0 <= wy < WINDOW):
        raise RulesError(f"move at {move.pos} maps outside the {WINDOW}x{WINDOW} window")
    a = _action_index(move.action)
    return ((wy * WINDOW + wx) * _ROTS + int(move.rotation)) * _ACTIONS + a


def encode_move(state: GameState, move: Move) -> int:
    """``Move`` -> action id in ``[0, ACTION_SPACE)``.

    Raises :class:`RulesError` if the move's position falls outside the window
    (impossible for legal moves on a base-game board; surfaced loudly if it ever
    happens).
    """
    return _encode_with_origin(move, window_origin(state))


def decode_move(state: GameState, idx: int) -> Move:
    """Action id -> ``Move`` (inverse of :func:`encode_move`)."""
    if not 0 <= idx < ACTION_SPACE:
        raise RulesError(f"action id {idx} out of range")
    a = idx % _ACTIONS
    rest = idx // _ACTIONS
    rot = rest % _ROTS
    rest //= _ROTS
    wx = rest % WINDOW
    wy = rest // WINDOW
    return Move(from_window(state, wx, wy), Rotation(rot), _decode_action(a))


def legal_mask(state: GameState) -> NDArray[np.bool_]:
    """Boolean mask of shape ``(ACTION_SPACE,)``, True exactly at legal move ids."""
    mask = np.zeros(ACTION_SPACE, dtype=np.bool_)
    origin = window_origin(state)  # hoisted: same for every move this turn (MCTS hot path)
    for move in legal_moves(state):
        mask[_encode_with_origin(move, origin)] = True
    return mask
