# tests/nn/test_actions.py
"""Crop geometry and the world-coordinate Move codec.

Correctness backbone: for ANY reachable state, the crop contains every legal
move, and encode/decode is a bijection between legal moves and flat indices.
"""
from __future__ import annotations

import random

import pytest

from carcassonne.core import GameConfig, apply, is_terminal, legal_moves, new_game
from carcassonne.core.types import MeepleKind, Move, PlaceMeeple, Pos, Rotation, RulesError
from carcassonne.nn.actions import (
    ACTIONS_PER_CELL,
    Crop,
    decode_move,
    encode_move,
    flat_index,
    key_move,
    legal_mask,
    move_key,
    rotate_key,
)


def _random_states(seed: int, n_states: int = 30):
    """Yield states sampled from a random playthrough (fresh game per exhaustion)."""
    rng = random.Random(seed)
    state = new_game(rng.randrange(10_000), GameConfig())
    for _ in range(n_states):
        if is_terminal(state):
            state = new_game(rng.randrange(10_000), GameConfig())
        yield state
        state = apply(state, rng.choice(legal_moves(state)))


def test_crop_is_bbox_plus_one_border() -> None:
    state = new_game(0, GameConfig())  # single start tile at (0, 0)
    crop = Crop.of(state)
    assert (crop.ox, crop.oy, crop.h, crop.w) == (-1, -1, 3, 3)
    assert crop.n_actions == 3 * 3 * ACTIONS_PER_CELL


def test_crop_contains_every_legal_move() -> None:
    for state in _random_states(1, n_states=60):
        crop = Crop.of(state)
        for m in legal_moves(state):
            gx, gy = m.pos.x - crop.ox, m.pos.y - crop.oy
            assert 0 <= gx < crop.w and 0 <= gy < crop.h


def test_encode_decode_roundtrip_all_legal_moves() -> None:
    for state in _random_states(2, n_states=40):
        for m in legal_moves(state):
            assert decode_move(state, encode_move(state, m)) == m


def test_legal_mask_bijection() -> None:
    """mask True exactly at the encodings of legal moves — no drops, no ghosts."""
    for state in _random_states(3, n_states=40):
        crop = Crop.of(state)
        mask = legal_mask(state)
        assert mask.shape == (crop.n_actions,)
        expected = {encode_move(state, m) for m in legal_moves(state)}
        got = {int(i) for i in mask.nonzero()[0]}
        assert got == expected


def test_padded_stride_indexing() -> None:
    """flat_index with a wider stride matches manual arithmetic (learner padding)."""
    state = new_game(0, GameConfig())
    crop = Crop.of(state)  # 3x3
    m = legal_moves(state)[0]
    key = move_key(m)
    idx = flat_index(crop, key, stride_w=10)
    gx, gy = key[0] - crop.ox, key[1] - crop.oy
    assert idx == ((gy * 10 + gx) * 4 + key[2]) * 14 + key[3]
    padded = legal_mask(state, pad_hw=(10, 10))
    assert padded.shape == (10 * 10 * ACTIONS_PER_CELL,)
    assert padded[idx]


def test_move_key_roundtrip_and_rotation() -> None:
    m = Move(Pos(3, -2), Rotation.R90, PlaceMeeple(2, MeepleKind.ABBOT))
    assert key_move(move_key(m)) == m
    k = move_key(m)
    k4 = k
    for _ in range(4):
        k4 = rotate_key(k4)
    assert k4 == k  # four quarter-turns are the identity
    assert rotate_key(k)[:3] == (-2, -3, 2)  # (x,y)->(y,-x), rot+1


def test_flat_index_raises_outside_crop() -> None:
    state = new_game(0, GameConfig())
    with pytest.raises(RulesError):
        flat_index(Crop.of(state), (99, 99, 0, 0))


def test_decode_move_raises_out_of_range() -> None:
    state = new_game(0, GameConfig())
    for idx in (-1, Crop.of(state).n_actions, 10**9):
        with pytest.raises(RulesError):
            decode_move(state, idx)


def test_legal_mask_rejects_padding_smaller_than_crop() -> None:
    state = new_game(0, GameConfig())  # crop is 3x3
    with pytest.raises(RulesError):
        legal_mask(state, pad_hw=(2, 2))


def test_rotate_key_negative_k_is_inverse() -> None:
    k = move_key(Move(Pos(3, -2), Rotation.R90, PlaceMeeple(2, MeepleKind.ABBOT)))
    assert rotate_key(rotate_key(k, 1), -1) == k
    assert rotate_key(k, -1) == rotate_key(k, 3)
