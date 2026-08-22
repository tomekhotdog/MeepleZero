from __future__ import annotations

import random

from carcassonne.core import apply, is_terminal, legal_moves, new_game
from carcassonne.core.engine import GameState
from carcassonne.core.types import Pos
from carcassonne.nn.actions import (
    ACTION_SPACE,
    decode_move,
    encode_move,
    legal_mask,
    to_window,
    window_origin,
)


def _sample_states(n: int, seed: int) -> list[GameState]:
    """States sampled along random games (each recorded before a random move)."""
    rng = random.Random(seed)
    states: list[GameState] = []
    state = new_game(rng.randint(0, 10_000))
    while len(states) < n:
        if is_terminal(state):
            state = new_game(rng.randint(0, 10_000))
            continue
        states.append(state)
        state = apply(state, rng.choice(legal_moves(state)))
    return states


def test_action_space_size() -> None:
    assert ACTION_SPACE == 53816


def test_window_origin_centres_fresh_game() -> None:
    state = new_game(0)
    assert list(state.board) == [Pos(0, 0)]  # only the start tile
    ox, oy = window_origin(state)
    assert (ox, oy) == (-15, -15)
    assert to_window(state, Pos(0, 0)) == (15, 15)


def test_round_trip_over_legal_moves() -> None:
    for state in _sample_states(50, seed=7):
        moves = legal_moves(state)
        ids = set()
        for move in moves:
            idx = encode_move(state, move)
            assert decode_move(state, idx) == move
            ids.add(idx)
        assert len(ids) == len(moves)  # encode is injective over the legal set


def test_legal_mask_matches_legal_moves() -> None:
    for state in _sample_states(50, seed=11):
        moves = legal_moves(state)
        mask = legal_mask(state)
        assert mask.shape == (ACTION_SPACE,)
        assert mask.dtype == bool
        assert int(mask.sum()) == len(moves)
        legal = set(moves)
        for idx in mask.nonzero()[0]:
            assert decode_move(state, int(idx)) in legal
