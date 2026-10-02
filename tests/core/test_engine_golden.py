"""Golden behaviour digest of the rules engine (M18b guard).

Hot-path optimisations must not change any game. The digest covers every
legal_moves() result -- content AND order, since agents pick by index -- and the
resulting scores/supplies/tile/discards, across seeded random games at 2 and 4
players. Recorded on the pre-optimisation engine; a changed digest is a
behaviour change, not a speed-up.
"""

from __future__ import annotations

import hashlib
import random

import pytest

from carcassonne.core import GameConfig, apply, is_terminal, legal_moves, new_game

GOLDEN = {
    2: "2a76ba2f32c4bda956ccd940fae1ee644e53c56c4a726e4ab6212936bf3967bc",
    4: "b19d27b458befab198554c83920acbac75f2e9e85529614d7d3882de2c3eecb7",
}


def _digest(num_players: int, seeds: range) -> str:
    h = hashlib.sha256()
    for seed in seeds:
        rng = random.Random(seed)
        state = new_game(seed, GameConfig(num_players=num_players))
        while not is_terminal(state):
            moves = legal_moves(state)
            h.update(repr(moves).encode())
            state = apply(state, rng.choice(moves))
            h.update(
                repr(
                    (state.scores, state.meeples, state.abbots, state.abbot_at,
                     state.current_tile, state.discarded, state.last_events)
                ).encode()
            )
    return h.hexdigest()


@pytest.mark.parametrize("num_players", [2, 4])
def test_engine_behaviour_is_unchanged(num_players: int) -> None:
    assert _digest(num_players, range(12)) == GOLDEN[num_players]
