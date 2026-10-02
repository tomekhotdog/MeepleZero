"""Golden behaviour digest of the rules engine (M18b guard).

Hot-path optimisations must not change any game. The digest covers every
legal_moves() result -- content AND order, since agents pick by index -- and the
resulting scores/supplies/tile/discards, across seeded random games at 2 and 4
players. Recorded on the pre-optimisation engine; a changed digest is a
behaviour change, not a speed-up. Tile sets are hashed sorted, so only
behaviour (not set construction order) can change the digest.
"""

from __future__ import annotations

import hashlib
import random

import pytest

from carcassonne.core import GameConfig, apply, is_terminal, legal_moves, new_game
from carcassonne.core.engine import ScoreEvent

GOLDEN = {
    2: "5b96e27600d0a55dc2029c286f3be1f18574cf7303a1719cf4d9f953d4b026ed",
    4: "4bf6c00e319dbee77c79e912affef53e0498ad5c68e83c8f6c0257c80de32052",
}


def _canon(events: tuple[ScoreEvent, ...]) -> tuple[object, ...]:
    """ScoreEvents with `tiles` as a sorted tuple: frozenset repr order depends
    on insertion order, which is not behaviour."""
    return tuple((e.player, e.points, e.kind.value, tuple(sorted(e.tiles))) for e in events)


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
                     state.current_tile, state.discarded, _canon(state.last_events))
                ).encode()
            )
    return h.hexdigest()


@pytest.mark.parametrize("num_players", [2, 4])
def test_engine_behaviour_is_unchanged(num_players: int) -> None:
    assert _digest(num_players, range(12)) == GOLDEN[num_players]
