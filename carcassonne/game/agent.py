"""Agent protocol: an agent maps a game state to a move.

Lives in game/ (not agents/) so the layering stays one-way: the protocol's
vocabulary (GameState, Move, Annot) is defined at or below this layer, and both
game.match and the agents package import it downward/sideways. The agent
registry and concrete agents live in carcassonne.agents.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

from carcassonne.core import GameState, Move
from carcassonne.game.serde import Annot


@dataclass(frozen=True, slots=True)
class TurnContext:
    """Per-game context handed to agents on every turn.

    `rng` is one seeded generator shared by both agents for the whole game
    (seeded by the match runner, distinct from the deck seed). When `annotate`
    is False, agents may skip expensive annotation work and return None.
    """

    rng: random.Random
    annotate: bool = False


class Agent(Protocol):
    name: str

    def choose(self, state: GameState, ctx: TurnContext) -> tuple[Move, Annot | None]: ...
