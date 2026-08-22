"""Uniform-random agent: the baseline opponent and smoke-test workhorse."""

from __future__ import annotations

from carcassonne.agents.base import TurnContext, register_agent
from carcassonne.core import GameState, Move, legal_moves
from carcassonne.game.serde import Annot


class RandomAgent:
    """Picks uniformly among legal moves; never annotates."""

    name: str = "random"

    def choose(self, state: GameState, ctx: TurnContext) -> tuple[Move, Annot | None]:
        return ctx.rng.choice(legal_moves(state)), None


register_agent("random", RandomAgent)
