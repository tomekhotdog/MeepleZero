"""Agent protocol and registry: an agent maps a game state to a move."""

from __future__ import annotations

import random
from collections.abc import Callable
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


_FACTORIES: dict[str, Callable[[], Agent]] = {}


def register_agent(spec: str, factory: Callable[[], Agent]) -> None:
    """Register a factory under a spec string ("random"; later: "greedy", "ckpt:<id>")."""
    _FACTORIES[spec] = factory


def make_agent(spec: str) -> Agent:
    """Build an agent from its spec string. ValueError on unknown spec."""
    factory = _FACTORIES.get(spec)
    if factory is None:
        known = ", ".join(sorted(_FACTORIES))
        raise ValueError(f"unknown agent spec {spec!r} (known: {known})")
    return factory()
