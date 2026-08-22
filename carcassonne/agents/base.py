"""Agent registry, plus re-export of the Agent protocol (defined in game.agent).

Import agents via the package (`carcassonne.agents`) so built-in registrations
run; importing this module directly leaves the registry empty.
"""

from __future__ import annotations

from collections.abc import Callable

from carcassonne.game.agent import Agent, TurnContext

__all__ = ["Agent", "TurnContext", "make_agent", "register_agent"]

_FACTORIES: dict[str, Callable[[], Agent]] = {}


def register_agent(spec: str, factory: Callable[[], Agent]) -> None:
    """Register a factory under a spec string ("random"; later: "greedy", "ckpt:<id>")."""
    existing = _FACTORIES.get(spec)
    if existing is not None and existing is not factory:
        raise ValueError(f"agent spec {spec!r} already registered with a different factory")
    _FACTORIES[spec] = factory


def make_agent(spec: str) -> Agent:
    """Build an agent from its spec string. ValueError on unknown spec."""
    factory = _FACTORIES.get(spec)
    if factory is None:
        known = ", ".join(sorted(_FACTORIES))
        raise ValueError(f"unknown agent spec {spec!r} (known: {known})")
    return factory()
