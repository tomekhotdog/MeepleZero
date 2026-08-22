"""Agent registry, plus re-export of the Agent protocol (defined in game.agent).

Import agents via the package (`carcassonne.agents`) so built-in registrations
run; importing this module directly leaves the registry empty.
"""

from __future__ import annotations

from collections.abc import Callable

from carcassonne.game.agent import Agent, TurnContext

__all__ = ["Agent", "TurnContext", "make_agent", "register_agent", "register_prefix"]

_FACTORIES: dict[str, Callable[[], Agent]] = {}
# Prefix factories handle whole *families* of specs whose id is only known at call
# time -- e.g. "ckpt:<id>". Keyed by prefix ("ckpt:"); the factory receives the
# full spec and parses the id itself.
_PREFIX_FACTORIES: dict[str, Callable[[str], Agent]] = {}


def register_agent(spec: str, factory: Callable[[], Agent]) -> None:
    """Register a factory under an exact spec string ("random", "greedy")."""
    existing = _FACTORIES.get(spec)
    if existing is not None and existing is not factory:
        raise ValueError(f"agent spec {spec!r} already registered with a different factory")
    _FACTORIES[spec] = factory


def register_prefix(prefix: str, factory: Callable[[str], Agent]) -> None:
    """Register a factory for every spec starting with ``prefix`` (e.g. "ckpt:").

    The factory is passed the *full* spec string and parses the id after the
    prefix. Exact specs (``register_agent``) always win over prefixes."""
    existing = _PREFIX_FACTORIES.get(prefix)
    if existing is not None and existing is not factory:
        raise ValueError(f"agent prefix {prefix!r} already registered with a different factory")
    _PREFIX_FACTORIES[prefix] = factory


def make_agent(spec: str) -> Agent:
    """Build an agent from its spec string. ValueError on unknown spec."""
    factory = _FACTORIES.get(spec)
    if factory is not None:
        return factory()
    for prefix, prefix_factory in _PREFIX_FACTORIES.items():
        if spec.startswith(prefix):
            return prefix_factory(spec)
    known = ", ".join(sorted(_FACTORIES) + [f"{p}<id>" for p in sorted(_PREFIX_FACTORIES)])
    raise ValueError(f"unknown agent spec {spec!r} (known: {known})")
