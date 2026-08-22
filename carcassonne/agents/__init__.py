"""Agents: policies mapping GameState -> Move.

Public API: Agent, TurnContext, RandomAgent, make_agent. Importing this package
registers the built-in agents with the registry in `base`.
"""

from carcassonne.agents.base import Agent, TurnContext, make_agent
from carcassonne.agents.random_agent import RandomAgent

__all__ = ["Agent", "RandomAgent", "TurnContext", "make_agent"]
