"""Agents: policies mapping GameState -> Move.

Public API: Agent, TurnContext, RandomAgent, GreedyAgent, make_agent. Importing
this package registers the built-in agents with the registry in `base`.
"""

from carcassonne.agents.base import Agent, TurnContext, make_agent
from carcassonne.agents.greedy import GreedyAgent
from carcassonne.agents.random_agent import RandomAgent

__all__ = ["Agent", "GreedyAgent", "RandomAgent", "TurnContext", "make_agent"]
