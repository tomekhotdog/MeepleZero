"""Agents: policies mapping GameState -> Move.

Public API: Agent, TurnContext, RandomAgent, GreedyAgent, MctsAgent, MctsConfig,
make_agent. Importing this package registers the parameterless built-in agents
with the registry in `base`. MctsAgent needs a net + device, so it is exported
here but *not* registered under a bare "mcts" spec -- checkpoint specs
("ckpt:<id>") arrive in Task 17.
"""

from carcassonne.agents.base import Agent, TurnContext, make_agent
from carcassonne.agents.greedy import GreedyAgent
from carcassonne.agents.mcts import MctsAgent, MctsConfig
from carcassonne.agents.random_agent import RandomAgent

__all__ = [
    "Agent",
    "GreedyAgent",
    "MctsAgent",
    "MctsConfig",
    "RandomAgent",
    "TurnContext",
    "make_agent",
]
