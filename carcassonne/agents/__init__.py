"""Agents: policies mapping GameState -> Move.

Public API: Agent, TurnContext, RandomAgent, GreedyAgent, MctsAgent, MctsConfig,
make_agent. Importing this package registers the parameterless built-in agents
with the registry in `base`, plus the ``ckpt:<id>`` checkpoint-opponent prefix
(see `checkpoints`), which resolves against a directory configured at app startup.
"""

from carcassonne.agents import checkpoints  # noqa: F401 -- registers the "ckpt:" prefix
from carcassonne.agents.base import Agent, TurnContext, make_agent
from carcassonne.agents.checkpoints import list_checkpoints, set_checkpoints_dir
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
    "list_checkpoints",
    "make_agent",
    "set_checkpoints_dir",
]
