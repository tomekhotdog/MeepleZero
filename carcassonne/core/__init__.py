"""Rules core. Public API: new_game, legal_moves, apply, is_terminal, final_scores."""

from carcassonne.core.config import GameConfig
from carcassonne.core.engine import GameState, ScoreEvent, apply, legal_moves
from carcassonne.core.game import final_scores, is_terminal, new_game
from carcassonne.core.types import IllegalMove, Move, PlaceMeeple, RetrieveAbbot, RulesError

__all__ = [
    "GameConfig",
    "GameState",
    "IllegalMove",
    "Move",
    "PlaceMeeple",
    "RetrieveAbbot",
    "RulesError",
    "ScoreEvent",
    "apply",
    "final_scores",
    "is_terminal",
    "legal_moves",
    "new_game",
]
