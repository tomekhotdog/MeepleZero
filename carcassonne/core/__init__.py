"""Rules core. Public API: new_game, legal_moves, apply, is_terminal, final_scores,
plus endgame_potential for heuristic agents."""

from carcassonne.core.config import GameConfig
from carcassonne.core.engine import GameState, ScoreEvent, apply, legal_moves
from carcassonne.core.game import (
    FeatureReport,
    endgame_potential,
    feature_report,
    final_scores,
    is_terminal,
    new_game,
)
from carcassonne.core.types import (
    IllegalMove,
    MeepleKind,
    Move,
    PlaceMeeple,
    Player,
    Pos,
    RetrieveAbbot,
    Rotation,
    RulesError,
)

__all__ = [
    "FeatureReport",
    "GameConfig",
    "GameState",
    "IllegalMove",
    "MeepleKind",
    "Move",
    "PlaceMeeple",
    "Player",
    "Pos",
    "RetrieveAbbot",
    "Rotation",
    "RulesError",
    "ScoreEvent",
    "apply",
    "endgame_potential",
    "feature_report",
    "final_scores",
    "is_terminal",
    "legal_moves",
    "new_game",
]
