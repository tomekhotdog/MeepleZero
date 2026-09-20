from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GameConfig:
    num_players: int = 2  # 2..5; validated by new_game
    meeples_per_player: int = 7
    # abbots are always 1 per player in the base+abbot ruleset
