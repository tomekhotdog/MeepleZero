from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GameConfig:
    meeples_per_player: int = 7
    # abbots are always 1 per player in the base+abbot ruleset
