from __future__ import annotations

from dataclasses import dataclass

# Supported table sizes (inclusive). The single source for every boundary that
# validates a player count: core.new_game, replay headers, the web session
# store and the CLI.
MIN_PLAYERS = 2
MAX_PLAYERS = 5


@dataclass(frozen=True, slots=True)
class GameConfig:
    num_players: int = MIN_PLAYERS  # MIN_PLAYERS..MAX_PLAYERS; validated by new_game
    meeples_per_player: int = 7
    # abbots are always 1 per player in the base+abbot ruleset
