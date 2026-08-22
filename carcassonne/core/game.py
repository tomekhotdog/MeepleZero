from __future__ import annotations

import random

from carcassonne.core.config import GameConfig
from carcassonne.core.engine import GameState, _neighbours8, draw_playable
from carcassonne.core.features import FeatureIndex
from carcassonne.core.state import empty_board_with_start
from carcassonne.core.tiles import DECK, START_TILE_ID
from carcassonne.core.types import FeatureKind, Player, Pos

_DEFAULT_CONFIG = GameConfig()


def new_game(seed: int, config: GameConfig = _DEFAULT_CONFIG) -> GameState:
    """Fresh game: start D tile at (0,0) R0, deck shuffled by `seed`, first tile drawn."""
    tiles = [
        t.id for t in DECK for _ in range(t.count - (1 if t.id == START_TILE_ID else 0))
    ]
    random.Random(seed).shuffle(tiles)
    board = empty_board_with_start()
    features = FeatureIndex.empty().with_tile(board, Pos(0, 0))
    current, deck, discarded = draw_playable(board, tuple(tiles))
    m = config.meeples_per_player
    return GameState(
        board=board,
        features=features,
        deck=deck,
        current_tile=current,
        current_player=0,
        scores=(0, 0),
        meeples=(m, m),
        abbots=(True, True),
        abbot_at=(None, None),
        discarded=discarded,
        last_events=(),
        turn=0,
    )


def is_terminal(state: GameState) -> bool:
    return state.current_tile is None


def final_scores(state: GameState) -> tuple[int, int]:
    """Live scores plus end-game value of every meepled feature. Pure; no events.

    Incomplete city: 1/tile + 1/shield; incomplete road: 1/tile — awarded to all
    tied majority holders. Monastery/garden with a piece: 1 + occupied
    8-neighbours, to the piece's owner.
    """
    scores = list(state.scores)
    for d in state.features.data.values():
        if not d.meeples:
            continue
        if d.kind in (FeatureKind.CITY, FeatureKind.ROAD):
            if d.open_edges == 0:
                continue  # completed features were scored live and cleared
            points = len(d.tiles) + (d.shields if d.kind is FeatureKind.CITY else 0)
            counts: dict[Player, int] = {}
            for m in d.meeples:
                counts[m.player] = counts.get(m.player, 0) + 1
            best = max(counts.values())
            for player, c in counts.items():
                if c == best:
                    scores[player] += points
        else:  # monastery / garden: exactly one tile, at most one piece
            (pos,) = d.tiles
            points = 1 + sum(1 for q in _neighbours8(pos) if q in state.board)
            scores[d.meeples[0].player] += points
    return (scores[0], scores[1])
