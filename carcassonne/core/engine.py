from __future__ import annotations

from dataclasses import dataclass

from carcassonne.core.features import FeatureData, FeatureIndex, Meeple, NodeId
from carcassonne.core.placement import fits, placements_for_tile
from carcassonne.core.state import Board, PlacedTile
from carcassonne.core.tiles import TILE_TYPES, TileType
from carcassonne.core.types import (
    FeatureKind,
    IllegalMove,
    MeepleKind,
    Move,
    PlaceMeeple,
    Player,
    Pos,
    RetrieveAbbot,
    Rotation,
)


@dataclass(frozen=True, slots=True)
class ScoreEvent:
    player: Player
    points: int
    kind: FeatureKind
    tiles: frozenset[Pos]


@dataclass(frozen=True)
class GameState:
    board: Board                     # treat as immutable; copied on apply
    features: FeatureIndex
    deck: tuple[str, ...]            # remaining tile type ids, pre-shuffled
    current_tile: str | None         # None <=> terminal
    current_player: Player
    scores: tuple[int, int]
    meeples: tuple[int, int]         # plain meeples in supply (start: 7, 7)
    abbots: tuple[bool, bool]        # abbot in supply?
    abbot_at: tuple[NodeId | None, NodeId | None]
    discarded: tuple[str, ...]       # unplaceable tiles set aside
    last_events: tuple[ScoreEvent, ...]
    turn: int


def legal_moves(state: GameState) -> tuple[Move, ...]:
    """All legal moves, deterministically ordered: (pos, rot) as produced by
    placements_for_tile; actions None, PlaceMeeple by (feature, MEEPLE < ABBOT),
    RetrieveAbbot last."""
    if state.current_tile is None:
        return ()
    tt = TILE_TYPES[state.current_tile]
    p = state.current_player
    any_piece = state.meeples[p] > 0 or state.abbots[p]
    can_retrieve = state.abbot_at[p] is not None
    moves: list[Move] = []
    for pos, rot in placements_for_tile(state.board, tt):
        moves.append(Move(pos, rot, None))
        if any_piece:
            _, idx = _integrated(state.board, state.features, state.current_tile, pos, rot)
            for i in range(len(tt.features)):
                for kind in (MeepleKind.MEEPLE, MeepleKind.ABBOT):
                    if _meeple_legal(state, idx, tt, pos, i, kind):
                        moves.append(Move(pos, rot, PlaceMeeple(i, kind)))
        if can_retrieve:
            moves.append(Move(pos, rot, RetrieveAbbot()))
    return tuple(moves)


def apply(state: GameState, move: Move) -> GameState:
    """Resolve one full turn: place -> integrate -> action -> score city/road
    completions -> monastery/garden sweep -> draw next playable tile -> advance."""
    if state.current_tile is None:
        raise IllegalMove("game is over")
    tt = TILE_TYPES[state.current_tile]
    if not fits(state.board, tt, move.pos, move.rotation):
        raise IllegalMove(f"{state.current_tile} does not fit at {move.pos} {move.rotation.name}")
    p = state.current_player
    board, idx = _integrated(
        state.board, state.features, state.current_tile, move.pos, move.rotation
    )

    scores = list(state.scores)
    meeples = list(state.meeples)
    abbots = list(state.abbots)
    abbot_at: list[NodeId | None] = list(state.abbot_at)
    events: list[ScoreEvent] = []

    # Resolve the meeple action against the post-merge index.
    action = move.action
    if isinstance(action, PlaceMeeple):
        if not _meeple_legal(state, idx, tt, move.pos, action.feature, action.kind):
            raise IllegalMove(f"cannot place {action.kind.value} on feature {action.feature}")
        node: NodeId = (move.pos, action.feature)
        idx = idx.with_meeple(node, Meeple(p, action.kind, node))
        if action.kind is MeepleKind.MEEPLE:
            meeples[p] -= 1
        else:
            abbots[p] = False
            abbot_at[p] = node
    elif isinstance(action, RetrieveAbbot):
        home = state.abbot_at[p]
        if home is None:
            raise IllegalMove("no abbot on the board")
        d = idx.root_data(home)
        points = 1 + sum(1 for q in neighbours8(home[0]) if q in board)
        events.append(ScoreEvent(p, points, d.kind, d.tiles))
        scores[p] += points
        idx = idx.without_feature_meeples(idx.find(home))
        abbots[p] = True
        abbot_at[p] = None

    # Score city/road features completed by this tile (majority; ties all score).
    for root, d in list(idx.data.items()):
        if (
            d.kind not in (FeatureKind.CITY, FeatureKind.ROAD)
            or d.open_edges > 0
            or move.pos not in d.tiles
        ):
            continue
        counts: dict[Player, int] = {}
        for m in d.meeples:
            counts[m.player] = counts.get(m.player, 0) + 1
            meeples[m.player] += 1  # city/road meeples are always plain
        if not counts:
            continue  # completes silently
        points = _score_value(d)
        best = max(counts.values())
        for leader in sorted(pl for pl, c in counts.items() if c == best):
            events.append(ScoreEvent(leader, points, d.kind, d.tiles))
            scores[leader] += points
        idx = idx.without_feature_meeples(root)

    # Monastery/garden sweep: any occupied one in the 3x3 around the new tile
    # that is now fully surrounded scores 9 and returns its piece.
    for q in sorted(neighbours8(move.pos) + (move.pos,)):
        placed = board.get(q)
        if placed is None or any(nq not in board for nq in neighbours8(q)):
            continue
        for i, f in enumerate(TILE_TYPES[placed.type_id].features):
            if f.kind not in (FeatureKind.MONASTERY, FeatureKind.GARDEN):
                continue
            d = idx.data[(q, i)]  # monastery/garden nodes are their own roots
            if not d.meeples:
                continue
            m = d.meeples[0]
            events.append(ScoreEvent(m.player, 9, f.kind, d.tiles))
            scores[m.player] += 9
            if m.kind is MeepleKind.ABBOT:
                abbots[m.player] = True
                abbot_at[m.player] = None
            else:
                meeples[m.player] += 1
            idx = idx.without_feature_meeples((q, i))

    # Draw the next playable tile; set aside unplaceable ones.
    current, deck, set_aside = draw_playable(board, state.deck)

    return GameState(
        board=board,
        features=idx,
        deck=deck,
        current_tile=current,
        current_player=1 - p,
        scores=(scores[0], scores[1]),
        meeples=(meeples[0], meeples[1]),
        abbots=(abbots[0], abbots[1]),
        abbot_at=(abbot_at[0], abbot_at[1]),
        discarded=state.discarded + set_aside,
        last_events=tuple(events),
        turn=state.turn + 1,
    )


def draw_playable(
    board: Board, deck: tuple[str, ...]
) -> tuple[str | None, tuple[str, ...], tuple[str, ...]]:
    """Draw until a playable tile appears (core-internal, not part of the public API).

    Returns (drawn tile or None, remaining deck, unplaceable tiles set aside)."""
    rest = list(deck)
    set_aside: list[str] = []
    while rest:
        candidate = rest.pop(0)
        if placements_for_tile(board, TILE_TYPES[candidate]):
            return candidate, tuple(rest), tuple(set_aside)
        set_aside.append(candidate)
    return None, (), tuple(set_aside)


_ALLOWED_ON: dict[MeepleKind, frozenset[FeatureKind]] = {
    MeepleKind.MEEPLE: frozenset(
        {FeatureKind.CITY, FeatureKind.ROAD, FeatureKind.MONASTERY}
    ),
    MeepleKind.ABBOT: frozenset({FeatureKind.MONASTERY, FeatureKind.GARDEN}),
}


def _integrated(
    board: Board, features: FeatureIndex, tile_id: str, pos: Pos, rot: Rotation
) -> tuple[Board, FeatureIndex]:
    """Copy of board with the tile placed, plus the post-merge feature index."""
    new_board = dict(board)
    new_board[pos] = PlacedTile(tile_id, rot)
    return new_board, features.with_tile(new_board, pos)


def _meeple_legal(
    state: GameState, idx: FeatureIndex, tt: TileType, pos: Pos, feature: int, kind: MeepleKind
) -> bool:
    """Is placing `kind` on the new tile's `feature` legal, given the post-merge index?"""
    if not 0 <= feature < len(tt.features):
        return False
    if tt.features[feature].kind not in _ALLOWED_ON[kind]:
        return False
    p = state.current_player
    if kind is MeepleKind.MEEPLE and state.meeples[p] <= 0:
        return False
    if kind is MeepleKind.ABBOT and not state.abbots[p]:
        return False
    return idx.root_data((pos, feature)).meeples == ()


def _score_value(d: FeatureData) -> int:
    """City/road value when the feature completes: cities double, roads flat."""
    if d.kind is FeatureKind.CITY:
        return 2 * len(d.tiles) + 2 * d.shields
    return len(d.tiles)  # road


def _endgame_value(d: FeatureData) -> int:
    """City/road value if the game ended now (incomplete): 1/tile, +1/shield for cities."""
    return len(d.tiles) + (d.shields if d.kind is FeatureKind.CITY else 0)


def neighbours8(pos: Pos) -> tuple[Pos, ...]:
    return tuple(
        Pos(pos.x + dx, pos.y + dy)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if (dx, dy) != (0, 0)
    )
