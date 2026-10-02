from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from carcassonne.core.features import FeatureData, FeatureIndex, Meeple, NodeId
from carcassonne.core.placement import fits, placements_for_tile
from carcassonne.core.state import Board, PlacedTile
from carcassonne.core.tiles import TILE_TYPES, TileFeature, TileType, rotated_feature_sides
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
    scores: tuple[int, ...]
    meeples: tuple[int, ...]         # plain meeples in supply (7 per seat at start)
    abbots: tuple[bool, ...]         # abbot in supply?
    abbot_at: tuple[NodeId | None, ...]
    discarded: tuple[str, ...]       # unplaceable tiles set aside
    last_events: tuple[ScoreEvent, ...]
    turn: int
    # Placements of current_tile on `board`, computed while drawing it (apply /
    # new_game) so legal_moves needn't recompute them. Self-validating: used only
    # while the recorded board object and tile id still match -- a
    # dataclasses.replace() with a new board (learner.rotate_state_90) leaves it
    # stale, and it is then ignored. compare=False: never part of state equality
    # (MCTS subtree reuse matches states with ==).
    placements_cache: tuple[Board, str, tuple[tuple[Pos, Rotation], ...]] | None = field(
        default=None, compare=False, repr=False
    )


def legal_moves(state: GameState) -> tuple[Move, ...]:
    """All legal moves, deterministically ordered: (pos, rot) as produced by
    placements_for_tile; actions None, PlaceMeeple by (feature, MEEPLE < ABBOT),
    RetrieveAbbot last."""
    if state.current_tile is None:
        return ()
    tt = TILE_TYPES[state.current_tile]
    p = state.current_player
    has_meeple = state.meeples[p] > 0
    has_abbot = state.abbots[p]
    can_retrieve = state.abbot_at[p] is not None
    moves: list[Move] = []
    for pos, rot in _current_placements(state):
        moves.append(Move(pos, rot, None))
        if has_meeple or has_abbot:
            for i, f in enumerate(tt.features):
                claimed: bool | None = None  # computed lazily, at most once per feature
                for kind in (MeepleKind.MEEPLE, MeepleKind.ABBOT):
                    if f.kind not in _ALLOWED_ON[kind]:
                        continue
                    if kind is MeepleKind.MEEPLE and not has_meeple:
                        continue
                    if kind is MeepleKind.ABBOT and not has_abbot:
                        continue
                    if claimed is None:
                        claimed = _claimed_on_placement(state.board, state.features, f, pos, rot)
                    if not claimed:
                        moves.append(Move(pos, rot, PlaceMeeple(i, kind)))
        if can_retrieve:
            moves.append(Move(pos, rot, RetrieveAbbot()))
    return tuple(moves)


def _current_placements(state: GameState) -> Sequence[tuple[Pos, Rotation]]:
    """Placements of the current tile -- carried from the draw when still valid."""
    cache = state.placements_cache
    if cache is not None and cache[0] is state.board and cache[1] == state.current_tile:
        return cache[2]
    assert state.current_tile is not None  # callers handle the terminal state first
    return placements_for_tile(state.board, TILE_TYPES[state.current_tile])


def _claimed_on_placement(
    board: Board, features: FeatureIndex, f: TileFeature, pos: Pos, rot: Rotation
) -> bool:
    """Would feature `f` of a tile placed at (pos, rot) join a feature that already
    holds a meeple?

    Equivalent to integrating the tile and reading the merged root's meeples
    (`_meeple_legal` on `_integrated`'s index): FeatureIndex.with_tile starts new
    nodes meeple-less and unions a new city/road node only with the first matching
    neighbour feature across each of its sides; monasteries/gardens never merge.
    So the merged root's meeples are the union of those neighbours' roots' meeples
    -- provided no sibling feature of the same kind can bridge in a further root,
    which holds because every tile with 2+ city (or road) features gives each a
    single edge (pinned by test_no_tile_can_bridge_neighbour_features_through_a_sibling).
    No board/index copy."""
    if f.kind not in (FeatureKind.CITY, FeatureKind.ROAD):
        return False
    for side in rotated_feature_sides(f, rot):
        npos = pos.neighbor(side)
        nplaced = board.get(npos)
        if nplaced is None:
            continue
        for j, nf in enumerate(TILE_TYPES[nplaced.type_id].features):
            if side.opposite in rotated_feature_sides(nf, nplaced.rotation):
                if features.root_data((npos, j)).meeples:
                    return True
                break
    return False


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
    current, deck, set_aside, placements = draw_playable(board, state.deck)

    return GameState(
        board=board,
        features=idx,
        deck=deck,
        current_tile=current,
        current_player=(p + 1) % len(state.scores),
        scores=tuple(scores),
        meeples=tuple(meeples),
        abbots=tuple(abbots),
        abbot_at=tuple(abbot_at),
        discarded=state.discarded + set_aside,
        last_events=tuple(events),
        turn=state.turn + 1,
        placements_cache=(board, current, placements) if current is not None else None,
    )


def draw_playable(
    board: Board, deck: tuple[str, ...]
) -> tuple[str | None, tuple[str, ...], tuple[str, ...], tuple[tuple[Pos, Rotation], ...]]:
    """Draw until a playable tile appears (core-internal, not part of the public API).

    Returns (drawn tile or None, remaining deck, unplaceable tiles set aside,
    the drawn tile's placements -- () when the deck ran out)."""
    rest = list(deck)
    set_aside: list[str] = []
    while rest:
        candidate = rest.pop(0)
        placements = placements_for_tile(board, TILE_TYPES[candidate])
        if placements:
            return candidate, tuple(rest), tuple(set_aside), tuple(placements)
        set_aside.append(candidate)
    return None, (), tuple(set_aside), ()


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
