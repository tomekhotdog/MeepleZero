"""Read-only JSON projections for the web API.

Every response body is built here as a plain dict, so the API surface lives in
one place and app.py stays a thin routing layer. Nothing here mutates state.
"""

from __future__ import annotations

from typing import Any

from carcassonne.core import (
    GameState,
    Move,
    ScoreEvent,
    feature_report,
    final_scores,
    is_terminal,
)
from carcassonne.core.tiles import DECK, derived_edges
from carcassonne.core.types import Pos
from carcassonne.game.replay import MoveRecord, ReplayHeader
from carcassonne.game.serde import Annot, annot_to_json, move_to_json


def state_view(state: GameState) -> dict[str, Any]:
    """StateView: the board plus everything a client needs to render a turn.

    Each meeple is attached to the tile it physically stands on (`Meeple.node`),
    with `feature` = the feature index on that tile, so the frontend can draw it
    on the right spot regardless of feature merges.
    """
    meeples_at: dict[Pos, list[dict[str, Any]]] = {}
    for d in state.features.data.values():
        for m in d.meeples:
            pos, feature = m.node
            report = feature_report(state, m.node)
            meeples_at.setdefault(pos, []).append(
                {
                    "player": m.player,
                    "kind": m.kind.value,
                    "feature": feature,
                    "feature_tiles": [[p.x, p.y] for p in report.tiles],
                    "score_now": report.current,
                    "score_potential": report.potential,
                    "feature_kind": report.kind.value,
                    "complete": report.complete,
                }
            )
    tiles = [
        {
            "x": pos.x,
            "y": pos.y,
            "type": placed.type_id,
            "rot": int(placed.rotation),
            "meeples": sorted(meeples_at.get(pos, []), key=lambda m: m["feature"]),
        }
        for pos, placed in sorted(state.board.items(), key=lambda kv: kv[0])
    ]
    terminal = is_terminal(state)
    view: dict[str, Any] = {
        "tiles": tiles,
        "scores": list(state.scores),
        "meeples": list(state.meeples),
        "abbots": list(state.abbots),
        "current_tile": state.current_tile,
        "current_player": state.current_player,
        "turn": state.turn,
        "last_events": [score_event_view(e) for e in state.last_events],
        "terminal": terminal,
    }
    if terminal:
        view["final_scores"] = list(final_scores(state))
    return view


def score_event_view(event: ScoreEvent) -> dict[str, Any]:
    return {
        "player": event.player,
        "points": event.points,
        "kind": event.kind.value,
        "tiles": sorted([p.x, p.y] for p in event.tiles),
    }


def legal_moves_view(moves: list[Move]) -> dict[str, Any]:
    """Each move carries its `idx` into this exact list; POST /move consumes it."""
    return {"moves": [{**move_to_json(m), "idx": i} for i, m in enumerate(moves)]}


def ai_move_view(move: Move, annot: Annot | None) -> dict[str, Any]:
    return {"move": move_to_json(move), "annot": None if annot is None else annot_to_json(annot)}


def hint_view(legal: list[Move], annot: Annot | None) -> dict[str, Any]:
    """Placement policy aggregated per (x, y, rot), meeple variants summed,
    renormalised to 1. Without an annot (e.g. the random agent) the policy is
    uniform over legal moves and the value is 0.

    ``top`` carries the raw per-move candidates -- prior AND visits -- so a hint
    from an MctsAgent shows the search's prior-vs-visits story, not just the
    aggregated heatmap. It is empty when there is no annot (random agent)."""
    if annot is None:
        weighted = [(m, 1.0) for m in legal]
        value = 0.0
        top: list[dict[str, Any]] = []
    else:
        weighted = [(m, prior) for m, prior, _ in annot.top]
        value = annot.value
        top = [
            {"move": move_to_json(m), "prior": prior, "visits": visits}
            for m, prior, visits in annot.top
        ]
    agg: dict[tuple[int, int, int], float] = {}
    for move, w in weighted:
        key = (move.pos.x, move.pos.y, int(move.rotation))
        agg[key] = agg.get(key, 0.0) + w
    total = sum(agg.values())
    policy = [
        {"x": x, "y": y, "rot": rot, "prob": w / total}
        for (x, y, rot), w in sorted(agg.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    return {"policy": policy, "value": value, "top": top}


def tiledefs_view() -> dict[str, Any]:
    """Static geometry of every tile type, enough to draw tiles procedurally."""
    return {
        "tiles": {
            tt.id: {
                "count": tt.count,
                "edges": [kind.value for kind in derived_edges(tt)],
                "features": [
                    {
                        "kind": f.kind.value,
                        "edges": sorted(int(s) for s in f.edges),
                        "shield": f.shield,
                    }
                    for f in tt.features
                ],
            }
            for tt in DECK
        }
    }


def replay_header_view(header: ReplayHeader) -> dict[str, Any]:
    return {
        "v": header.v,
        "seed": header.seed,
        "config": {"meeples_per_player": header.config.meeples_per_player},
        "agents": list(header.agents),
        "checkpoint": header.checkpoint,
        "started_at": header.started_at,
    }


def move_record_view(rec: MoveRecord) -> dict[str, Any]:
    return {
        "n": rec.n,
        "player": rec.player,
        "tile": rec.tile,
        "move": move_to_json(rec.move),
        "score_after": list(rec.score_after),
        "annot": None if rec.annot is None else annot_to_json(rec.annot),
    }
