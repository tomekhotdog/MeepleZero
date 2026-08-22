"""1-ply greedy heuristic: the hand-written yardstick the trained AI must beat.

Every legal move is applied (`core.apply`) and the resulting state is valued
from the mover's perspective:

    value(move) = my_score_delta - opp_score_delta
                + 0.7 * Δ(my endgame_potential) - 0.7 * Δ(opp endgame_potential)
                + 0.15 * my pieces in supply after the move (meeples + abbot)

where `endgame_potential` (core) is a player's share of final_scores' end-game
component. The score deltas reward completions; the potential terms reward
claiming/growing features and blocking the opponent's; the supply term nudges
against squandering pieces on low-value features.
"""

from __future__ import annotations

import math
import time
from dataclasses import replace

from carcassonne.agents.base import TurnContext, register_agent
from carcassonne.core import GameState, Move, Player, apply, endgame_potential, legal_moves
from carcassonne.game.serde import Annot

_W_POTENTIAL = 0.7
_W_SUPPLY = 0.15
_TOP_K = 8
# Annot.value must land in [-1, 1] like a network value head: tanh(best / 10)
# maps a typical one-turn swing (a 10-point coup ~> 0.76) onto that range.
_VALUE_SCALE = 10.0


class GreedyAgent:
    """Argmax of the 1-ply value above; ties broken uniformly by ctx.rng."""

    name: str = "greedy"

    def choose(self, state: GameState, ctx: TurnContext) -> tuple[Move, Annot | None]:
        start = time.perf_counter()
        me: Player = state.current_player
        opp: Player = 1 - me
        pot_me = endgame_potential(state, me)
        pot_opp = endgame_potential(state, opp)
        moves = legal_moves(state)
        # Evaluate on a deckless copy: apply's post-move draw scans the board for
        # a placeable tile, which dominates runtime and cannot affect the value
        # (scores, supplies, and features are settled before the draw).
        probe = replace(state, deck=())
        values = [_value(probe, move, me, opp, pot_me, pot_opp) for move in moves]
        best = max(values)
        move = ctx.rng.choice([m for m, v in zip(moves, values, strict=True) if v == best])
        if not ctx.annotate:
            return move, None
        think_ms = int((time.perf_counter() - start) * 1000)
        return move, _annot(moves, values, move, best, think_ms)


def _value(
    state: GameState, move: Move, me: Player, opp: Player, pot_me: int, pot_opp: int
) -> float:
    nxt = apply(state, move)
    supply = nxt.meeples[me] + (1 if nxt.abbots[me] else 0)
    return (
        (nxt.scores[me] - state.scores[me])
        - (nxt.scores[opp] - state.scores[opp])
        + _W_POTENTIAL * (endgame_potential(nxt, me) - pot_me)
        - _W_POTENTIAL * (endgame_potential(nxt, opp) - pot_opp)
        + _W_SUPPLY * supply
    )


def _annot(
    moves: tuple[Move, ...], values: list[float], chosen: Move, best: float, think_ms: int
) -> Annot:
    """UI annotation: value = tanh(best / 10) in [-1, 1]; top = up to 8 moves with
    softmax(values) weights (normalised over ALL evaluated moves, so the shown
    weights only sum to 1 when every move fits in the list) and 0 visits.

    The chosen move leads the list even among softmax ties, so top[0] is always
    what was played.
    """
    weights = [math.exp(v - best) for v in values]
    total = sum(weights)
    ranked = sorted(
        zip(moves, weights, strict=True), key=lambda mw: (-mw[1], mw[0] != chosen)
    )[:_TOP_K]
    top = tuple((move, weight / total, 0) for move, weight in ranked)
    return Annot(value=math.tanh(best / _VALUE_SCALE), sims=0, think_ms=think_ms, top=top)


register_agent("greedy", GreedyAgent)
