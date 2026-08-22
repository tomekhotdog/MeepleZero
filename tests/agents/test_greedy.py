"""GreedyAgent: endgame_potential, obvious-completion pick, annotation, yardstick."""

from __future__ import annotations

import math
import random

import pytest

from carcassonne.agents import GreedyAgent, RandomAgent, TurnContext, make_agent
from carcassonne.core import GameState, Move, Pos, Rotation, endgame_potential, legal_moves
from carcassonne.core.features import FeatureIndex, Meeple, NodeId
from carcassonne.core.state import Board, PlacedTile
from carcassonne.core.types import MeepleKind
from carcassonne.game.arena import run_match


def _build(*tiles: tuple[Pos, str, Rotation]) -> tuple[Board, FeatureIndex]:
    board: Board = {}
    idx = FeatureIndex.empty()
    for pos, tid, rot in tiles:
        board[pos] = PlacedTile(tid, rot)
        idx = idx.with_tile(board, pos)
    return board, idx


def _state(
    board: Board,
    features: FeatureIndex,
    *,
    current_tile: str | None = None,
    meeples: tuple[int, int] = (7, 7),
) -> GameState:
    return GameState(
        board=board,
        features=features,
        deck=(),
        current_tile=current_tile,
        current_player=0,
        scores=(0, 0),
        meeples=meeples,
        abbots=(True, True),
        abbot_at=(None, None),
        discarded=(),
        last_events=(),
        turn=0,
    )


# --- endgame_potential ---


def test_endgame_potential_city_majority_and_monastery() -> None:
    # D city extended north by F (2 tiles + 1 shield, incomplete); B monastery
    # south of D (1 + occupied 8-neighbours {(0,0),(1,0)} = 3); U road east.
    board, idx = _build(
        (Pos(0, 0), "D", Rotation.R0),
        (Pos(0, 1), "F", Rotation.R90),
        (Pos(1, 0), "U", Rotation.R90),
        (Pos(0, -1), "B", Rotation.R0),
    )
    city: NodeId = (Pos(0, 0), 0)
    monastery: NodeId = (Pos(0, -1), 0)
    idx = idx.with_meeple(city, Meeple(0, MeepleKind.MEEPLE, city))
    idx = idx.with_meeple(monastery, Meeple(1, MeepleKind.MEEPLE, monastery))
    state = _state(board, idx)
    assert endgame_potential(state, 0) == 3  # sole majority on the city
    assert endgame_potential(state, 1) == 3  # monastery goes to its piece's owner


def test_endgame_potential_contested_feature_counts_for_both() -> None:
    board, idx = _build((Pos(0, 0), "D", Rotation.R0), (Pos(0, 1), "F", Rotation.R90))
    city: NodeId = (Pos(0, 0), 0)
    idx = idx.with_meeple(city, Meeple(0, MeepleKind.MEEPLE, city))
    idx = idx.with_meeple(city, Meeple(1, MeepleKind.MEEPLE, (Pos(0, 1), 0)))
    state = _state(board, idx)
    assert endgame_potential(state, 0) == 3
    assert endgame_potential(state, 1) == 3


def test_endgame_potential_empty_and_completed_features_count_for_none() -> None:
    # E (city S after R180) atop D closes the city: completed features are
    # excluded even if a meeple is (hand-)left on them; unmeepled ones never count.
    board, idx = _build((Pos(0, 0), "D", Rotation.R0), (Pos(0, 1), "E", Rotation.R180))
    state = _state(board, idx)
    assert endgame_potential(state, 0) == 0
    assert endgame_potential(state, 1) == 0
    city: NodeId = (Pos(0, 0), 0)
    stale = _state(board, idx.with_meeple(city, Meeple(0, MeepleKind.MEEPLE, city)))
    assert endgame_potential(stale, 0) == 0


# --- GreedyAgent.choose ---


def _completion_state() -> GameState:
    """Player 0 has a meeple on D's 1-tile city; the drawn E completes it at
    (0,1) R180 for 4 points. All other placements score nothing now."""
    board, idx = _build((Pos(0, 0), "D", Rotation.R0))
    city: NodeId = (Pos(0, 0), 0)
    idx = idx.with_meeple(city, Meeple(0, MeepleKind.MEEPLE, city))
    return _state(board, idx, current_tile="E", meeples=(6, 7))


def test_greedy_picks_the_obvious_completion() -> None:
    state = _completion_state()
    completing = Move(Pos(0, 1), Rotation.R180, None)
    moves = legal_moves(state)
    assert completing in moves
    assert len(moves) > 1  # there are real alternatives to reject
    move, annot = GreedyAgent().choose(state, TurnContext(rng=random.Random(0)))
    assert move == completing
    assert annot is None  # annotate defaults to False


def test_greedy_moves_are_always_legal_and_deterministic() -> None:
    state = _completion_state()
    a = GreedyAgent().choose(state, TurnContext(rng=random.Random(3)))
    b = GreedyAgent().choose(state, TurnContext(rng=random.Random(3)))
    assert a == b
    assert a[0] in legal_moves(state)


def test_greedy_annotates_when_asked() -> None:
    state = _completion_state()
    move, annot = GreedyAgent().choose(state, TurnContext(rng=random.Random(0), annotate=True))
    assert annot is not None
    assert annot.sims == 0
    assert annot.think_ms >= 0
    assert -1.0 <= annot.value <= 1.0
    # Completion: +4 score, -1 own potential (weight 0.7), 8 pieces left (0.15).
    assert annot.value == pytest.approx(math.tanh((4 - 0.7 + 0.15 * 8) / 10))
    assert 1 <= len(annot.top) <= 8
    assert annot.top[0][0] == move  # best move leads the list
    priors = [prior for _, prior, _ in annot.top]
    assert all(0.0 < prior <= 1.0 for prior in priors)
    assert sum(priors) <= 1.0 + 1e-9
    assert all(visits == 0 for _, _, visits in annot.top)


def test_make_agent_greedy() -> None:
    agent = make_agent("greedy")
    assert isinstance(agent, GreedyAgent)
    assert agent.name == "greedy"


# --- the yardstick ---


def test_greedy_crushes_random() -> None:
    r = run_match(GreedyAgent(), RandomAgent(), n_games=20, base_seed=7)
    assert r.games == 20
    assert r.wins[0] >= 17  # greedy wins >=85% vs random
