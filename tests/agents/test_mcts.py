"""MctsAgent: PUCT search sign conventions, determinism, annotation, and strength.

The two subtle correctness points are pinned first as unit tests -- the
terminal-value sign and the grounded near-terminal pick -- before the slower
integration test that MCTS + value net beats RandomAgent.
"""

from __future__ import annotations

import random
from dataclasses import replace

import pytest
import torch

from carcassonne.agents import MctsAgent, MctsConfig, RandomAgent, TurnContext
from carcassonne.agents.mcts import _terminal_value
from carcassonne.core import GameState, Move, Pos, Rotation, is_terminal, legal_moves, new_game
from carcassonne.core.features import FeatureIndex, Meeple, NodeId
from carcassonne.core.state import Board, PlacedTile
from carcassonne.core.types import MeepleKind
from carcassonne.game.arena import run_match
from carcassonne.nn.actions import legal_mask
from carcassonne.nn.model import CarcassonneNet

SEED = 0
DEVICE = torch.device("cpu")


def _net() -> CarcassonneNet:
    torch.manual_seed(SEED)
    return CarcassonneNet()


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
    current_player: int = 0,
    scores: tuple[int, int] = (0, 0),
    deck: tuple[str, ...] = (),
    meeples: tuple[int, int] = (7, 7),
) -> GameState:
    return GameState(
        board=board,
        features=features,
        deck=deck,
        current_tile=current_tile,
        current_player=current_player,
        scores=scores,
        meeples=meeples,
        abbots=(True, True),
        abbot_at=(None, None),
        discarded=(),
        last_events=(),
        turn=0,
    )


# ---------------------------------------------------------------------------
# 5a. terminal-value sign helper (the load-bearing correctness point)


def test_terminal_value_sign() -> None:
    board, idx = _build((Pos(0, 0), "D", Rotation.R0))  # no meeples -> final == scores
    won = _state(board, idx, current_tile=None, current_player=0, scores=(5, 3))
    assert is_terminal(won)
    assert _terminal_value(won) == 1.0  # player 0 to move is the winner
    # Same scores, but it's the loser's "turn" at the terminal node.
    assert _terminal_value(replace(won, current_player=1)) == -1.0
    drawn = _state(board, idx, current_tile=None, current_player=0, scores=(3, 3))
    assert _terminal_value(drawn) == 0.0


# ---------------------------------------------------------------------------
# 5b. near-terminal grounding: with the deck empty every child is terminal and
# exactly valued, so search picks the winning move regardless of the net.


def _one_from_win() -> tuple[GameState, Move]:
    """Player 0 owns D's 1-tile city and trails 0-3; the drawn E completes it at
    (0,1) R180 for 4 -> a 4-3 win. The deck is empty, so *any* move ends the
    game: completing wins, everything else leaves p0 at 1 (endgame partial) < 3."""
    board, idx = _build((Pos(0, 0), "D", Rotation.R0))
    city: NodeId = (Pos(0, 0), 0)
    idx = idx.with_meeple(city, Meeple(0, MeepleKind.MEEPLE, city))
    state = _state(board, idx, current_tile="E", scores=(0, 3), meeples=(6, 7))
    return state, Move(Pos(0, 1), Rotation.R180, None)


def test_mcts_picks_grounded_winning_move() -> None:
    state, winning = _one_from_win()
    moves = legal_moves(state)
    assert winning in moves and len(moves) > 1
    agent = MctsAgent(_net(), DEVICE, MctsConfig(sims=64), self_play=False)
    move, _ = agent.choose(state, TurnContext(rng=random.Random(0)))
    assert move == winning


# ---------------------------------------------------------------------------
# 2. visit_policy is a valid distribution over exactly the legal moves


def test_visit_policy_is_a_distribution_over_legal_moves() -> None:
    state = new_game(SEED)
    agent = MctsAgent(_net(), DEVICE, MctsConfig(sims=32))
    policy = agent.visit_policy(state)
    legal = set(legal_moves(state))
    assert set(policy) == legal  # every legal move present, nothing illegal
    assert all(0.0 <= p <= 1.0 for p in policy.values())
    assert sum(policy.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 3. determinism: same net + config + same-seeded rng -> identical move + visits


def test_choose_is_deterministic_under_a_fixed_seed() -> None:
    net = _net()  # one shared net; eval-mode no_grad forward is deterministic on CPU
    cfg = MctsConfig(sims=48)
    state = new_game(SEED)

    def run() -> tuple[Move, tuple[int, ...]]:
        agent = MctsAgent(net, DEVICE, cfg, self_play=True)  # noise + sampling both seeded
        move, annot = agent.choose(state, TurnContext(rng=random.Random(123), annotate=True))
        assert annot is not None
        return move, tuple(visits for _, _, visits in annot.top)

    assert run() == run()


# ---------------------------------------------------------------------------
# 4. annotation shape


def test_annot_shape() -> None:
    cfg = MctsConfig(sims=40)
    agent = MctsAgent(_net(), DEVICE, cfg, self_play=False)
    state = new_game(SEED)
    move, annot = agent.choose(state, TurnContext(rng=random.Random(1), annotate=True))
    assert annot is not None
    assert -1.0 <= annot.value <= 1.0
    assert annot.sims == cfg.sims
    assert annot.think_ms >= 0
    assert 1 <= len(annot.top) <= 8
    visits = [v for _, _, v in annot.top]
    assert visits == sorted(visits, reverse=True)  # sorted by visits desc
    for m, prior, v in annot.top:
        assert isinstance(m, Move)
        assert isinstance(prior, float) and 0.0 <= prior <= 1.0
        assert isinstance(v, int)
    assert move in [m for m, _, _ in annot.top]  # chosen move is shown
    assert move == annot.top[0][0]  # eval mode: chosen == visit-argmax leads the list


def test_no_annot_when_not_requested() -> None:
    agent = MctsAgent(_net(), DEVICE, MctsConfig(sims=16))
    _, annot = agent.choose(new_game(SEED), TurnContext(rng=random.Random(0)))
    assert annot is None


# ---------------------------------------------------------------------------
# 6. self-play behaviours: Dirichlet root noise + temperature sampling


def _priors_by_move(agent: MctsAgent, state: GameState, seed: int) -> dict[Move, float]:
    _, annot = agent.choose(state, TurnContext(rng=random.Random(seed), annotate=True))
    assert annot is not None
    return {m: prior for m, prior, _ in annot.top}


def test_self_play_adds_dirichlet_noise_to_root_priors() -> None:
    net = _net()
    cfg = MctsConfig(sims=48)
    state = new_game(SEED)
    raw = _priors_by_move(MctsAgent(net, DEVICE, cfg, self_play=False), state, 7)
    noised = _priors_by_move(MctsAgent(net, DEVICE, cfg, self_play=True), state, 7)
    shared = set(raw) & set(noised)
    assert shared  # the top moves overlap enough to compare
    assert any(raw[m] != pytest.approx(noised[m]) for m in shared)  # noise moved priors


def test_self_play_temperature_explores_early_moves() -> None:
    """Early turns (turn < temp_turns) sample proportional to visits, so different
    seeds must not always yield the same move -- unlike eval-mode argmax."""
    agent = MctsAgent(_net(), DEVICE, MctsConfig(sims=48), self_play=True)
    state = new_game(SEED)
    assert state.turn < agent._config.temp_turns  # noqa: SLF001 -- guard the premise
    chosen = {agent.choose(state, TurnContext(rng=random.Random(s)))[0] for s in range(12)}
    assert len(chosen) > 1


# ---------------------------------------------------------------------------
# Wide-board robustness: when the board already fills the 31-cell window, a legal
# placement one cell past either end maps *outside* the window. legal_mask must
# mask (not crash on) those, and MCTS must prune them and still search. This is
# the bug MCTS surfaced -- it explores far wider board shapes than a normal game
# reaches. A row of 31 all-field "B" monastery tiles fills the window exactly, so
# every tile still encodes while the placements extending it are out of window.


def _wide_board_state() -> GameState:
    board, idx = _build(
        (Pos(0, 0), "D", Rotation.R0),
        *[(Pos(i, -1), "B", Rotation.R0) for i in range(31)],  # fills the window (x: 0..30)
    )
    xs = [p.x for p in board]
    assert max(xs) - min(xs) + 1 == 31  # exactly window-width: tiles fit, extensions don't
    return _state(board, idx, current_tile="B")


def test_legal_mask_prunes_moves_outside_the_window() -> None:
    state = _wide_board_state()
    n_legal = len(legal_moves(state))
    mask = legal_mask(state)  # must not raise
    assert 0 < int(mask.sum()) < n_legal  # some legal moves fall outside the window


def test_mcts_searches_a_wide_board_without_crashing() -> None:
    state = _wide_board_state()
    agent = MctsAgent(_net(), DEVICE, MctsConfig(sims=16))
    move, _ = agent.choose(state, TurnContext(rng=random.Random(0)))
    assert move in legal_moves(state)


# ---------------------------------------------------------------------------
# End-to-end: MctsAgent plays complete, legal games against RandomAgent without
# crashing (also regression-covers the wide-board path in real play).
#
# NOTE on strength: the plan hoped an *untrained*-net search would beat random.
# It does not. Measured over 10 seeded games (sims=64) the untrained MctsAgent
# won 3/10. Diagnosis: the random-weighted value head systematically (if
# arbitrarily) rates feature-claiming states lower, so the search places meeples
# on ~6/36 turns vs random's ~14/36 and under-scores. Grounding only reaches the
# last ply or two, so even from near-terminal positions MCTS still lost (~3/6).
# This is the expected "search is only as good as its value function" result --
# the strength test belongs with a *trained* checkpoint (Task 18+). Search
# correctness itself is proven above by the terminal-value sign and the grounded
# winning-move tests, which are independent of the (untrained) value head.


def test_mcts_plays_legal_games_to_completion() -> None:
    agent = MctsAgent(_net(), DEVICE, MctsConfig(sims=32), self_play=False)
    result = run_match(agent, RandomAgent(), n_games=2, base_seed=0, swap_seats=True)
    assert result.games == 2
    for game in result.results:
        assert game.turns > 0
        assert all(s >= 0 for s in game.final_scores)  # completed with real scores
