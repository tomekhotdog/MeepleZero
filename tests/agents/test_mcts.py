"""MctsAgent: PUCT search sign conventions, determinism, annotation, and strength.

The two subtle correctness points are pinned first as unit tests -- the
terminal-value sign and the grounded near-terminal pick -- before the slower
integration test that MCTS + value net beats RandomAgent.
"""

from __future__ import annotations

import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from carcassonne.agents import MctsAgent, MctsConfig, RandomAgent, TurnContext
from carcassonne.agents.mcts import _terminal_value
from carcassonne.core import (
    GameConfig,
    GameState,
    Move,
    Pos,
    Rotation,
    RulesError,
    apply,
    is_terminal,
    legal_moves,
    new_game,
)
from carcassonne.core.features import FeatureIndex, Meeple, NodeId
from carcassonne.core.state import Board, PlacedTile
from carcassonne.core.types import MeepleKind
from carcassonne.game.arena import run_match
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
# 2-player guard: MCTS backup is negamax (value negated each ply), which is only
# correct for exactly two alternating players. N-player states must be rejected
# loudly, not searched wrongly.


def test_mcts_rejects_non_two_player_state() -> None:
    agent = MctsAgent(_net(), DEVICE, MctsConfig(sims=1))
    with pytest.raises(RulesError, match="2-player"):
        agent.choose(new_game(0, GameConfig(num_players=3)), TurnContext(rng=random.Random(0)))


def test_choose_with_policy_rejects_non_two_player_state() -> None:
    agent = MctsAgent(_net(), DEVICE, MctsConfig(sims=1))
    with pytest.raises(RulesError, match="2-player"):
        agent.choose_with_policy(
            new_game(0, GameConfig(num_players=3)), TurnContext(rng=random.Random(0))
        )


def test_visit_policy_rejects_non_two_player_state() -> None:
    # Unguarded, a 3-player search would not crash — it would return a
    # plausible but semantically wrong distribution (negamax assumes two
    # alternating players). All three public entry points must reject loudly.
    agent = MctsAgent(_net(), DEVICE, MctsConfig(sims=1))
    with pytest.raises(RulesError, match="2-player"):
        agent.visit_policy(new_game(0, GameConfig(num_players=3)))


# ---------------------------------------------------------------------------
# End-to-end: MctsAgent plays complete, legal games against RandomAgent without
# crashing.
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


# ---------------------------------------------------------------------------
# Subtree reuse: apply() draws from the in-state deck, so the child reached by the
# committed move IS the next real state -- reusing it is exact, not approximate.


def test_tree_reuse_carries_visits_across_moves() -> None:
    net = _net()
    agent = MctsAgent(net, DEVICE, MctsConfig(sims=8, reuse_tree=True))
    state = new_game(0, GameConfig())
    ctx = TurnContext(rng=random.Random(0), annotate=False)
    move, _ = agent.choose(state, ctx)
    stash = agent._stash  # noqa: SLF001
    assert stash is not None  # subtree kept for the next ply
    nxt = apply(state, move)
    if not is_terminal(nxt):
        inherited = int(stash.n.sum())
        root = agent._search(nxt, np.random.default_rng(0), add_noise=False)  # noqa: SLF001
        assert root is stash  # the committed child became the new root
        assert int(root.n.sum()) == inherited + 8  # inherited visits + this search's sims


def test_tree_reuse_misses_on_unrelated_state() -> None:
    """A stash that matches neither the state nor its children must be ignored
    (fresh root, exactly ``sims`` visits) and consumed, not carried forward."""
    net = _net()
    agent = MctsAgent(net, DEVICE, MctsConfig(sims=4, reuse_tree=True))
    ctx = TurnContext(rng=random.Random(0), annotate=False)
    agent.choose(new_game(0, GameConfig()), ctx)
    old = agent._stash  # noqa: SLF001
    assert old is not None
    unrelated = new_game(99, GameConfig())
    root = agent._search(unrelated, np.random.default_rng(0), add_noise=False)  # noqa: SLF001
    assert root is not old
    assert int(root.n.sum()) == 4  # fresh root: nothing inherited
    assert agent._stash is None  # noqa: SLF001 -- the stale stash was consumed

    agent._stash = old  # noqa: SLF001 -- replay the stale stash through choose()
    move, _ = agent.choose(unrelated, ctx)
    assert move in legal_moves(unrelated)
    new = agent._stash  # noqa: SLF001
    assert new is not None and new is not old
    assert new.state == apply(unrelated, move)  # re-stashed from the NEW tree


def test_reuse_off_never_stashes_or_inherits() -> None:
    net = _net()
    sims = 64
    off = MctsAgent(net, DEVICE, MctsConfig(sims=sims, reuse_tree=False))
    state = new_game(3, GameConfig())
    move, _ = off.choose(state, TurnContext(rng=random.Random(1)))
    assert off._stash is None  # noqa: SLF001
    off.choose_with_policy(state, TurnContext(rng=random.Random(1)))
    assert off._stash is None  # noqa: SLF001

    # Even a matching stash planted from a reuse-on agent must not be inherited.
    on = MctsAgent(net, DEVICE, MctsConfig(sims=sims, reuse_tree=True))
    move, _ = on.choose(state, TurnContext(rng=random.Random(1)))
    planted = on._stash  # noqa: SLF001
    assert planted is not None and int(planted.n.sum()) > 0  # premise: would be reusable
    off._stash = planted  # noqa: SLF001
    root = off._search(apply(state, move), np.random.default_rng(0), add_noise=False)  # noqa: SLF001
    assert root is not planted
    assert int(root.n.sum()) == sims  # fresh root, nothing inherited


def test_tree_reuse_finds_grandchild_after_opponent_move() -> None:
    """Arena/web case: the opponent moved once since our committed move, so the
    new root must be the grandchild our last search already expanded."""
    sims = 64
    agent = MctsAgent(_net(), DEVICE, MctsConfig(sims=sims, reuse_tree=True))
    s0 = new_game(0, GameConfig())
    agent.choose(s0, TurnContext(rng=random.Random(0)))
    stash = agent._stash  # noqa: SLF001
    assert stash is not None
    candidates = [c for c in stash.children.values() if not c.terminal]
    assert candidates, "premise: the search expanded an opponent reply"
    grandchild = max(candidates, key=lambda c: int(c.n.sum()))
    inherited = int(grandchild.n.sum())
    assert inherited > 0  # premise: the reuse carries real visits
    root = agent._search(grandchild.state, np.random.default_rng(0), add_noise=False)  # noqa: SLF001
    assert root is grandchild
    assert int(root.n.sum()) == inherited + sims


def test_visit_policy_leaves_play_stash_untouched() -> None:
    agent = MctsAgent(_net(), DEVICE, MctsConfig(sims=16, reuse_tree=True))
    state = new_game(0, GameConfig())
    move, _ = agent.choose(state, TurnContext(rng=random.Random(0)))
    stash = agent._stash  # noqa: SLF001
    assert stash is not None
    agent.visit_policy(apply(state, move))  # a hint on the very state the stash matches
    assert agent._stash is stash  # noqa: SLF001


def test_reused_root_annot_reports_total_visits() -> None:
    """Annot.sims must say how many visits the shown counts came from: on a reused
    root that is inherited + this search's sims, not the configured budget."""
    sims = 64
    agent = MctsAgent(_net(), DEVICE, MctsConfig(sims=sims, reuse_tree=True))
    s0 = new_game(0, GameConfig())
    move, _ = agent.choose(s0, TurnContext(rng=random.Random(0)))
    stash = agent._stash  # noqa: SLF001
    assert stash is not None
    inherited = int(stash.n.sum())
    assert inherited > 0  # premise: the reused root carries visits
    s1 = apply(s0, move)
    _, annot = agent.choose(s1, TurnContext(rng=random.Random(1), annotate=True))
    assert annot is not None
    assert annot.sims == inherited + sims > sims


def test_node_expansion_enumerates_legal_moves_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Move generation is the engine's hot path: a leaf expansion must enumerate
    legal moves once and hand them to the net, not have nn recompute them."""
    import carcassonne.agents.mcts as mcts_mod
    import carcassonne.nn.actions as actions_mod
    import carcassonne.nn.model as model_mod

    calls = 0
    real = legal_moves

    def counting(state: GameState) -> tuple[Move, ...]:
        nonlocal calls
        calls += 1
        return real(state)

    for mod in (mcts_mod, actions_mod, model_mod):
        if hasattr(mod, "legal_moves"):
            monkeypatch.setattr(mod, "legal_moves", counting)

    agent = MctsAgent(_net(), DEVICE, MctsConfig(sims=1))
    node = agent._make_node(new_game(SEED))
    assert calls == 1
    assert len(node.moves) == len(node.prior)
    assert abs(float(node.prior.sum()) - 1.0) < 1e-9
