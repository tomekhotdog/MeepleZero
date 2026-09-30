"""Self-play worker: ONE tiny real game (the only real-self-play test in the
suite -- kept to a single game at sims=4 with a minimal net per the perf guard)."""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from carcassonne.agents.mcts import MctsAgent, MctsConfig
from carcassonne.core import legal_moves, new_game
from carcassonne.game.agent import TurnContext
from carcassonne.game.replay import load_replay, replay_states
from carcassonne.nn.actions import key_move, move_key
from carcassonne.nn.model import CarcassonneNet
from carcassonne.training.buffer import ReplayBuffer
from carcassonne.training.run import TrainingRun
from carcassonne.training.selfplay import _winner, play_selfplay_game

DEVICE = torch.device("cpu")


def _tiny_net() -> CarcassonneNet:
    torch.manual_seed(0)
    return CarcassonneNet(channels=8, n_blocks=1)


def test_one_tiny_selfplay_game_returns_examples_and_replay(tmp_path: Path) -> None:
    run = TrainingRun.create(tmp_path / "run", config={"note": "test"})

    game_id, replay_path_str, examples = play_selfplay_game(
        net=_tiny_net(),
        device=DEVICE,
        seed=1,
        run=run,
        mcts_config=MctsConfig(sims=4),
        game_id="game0",
        started_at="2026-08-22T00:00:00+00:00",
    )
    assert game_id == "game0"

    # Replay was written where the worker says it was, and verifies end-to-end
    # through the rules core (callers use the returned path, never re-derive).
    replay_path = Path(replay_path_str)
    assert replay_path.parent == run.replays_dir
    replay = load_replay(replay_path)
    states = list(replay_states(replay))  # raises if any record desyncs
    n_moves = len(replay.moves)
    assert n_moves > 0
    assert len(states) == n_moves + 1  # a state before each move, then terminal

    # One example per move; every policy key is a world-coordinate 4-tuple.
    assert len(examples) == n_moves
    assert all(
        isinstance(k, tuple) and len(k) == 4 for ex in examples for k in ex.policy
    )
    # Keys are the engine's own coordinates: each decodes to a legal move of the
    # state it was recorded at.
    for ex in examples:
        legal_keys = {move_key(m) for m in legal_moves(states[ex.move_n])}
        assert set(ex.policy) <= legal_keys

    # Buffer insertion is the caller's job now; add and check round-trip.
    buf = ReplayBuffer(run.buffer_path)
    buf.add_game(game_id, str(replay_path), examples, order_key=0)
    assert buf.n_games() == 1
    assert len(buf) == n_moves

    # Sample and check target invariants + the value-sign rule against the replay.
    movers = {rec.n: rec.player for rec in replay.moves}
    winner = replay.winners[0] if len(replay.winners) == 1 else None
    batch = buf.sample(30, random.Random(0))
    seen: set[int] = set()
    for ex in batch:
        assert sum(ex.policy.values()) == pytest.approx(1.0)
        expected = 0.0 if winner is None else (1.0 if movers[ex.move_n] == winner else -1.0)
        assert ex.value == expected
        seen.add(ex.move_n)
    assert len(seen) >= 2  # value-sign checked on several distinct positions
    buf.close()


def test_choose_with_policy_single_search_matches_chosen_move() -> None:
    agent = MctsAgent(_tiny_net(), DEVICE, MctsConfig(sims=4), self_play=False)
    state = new_game(0)
    move, annot, policy = agent.choose_with_policy(state, TurnContext(rng=random.Random(0)))

    legal = set(legal_moves(state))
    assert set(policy) == legal  # distribution over exactly the legal moves
    assert sum(policy.values()) == pytest.approx(1.0)
    # eval-mode pick is the visit-argmax, so it must be the policy's argmax.
    assert move == max(policy, key=lambda m: policy[m])
    assert annot.sims == 4
    # every policy move round-trips through the world-key codec (self-play stores keys).
    assert all(key_move(move_key(m)) == m for m in policy)


def test_winner_refuses_non_two_player_scores() -> None:
    """The +1/-1/0 outcome labels are 2-player negamax targets; a 3-player score
    tuple must fail loud here rather than silently crown seat 0 or 1."""
    assert _winner((5, 3)) == 0
    assert _winner((3, 5)) == 1
    assert _winner((4, 4)) is None
    with pytest.raises(ValueError, match="2-player"):
        _winner((1, 9, 2))
