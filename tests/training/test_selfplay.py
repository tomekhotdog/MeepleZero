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
from carcassonne.nn.actions import ACTION_SPACE, encode_move
from carcassonne.nn.model import CarcassonneNet
from carcassonne.training.buffer import ReplayBuffer
from carcassonne.training.run import TrainingRun
from carcassonne.training.selfplay import play_selfplay_game

DEVICE = torch.device("cpu")


def _tiny_net() -> CarcassonneNet:
    torch.manual_seed(0)
    return CarcassonneNet(channels=8, n_blocks=1)


def test_one_tiny_selfplay_game_populates_buffer_and_replay(tmp_path: Path) -> None:
    run = TrainingRun.create(tmp_path / "run", config={"note": "test"})
    buf = ReplayBuffer(run.buffer_path)

    game_id = play_selfplay_game(
        net=_tiny_net(),
        device=DEVICE,
        seed=1,
        run=run,
        mcts_config=MctsConfig(sims=4),
        game_id="game0",
        started_at="2026-08-22T00:00:00+00:00",
        order_key=0,
        buffer=buf,
    )
    assert game_id == "game0"

    # Replay was written and verifies end-to-end through the rules core.
    replay_path = run.replays_dir / "game0.jsonl"
    replay = load_replay(replay_path)
    states = list(replay_states(replay))  # raises if any record desyncs
    n_moves = len(replay.moves)
    assert n_moves > 0
    assert len(states) == n_moves + 1  # a state before each move, then terminal

    # Buffer holds exactly one example per move.
    assert buf.n_games() == 1
    assert len(buf) == n_moves

    # Sample and check target invariants + the value-sign rule against the replay.
    movers = {rec.n: rec.player for rec in replay.moves}
    winner = replay.winner
    batch = buf.sample(30, random.Random(0))
    seen: set[int] = set()
    for ex in batch:
        assert sum(ex.policy.values()) == pytest.approx(1.0)
        for action_id in ex.policy:
            assert 0 <= action_id < ACTION_SPACE
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
    # every policy move encodes into the action space (self-play stores these ids).
    assert all(0 <= encode_move(state, m) < ACTION_SPACE for m in policy)
