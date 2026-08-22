"""RandomAgent and match runner: legality, determinism, verified replay output."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from carcassonne.agents import RandomAgent, TurnContext, make_agent
from carcassonne.core import apply, final_scores, is_terminal, legal_moves, new_game
from carcassonne.game.match import play_game
from carcassonne.game.replay import load_replay, replay_states

SEED = 11
STARTED_AT = "2026-08-22T00:00:00+00:00"


# --- RandomAgent ---


def test_random_agent_returns_legal_move_and_no_annot() -> None:
    state = new_game(SEED)
    agent = RandomAgent()
    ctx = TurnContext(rng=random.Random(0))
    for _ in range(5):
        move, annot = agent.choose(state, ctx)
        assert annot is None
        assert move in legal_moves(state)
        state = apply(state, move)


def test_random_agent_deterministic_given_same_seeded_rng() -> None:
    state = new_game(SEED)
    a = RandomAgent().choose(state, TurnContext(rng=random.Random(7)))
    b = RandomAgent().choose(state, TurnContext(rng=random.Random(7)))
    assert a == b


# --- registry ---


def test_make_agent_random() -> None:
    agent = make_agent("random")
    assert isinstance(agent, RandomAgent)
    assert agent.name == "random"


def test_make_agent_unknown_spec_lists_known_specs() -> None:
    with pytest.raises(ValueError, match="random"):
        make_agent("skynet")


# --- play_game ---


def test_play_game_completes_and_result_is_consistent() -> None:
    result = play_game((RandomAgent(), RandomAgent()), seed=SEED)
    assert result.replay_path is None
    assert result.turns > 0
    s0, s1 = result.final_scores
    if s0 == s1:
        assert result.winner is None
    else:
        assert result.winner == (0 if s0 > s1 else 1)


def test_play_game_deterministic_for_seed() -> None:
    r1 = play_game((RandomAgent(), RandomAgent()), seed=SEED)
    r2 = play_game((RandomAgent(), RandomAgent()), seed=SEED)
    assert (r1.final_scores, r1.winner, r1.turns) == (r2.final_scores, r2.winner, r2.turns)


def test_play_game_writes_fully_verifiable_replay(tmp_path: Path) -> None:
    result = play_game(
        (RandomAgent(), RandomAgent()),
        seed=SEED,
        replay_dir=tmp_path,
        started_at=STARTED_AT,
    )
    assert result.replay_path is not None
    assert result.replay_path.parent == tmp_path

    replay = load_replay(result.replay_path)
    assert replay.header.seed == SEED
    assert replay.header.agents == ("random", "random")
    assert replay.header.started_at == STARTED_AT
    assert replay.final_scores == result.final_scores
    assert replay.winner == result.winner
    assert len(replay.moves) == result.turns

    states = list(replay_states(replay))  # re-derives and verifies every move
    assert len(states) == result.turns + 1
    assert is_terminal(states[-1])
    assert final_scores(states[-1]) == result.final_scores


def test_replay_filenames_are_sanitised_and_unique(tmp_path: Path) -> None:
    r1 = play_game(
        (RandomAgent(), RandomAgent()), seed=SEED, replay_dir=tmp_path, started_at=STARTED_AT
    )
    r2 = play_game(
        (RandomAgent(), RandomAgent()), seed=SEED, replay_dir=tmp_path, started_at=STARTED_AT
    )
    assert r1.replay_path is not None and r2.replay_path is not None
    assert r1.replay_path != r2.replay_path  # same metadata must not clobber
    for path in (r1.replay_path, r2.replay_path):
        assert path.suffix == ".jsonl"
        assert ":" not in path.name and "+" not in path.name
        load_replay(path)  # both files are complete and loadable
