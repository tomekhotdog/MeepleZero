"""play_game generalised to N seats; GameResult carries a winners tie set."""

from __future__ import annotations

from pathlib import Path

from carcassonne.agents import GreedyAgent, RandomAgent
from carcassonne.core import GameConfig
from carcassonne.game.match import play_game
from carcassonne.game.replay import load_replay, replay_states, winners_of


def test_three_player_game_runs_and_records(tmp_path: Path) -> None:
    result = play_game(
        (GreedyAgent(), RandomAgent(), RandomAgent()),
        seed=9,
        replay_dir=tmp_path,
        config=GameConfig(num_players=3),
    )
    assert len(result.final_scores) == 3
    assert result.winners == winners_of(result.final_scores)
    replay = load_replay(result.replay_path)
    assert replay.header.agents == ("greedy", "random", "random")
    list(replay_states(replay))


def test_agent_count_must_match_config() -> None:
    import pytest

    from carcassonne.core import RulesError

    with pytest.raises(RulesError):
        play_game((RandomAgent(), RandomAgent()), seed=1, config=GameConfig(num_players=3))
