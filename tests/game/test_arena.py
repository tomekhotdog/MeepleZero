"""Arena: seat-swapping with results mapped back to (a, b) order; determinism."""

from __future__ import annotations

from pathlib import Path

from carcassonne.agents import GreedyAgent, RandomAgent
from carcassonne.game.arena import run_match
from carcassonne.game.match import play_game
from carcassonne.game.replay import load_replay


def test_run_match_swaps_seats_and_maps_results_back(tmp_path: Path) -> None:
    r = run_match(GreedyAgent(), RandomAgent(), n_games=2, base_seed=3, replay_dir=tmp_path)
    assert r.games == 2 and len(r.results) == 2
    assert r.wins[0] + r.wins[1] + r.draws == 2

    # Replays record actual seating: game 0 (a, b); game 1 swapped to (b, a).
    headers = {h.seed: h.agents for h in (load_replay(p).header for p in tmp_path.glob("*.jsonl"))}
    assert headers == {3: ("greedy", "random"), 4: ("random", "greedy")}

    # The swapped game's result is mapped back to (a, b) order.
    raw = play_game((RandomAgent(), GreedyAgent()), seed=4)
    assert r.results[1].final_scores == (raw.final_scores[1], raw.final_scores[0])
    winner = None if raw.winner is None else 1 - raw.winner
    assert r.results[1].winner == winner


def test_run_match_no_swap_is_deterministic_with_correct_aggregates() -> None:
    r1 = run_match(RandomAgent(), RandomAgent(), n_games=3, base_seed=5, swap_seats=False)
    r2 = run_match(RandomAgent(), RandomAgent(), n_games=3, base_seed=5, swap_seats=False)
    assert r1 == r2

    # Without swapping, game i is exactly play_game((a, b), base_seed + i).
    for i, result in enumerate(r1.results):
        direct = play_game((RandomAgent(), RandomAgent()), seed=5 + i)
        assert result == direct

    assert r1.wins[0] + r1.wins[1] + r1.draws == r1.games == 3
    for side in (0, 1):
        mean = sum(g.final_scores[side] for g in r1.results) / 3
        assert r1.mean_scores[side] == mean
