"""Arena: head-to-head evaluation of two agents over a series of seeded games."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from carcassonne.game.agent import Agent
from carcassonne.game.match import GameResult, play_game


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Aggregate of run_match. Everything — wins, mean_scores, and each entry in
    results — is in (a, b) order regardless of who sat in seat 0; replay files
    are the only record of actual seating."""

    wins: tuple[int, int]
    draws: int
    games: int
    mean_scores: tuple[float, float]
    results: tuple[GameResult, ...]


def run_match(
    a: Agent,
    b: Agent,
    n_games: int,
    base_seed: int,
    replay_dir: Path | None = None,
    swap_seats: bool = True,
) -> MatchResult:
    """Play n_games between `a` and `b`; deterministic given base_seed.

    Game i uses seed base_seed + i. With swap_seats (default), `b` takes seat 0
    on odd games to cancel first-move advantage; each such result is flipped
    back so the returned MatchResult stays in (a, b) order.
    """
    wins = [0, 0]
    draws = 0
    totals = [0.0, 0.0]
    results: list[GameResult] = []
    for i in range(n_games):
        swapped = swap_seats and i % 2 == 1
        seats = (b, a) if swapped else (a, b)
        result = play_game(seats, base_seed + i, replay_dir=replay_dir)
        if swapped:
            result = _flipped(result)
        results.append(result)
        if result.winner is None:
            draws += 1
        else:
            wins[result.winner] += 1
        totals[0] += result.final_scores[0]
        totals[1] += result.final_scores[1]
    n = max(n_games, 1)  # 0 games -> means of 0.0, not ZeroDivisionError
    return MatchResult(
        wins=(wins[0], wins[1]),
        draws=draws,
        games=n_games,
        mean_scores=(totals[0] / n, totals[1] / n),
        results=tuple(results),
    )


def _flipped(r: GameResult) -> GameResult:
    return GameResult(
        final_scores=(r.final_scores[1], r.final_scores[0]),
        winner=None if r.winner is None else 1 - r.winner,
        turns=r.turns,
        replay_path=r.replay_path,
    )
