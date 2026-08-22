"""Arena gating for the training loop -- the AlphaGo-Zero promotion decision.

Two thin wrappers over :func:`carcassonne.game.arena.run_match` (which owns the
seat-swapping and determinism; we do **not** re-implement it here):

* :func:`gate` -- candidate vs current-best over ``n_games``; the candidate is
  promoted only if it wins more than 55% of *all* games. Draws count as
  non-wins, so ``win_rate = wins / games`` (not ``wins / (wins + losses)``): a
  candidate that mostly draws has not proven itself the stronger net.
* :func:`yardstick` -- candidate vs :class:`GreedyAgent`, an *absolute* progress
  signal independent of the (moving) best net.

Both agents search noise-free (``self_play=False``): gating measures strength,
not exploration, so no Dirichlet noise and always the visit-argmax move.
"""

from __future__ import annotations

import torch

from carcassonne.agents.greedy import GreedyAgent
from carcassonne.agents.mcts import MctsAgent, MctsConfig
from carcassonne.game.arena import run_match
from carcassonne.nn.model import CarcassonneNet

PROMOTE_THRESHOLD = 0.55


def gate(
    candidate_net: CarcassonneNet,
    best_net: CarcassonneNet,
    device: torch.device,
    mcts_config: MctsConfig,
    n_games: int,
    base_seed: int,
) -> tuple[float, bool]:
    """Play ``candidate`` vs ``best`` (both :class:`MctsAgent`, ``self_play=False``)
    over ``n_games`` seat-swapped games. Return ``(win_rate, promote)`` where
    ``win_rate = candidate_wins / n_games`` (draws are non-wins) and
    ``promote = win_rate > 0.55``.
    """
    candidate = MctsAgent(candidate_net, device, mcts_config, self_play=False, name="candidate")
    best = MctsAgent(best_net, device, mcts_config, self_play=False, name="best")
    result = run_match(candidate, best, n_games, base_seed, swap_seats=True)
    win_rate = result.wins[0] / result.games if result.games else 0.0
    return win_rate, win_rate > PROMOTE_THRESHOLD


def yardstick(
    candidate_net: CarcassonneNet,
    device: torch.device,
    mcts_config: MctsConfig,
    n_games: int,
    base_seed: int,
) -> float:
    """Candidate (:class:`MctsAgent`, ``self_play=False``) vs :class:`GreedyAgent`
    over ``n_games`` seat-swapped games. Return the candidate's win rate in
    ``[0, 1]`` -- absolute progress against a fixed hand-written opponent."""
    candidate = MctsAgent(candidate_net, device, mcts_config, self_play=False, name="candidate")
    greedy = GreedyAgent()
    result = run_match(candidate, greedy, n_games, base_seed, swap_seats=True)
    return result.wins[0] / result.games if result.games else 0.0
