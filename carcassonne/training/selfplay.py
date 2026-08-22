"""Self-play game generation: one :class:`MctsAgent` plays both sides of a game,
producing a replay on disk and (policy, value) training targets in the buffer.

One search per move (via :meth:`MctsAgent.choose_with_policy`): the visit
distribution is both recorded as the replay annotation and stored as the sparse
policy target ``{action_id: prob}``. After the game the winner is read from the
final scores and every stored move gets a value target of +1/-1/0 from *its
mover's* perspective (mover == winner -> +1). This matches the value head and
MCTS terminal-value convention, so the learner and search agree on sign.

D4 augmentation is deliberately NOT applied here -- raw targets are stored and
augmentation happens at sampling time (Task 19). No multiprocessing here either
(Task 20 orchestrates the pool); this is the single-game worker.
"""

from __future__ import annotations

import random

import torch

from carcassonne.agents.mcts import MctsAgent, MctsConfig
from carcassonne.core import GameConfig, apply, final_scores, is_terminal, new_game
from carcassonne.game.agent import TurnContext
from carcassonne.game.replay import REPLAY_VERSION, ReplayHeader, ReplayWriter
from carcassonne.nn.actions import encode_move
from carcassonne.nn.model import CarcassonneNet
from carcassonne.training.buffer import Example, ReplayBuffer
from carcassonne.training.run import TrainingRun


def play_selfplay_game(
    net: CarcassonneNet,
    device: torch.device,
    seed: int,
    run: TrainingRun,
    mcts_config: MctsConfig,
    game_id: str,
    started_at: str,
    order_key: int,
    buffer: ReplayBuffer | None = None,
) -> str:
    """Play one MctsAgent-vs-itself game; write its replay into ``run.replays_dir``
    and, if ``buffer`` is given, insert its training examples. Returns ``game_id``."""
    config = GameConfig()
    agent = MctsAgent(net, device, mcts_config, self_play=True)
    ctx = TurnContext(rng=random.Random(seed), annotate=True)
    state = new_game(seed, config)

    replay_path = run.replays_dir / f"{game_id}.jsonl"
    header = ReplayHeader(
        v=REPLAY_VERSION,
        seed=seed,
        config=config,
        agents=(agent.name, agent.name),
        checkpoint=None,
        started_at=started_at,
    )

    # (move_n, sparse policy target, mover) captured per move; value filled after.
    pending: list[tuple[int, dict[int, float], int]] = []
    n = 0
    with ReplayWriter(replay_path, header) as writer:
        while not is_terminal(state):
            player, tile = state.current_player, state.current_tile
            assert tile is not None  # non-terminal <=> a tile is drawn
            move, annot, policy = agent.choose_with_policy(state, ctx)
            sparse = {encode_move(state, m): p for m, p in policy.items()}
            pending.append((n, sparse, player))
            state = apply(state, move)
            writer.record(n, player, tile, move, state.scores, annot)
            n += 1
        finals = final_scores(state)
        writer.finish(finals)

    winner = _winner(finals)
    examples = [
        Example(move_n=move_n, policy=sparse, value=_value_for(mover, winner))
        for move_n, sparse, mover in pending
    ]
    if buffer is not None:
        buffer.add_game(game_id, str(replay_path), examples, order_key)
    return game_id


def _winner(scores: tuple[int, int]) -> int | None:
    if scores[0] == scores[1]:
        return None
    return 0 if scores[0] > scores[1] else 1


def _value_for(mover: int, winner: int | None) -> float:
    """Outcome from ``mover``'s perspective: +1 win, -1 loss, 0 draw."""
    if winner is None:
        return 0.0
    return 1.0 if mover == winner else -1.0
