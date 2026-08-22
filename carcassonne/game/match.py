"""Match runner: play one full game between two agents, optionally recording a replay.

This is the orchestration layer — wall-clock timestamps live here; everything
below it (core, replay) is deterministic.
"""

from __future__ import annotations

import contextlib
import random
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from carcassonne.agents.base import Agent, TurnContext
from carcassonne.core import GameConfig, apply, final_scores, is_terminal, new_game
from carcassonne.game.replay import REPLAY_VERSION, ReplayHeader, ReplayWriter

_DEFAULT_CONFIG = GameConfig()
_RNG_SALT = 0x5EED  # agents' rng stream must differ from the deck-shuffle seed


@dataclass(frozen=True, slots=True)
class GameResult:
    final_scores: tuple[int, int]
    winner: int | None  # 0 | 1 | None (draw)
    turns: int
    replay_path: Path | None


def play_game(
    agents: tuple[Agent, Agent],
    seed: int,
    replay_dir: Path | None = None,
    config: GameConfig = _DEFAULT_CONFIG,
    started_at: str | None = None,
) -> GameResult:
    """Run one game to completion; agents share a single rng seeded from `seed`.

    With `replay_dir` set, writes a uniquely named JSONL replay there. If an
    agent or the engine raises mid-game, the writer's context manager leaves an
    incomplete file behind as a crash artifact (which `load_replay` refuses).
    """
    if started_at is None:
        started_at = datetime.now(UTC).isoformat(timespec="seconds")
    state = new_game(seed, config)
    ctx = TurnContext(rng=random.Random(seed ^ _RNG_SALT))
    writer: ReplayWriter | None = None
    path: Path | None = None
    if replay_dir is not None:
        replay_dir.mkdir(parents=True, exist_ok=True)
        path = _unique_replay_path(replay_dir, started_at, seed, agents)
        header = ReplayHeader(
            v=REPLAY_VERSION,
            seed=seed,
            config=config,
            agents=(agents[0].name, agents[1].name),
            checkpoint=None,
            started_at=started_at,
        )
        writer = ReplayWriter(path, header)

    n = 0
    with contextlib.ExitStack() as stack:
        if writer is not None:
            stack.enter_context(writer)
        while not is_terminal(state):
            player, tile = state.current_player, state.current_tile
            assert tile is not None  # non-terminal <=> a tile is drawn
            move, annot = agents[player].choose(state, ctx)
            state = apply(state, move)
            if writer is not None:
                writer.record(n, player, tile, move, state.scores, annot)
            n += 1
        finals = final_scores(state)
        if writer is not None:
            writer.finish(finals)
    return GameResult(final_scores=finals, winner=_winner(finals), turns=n, replay_path=path)


def _winner(scores: tuple[int, int]) -> int | None:
    if scores[0] == scores[1]:
        return None
    return 0 if scores[0] > scores[1] else 1


def _unique_replay_path(
    replay_dir: Path, started_at: str, seed: int, agents: tuple[Agent, Agent]
) -> Path:
    raw = f"{started_at or 'game'}_{seed}_{agents[0].name}-vs-{agents[1].name}"
    base = re.sub(r"[^A-Za-z0-9._-]", "-", raw)
    path = replay_dir / f"{base}.jsonl"
    counter = 1
    while path.exists():
        path = replay_dir / f"{base}_{counter}.jsonl"
        counter += 1
    return path
