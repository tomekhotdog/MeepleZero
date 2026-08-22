"""In-memory game sessions: one human-vs-agent game each, replay-recorded live.

The store is a plain dict with no locking: the app runs in a single uvicorn
worker (see the `serve` subcommand), so sessions are process-local by design.
Everything durable is the replay file, written move by move and finished with
the end line the moment the game terminates.
"""

from __future__ import annotations

import random
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from carcassonne.agents import make_agent
from carcassonne.core import (
    GameConfig,
    GameState,
    IllegalMove,
    Move,
    ScoreEvent,
    apply,
    final_scores,
    is_terminal,
    legal_moves,
    new_game,
)
from carcassonne.game.agent import Agent, TurnContext
from carcassonne.game.replay import REPLAY_VERSION, ReplayHeader, ReplayWriter
from carcassonne.game.serde import Annot

_RNG_SALT = 0x5EED  # same salt as game.match: agents' rng stream != deck-shuffle seed


class NotFound(Exception):
    """Unknown game id or replay name -> HTTP 404."""


class UnknownOpponent(Exception):
    """Unknown agent spec in game creation -> HTTP 422."""


@dataclass
class Session:
    id: str
    state: GameState
    human_player: int
    opponent: Agent
    rng: random.Random
    writer: ReplayWriter | None  # None once the end line is written
    legal_cache: list[Move]  # rebuilt after every state change; /move idx points here
    move_n: int
    replay_name: str

    @property
    def terminal(self) -> bool:
        return is_terminal(self.state)

    def apply_human(self, idx: int) -> tuple[ScoreEvent, ...]:
        """Apply the human's move by index into the current legal_cache.

        IllegalMove on a stale or out-of-range idx (including moves on a
        finished game, whose cache is empty)."""
        if not 0 <= idx < len(self.legal_cache):
            raise IllegalMove(
                f"move index {idx} is stale or invalid:"
                f" the current position has {len(self.legal_cache)} legal moves"
            )
        return self._apply(self.legal_cache[idx], annot=None)

    def ai_step(self) -> tuple[Move, Annot | None, tuple[ScoreEvent, ...]]:
        """Let the opponent take its turn, always with a real annot recorded."""
        move, annot = self.opponent.choose(self.state, TurnContext(self.rng, annotate=True))
        events = self._apply(move, annot)
        return move, annot, events

    def hint_annot(self) -> Annot | None:
        """The opponent's read on the human's current position.

        Uses a throwaway rng: hints must not perturb the session's agent rng
        stream, or replays would depend on how often the human peeked."""
        _, annot = self.opponent.choose(self.state, TurnContext(random.Random(0), annotate=True))
        return annot

    def _apply(self, move: Move, annot: Annot | None) -> tuple[ScoreEvent, ...]:
        player, tile = self.state.current_player, self.state.current_tile
        assert tile is not None  # both callers guarantee a non-terminal state
        self.state = apply(self.state, move)
        if self.writer is not None:
            self.writer.record(self.move_n, player, tile, move, self.state.scores, annot)
        self.move_n += 1
        self.legal_cache = list(legal_moves(self.state))
        if is_terminal(self.state) and self.writer is not None:
            self.writer.finish(final_scores(self.state))
            self.writer = None
        return self.state.last_events


@dataclass
class SessionStore:
    replays_dir: Path
    _sessions: dict[str, Session] = field(default_factory=dict)

    def create(self, opponent_spec: str, human_player: int, seed: int) -> Session:
        try:
            opponent = make_agent(opponent_spec)
        except ValueError as e:
            raise UnknownOpponent(str(e)) from e
        config = GameConfig()
        state = new_game(seed, config)
        started_at = datetime.now(UTC).isoformat(timespec="seconds")
        names = ("human", opponent.name) if human_player == 0 else (opponent.name, "human")
        self.replays_dir.mkdir(parents=True, exist_ok=True)
        path = _unique_replay_path(self.replays_dir, started_at, seed, names)
        header = ReplayHeader(
            v=REPLAY_VERSION,
            seed=seed,
            config=config,
            agents=names,
            checkpoint=None,
            started_at=started_at,
        )
        session = Session(
            id=uuid.uuid4().hex,
            state=state,
            human_player=human_player,
            opponent=opponent,
            rng=random.Random(seed ^ _RNG_SALT),
            writer=ReplayWriter(path, header),
            legal_cache=list(legal_moves(state)),
            move_n=0,
            replay_name=path.name,
        )
        self._sessions[session.id] = session
        return session

    def get(self, game_id: str) -> Session:
        session = self._sessions.get(game_id)
        if session is None:
            raise NotFound(f"no such game: {game_id}")
        return session


def _unique_replay_path(
    replay_dir: Path, started_at: str, seed: int, names: tuple[str, str]
) -> Path:
    """Same naming convention as game.match: <started_at>_<seed>_<p0>-vs-<p1>.jsonl."""
    raw = f"{started_at}_{seed}_{names[0]}-vs-{names[1]}"
    base = re.sub(r"[^A-Za-z0-9._-]", "-", raw)
    path = replay_dir / f"{base}.jsonl"
    counter = 1
    while path.exists():
        path = replay_dir / f"{base}_{counter}.jsonl"
        counter += 1
    return path
