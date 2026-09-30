"""In-memory game sessions: one human vs 1-4 AI seats each, replay-recorded live.

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


class HintUnavailable(Exception):
    """Hint asked for in a game with several AI seats -> HTTP 409."""


@dataclass
class Session:
    id: str
    state: GameState
    human_player: int
    agents: dict[int, Agent]  # seat -> agent for every AI seat; the human seat is absent
    seat_names: tuple[str, ...]  # per-seat names, "human" at human_player
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

    def run_ai_turns(self) -> list[tuple[int, Move, Annot | None, tuple[ScoreEvent, ...]]]:
        """Step every consecutive AI seat until it's the human's turn or the game ends.

        Each step records a real annot; returns (seat, move, annot, events) per step."""
        out: list[tuple[int, Move, Annot | None, tuple[ScoreEvent, ...]]] = []
        while not self.terminal and self.state.current_player != self.human_player:
            seat = self.state.current_player
            move, annot = self.agents[seat].choose(self.state, TurnContext(self.rng, annotate=True))
            out.append((seat, move, annot, self._apply(move, annot)))
        return out

    def hint_annot(self) -> Annot | None:
        """The opponent's read on the human's current position (2-player only).

        With several AI seats there is no single "opponent" to ask, so
        HintUnavailable. None means the opponent doesn't annotate (e.g. random).
        Uses a throwaway rng: hints must not perturb the session's agent rng
        stream, or replays would depend on how often the human peeked."""
        if len(self.agents) != 1:
            raise HintUnavailable(
                f"hints are a 2-player feature (this game has {len(self.seat_names)} players)"
            )
        opponent = next(iter(self.agents.values()))
        _, annot = opponent.choose(self.state, TurnContext(random.Random(0), annotate=True))
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


_MAX_SESSIONS = 32  # local single-user tool; evict oldest beyond this


@dataclass
class SessionStore:
    replays_dir: Path
    _sessions: dict[str, Session] = field(default_factory=dict)

    def _evict_for_capacity(self) -> None:
        """Keep the store bounded: drop finished sessions first, then the oldest.

        An abandoned mid-game session holds an open ReplayWriter; closing it via
        the context-manager path leaves a crash artifact that load_replay refuses,
        which is the correct record of an unfinished game."""
        while len(self._sessions) >= _MAX_SESSIONS:
            victim_id = next(
                (sid for sid, s in self._sessions.items() if s.writer is None),
                next(iter(self._sessions)),  # dicts preserve insertion order: oldest first
            )
            victim = self._sessions.pop(victim_id)
            if victim.writer is not None:
                victim.writer.__exit__(None, None, None)
                victim.writer = None

    def create(self, opponent_specs: list[str], human_player: int, seed: int) -> Session:
        """Seat the human at ``human_player`` and one agent per spec, in seat order,
        in the remaining seats (so ``len(opponent_specs) + 1`` players).

        Checkpoint agents are 2-player only (the RL stack is): seating one in a
        larger game is an UnknownOpponent, the web-level mirror of that guard."""
        num_players = len(opponent_specs) + 1
        if num_players > 2:
            for spec in opponent_specs:
                if spec.startswith("ckpt:"):
                    raise UnknownOpponent(
                        f"checkpoint agents are 2-player only; cannot seat {spec!r}"
                        f" in a {num_players}-player game"
                    )
        self._evict_for_capacity()
        try:
            opponents = [make_agent(s) for s in opponent_specs]
        except ValueError as e:
            raise UnknownOpponent(str(e)) from e
        config = GameConfig(num_players=num_players)
        state = new_game(seed, config)
        ai_seats = [s for s in range(num_players) if s != human_player]
        agents = dict(zip(ai_seats, opponents, strict=True))
        names_list = ["human"] * num_players
        for seat, agent in agents.items():
            names_list[seat] = agent.name
        names = tuple(names_list)
        started_at = datetime.now(UTC).isoformat(timespec="seconds")
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
            agents=agents,
            seat_names=names,
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
    replay_dir: Path, started_at: str, seed: int, names: tuple[str, ...]
) -> Path:
    """Same naming convention as game.match: <started_at>_<seed>_<p0>-vs-<p1>[-vs-...].jsonl."""
    raw = f"{started_at}_{seed}_{'-vs-'.join(names)}"
    base = re.sub(r"[^A-Za-z0-9._-]", "-", raw)
    path = replay_dir / f"{base}.jsonl"
    counter = 1
    while path.exists():
        path = replay_dir / f"{base}_{counter}.jsonl"
        counter += 1
    return path
