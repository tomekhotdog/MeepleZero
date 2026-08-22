"""``ReplayBuffer``: an on-disk sqlite window of self-play training targets.

States are **not** stored. Each example is (policy target, value target) tagged
with the game and move number it came from; the *state* is re-derived on demand
by replaying the referenced game through the deterministic rules core
(:func:`replay_states`), with an LRU cache of decoded games so repeated samples
of the same game are cheap. This keeps the buffer tiny (JSON + a float per move)
and leans on engine determinism -- the same idea the replay format is built on.

Perspective: ``value`` is the game outcome from the perspective of the player to
move at that position (+1 win / -1 loss / 0 draw), matching the value head and
:func:`carcassonne.agents.mcts._terminal_value`. ``policy`` is the MCTS visit
distribution as a sparse ``{action_id: prob}`` map over the action space.

No wall-clock and no global RNG live here: ``order_key`` (recency, for eviction)
is passed in by the caller, and :meth:`sample` takes an explicit ``random.Random``
-- so buffer behaviour is fully deterministic and testable.
"""

from __future__ import annotations

import functools
import json
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from carcassonne.core import GameState
from carcassonne.game.replay import load_replay, replay_states


@dataclass(frozen=True)
class Example:
    """One training example: a move's visit-policy target and value target."""

    move_n: int
    policy: dict[int, float]  # sparse {action_id: prob}, sums to ~1 over legal
    value: float  # +1 / -1 / 0 from this position's mover's perspective


@dataclass(frozen=True)
class SampledExample:
    """An :class:`Example` plus the game it was drawn from (for state re-derivation)."""

    game_id: str
    move_n: int
    policy: dict[int, float]
    value: float


@functools.lru_cache(maxsize=64)
def _decode_states(game_id: str, replay_path: str) -> tuple[GameState, ...]:
    """Decode a game's replay into the sequence of states before each move, then
    the terminal state (see :func:`replay_states`). Cached by (game_id, path) so
    repeated samples of the same game do not re-parse or re-simulate it."""
    return tuple(replay_states(load_replay(Path(replay_path))))


class ReplayBuffer:
    def __init__(self, path: Path) -> None:
        self._conn = sqlite3.connect(str(path))
        self._create_tables()

    def _create_tables(self) -> None:
        with self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS games ("
                "game_id TEXT PRIMARY KEY, replay_path TEXT, n_moves INT, order_key INT)"
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS examples ("
                "game_id TEXT, move_n INT, policy TEXT, value REAL, "
                "PRIMARY KEY(game_id, move_n))"
            )

    def add_game(
        self, game_id: str, replay_path: str, examples: list[Example], order_key: int
    ) -> None:
        """Insert one game and all its examples in a single transaction."""
        with self._conn:
            self._conn.execute(
                "INSERT INTO games(game_id, replay_path, n_moves, order_key) VALUES (?, ?, ?, ?)",
                (game_id, replay_path, len(examples), order_key),
            )
            self._conn.executemany(
                "INSERT INTO examples(game_id, move_n, policy, value) VALUES (?, ?, ?, ?)",
                [
                    (game_id, ex.move_n, _policy_to_json(ex.policy), ex.value)
                    for ex in examples
                ],
            )

    def __len__(self) -> int:
        (n,) = self._conn.execute("SELECT COUNT(*) FROM examples").fetchone()
        return int(n)

    def n_games(self) -> int:
        (n,) = self._conn.execute("SELECT COUNT(*) FROM games").fetchone()
        return int(n)

    def sample(self, batch_size: int, rng: random.Random) -> list[SampledExample]:
        """Uniformly sample ``batch_size`` examples (with replacement). Deterministic
        for a given ``rng`` and buffer contents (rows read in a stable order)."""
        rows = self._conn.execute(
            "SELECT game_id, move_n, policy, value FROM examples ORDER BY game_id, move_n"
        ).fetchall()
        if not rows:
            raise ValueError("cannot sample from an empty replay buffer")
        chosen = rng.choices(rows, k=batch_size)
        return [
            SampledExample(
                game_id=row[0],
                move_n=int(row[1]),
                policy=_policy_from_json(row[2]),
                value=float(row[3]),
            )
            for row in chosen
        ]

    def state_for(self, game_id: str, move_n: int) -> GameState:
        """The ``GameState`` *before* move ``move_n`` of ``game_id``, re-derived by
        replaying the referenced game (LRU-cached)."""
        row = self._conn.execute(
            "SELECT replay_path FROM games WHERE game_id = ?", (game_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no such game in buffer: {game_id!r}")
        states = _decode_states(game_id, row[0])
        return states[move_n]

    def evict_to(self, max_games: int) -> int:
        """Delete oldest games (smallest ``order_key``) and their examples until at
        most ``max_games`` remain. Returns the number of games deleted."""
        surplus = self.n_games() - max_games
        if surplus <= 0:
            return 0
        victims = [
            row[0]
            for row in self._conn.execute(
                "SELECT game_id FROM games ORDER BY order_key ASC, game_id ASC LIMIT ?",
                (surplus,),
            ).fetchall()
        ]
        with self._conn:
            self._conn.executemany(
                "DELETE FROM examples WHERE game_id = ?", [(g,) for g in victims]
            )
            self._conn.executemany(
                "DELETE FROM games WHERE game_id = ?", [(g,) for g in victims]
            )
        return len(victims)

    def close(self) -> None:
        self._conn.close()


def _policy_to_json(policy: dict[int, float]) -> str:
    return json.dumps({str(k): v for k, v in policy.items()}, separators=(",", ":"))


def _policy_from_json(raw: str) -> dict[int, float]:
    return {int(k): float(v) for k, v in json.loads(raw).items()}
