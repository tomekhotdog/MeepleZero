"""JSONL replay format: one file per game, write-as-you-go, verifying loader.

Line 1 is a header, then one line per move, then an end line with final scores.
`load_replay` validates structure; `replay_states` re-derives every state through
the rules core and cross-checks each record — a tampered or desynced file cannot
be replayed silently.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Self, TextIO

from carcassonne.core import (
    GameConfig,
    GameState,
    IllegalMove,
    Move,
    RulesError,
    apply,
    final_scores,
    is_terminal,
    new_game,
)
from carcassonne.core.types import Player
from carcassonne.game.serde import (
    Annot,
    annot_from_json,
    annot_to_json,
    move_from_json,
    move_to_json,
)

REPLAY_VERSION = 1


@dataclass(frozen=True, slots=True)
class ReplayHeader:
    v: int
    seed: int
    config: GameConfig
    agents: tuple[str, str]
    checkpoint: str | None
    started_at: str  # caller-provided ISO-8601 string


@dataclass(frozen=True, slots=True)
class MoveRecord:
    n: int
    player: Player
    tile: str
    move: Move
    score_after: tuple[int, int]
    annot: Annot | None


@dataclass(frozen=True, slots=True)
class Replay:
    header: ReplayHeader
    moves: tuple[MoveRecord, ...]
    final_scores: tuple[int, int]
    winner: int | None  # 0 | 1 | None (draw)


class ReplayWriter:
    """Append-only JSONL writer. Writes the header immediately; every record is
    flushed. Closing without `finish` (e.g. via the context manager on a crash)
    leaves an incomplete file that `load_replay` refuses."""

    def __init__(self, path: Path, header: ReplayHeader) -> None:
        self._file: TextIO = path.open("w", encoding="utf-8")
        self._write(
            {
                "v": header.v,
                "seed": header.seed,
                "config": {"meeples_per_player": header.config.meeples_per_player},
                "agents": list(header.agents),
                "checkpoint": header.checkpoint,
                "started_at": header.started_at,
            }
        )

    def record(
        self,
        n: int,
        player: Player,
        tile: str,
        move: Move,
        score_after: tuple[int, int],
        annot: Annot | None = None,
    ) -> None:
        self._write(
            {
                "n": n,
                "player": player,
                "tile": tile,
                "move": move_to_json(move),
                "score_after": list(score_after),
                "annot": None if annot is None else annot_to_json(annot),
            }
        )

    def finish(self, final_scores: tuple[int, int]) -> None:
        self._write(
            {
                "end": True,
                "final_scores": list(final_scores),
                "winner": _winner(final_scores),
            }
        )
        self._file.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if not self._file.closed:
            self._file.close()

    def _write(self, obj: dict[str, Any]) -> None:
        self._file.write(json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n")
        self._file.flush()


def load_replay(path: Path) -> Replay:
    """Parse and structurally validate a replay file. RulesError on any defect,
    including a missing end line (an incomplete replay does not load)."""
    rows: list[Any] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise RulesError(f"line {i + 1}: invalid JSON: {e}") from e
    if len(rows) < 2:
        raise RulesError("replay must have a header line and an end line")
    header = _header_from_json(rows[0])
    end = rows[-1]
    if not (isinstance(end, dict) and end.get("end") is True):
        raise RulesError("replay has no end line (incomplete replays do not load)")
    moves = tuple(
        _move_record_from_json(row, expected_n=i) for i, row in enumerate(rows[1:-1])
    )
    finals = _scores_field(end, "final_scores")
    winner = end.get("winner")
    if isinstance(winner, bool) or winner not in (0, 1, None):
        raise RulesError(f"field 'winner' must be 0, 1 or null, got {winner!r}")
    return Replay(header=header, moves=moves, final_scores=finals, winner=winner)


def replay_states(replay: Replay) -> Iterator[GameState]:
    """Yield the state before each recorded move, then the terminal state,
    re-deriving everything via the rules core and verifying each record
    (tile drawn, player to move, score after, final scores, winner).
    RulesError on any mismatch."""
    state = new_game(replay.header.seed, replay.header.config)
    for rec in replay.moves:
        if rec.tile != state.current_tile:
            raise RulesError(
                f"move {rec.n}: recorded tile {rec.tile!r},"
                f" engine drew {state.current_tile!r}"
            )
        if rec.player != state.current_player:
            raise RulesError(
                f"move {rec.n}: recorded player {rec.player},"
                f" engine expects {state.current_player}"
            )
        yield state
        try:
            state = apply(state, rec.move)
        except IllegalMove as e:
            raise RulesError(f"move {rec.n}: illegal move: {e}") from e
        if rec.score_after != state.scores:
            raise RulesError(
                f"move {rec.n}: recorded score_after {list(rec.score_after)},"
                f" engine scored {list(state.scores)}"
            )
    if not is_terminal(state):
        raise RulesError(f"replay ends after {len(replay.moves)} moves but the game is not over")
    finals = final_scores(state)
    if finals != replay.final_scores:
        raise RulesError(
            f"recorded final_scores {list(replay.final_scores)},"
            f" engine computed {list(finals)}"
        )
    if replay.winner != _winner(finals):
        raise RulesError(f"recorded winner {replay.winner!r}, engine computed {_winner(finals)!r}")
    yield state


def _winner(scores: tuple[int, int]) -> int | None:
    if scores[0] == scores[1]:
        return None
    return 0 if scores[0] > scores[1] else 1


def _header_from_json(row: object) -> ReplayHeader:
    if not isinstance(row, dict):
        raise RulesError(f"header must be an object, got {type(row).__name__}")
    v = _int_field(row, "v")
    if v != REPLAY_VERSION:
        raise RulesError(f"unsupported replay version {v} (expected {REPLAY_VERSION})")
    seed = _int_field(row, "seed")
    config_raw = row.get("config")
    if not isinstance(config_raw, dict) or set(config_raw) != {"meeples_per_player"}:
        raise RulesError(
            f"field 'config' must be {{'meeples_per_player': int}}, got {config_raw!r}"
        )
    config = GameConfig(meeples_per_player=_int_field(config_raw, "meeples_per_player"))
    agents_raw = row.get("agents")
    if not (
        isinstance(agents_raw, list)
        and len(agents_raw) == 2
        and all(isinstance(a, str) for a in agents_raw)
    ):
        raise RulesError(f"field 'agents' must be a list of two strings, got {agents_raw!r}")
    checkpoint = row.get("checkpoint")
    if checkpoint is not None and not isinstance(checkpoint, str):
        raise RulesError(f"field 'checkpoint' must be a string or null, got {checkpoint!r}")
    started_at = row.get("started_at")
    if not isinstance(started_at, str):
        raise RulesError(f"field 'started_at' must be a string, got {started_at!r}")
    return ReplayHeader(
        v=v,
        seed=seed,
        config=config,
        agents=(agents_raw[0], agents_raw[1]),
        checkpoint=checkpoint,
        started_at=started_at,
    )


def _move_record_from_json(row: object, expected_n: int) -> MoveRecord:
    if not isinstance(row, dict):
        raise RulesError(f"move record must be an object, got {type(row).__name__}")
    n = _int_field(row, "n")
    if n != expected_n:
        raise RulesError(f"move records out of order: expected n={expected_n}, got n={n}")
    player = _int_field(row, "player")
    if player not in (0, 1):
        raise RulesError(f"move {n}: field 'player' must be 0 or 1, got {player}")
    tile = row.get("tile")
    if not isinstance(tile, str):
        raise RulesError(f"move {n}: field 'tile' must be a string, got {tile!r}")
    annot_raw = row.get("annot")
    return MoveRecord(
        n=n,
        player=player,
        tile=tile,
        move=move_from_json(row.get("move")),
        score_after=_scores_field(row, "score_after"),
        annot=None if annot_raw is None else annot_from_json(annot_raw),
    )


def _scores_field(d: dict[Any, Any], key: str) -> tuple[int, int]:
    v = d.get(key)
    if not (
        isinstance(v, list)
        and len(v) == 2
        and all(isinstance(s, int) and not isinstance(s, bool) for s in v)
    ):
        raise RulesError(f"field {key!r} must be a list of two ints, got {v!r}")
    return (v[0], v[1])


def _int_field(d: dict[Any, Any], key: str) -> int:
    v = d.get(key)
    if isinstance(v, bool) or not isinstance(v, int):
        raise RulesError(f"field {key!r} must be an int, got {v!r}")
    return v
