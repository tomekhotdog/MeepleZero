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
    Player,
    RulesError,
    apply,
    final_scores,
    is_terminal,
    new_game,
)
from carcassonne.game.serde import (
    Annot,
    annot_from_json,
    annot_to_json,
    move_from_json,
    move_to_json,
)

REPLAY_VERSION = 2
_SUPPORTED_VERSIONS = (1, 2)


@dataclass(frozen=True, slots=True)
class ReplayHeader:
    v: int
    seed: int
    config: GameConfig
    agents: tuple[str, ...]  # one name per seat
    checkpoint: str | None
    started_at: str  # caller-provided ISO-8601 string


@dataclass(frozen=True, slots=True)
class MoveRecord:
    n: int
    player: Player
    tile: str
    move: Move
    score_after: tuple[int, ...]
    annot: Annot | None


@dataclass(frozen=True, slots=True)
class Replay:
    header: ReplayHeader
    moves: tuple[MoveRecord, ...]
    final_scores: tuple[int, ...]
    winners: tuple[int, ...]  # tie set: singleton = outright win, larger = shared


def winners_of(scores: tuple[int, ...]) -> tuple[int, ...]:
    """Tie set of top scorers: singleton = outright win, larger = shared."""
    best = max(scores)
    return tuple(p for p, s in enumerate(scores) if s == best)


class ReplayWriter:
    """Append-only JSONL writer. Writes the header immediately; every record is
    flushed. Closing without `finish` (e.g. via the context manager on a crash)
    leaves an incomplete file that `load_replay` refuses."""

    def __init__(self, path: Path, header: ReplayHeader) -> None:
        if header.v != REPLAY_VERSION:
            raise RulesError(f"unsupported replay version {header.v} (writer is v{REPLAY_VERSION})")
        if len(header.agents) != header.config.num_players:
            raise RulesError(
                f"header names {len(header.agents)} agents"
                f" for {header.config.num_players} players"
            )
        self._file: TextIO = path.open("w", encoding="utf-8")
        self._write(
            {
                "v": header.v,
                "seed": header.seed,
                "config": {
                    "meeples_per_player": header.config.meeples_per_player,
                    "num_players": header.config.num_players,
                },
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
        score_after: tuple[int, ...],
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

    def finish(self, final_scores: tuple[int, ...]) -> None:
        self._write(
            {
                "end": True,
                "final_scores": list(final_scores),
                "winners": list(winners_of(final_scores)),
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
    num_players = header.config.num_players
    end = rows[-1]
    if not (isinstance(end, dict) and end.get("end") is True):
        raise RulesError("replay has no end line (incomplete replays do not load)")
    moves = tuple(
        _move_record_from_json(row, expected_n=i, num_players=num_players)
        for i, row in enumerate(rows[1:-1])
    )
    finals = _scores_field(end, "final_scores", num_players)
    winners = _winners_field(end, header.v, num_players)
    return Replay(header=header, moves=moves, final_scores=finals, winners=winners)


def _winners_field(end: dict[Any, Any], v: int, num_players: int) -> tuple[int, ...]:
    if v == 1:  # v1 wrote a single 'winner'; null meant a two-way tie
        winner = end.get("winner")
        if isinstance(winner, bool) or winner not in (0, 1, None):
            raise RulesError(f"field 'winner' must be 0, 1 or null, got {winner!r}")
        return (0, 1) if winner is None else (winner,)
    raw = end.get("winners")
    if not (
        isinstance(raw, list)
        and raw
        and all(isinstance(w, int) and not isinstance(w, bool) for w in raw)
        and all(0 <= w < num_players for w in raw)
        and all(a < b for a, b in zip(raw, raw[1:], strict=False))
    ):
        raise RulesError(
            f"field 'winners' must be a non-empty strictly increasing list of"
            f" player indices in 0..{num_players - 1}, got {raw!r}"
        )
    return tuple(raw)


def replay_states(replay: Replay) -> Iterator[GameState]:
    """Yield the state before each recorded move, then the terminal state,
    re-deriving everything via the rules core and verifying each record
    (tile drawn, player to move, score after, final scores, winners).
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
    if replay.winners != winners_of(finals):
        raise RulesError(
            f"recorded winners {list(replay.winners)},"
            f" engine computed {list(winners_of(finals))}"
        )
    yield state


def _header_from_json(row: object) -> ReplayHeader:
    if not isinstance(row, dict):
        raise RulesError(f"header must be an object, got {type(row).__name__}")
    v = _int_field(row, "v")
    if v not in _SUPPORTED_VERSIONS:
        raise RulesError(f"unsupported replay version {v} (supported: {_SUPPORTED_VERSIONS})")
    seed = _int_field(row, "seed")
    config_raw = row.get("config")
    if v == 1:  # v1 predates multiplayer: config has no num_players, always 2 seats
        if not isinstance(config_raw, dict) or set(config_raw) != {"meeples_per_player"}:
            raise RulesError(
                f"field 'config' must be {{'meeples_per_player': int}}, got {config_raw!r}"
            )
        config = GameConfig(
            meeples_per_player=_int_field(config_raw, "meeples_per_player"), num_players=2
        )
    else:
        if not isinstance(config_raw, dict) or set(config_raw) != {
            "meeples_per_player",
            "num_players",
        }:
            raise RulesError(
                f"field 'config' must be {{'meeples_per_player': int, 'num_players': int}},"
                f" got {config_raw!r}"
            )
        config = GameConfig(
            meeples_per_player=_int_field(config_raw, "meeples_per_player"),
            num_players=_int_field(config_raw, "num_players"),
        )
    agents_raw = row.get("agents")
    if not (
        isinstance(agents_raw, list)
        and len(agents_raw) == config.num_players
        and all(isinstance(a, str) for a in agents_raw)
    ):
        raise RulesError(
            f"field 'agents' must be a list of {config.num_players} strings, got {agents_raw!r}"
        )
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
        agents=tuple(agents_raw),
        checkpoint=checkpoint,
        started_at=started_at,
    )


def _move_record_from_json(row: object, expected_n: int, num_players: int) -> MoveRecord:
    if not isinstance(row, dict):
        raise RulesError(f"move record must be an object, got {type(row).__name__}")
    n = _int_field(row, "n")
    if n != expected_n:
        raise RulesError(f"move records out of order: expected n={expected_n}, got n={n}")
    player = _int_field(row, "player")
    if not 0 <= player < num_players:
        raise RulesError(
            f"move {n}: field 'player' must be in 0..{num_players - 1}, got {player}"
        )
    tile = row.get("tile")
    if not isinstance(tile, str):
        raise RulesError(f"move {n}: field 'tile' must be a string, got {tile!r}")
    annot_raw = row.get("annot")
    return MoveRecord(
        n=n,
        player=player,
        tile=tile,
        move=move_from_json(row.get("move")),
        score_after=_scores_field(row, "score_after", num_players),
        annot=None if annot_raw is None else annot_from_json(annot_raw),
    )


def _scores_field(d: dict[Any, Any], key: str, num_players: int) -> tuple[int, ...]:
    v = d.get(key)
    if not (
        isinstance(v, list)
        and len(v) == num_players
        and all(isinstance(s, int) and not isinstance(s, bool) for s in v)
    ):
        raise RulesError(f"field {key!r} must be a list of {num_players} ints, got {v!r}")
    return tuple(v)


def _int_field(d: dict[Any, Any], key: str) -> int:
    v = d.get(key)
    if isinstance(v, bool) or not isinstance(v, int):
        raise RulesError(f"field {key!r} must be an int, got {v!r}")
    return v
