"""JSON codecs for Move and Annot.

Total and explicit: any malformed input raises RulesError; round-trips are exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from carcassonne.core import MeepleKind, Move, PlaceMeeple, Pos, RetrieveAbbot, Rotation, RulesError


@dataclass(frozen=True, slots=True)
class Annot:
    """Per-move agent annotation: search value, effort, and top considered moves."""

    value: float
    sims: int
    think_ms: int
    top: tuple[tuple[Move, float, int], ...]  # (move, prior, visits)


def move_to_json(move: Move) -> dict[str, Any]:
    action: dict[str, Any] | None
    match move.action:
        case None:
            action = None
        case PlaceMeeple(feature=feature, kind=kind):
            action = {"type": "meeple", "feature": feature, "kind": kind.value}
        case RetrieveAbbot():
            action = {"type": "retrieve_abbot"}
    return {"x": move.pos.x, "y": move.pos.y, "rot": int(move.rotation), "action": action}


def move_from_json(d: object) -> Move:
    if not isinstance(d, dict):
        raise RulesError(f"move must be an object, got {type(d).__name__}")
    x = _int_field(d, "x")
    y = _int_field(d, "y")
    rot = _int_field(d, "rot")
    if rot not in (0, 1, 2, 3):
        raise RulesError(f"field 'rot' must be 0..3, got {rot}")
    if "action" not in d:
        raise RulesError("move missing field 'action'")
    return Move(Pos(x, y), Rotation(rot), _action_from_json(d["action"]))


def annot_to_json(annot: Annot) -> dict[str, Any]:
    return {
        "value": annot.value,
        "sims": annot.sims,
        "think_ms": annot.think_ms,
        "top": [
            {"move": move_to_json(move), "prior": prior, "visits": visits}
            for move, prior, visits in annot.top
        ],
    }


def annot_from_json(d: object) -> Annot:
    if not isinstance(d, dict):
        raise RulesError(f"annot must be an object, got {type(d).__name__}")
    value = _float_field(d, "value")
    sims = _int_field(d, "sims")
    think_ms = _int_field(d, "think_ms")
    top_raw = d.get("top")
    if not isinstance(top_raw, list):
        raise RulesError(f"field 'top' must be a list, got {top_raw!r}")
    top: list[tuple[Move, float, int]] = []
    for entry in top_raw:
        if not isinstance(entry, dict):
            raise RulesError(f"top entry must be an object, got {type(entry).__name__}")
        top.append(
            (
                move_from_json(entry.get("move")),
                _float_field(entry, "prior"),
                _int_field(entry, "visits"),
            )
        )
    return Annot(value=value, sims=sims, think_ms=think_ms, top=tuple(top))


def _action_from_json(a: object) -> PlaceMeeple | RetrieveAbbot | None:
    if a is None:
        return None
    if not isinstance(a, dict):
        raise RulesError(f"action must be an object or null, got {type(a).__name__}")
    kind = a.get("type")
    if kind == "meeple":
        meeple_kind = a.get("kind")
        if meeple_kind not in ("meeple", "abbot"):
            raise RulesError(f"field 'kind' must be 'meeple' or 'abbot', got {meeple_kind!r}")
        return PlaceMeeple(_int_field(a, "feature"), MeepleKind(meeple_kind))
    if kind == "retrieve_abbot":
        return RetrieveAbbot()
    raise RulesError(f"unknown action type: {kind!r}")


def _int_field(d: dict[Any, Any], key: str) -> int:
    v = d.get(key)
    if isinstance(v, bool) or not isinstance(v, int):
        raise RulesError(f"field {key!r} must be an int, got {v!r}")
    return v


def _float_field(d: dict[Any, Any], key: str) -> float:
    v = d.get(key)
    if isinstance(v, bool) or not isinstance(v, int | float):
        raise RulesError(f"field {key!r} must be a number, got {v!r}")
    return float(v)
