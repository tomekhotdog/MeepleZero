from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum

type Player = int  # 0 | 1


class Side(IntEnum):
    N = 0
    E = 1
    S = 2
    W = 3

    @property
    def opposite(self) -> Side:
        return Side((self + 2) % 4)


class Rotation(IntEnum):
    """Clockwise quarter-turns."""

    R0 = 0
    R90 = 1
    R180 = 2
    R270 = 3


class EdgeKind(Enum):
    CITY = "city"
    ROAD = "road"
    FIELD = "field"


class FeatureKind(Enum):
    CITY = "city"
    ROAD = "road"
    MONASTERY = "monastery"
    GARDEN = "garden"


class MeepleKind(Enum):
    MEEPLE = "meeple"
    ABBOT = "abbot"


@dataclass(frozen=True, slots=True, order=True)
class Pos:
    x: int
    y: int

    def neighbor(self, side: Side) -> Pos:
        dx, dy = ((0, 1), (1, 0), (0, -1), (-1, 0))[side]
        return Pos(self.x + dx, self.y + dy)


# --- Move: one full turn decision ---


@dataclass(frozen=True, slots=True)
class PlaceMeeple:
    feature: int  # index into the placed tile's TileType.features
    kind: MeepleKind


@dataclass(frozen=True, slots=True)
class RetrieveAbbot:
    pass  # retrieve own abbot (its location is in state), score it now


type TurnAction = PlaceMeeple | RetrieveAbbot | None


@dataclass(frozen=True, slots=True)
class Move:
    pos: Pos
    rotation: Rotation
    action: TurnAction


class IllegalMove(Exception): ...


class RulesError(Exception): ...
