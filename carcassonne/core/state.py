from __future__ import annotations

from dataclasses import dataclass

from carcassonne.core.tiles import START_TILE_ID
from carcassonne.core.types import Pos, Rotation


@dataclass(frozen=True, slots=True)
class PlacedTile:
    type_id: str
    rotation: Rotation


type Board = dict[Pos, PlacedTile]


def empty_board_with_start() -> Board:
    return {Pos(0, 0): PlacedTile(START_TILE_ID, Rotation.R0)}
