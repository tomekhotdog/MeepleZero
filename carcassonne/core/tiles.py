from __future__ import annotations

from dataclasses import dataclass, field

from carcassonne.core.types import EdgeKind, FeatureKind, Rotation, Side


@dataclass(frozen=True, slots=True)
class TileFeature:
    kind: FeatureKind
    edges: frozenset[Side] = field(default_factory=frozenset)  # empty for monastery/garden
    shield: bool = False


@dataclass(frozen=True, slots=True)
class TileType:
    id: str  # "A".."X", garden variants like "U_G"
    count: int
    features: tuple[TileFeature, ...]


_EDGE_OF = {FeatureKind.CITY: EdgeKind.CITY, FeatureKind.ROAD: EdgeKind.ROAD}


def derived_edges(tt: TileType) -> tuple[EdgeKind, EdgeKind, EdgeKind, EdgeKind]:
    edges = [EdgeKind.FIELD] * 4
    for f in tt.features:
        for s in f.edges:
            edges[s] = _EDGE_OF[f.kind]
    return tuple(edges)  # type: ignore[return-value]


def world_edge(tt: TileType, rot: Rotation, world_side: Side) -> EdgeKind:
    return derived_edges(tt)[(world_side - rot) % 4]


def rotated_feature_sides(f: TileFeature, rot: Rotation) -> frozenset[Side]:
    return frozenset(Side((s + rot) % 4) for s in f.edges)


def _city(*sides: Side, shield: bool = False) -> TileFeature:
    return TileFeature(FeatureKind.CITY, frozenset(sides), shield)


def _road(*sides: Side) -> TileFeature:
    return TileFeature(FeatureKind.ROAD, frozenset(sides))


_MON = TileFeature(FeatureKind.MONASTERY)
_GAR = TileFeature(FeatureKind.GARDEN)

# NOTE: garden distribution (8 gardens, on which copies) is our approximation of
# the new-edition (C3) layout — verify against the physical rulebook and adjust
# counts here if it differs; nothing else depends on the exact split.
DECK: tuple[TileType, ...] = (
    TileType("A", 2, (_MON, _road(Side.S))),
    TileType("B", 4, (_MON,)),
    TileType("C", 1, (_city(Side.N, Side.E, Side.S, Side.W, shield=True),)),
    TileType("D", 4, (_city(Side.N), _road(Side.E, Side.W))),          # start tile type
    TileType("E", 4, (_city(Side.N),)),
    TileType("E_G", 1, (_city(Side.N), _GAR)),
    TileType("F", 2, (_city(Side.E, Side.W, shield=True),)),
    TileType("G", 1, (_city(Side.N, Side.S),)),
    TileType("H", 2, (_city(Side.E), _city(Side.W))),
    TileType("H_G", 1, (_city(Side.E), _city(Side.W), _GAR)),
    TileType("I", 2, (_city(Side.E), _city(Side.S))),
    TileType("J", 3, (_city(Side.N), _road(Side.E, Side.S))),
    TileType("K", 3, (_city(Side.N), _road(Side.S, Side.W))),
    TileType("L", 3, (_city(Side.N), _road(Side.E), _road(Side.S), _road(Side.W))),
    TileType("M", 2, (_city(Side.N, Side.W, shield=True),)),
    TileType("N", 2, (_city(Side.N, Side.W),)),
    TileType("N_G", 1, (_city(Side.N, Side.W), _GAR)),
    TileType("O", 2, (_city(Side.N, Side.W, shield=True), _road(Side.E, Side.S))),
    TileType("P", 2, (_city(Side.N, Side.W), _road(Side.E, Side.S))),
    TileType("P_G", 1, (_city(Side.N, Side.W), _road(Side.E, Side.S), _GAR)),
    TileType("Q", 1, (_city(Side.N, Side.E, Side.W, shield=True),)),
    TileType("R", 3, (_city(Side.N, Side.E, Side.W),)),
    TileType("S", 2, (_city(Side.N, Side.E, Side.W, shield=True), _road(Side.S))),
    TileType("T", 1, (_city(Side.N, Side.E, Side.W), _road(Side.S))),
    TileType("U", 6, (_road(Side.N, Side.S),)),
    TileType("U_G", 2, (_road(Side.N, Side.S), _GAR)),
    TileType("V", 7, (_road(Side.S, Side.W),)),
    TileType("V_G", 2, (_road(Side.S, Side.W), _GAR)),
    TileType("W", 4, (_road(Side.E), _road(Side.S), _road(Side.W))),
    TileType("X", 1, (_road(Side.N), _road(Side.E), _road(Side.S), _road(Side.W))),
)

TILE_TYPES: dict[str, TileType] = {t.id: t for t in DECK}
START_TILE_ID = "D"
