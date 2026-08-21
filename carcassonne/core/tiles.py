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
