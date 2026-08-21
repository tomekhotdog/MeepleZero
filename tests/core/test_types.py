from carcassonne.core.tiles import TileFeature, TileType, derived_edges, world_edge
from carcassonne.core.types import EdgeKind, FeatureKind, Pos, Rotation, Side


def test_pos_neighbors() -> None:
    p = Pos(0, 0)
    assert p.neighbor(Side.N) == Pos(0, 1)
    assert p.neighbor(Side.S) == Pos(0, -1)
    assert p.neighbor(Side.E) == Pos(1, 0)
    assert Side.N.opposite is Side.S


def test_rotation_maps_tile_side_to_world_side() -> None:
    # a tile rotated 90 degrees clockwise: its N edge now faces E (world)
    tt = TileType(id="E", count=5, features=(TileFeature(FeatureKind.CITY, frozenset({Side.N})),))
    assert world_edge(tt, Rotation.R0, Side.N) is EdgeKind.CITY
    assert world_edge(tt, Rotation.R90, Side.E) is EdgeKind.CITY
    assert world_edge(tt, Rotation.R90, Side.N) is EdgeKind.FIELD


def test_derived_edges() -> None:
    tt = TileType(
        id="D",
        count=4,
        features=(
            TileFeature(FeatureKind.CITY, frozenset({Side.N})),
            TileFeature(FeatureKind.ROAD, frozenset({Side.E, Side.W})),
        ),
    )
    assert derived_edges(tt) == (EdgeKind.CITY, EdgeKind.ROAD, EdgeKind.FIELD, EdgeKind.ROAD)
