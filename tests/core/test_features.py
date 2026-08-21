from carcassonne.core.features import FeatureIndex, Meeple
from carcassonne.core.placement import fits
from carcassonne.core.state import Board, PlacedTile, empty_board_with_start
from carcassonne.core.tiles import TILE_TYPES
from carcassonne.core.types import FeatureKind, MeepleKind, Pos, Rotation


def _integrate(board: Board, idx: FeatureIndex, pos: Pos, tile: PlacedTile) -> FeatureIndex:
    board[pos] = tile
    return idx.with_tile(board, pos)


def test_two_city_caps_complete_a_city() -> None:
    board = empty_board_with_start()  # D at (0,0), R0: city N (feature 0), road E-W
    idx = FeatureIndex.empty().with_tile(board, Pos(0, 0))
    assert idx.completed_now == ()

    # E at R180: its N-city faces S (world), closing D's city from above.
    idx = _integrate(board, idx, Pos(0, 1), PlacedTile("E", Rotation.R180))

    assert len(idx.completed_now) == 1
    city = idx.completed_now[0]
    assert city.kind is FeatureKind.CITY
    assert city.tiles == frozenset({Pos(0, 0), Pos(0, 1)})
    assert city.open_edges == 0


def test_road_loop_self_union() -> None:
    # Four V tiles (road S-W at R0) forming a 2x2 loop.
    loop = (
        (Pos(0, 0), Rotation.R270),  # roads E, S
        (Pos(1, 0), Rotation.R0),  # roads S, W
        (Pos(1, -1), Rotation.R90),  # roads W, N
        (Pos(0, -1), Rotation.R180),  # roads N, E
    )
    board: Board = {}
    idx = FeatureIndex.empty()
    for pos, rot in loop:
        if board:  # sanity: every placement after the first is edge-compatible
            assert fits(board, TILE_TYPES["V"], pos, rot)
        idx = _integrate(board, idx, pos, PlacedTile("V", rot))

    assert len(idx.completed_now) == 1
    road = idx.completed_now[0]
    assert road.kind is FeatureKind.ROAD
    assert road.tiles == frozenset(pos for pos, _ in loop)
    assert road.open_edges == 0


def test_merge_two_meepled_cities_keeps_both_meeples() -> None:
    board = empty_board_with_start()  # D city N at feature index 0
    idx = FeatureIndex.empty().with_tile(board, Pos(0, 0))
    # Separate city two tiles up, facing back down.
    idx = _integrate(board, idx, Pos(0, 2), PlacedTile("E", Rotation.R180))

    m0 = Meeple(player=0, kind=MeepleKind.MEEPLE, node=(Pos(0, 0), 0))
    m1 = Meeple(player=1, kind=MeepleKind.MEEPLE, node=(Pos(0, 2), 0))
    idx = idx.with_meeple((Pos(0, 0), 0), m0)
    idx = idx.with_meeple((Pos(0, 2), 0), m1)
    assert idx.root_data((Pos(0, 0), 0)) is not idx.root_data((Pos(0, 2), 0))

    # G (city N-S) bridges the two cities — a contested feature.
    idx = _integrate(board, idx, Pos(0, 1), PlacedTile("G", Rotation.R0))

    merged = idx.root_data((Pos(0, 0), 0))
    assert merged is idx.root_data((Pos(0, 2), 0))
    assert set(merged.meeples) == {m0, m1}
    assert merged.open_edges == 0
    assert idx.completed_now == (merged,)


def test_with_meeple_and_without_feature_meeples_roundtrip() -> None:
    board = empty_board_with_start()
    idx = FeatureIndex.empty().with_tile(board, Pos(0, 0))
    idx = _integrate(board, idx, Pos(0, 1), PlacedTile("E", Rotation.R180))

    non_root = (Pos(0, 0), 0)
    root = idx.find(non_root)
    assert root != non_root  # the new tile's node became the root

    m = Meeple(player=0, kind=MeepleKind.MEEPLE, node=non_root)
    with_m = idx.with_meeple(non_root, m)
    assert with_m.root_data(non_root).meeples == (m,)  # landed on root data
    assert with_m.completed_now == idx.completed_now  # carried over
    assert idx.root_data(non_root).meeples == ()  # original untouched

    cleared = with_m.without_feature_meeples(root)
    assert cleared.root_data(non_root).meeples == ()
    assert with_m.root_data(non_root).meeples == (m,)  # original untouched


def test_monastery_nodes_never_union() -> None:
    board = empty_board_with_start()  # D at (0,0): road E-W
    idx = FeatureIndex.empty().with_tile(board, Pos(0, 0))
    # A at R270: monastery (feature 0) + road stub facing E onto D's W road edge.
    assert fits(board, TILE_TYPES["A"], Pos(-1, 0), Rotation.R270)
    idx = _integrate(board, idx, Pos(-1, 0), PlacedTile("A", Rotation.R270))

    monastery = (Pos(-1, 0), 0)
    assert idx.find(monastery) == monastery  # stays its own root
    d = idx.root_data(monastery)
    assert d.kind is FeatureKind.MONASTERY
    assert d.open_edges == 0
    assert idx.completed_now == ()  # never reported complete here; road still open

    # The road stub did union with D's road and remains open.
    assert idx.find((Pos(-1, 0), 1)) == idx.find((Pos(0, 0), 1))
    assert idx.root_data((Pos(-1, 0), 1)).open_edges == 1
