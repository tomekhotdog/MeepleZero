from carcassonne.core.engine import GameState
from carcassonne.core.features import FeatureIndex, Meeple, NodeId
from carcassonne.core.game import feature_report
from carcassonne.core.state import Board, PlacedTile
from carcassonne.core.types import FeatureKind, MeepleKind, Pos, Rotation


def _build(*tiles: tuple[Pos, str, Rotation]) -> tuple[Board, FeatureIndex]:
    board: Board = {}
    idx = FeatureIndex.empty()
    for pos, tid, rot in tiles:
        board[pos] = PlacedTile(tid, rot)
        idx = idx.with_tile(board, pos)
    return board, idx


def _state(board: Board, features: FeatureIndex) -> GameState:
    """Minimal GameState; feature_report only reads `board` and `features`."""
    return GameState(
        board=board,
        features=features,
        deck=(),
        current_tile=None,
        current_player=0,
        scores=(0, 0),
        meeples=(7, 7),
        abbots=(True, True),
        abbot_at=(None, None),
        discarded=(),
        last_events=(),
        turn=0,
    )


def test_incomplete_shielded_city_reports_endgame_and_completed_values() -> None:
    # F = city E-W with a shield; cap only its W edge with an E city-cap facing E.
    # The E edge stays open -> a 2-tile, 1-shield incomplete city.
    board, idx = _build(
        (Pos(0, 0), "F", Rotation.R0),  # city E-W, shield
        (Pos(-1, 0), "E", Rotation.R90),  # city faces E, caps F's W edge
    )
    d = idx.root_data((Pos(0, 0), 0))
    assert d.kind is FeatureKind.CITY and d.open_edges == 1  # still incomplete

    rep = feature_report(_state(board, idx), (Pos(0, 0), 0))

    assert rep.kind is FeatureKind.CITY
    assert rep.tiles == (Pos(-1, 0), Pos(0, 0))  # sorted footprint
    assert rep.complete is False
    # incomplete city = tiles + shields = 2 + 1 = 3
    assert rep.current == 3
    # completed city = 2*tiles + 2*shields = 4 + 2 = 6
    assert rep.potential == 6


def test_completed_city_reports_equal_current_and_potential() -> None:
    # E-cap + F(shield) + E-cap = 3 tiles, 1 shield, fully closed.
    board, idx = _build(
        (Pos(-1, 0), "E", Rotation.R90),  # city faces E
        (Pos(0, 0), "F", Rotation.R0),  # city E-W, shield
        (Pos(1, 0), "E", Rotation.R270),  # city faces W, closes it
    )
    d = idx.root_data((Pos(0, 0), 0))
    assert d.open_edges == 0

    rep = feature_report(_state(board, idx), (Pos(0, 0), 0))

    assert rep.complete is True
    # completed city = 2*3 + 2*1 = 8
    assert rep.current == 8
    assert rep.potential == 8


def test_incomplete_road_current_equals_potential_equals_tiles() -> None:
    board, idx = _build((Pos(0, 0), "D", Rotation.R0))  # road E-W, feature 1
    d = idx.root_data((Pos(0, 0), 1))
    assert d.kind is FeatureKind.ROAD and d.open_edges == 2  # both ends open

    rep = feature_report(_state(board, idx), (Pos(0, 0), 1))

    assert rep.kind is FeatureKind.ROAD
    assert rep.complete is False
    assert rep.current == 1  # 1 tile
    assert rep.potential == 1  # roads score the same either way


def test_completed_road_loop_reports_tile_count() -> None:
    # Four V tiles (road S-W at R0) forming a closed 2x2 loop of 4 tiles.
    board, idx = _build(
        (Pos(0, 0), "V", Rotation.R270),
        (Pos(1, 0), "V", Rotation.R0),
        (Pos(1, -1), "V", Rotation.R90),
        (Pos(0, -1), "V", Rotation.R180),
    )
    d = idx.root_data((Pos(0, 0), 0))
    assert d.kind is FeatureKind.ROAD and d.open_edges == 0

    rep = feature_report(_state(board, idx), (Pos(0, 0), 0))

    assert rep.complete is True
    assert rep.current == 4
    assert rep.potential == 4


def test_monastery_reports_neighbour_count_and_completion() -> None:
    # B = monastery, all-field edges. Place it with exactly 3 occupied neighbours.
    neighbours = [Pos(1, 0), Pos(-1, 0), Pos(0, 1)]
    board, idx = _build(
        (Pos(0, 0), "B", Rotation.R0),
        *((p, "B", Rotation.R0) for p in neighbours),
    )

    rep = feature_report(_state(board, idx), (Pos(0, 0), 0))

    assert rep.kind is FeatureKind.MONASTERY
    assert rep.tiles == (Pos(0, 0),)
    assert rep.current == 1 + 3  # 1 + occupied 8-neighbours
    assert rep.potential == 9
    assert rep.complete is False


def test_monastery_fully_surrounded_is_complete() -> None:
    ring = [Pos(x, y) for x in (-1, 0, 1) for y in (-1, 0, 1) if Pos(x, y) != Pos(0, 0)]
    board, idx = _build(
        (Pos(0, 0), "B", Rotation.R0),
        *((p, "B", Rotation.R0) for p in ring),
    )

    rep = feature_report(_state(board, idx), (Pos(0, 0), 0))

    assert rep.current == 9  # 1 + 8
    assert rep.potential == 9
    assert rep.complete is True


def test_report_resolves_non_root_node_of_merged_feature() -> None:
    # D city N (feature 0) capped by E from above -> a merged, completed 2-tile city.
    # The new tile's node becomes the union-find root, so (Pos(0,0),0) is a NON-root.
    board, idx = _build(
        (Pos(0, 0), "D", Rotation.R0),
        (Pos(0, 1), "E", Rotation.R180),  # city faces S, caps D's N city
    )
    non_root: NodeId = (Pos(0, 0), 0)
    assert idx.find(non_root) != non_root  # sanity: really not the root

    # add a meeple so it mirrors how the web view passes m.node
    idx = idx.with_meeple(non_root, Meeple(0, MeepleKind.MEEPLE, non_root))

    rep = feature_report(_state(board, idx), non_root)

    assert rep.kind is FeatureKind.CITY
    assert rep.tiles == (Pos(0, 0), Pos(0, 1))  # the WHOLE merged feature
    assert rep.complete is True
    assert rep.current == 4  # 2*2 + 0 shields
    assert rep.potential == 4
