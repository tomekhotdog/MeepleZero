from collections import Counter

from carcassonne.core.tiles import DECK, START_TILE_ID, TILE_TYPES
from carcassonne.core.types import FeatureKind, Side


def test_deck_has_72_tiles() -> None:
    assert sum(t.count for t in DECK) == 72


def test_official_letter_counts() -> None:
    # counts per base letter (garden variants fold into their letter)
    expect = {"A": 2, "B": 4, "C": 1, "D": 4, "E": 5, "F": 2, "G": 1, "H": 3,
              "I": 2, "J": 3, "K": 3, "L": 3, "M": 2, "N": 3, "O": 2, "P": 3,
              "Q": 1, "R": 3, "S": 2, "T": 1, "U": 8, "V": 9, "W": 4, "X": 1}
    got: Counter[str] = Counter()
    for t in DECK:
        got[t.id.split("_")[0]] += t.count
    assert dict(got) == expect


def test_every_feature_edge_consistent() -> None:
    for t in DECK:
        # no two features may share an edge slot
        seen: set[Side] = set()
        for f in t.features:
            assert not (f.edges & seen), t.id
            seen |= f.edges


def test_no_duplicate_tile_ids() -> None:
    # a duplicate id would silently drop a variant from the TILE_TYPES lookup
    assert len(TILE_TYPES) == len(DECK)


def test_shields_total_and_gardens() -> None:
    assert sum(t.count for t in DECK for f in t.features if f.shield) == 10
    gardens = sum(t.count for t in DECK for f in t.features
                  if f.kind is FeatureKind.GARDEN)
    assert gardens == 8


def test_start_tile() -> None:
    assert START_TILE_ID == "D"
    assert TILE_TYPES["D"].count == 4
