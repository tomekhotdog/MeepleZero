from __future__ import annotations

from dataclasses import dataclass, replace

from carcassonne.core.state import Board
from carcassonne.core.tiles import TILE_TYPES, rotated_feature_sides
from carcassonne.core.types import FeatureKind, MeepleKind, Player, Pos, Side

type NodeId = tuple[Pos, int]


@dataclass(frozen=True, slots=True)
class Meeple:
    player: Player
    kind: MeepleKind
    node: NodeId  # where it was physically placed (for rendering)


@dataclass(frozen=True, slots=True)
class FeatureData:
    kind: FeatureKind
    open_edges: int
    tiles: frozenset[Pos]
    shields: int
    meeples: tuple[Meeple, ...]


@dataclass(frozen=True)
class FeatureIndex:
    parent: dict[NodeId, NodeId]
    data: dict[NodeId, FeatureData]  # roots only
    completed_now: tuple[FeatureData, ...] = ()  # CITY/ROAD completions from the last with_tile

    @staticmethod
    def empty() -> FeatureIndex:
        return FeatureIndex({}, {})

    def find(self, n: NodeId) -> NodeId:
        while self.parent[n] != n:
            n = self.parent[n]
        return n

    def with_tile(self, board: Board, pos: Pos) -> FeatureIndex:
        """Return a new index with `pos`'s tile integrated; sets completed_now."""
        placed = board[pos]
        tt = TILE_TYPES[placed.type_id]
        parent = dict(self.parent)
        data = dict(self.data)
        new_nodes: list[tuple[NodeId, frozenset[Side]]] = []
        for i, f in enumerate(tt.features):
            node: NodeId = (pos, i)
            parent[node] = node
            sides = rotated_feature_sides(f, placed.rotation)
            data[node] = FeatureData(f.kind, len(sides), frozenset({pos}), int(f.shield), ())
            if f.kind in (FeatureKind.CITY, FeatureKind.ROAD):
                new_nodes.append((node, sides))

        def find(n: NodeId) -> NodeId:
            while parent[n] != n:
                n = parent[n]
            return n

        def union(a: NodeId, b: NodeId) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:  # loop closes: two open slots consumed
                d = data[ra]
                data[ra] = replace(d, open_edges=d.open_edges - 2)
                return
            d_rb, d_ra = data.pop(rb), data[ra]
            data[ra] = FeatureData(
                d_ra.kind,
                d_ra.open_edges + d_rb.open_edges - 2,
                d_ra.tiles | d_rb.tiles,
                d_ra.shields + d_rb.shields,
                d_ra.meeples + d_rb.meeples,
            )
            parent[rb] = ra

        for node, sides in new_nodes:
            for side in sides:
                npos = pos.neighbor(side)
                nplaced = board.get(npos)
                if nplaced is None:
                    continue
                ntt = TILE_TYPES[nplaced.type_id]
                for j, nf in enumerate(ntt.features):
                    if side.opposite in rotated_feature_sides(nf, nplaced.rotation):
                        union(node, (npos, j))
                        break

        completed = tuple(
            d
            for d in data.values()
            if d.open_edges == 0
            and pos in d.tiles
            and d.kind in (FeatureKind.CITY, FeatureKind.ROAD)
        )
        return FeatureIndex(parent, data, completed)

    def root_data(self, node: NodeId) -> FeatureData:
        return self.data[self.find(node)]

    def with_meeple(self, node: NodeId, m: Meeple) -> FeatureIndex:
        root = self.find(node)
        data = dict(self.data)
        d = data[root]
        data[root] = replace(d, meeples=d.meeples + (m,))
        return FeatureIndex(dict(self.parent), data, self.completed_now)

    def without_feature_meeples(self, root: NodeId) -> FeatureIndex:
        data = dict(self.data)
        data[root] = replace(data[root], meeples=())
        return FeatureIndex(dict(self.parent), data, self.completed_now)
