# ProjectCarcasonne Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use tomek-superpowers:build to implement this plan task-by-task.

**Goal:** Carcassonne simulator (base game + abbots, no farmers, 2 players) with a
pure rules core, Replay files as interchange, an RL-microscope web UI, and a
resumable AlphaZero-style training loop that runs on a Mac and a Raspberry Pi 5.

**Architecture:** one layered package `carcassonne` (`core → game → agents/nn →
training/web`), strict one-way imports. Engine is deterministic from `(seed, moves)`.
See `docs/plans/2026-08-21-project-carcasonne-design.md`.

**Tech stack:** Python 3.12, pytest + hypothesis, PyTorch (mps/cpu), FastAPI +
uvicorn, vanilla JS canvas (no build step), ruff + mypy.

**Design:** `docs/plans/2026-08-21-project-carcasonne-design.md`

**Plan granularity note (deliberate, flagged):** Phases 0–2 (engine, replay,
agents, CLI) are specified to near-complete code — they are the correctness-
critical foundation. Phases 3–6 specify exact contracts, schemas, algorithms,
and complete *test* code, with implementation left to the task agent within
those contracts. Expanding those phases to full code now would front-load
thousands of lines that will shift once the engine exists; each phase's first
task includes a "re-read design §N" step instead.

**Design correction (applied to design doc + language.md):** `RetrieveAbbot` is
not a standalone move — a turn always places a tile. The Move type is
`Move(pos, rotation, action)` with `action: None | PlaceMeeple(feature, kind) | RetrieveAbbot()`.

---

## Phase 0 — Scaffolding

### Task 1: Project scaffolding ✅ DONE (71e6d8f; torch → optional `ml` extra: Intel Mac, no torch≥2.4 wheels — relax to `torch>=2.2` when Phase 4 starts)
**Depends on:** none

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `carcassonne/__init__.py`, and empty
  `__init__.py` in `carcassonne/{core,game,agents,nn,training,web,cli}/`
- Create: `tests/__init__.py`, `tests/test_scaffolding.py`

**Step 1: Write the failing test**
```python
# tests/test_scaffolding.py
def test_package_imports():
    import carcassonne.core
    import carcassonne.game
    import carcassonne.agents
```

**Step 2:** Run `uv run pytest tests/test_scaffolding.py` → FAIL (no package).

**Step 3: Implementation**

`pyproject.toml`:
```toml
[project]
name = "carcassonne"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["fastapi>=0.115", "uvicorn>=0.30", "torch>=2.4", "numpy>=2.0"]

[project.optional-dependencies]
dev = ["pytest>=8", "hypothesis>=6.100", "ruff>=0.6", "mypy>=1.11", "httpx>=0.27"]

[project.scripts]
carcassonne = "carcassonne.cli.main:main"

[tool.ruff]
line-length = 100
[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "TID251"]  # TID: enforce layering later
[tool.mypy]
strict = true
[tool.pytest.ini_options]
testpaths = ["tests"]
```

`.gitignore`: `__pycache__/`, `.venv/`, `*.egg-info/`, `runs/`, `replays/`, `.mypy_cache/`, `.ruff_cache/`, `.pytest_cache/`.

Create the package directories with empty `__init__.py` files. Set up env:
`uv venv && uv pip install -e '.[dev]'` (use `uv`; fall back to plain venv+pip if unavailable).

**Step 4:** `uv run pytest` → 1 passed.
**Step 5:** `git add -A && git commit -m "chore: project scaffolding"`

---

## Phase 1 — Core engine

### Task 2: Core types and tile schema ✅ DONE (f25ac7d + review fix: direct rotated_feature_sides test)
**Depends on:** Task 1

**Files:**
- Create: `carcassonne/core/types.py`, `carcassonne/core/tiles.py` (schema only)
- Test: `tests/core/test_types.py`

**Step 1: Failing test**
```python
# tests/core/test_types.py
from carcassonne.core.types import Pos, Side, Rotation, EdgeKind
from carcassonne.core.tiles import TileType, TileFeature, derived_edges
from carcassonne.core.types import FeatureKind

def test_pos_neighbors():
    p = Pos(0, 0)
    assert p.neighbor(Side.N) == Pos(0, 1)
    assert p.neighbor(Side.S) == Pos(0, -1)
    assert p.neighbor(Side.E) == Pos(1, 0)
    assert Side.N.opposite is Side.S

def test_rotation_maps_tile_side_to_world_side():
    # a tile rotated 90° clockwise: its N edge now faces E (world)
    from carcassonne.core.tiles import world_edge
    tt = TileType(id="E", count=5, features=(TileFeature(FeatureKind.CITY, frozenset({Side.N})),))
    assert world_edge(tt, Rotation.R0, Side.N) is EdgeKind.CITY
    assert world_edge(tt, Rotation.R90, Side.E) is EdgeKind.CITY
    assert world_edge(tt, Rotation.R90, Side.N) is EdgeKind.FIELD

def test_derived_edges():
    tt = TileType(id="D", count=4, features=(
        TileFeature(FeatureKind.CITY, frozenset({Side.N})),
        TileFeature(FeatureKind.ROAD, frozenset({Side.E, Side.W})),
    ))
    assert derived_edges(tt) == (EdgeKind.CITY, EdgeKind.ROAD, EdgeKind.FIELD, EdgeKind.ROAD)
```

**Step 2:** `uv run pytest tests/core/test_types.py` → FAIL.

**Step 3: Implementation**

`carcassonne/core/types.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum

type Player = int  # 0 | 1

class Side(IntEnum):
    N = 0; E = 1; S = 2; W = 3

    @property
    def opposite(self) -> "Side":
        return Side((self + 2) % 4)

class Rotation(IntEnum):
    R0 = 0; R90 = 1; R180 = 2; R270 = 3  # clockwise quarter-turns

class EdgeKind(Enum):
    CITY = "city"; ROAD = "road"; FIELD = "field"

class FeatureKind(Enum):
    CITY = "city"; ROAD = "road"; MONASTERY = "monastery"; GARDEN = "garden"

class MeepleKind(Enum):
    MEEPLE = "meeple"; ABBOT = "abbot"

@dataclass(frozen=True, slots=True, order=True)
class Pos:
    x: int
    y: int

    def neighbor(self, side: Side) -> "Pos":
        dx, dy = ((0, 1), (1, 0), (0, -1), (-1, 0))[side]
        return Pos(self.x + dx, self.y + dy)

# --- Move: one full turn decision ---

@dataclass(frozen=True, slots=True)
class PlaceMeeple:
    feature: int          # index into the placed tile's TileType.features
    kind: MeepleKind

@dataclass(frozen=True, slots=True)
class RetrieveAbbot:
    pass                  # retrieve own abbot (its location is in state), score it now

type TurnAction = PlaceMeeple | RetrieveAbbot | None

@dataclass(frozen=True, slots=True)
class Move:
    pos: Pos
    rotation: Rotation
    action: TurnAction

class IllegalMove(Exception): ...
class RulesError(Exception): ...
```

`carcassonne/core/tiles.py` (schema part):
```python
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from carcassonne.core.types import EdgeKind, FeatureKind, Rotation, Side

@dataclass(frozen=True, slots=True)
class TileFeature:
    kind: FeatureKind
    edges: frozenset[Side] = field(default_factory=frozenset)  # empty for monastery/garden
    shield: bool = False

@dataclass(frozen=True, slots=True)
class TileType:
    id: str                       # "A".."X", garden variants like "U_G"
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
```
Wrap `derived_edges` in `lru_cache` keyed by tile id via a module-level dict if
profiling ever demands it — not now.

**Step 4:** `uv run pytest tests/core` → PASS.
**Step 5:** `git commit -am "feat(core): tiny types, Move, tile schema with derived edges"`

---

### Task 3: Full base-game deck data ✅ DONE
**Depends on:** Task 2

**Files:**
- Modify: `carcassonne/core/tiles.py` (append `DECK`, `TILE_TYPES`, `START_TILE_ID`)
- Test: `tests/core/test_deck.py`

**Step 1: Failing test**
```python
# tests/core/test_deck.py
from collections import Counter
from carcassonne.core.tiles import DECK, TILE_TYPES, START_TILE_ID, derived_edges
from carcassonne.core.types import FeatureKind, Side

def test_deck_has_72_tiles():
    assert sum(t.count for t in DECK) == 72

def test_official_letter_counts():
    # counts per base letter (garden variants fold into their letter)
    expect = {"A": 2, "B": 4, "C": 1, "D": 4, "E": 5, "F": 2, "G": 1, "H": 3,
              "I": 2, "J": 3, "K": 3, "L": 3, "M": 2, "N": 3, "O": 2, "P": 3,
              "Q": 1, "R": 3, "S": 2, "T": 1, "U": 8, "V": 9, "W": 4, "X": 1}
    got = Counter()
    for t in DECK:
        got[t.id.split("_")[0]] += t.count
    assert dict(got) == expect

def test_every_feature_edge_consistent():
    for t in DECK:
        derived_edges(t)  # must not raise
        # no two same-kind features may share an edge slot
        seen: set[Side] = set()
        for f in t.features:
            assert not (f.edges & seen), t.id
            seen |= f.edges

def test_shields_total_and_gardens():
    shields = sum(f.shield for t in DECK for f in t.features) and \
              sum(t.count for t in DECK for f in t.features if f.shield)
    assert sum(t.count for t in DECK for f in t.features if f.shield) == 10
    gardens = sum(t.count for t in DECK for f in t.features
                  if f.kind is FeatureKind.GARDEN)
    assert gardens == 8

def test_start_tile():
    assert START_TILE_ID == "D"
    assert TILE_TYPES["D"].count == 4
```

**Step 2:** run → FAIL.

**Step 3: Implementation** — append to `tiles.py`. Helper local names for brevity:
```python
from carcassonne.core.types import Side as S

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
    TileType("A", 2, (_MON, _road(S.S))),
    TileType("B", 4, (_MON,)),
    TileType("C", 1, (_city(S.N, S.E, S.S, S.W, shield=True),)),
    TileType("D", 4, (_city(S.N), _road(S.E, S.W))),          # start tile type
    TileType("E", 4, (_city(S.N),)),
    TileType("E_G", 1, (_city(S.N), _GAR)),
    TileType("F", 2, (_city(S.E, S.W, shield=True),)),
    TileType("G", 1, (_city(S.N, S.S),)),
    TileType("H", 2, (_city(S.E), _city(S.W))),
    TileType("H_G", 1, (_city(S.E), _city(S.W), _GAR)),
    TileType("I", 2, (_city(S.E), _city(S.S))),
    TileType("J", 3, (_city(S.N), _road(S.E, S.S))),
    TileType("K", 3, (_city(S.N), _road(S.S, S.W))),
    TileType("L", 3, (_city(S.N), _road(S.E), _road(S.S), _road(S.W))),
    TileType("M", 2, (_city(S.N, S.W, shield=True),)),
    TileType("N", 2, (_city(S.N, S.W),)),
    TileType("N_G", 1, (_city(S.N, S.W), _GAR)),
    TileType("O", 2, (_city(S.N, S.W, shield=True), _road(S.E, S.S))),
    TileType("P", 2, (_city(S.N, S.W), _road(S.E, S.S))),
    TileType("P_G", 1, (_city(S.N, S.W), _road(S.E, S.S), _GAR)),
    TileType("Q", 1, (_city(S.N, S.E, S.W, shield=True),)),
    TileType("R", 3, (_city(S.N, S.E, S.W),)),
    TileType("S", 2, (_city(S.N, S.E, S.W, shield=True), _road(S.S))),
    TileType("T", 1, (_city(S.N, S.E, S.W), _road(S.S))),
    TileType("U", 6, (_road(S.N, S.S),)),
    TileType("U_G", 2, (_road(S.N, S.S), _GAR)),
    TileType("V", 7, (_road(S.S, S.W),)),
    TileType("V_G", 2, (_road(S.S, S.W), _GAR)),
    TileType("W", 4, (_road(S.E), _road(S.S), _road(S.W))),
    TileType("X", 1, (_road(S.N), _road(S.E), _road(S.S), _road(S.W))),
)

TILE_TYPES: dict[str, TileType] = {t.id: t for t in DECK}
START_TILE_ID = "D"
```
Shield check: C(1) + F(2) + M(2) + O(2) + Q(1) + S(2) = 10 ✓. Gardens: 1+1+1+2+2+1 = 8 ✓.

**Step 4:** `uv run pytest tests/core/test_deck.py` → PASS.
**Step 5:** `git commit -am "feat(core): full 72-tile base deck with abbot gardens"`

---

### Task 4: GameState and placement legality ✅ DONE (af4b65b)
**Depends on:** Task 3

**Files:**
- Create: `carcassonne/core/state.py`, `carcassonne/core/placement.py`
- Test: `tests/core/test_placement.py`

**Step 1: Failing test**
```python
# tests/core/test_placement.py
from carcassonne.core.types import Pos, Rotation, Side
from carcassonne.core.state import PlacedTile, empty_board_with_start
from carcassonne.core.placement import placements_for_tile
from carcassonne.core.tiles import TILE_TYPES

def test_start_board_and_first_tile_placements():
    board = empty_board_with_start()          # D at (0,0), R0: city N, road E-W
    # tile E (city N): fits south of start at R180? E's city N at R180 faces S —
    # touching start's N edge (city). So (0,1) requires our S edge == start N edge (CITY).
    placs = placements_for_tile(board, TILE_TYPES["E"])
    assert (Pos(0, 1), Rotation.R180) in placs      # city faces down onto start's city
    assert (Pos(0, 1), Rotation.R0) not in placs    # field S vs city N mismatch
    assert all(p != Pos(0, 0) for p, _ in placs)     # occupied

def test_no_floating_placements():
    board = empty_board_with_start()
    placs = placements_for_tile(board, TILE_TYPES["B"])
    assert all(abs(p.x) + abs(p.y) == 1 for p, _ in placs)  # only adjacent to start
```

**Step 2:** run → FAIL.

**Step 3: Implementation**

`carcassonne/core/state.py` (state shell; features/scores wired in Tasks 5–7):
```python
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
```

`carcassonne/core/placement.py`:
```python
from __future__ import annotations

from carcassonne.core.state import Board
from carcassonne.core.tiles import TILE_TYPES, TileType, world_edge
from carcassonne.core.types import Pos, Rotation, Side

def frontier(board: Board) -> set[Pos]:
    """Empty positions adjacent to at least one placed tile."""
    out: set[Pos] = set()
    for pos in board:
        for side in Side:
            n = pos.neighbor(side)
            if n not in board:
                out.add(n)
    return out

def fits(board: Board, tt: TileType, pos: Pos, rot: Rotation) -> bool:
    touching = False
    for side in Side:
        n = pos.neighbor(side)
        placed = board.get(n)
        if placed is None:
            continue
        touching = True
        mine = world_edge(tt, rot, side)
        theirs = world_edge(TILE_TYPES[placed.type_id], placed.rotation, side.opposite)
        if mine is not theirs:
            return False
    return touching

def placements_for_tile(board: Board, tt: TileType) -> list[tuple[Pos, Rotation]]:
    return [(p, r) for p in sorted(frontier(board)) for r in Rotation if fits(board, tt, p, r)]
```
(Sorted frontier ⇒ deterministic move ordering, which ActionIndexer and replay
tests rely on.)

**Step 4:** run → PASS.  **Step 5:** `git commit -am "feat(core): board, edge-matching placement legality"`

---

### Task 5: Feature connectivity (union-find) 
**Depends on:** Task 3 (parallel with Task 4)

**Files:**
- Create: `carcassonne/core/features.py`
- Test: `tests/core/test_features.py`

Semantics: every CITY/ROAD `TileFeature` instance on a placed tile is a node
`(pos, feature_index)`. Placing a tile unions its features with matching
neighbour features across shared edges. `open_edges` counts unmatched
city/road edge-slots; a feature is **complete** when `open_edges == 0`.
Union bookkeeping: `merged.open = a.open + b.open - 2`; self-union (loop):
`open -= 2`. Monastery/garden features are nodes with no edges (never unioned;
completion = 8 neighbours, handled in Task 6).

**Step 1: Failing test**
```python
# tests/core/test_features.py
from carcassonne.core.features import FeatureIndex
from carcassonne.core.state import PlacedTile, empty_board_with_start
from carcassonne.core.types import Pos, Rotation, FeatureKind

def _add(fi, board, pos, type_id, rot):
    board[pos] = PlacedTile(type_id, rot)
    return fi.with_tile(board, pos)

def test_two_city_caps_complete_a_city():
    board = empty_board_with_start()                 # D: city N (feature 0)
    fi = FeatureIndex.empty().with_tile(board, Pos(0, 0))
    # E at (0,1) rotated 180: its city faces S, closing D's city
    fi = _add(fi, board, Pos(0, 1), "E", Rotation.R180)
    completed = fi.completed_now
    assert len(completed) == 1
    (data,) = completed
    assert data.kind is FeatureKind.CITY and len(data.tiles) == 2 and data.open_edges == 0

def test_road_loop_self_union():
    # four V curves (road S-W) forming a 2x2 loop closes a road
    board = {}
    fi = FeatureIndex.empty()
    fi = _add(fi, board, Pos(0, 0), "V", Rotation.R90)    # road W->S becomes N->W? verify: R90 rotates S,W -> W,N => road N-W
    fi = _add(fi, board, Pos(1, 0), "V", Rotation.R0)     # road S-W
    # ... build the loop; final placement must yield a completed ROAD with 4 tiles
```
(The loop test enumerates the four placements with explicit rotations; the task
agent works them out from `rotated_feature_sides` and asserts one completed
ROAD, 4 tiles, `open_edges == 0`.)

**Step 2:** run → FAIL.

**Step 3: Implementation**

```python
# carcassonne/core/features.py
from __future__ import annotations

from dataclasses import dataclass, field, replace

from carcassonne.core.state import Board
from carcassonne.core.tiles import TILE_TYPES, rotated_feature_sides
from carcassonne.core.types import FeatureKind, MeepleKind, Player, Pos, Side

type NodeId = tuple[Pos, int]

@dataclass(frozen=True, slots=True)
class Meeple:
    player: Player
    kind: MeepleKind
    node: NodeId          # where it was physically placed (for rendering)

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
    data: dict[NodeId, FeatureData]      # roots only
    completed_now: tuple[FeatureData, ...] = ()   # completions from the last with_tile

    @staticmethod
    def empty() -> "FeatureIndex":
        return FeatureIndex({}, {})

    def find(self, n: NodeId) -> NodeId:
        while self.parent[n] != n:
            n = self.parent[n]
        return n

    def with_tile(self, board: Board, pos: Pos) -> "FeatureIndex":
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
            data[node] = FeatureData(f.kind, len(sides), frozenset({pos}),
                                     int(f.shield), ())
            if f.kind in (FeatureKind.CITY, FeatureKind.ROAD):
                new_nodes.append((node, sides))

        def find(n: NodeId) -> NodeId:
            while parent[n] != n:
                n = parent[n]
            return n

        def union(a: NodeId, b: NodeId) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:                      # loop closes: two open slots consumed
                d = data[ra]
                data[ra] = replace(d, open_edges=d.open_edges - 2)
                return
            da, db = data.pop(rb), data[ra]
            data[ra] = FeatureData(db.kind, da.open_edges + db.open_edges - 2,
                                   da.tiles | db.tiles, da.shields + db.shields,
                                   da.meeples + db.meeples)
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

        completed = tuple(d for d in data.values()
                          if d.open_edges == 0 and pos in d.tiles
                          and d.kind in (FeatureKind.CITY, FeatureKind.ROAD))
        return FeatureIndex(parent, data, completed)

    def root_data(self, node: NodeId) -> FeatureData:
        return self.data[self.find(node)]

    def with_meeple(self, node: NodeId, m: Meeple) -> "FeatureIndex": ...
        # copy dicts, append m to root's FeatureData.meeples

    def without_feature_meeples(self, root: NodeId) -> "FeatureIndex": ...
        # copy dicts, clear meeples on root (after scoring)
```
(`with_meeple`/`without_feature_meeples` are 5-line dict-copy methods — implement
fully, same pattern as `with_tile`.)

**Step 4:** run → PASS.  **Step 5:** `git commit -am "feat(core): incremental union-find feature index"`

---

### Task 6: apply() — full turn resolution and scoring
**Depends on:** Tasks 4, 5

**Files:**
- Create: `carcassonne/core/engine.py` (GameState + apply + legal_moves)
- Test: `tests/core/test_engine.py`

**GameState (final shape):**
```python
@dataclass(frozen=True, slots=True)
class ScoreEvent:
    player: Player
    points: int
    kind: FeatureKind
    tiles: frozenset[Pos]

@dataclass(frozen=True)
class GameState:
    board: Board                     # treat as immutable; copied on apply
    features: FeatureIndex
    deck: tuple[str, ...]            # remaining tile type ids, pre-shuffled
    current_tile: str | None         # None <=> terminal
    current_player: Player
    scores: tuple[int, int]
    meeples: tuple[int, int]         # plain meeples in supply (start: 7, 7)
    abbots: tuple[bool, bool]        # abbot in supply?
    abbot_at: tuple[NodeId | None, NodeId | None]
    discarded: tuple[str, ...]       # unplaceable tiles set aside
    last_events: tuple[ScoreEvent, ...]
    turn: int
```

**Scoring rules implemented here (single source of truth):**
- City complete: `2×tiles + 2×shields`; Road complete: `1×tiles`.
- Majority: meeple-count per player on the feature; all tied leaders score full
  points. All meeples on the feature return to supply.
- Monastery/garden complete (8 neighbours): 9 points to the meeple/abbot owner,
  piece returns. Checked for every monastery/garden node whose tile is in the
  3×3 around the new tile (and the new tile itself if it has one and a meeple
  was just placed — a monastery placed into a full hole scores immediately).
- `RetrieveAbbot`: score `1 + occupied neighbours` of the abbot's tile now,
  return abbot to supply.
- Placing a meeple requires the (post-merge) feature to have no meeples;
  monastery/garden must be unoccupied; ABBOT only on MONASTERY/GARDEN; MEEPLE
  on CITY/ROAD/MONASTERY (not GARDEN); supply must have the piece.

**legal_moves:** for each `(pos, rot)` from `placements_for_tile(board, current)`:
option `None`; every legal `PlaceMeeple(i, kind)` (evaluate occupancy against a
speculative merge — implement by running `with_tile` on a scratch board and
checking the merged root has no meeples); plus the same `(pos, rot)` with
`RetrieveAbbot()` if the player's abbot is on the board. Deterministic order:
positions sorted, rotations ascending, actions ordered `None, PlaceMeeple(0..),
RetrieveAbbot`.

**apply(state, move):** raise `IllegalMove` unless `move in legal_moves(state)`
(fast path allowed later; correctness first) → place tile → integrate features →
resolve meeple action → score completions + monasteries → draw next playable
tile (discarding unplaceable ones into `discarded`) → advance player/turn.

**Step 1: Failing tests** (complete list; each is a compact scenario built by
scripted `apply` calls with a fixed deck injected via `dataclasses.replace`):
```python
def test_city_completion_scores_majority_owner(): ...   # 2 tiles + shield = 6 pts
def test_road_junction_splits_roads(): ...              # W tile: 3 separate roads
def test_cannot_place_meeple_on_occupied_merged_city(): ...
def test_tie_majority_both_score(): ...
def test_monastery_scores_9_when_surrounded(): ...
def test_abbot_on_garden_and_retrieve_scores_current_value(): ...
def test_meeple_returns_to_supply_after_scoring(): ...
def test_unplaceable_tile_discarded_and_next_drawn(): ...
def test_apply_rejects_illegal_move(): ...              # raises IllegalMove
```

**Step 2:** run → FAIL. **Step 3:** implement `engine.py` per the spec above
(~200 lines; every rule stated here, no interpretation needed).
**Step 4:** `uv run pytest tests/core` → all PASS.
**Step 5:** `git commit -am "feat(core): apply() with full turn resolution and scoring"`

---

### Task 7: Game lifecycle + property tests — the five-function API
**Depends on:** Task 6

**Files:**
- Create: `carcassonne/core/__init__.py` exports; `carcassonne/core/game.py`
  (`new_game`, `is_terminal`, `final_scores`); `carcassonne/core/config.py`
  (`GameConfig(seed-independent knobs: meeples=7, window handled in nn)`)
- Test: `tests/core/test_game.py`, `tests/core/test_properties.py`

`new_game(seed, config)` shuffles the 71 non-start tiles with
`random.Random(seed)`, places the start D tile, draws the first playable tile.
`final_scores` adds end-game scoring: incomplete city `1×tiles + 1×shields`,
incomplete road `1×tiles`, monastery/garden `1 + neighbours` — pure function,
does not mutate state.

**Property tests (hypothesis, the correctness backstop):**
```python
# tests/core/test_properties.py
from hypothesis import given, settings, strategies as st
from carcassonne import core

@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 10_000))
def test_random_game_invariants(seed):
    state = core.new_game(seed)
    rng = __import__("random").Random(seed)
    meeple_total = {0: 8, 1: 8}                      # 7 meeples + 1 abbot each
    while not core.is_terminal(state):
        moves = core.legal_moves(state)
        assert moves, "legal_moves must be non-empty on non-terminal state"
        prev = state.scores
        state = core.apply(state, rng.choice(moves))
        assert state.scores[0] >= prev[0] and state.scores[1] >= prev[1]
        for p in (0, 1):
            on_board = sum(m.player == p for d in state.features.data.values()
                           for m in d.meeples)
            in_supply = state.meeples[p] + int(state.abbots[p])
            assert on_board + in_supply == meeple_total[p]
    assert core.final_scores(state) >= (0, 0)

@settings(max_examples=10, deadline=None)
@given(seed=st.integers(0, 10_000))
def test_replay_determinism(seed):
    s1, s2 = core.new_game(seed), core.new_game(seed)
    rng = __import__("random").Random(seed)
    while not core.is_terminal(s1):
        mv = rng.choice(core.legal_moves(s1))
        s1, s2 = core.apply(s1, mv), core.apply(s2, mv)
        assert s1.scores == s2.scores and s1.board == s2.board
```

Also `tests/core/test_game.py`: a full fixed-seed game snapshot test — play seed
42 with seeded-random moves to completion, assert exact final scores (pin the
value on first green run; guards future refactors).

**Steps:** test-fail-implement-pass, then
`git commit -am "feat(core): five-function public API with property tests"`.

---

## Phase 2 — Replay, agents, CLI

### Task 8: Replay format (writer/loader)
**Depends on:** Task 7

**Files:**
- Create: `carcassonne/game/replay.py`, `carcassonne/game/serde.py` (Move/state JSON codecs)
- Test: `tests/game/test_replay.py`

**Format (JSONL), one file per game:**
```json
{"v": 1, "seed": 42, "config": {}, "agents": ["human", "greedy"], "checkpoint": null, "started_at": "..."}
{"n": 0, "player": 0, "tile": "V", "move": {"x": 1, "y": 0, "rot": 1, "action": null}, "score_after": [0, 0], "annot": null}
{"n": 1, "player": 1, "tile": "E", "move": {"x": 0, "y": 1, "rot": 2, "action": {"type": "meeple", "feature": 0, "kind": "meeple"}}, "score_after": [0, 0], "annot": {"value": 0.12, "sims": 200, "think_ms": 310, "top": [{"move": {...}, "prior": 0.4, "visits": 88}]}}
```
Final line: `{"end": true, "final_scores": [34, 41], "winner": 1}`.

API: `ReplayWriter(path, header)` / `.record(n, player, tile, move, score_after, annot)` /
`.finish(final_scores)`; `load_replay(path) -> Replay` (frozen dataclass);
`replay_states(replay) -> Iterator[GameState]` re-derives every state via the
engine and **asserts** each recorded `score_after` matches — loading is itself
a verification.

**Tests:** round-trip (simulate seeded random game → write → load →
`replay_states` reproduces final scores); tamper test (corrupt a score in the
file → loader raises `RulesError`).

Commit: `git commit -am "feat(game): JSONL replay format with verifying loader"`

---

### Task 9: Agent protocol, RandomAgent, `simulate` CLI
**Depends on:** Task 8

**Files:**
- Create: `carcassonne/agents/base.py`, `carcassonne/agents/random_agent.py`,
  `carcassonne/game/match.py` (`play_game(agents, seed, replay_path?) -> Result`),
  `carcassonne/cli/main.py` (argparse subcommands; `simulate` first)
- Test: `tests/agents/test_random.py`, `tests/cli/test_simulate.py`

```python
# agents/base.py
class TurnContext(Protocol-ish dataclass): rng: random.Random; annotate: bool
class Agent(Protocol):
    name: str
    def choose(self, state: GameState, ctx: TurnContext) -> tuple[Move, Annot | None]: ...
```
`Annot` = frozen dataclass mirroring the replay `annot` object.

CLI: `carcassonne simulate --p0 random --p1 random --seed 42 --out replays/`
prints final scores and the replay path; e2e test runs it via `main([...])` and
loads the produced replay.

Commit: `git commit -am "feat(agents,cli): Agent protocol, RandomAgent, simulate command"`

---

### Task 10: GreedyAgent + `evaluate` CLI
**Depends on:** Task 9

**Files:**
- Create: `carcassonne/agents/greedy.py`, `carcassonne/game/arena.py`
  (`run_match(a, b, n_games, base_seed) -> MatchResult` with seat-swapping),
  CLI `evaluate` subcommand
- Test: `tests/agents/test_greedy.py`

**GreedyAgent value function** (1-ply, exact spec):
```
value(move) = my_score_delta - opp_score_delta
            + 0.7 * Δ(my potential) - 0.7 * Δ(opp potential)
            + 0.15 * meeples_in_supply_after
potential(player) = Σ over player-majority incomplete features of
                    current end-game value of that feature
```
Ties broken by `ctx.rng` for variety. `MatchResult`: wins/draws/mean scores,
per-game replays written.

**Sanity test (the yardstick works):**
```python
def test_greedy_crushes_random():
    r = run_match(GreedyAgent(), RandomAgent(), n_games=20, base_seed=7)
    assert r.wins[0] >= 17   # greedy wins ≥85% vs random
```
CLI e2e: `carcassonne evaluate --p0 greedy --p1 random --games 10 --seed 1`
prints a result table.

Commit: `git commit -am "feat(agents): greedy heuristic agent + evaluate arena"`

---

## Phase 3 — Web app (Play + Replay views)

*Re-read design §5 before starting this phase.*

### Task 11: FastAPI backend
**Depends on:** Task 10

**Files:**
- Create: `carcassonne/web/app.py`, `carcassonne/web/views.py` (state→JSON),
  `carcassonne/web/sessions.py`; CLI `serve` subcommand (uvicorn)
- Test: `tests/web/test_api.py` (httpx TestClient)

**Endpoints (exact contract):**
| Method/Path | Body → Response |
|---|---|
| `POST /api/games` | `{opponent: "random"\|"greedy"\|"ckpt:<id>", human_player: 0\|1, seed?: int}` → `{game_id, state}` (AI moves first if human is 1) |
| `GET /api/games/{id}` | → `{state}` |
| `GET /api/games/{id}/legal` | → `{moves: [{x,y,rot,action,idx}]}` |
| `POST /api/games/{id}/move` | `{idx}` (index into last legal list) → `{state, ai_move: {move, annot} \| null, events: [...]}` — applies human move, then AI reply, saves replay on game end |
| `GET /api/games/{id}/hint` | → `{policy: [{x,y,rot,prob}], value: float}` from the opponent agent (404 until Task 17 wires MCTS; greedy returns normalised move values) |
| `GET /api/replays` / `GET /api/replays/{name}` | list / full parsed replay incl. per-move board states |
| `GET /api/health` | `{ok: true}` |

**State JSON shape** (`views.py`): `{tiles: [{x, y, type, rot, meeples: [{player, kind, feature}]}], scores, meeples, abbots, current_tile, current_player, turn, last_events, terminal, final_scores?}`.

Errors: `IllegalMove` → 409 with `{error, explanation}`; unknown game → 404.
Sessions: in-memory dict keyed by uuid4 hex; replays written to `./replays/`.

Tests: full API round-trip playing a scripted game vs random to completion via
TestClient; illegal move returns 409.

Commit: `git commit -am "feat(web): FastAPI game/replay API"`

---

### Task 12: Play view (canvas frontend)
**Depends on:** Task 11

**Files:**
- Create: `carcassonne/web/static/index.html`, `static/app.js`, `static/tiles.js`
  (procedural tile drawing), `static/style.css`
- Test: `tests/web/test_static.py` (assets served; JS syntax-checked with `node --check` if node present, else skipped)

**Contract (implementation freedom within it):**
- Canvas board, pan (drag) + zoom (wheel); 64px tiles at zoom 1.
- `tiles.js` exposes `drawTile(ctx, tileType, rotation, x, y, size)` rendering
  from the same feature data (served once via `GET /api/tiledefs`, added to the
  backend in this task: `{id, edges, features: [{kind, edges, shield}]}` for all
  types): field = green base, city = tan polygon over its edges w/ shield badge,
  road = grey path between edge midpoints (junction = centre dot), monastery =
  dark red house glyph, garden = ring of green dots. Meeples: coloured discs
  (P0 blue, P1 red), abbot = disc + cross.
- Turn flow: current tile shown in sidebar; `R` or click rotates; legal
  placements pulled from `/legal` and shown as pulsing outlines **for the
  current rotation**; click places; if meeple options exist for that placement
  a small radial picker appears over the placed tile (feature glyphs + "skip" +
  "retrieve abbot" when legal); then POST `/move`; AI reply animates in.
- Sidebar: scores, supplies, deck-remaining, last score events log, hint toggle
  (renders `/hint` policy as colour-graded cell overlays), "new game" dialog
  (opponent, seat, seed).
- End of game: banner with final scores + link to replay view.

Definition of done (manual e2e, recorded in commit message): play a full game
vs greedy in the browser, replay file appears in `./replays/`.

Commit: `git commit -am "feat(web): playable canvas UI vs any agent"`

---

### Task 13: Replay view
**Depends on:** Task 12

**Files:**
- Create: `static/replay.html`, `static/replay.js` (reuses `tiles.js`)
- Test: extend `tests/web/test_api.py` for `/api/replays/{name}` board-state payloads

**Contract:** replay picker (list from `/api/replays`); board scrubber
(first/prev/next/last, arrow keys, autoplay); per-move panel: player, tile,
score delta, and when `annot` present the **search panel** — horizontal bar
pairs (prior vs visits, normalised) for top-k moves with the chosen move
highlighted — plus a win-probability sparkline across all moves (from
`annot.value`, sign-adjusted to P0 perspective). Panel shows placeholders for
human/greedy moves (annot null or heuristic-only).

Commit: `git commit -am "feat(web): replay viewer with search panel + win-prob chart"`

---

## Phase 4 — Network (parallel with Phase 3)

*Re-read design §2 before starting this phase.*

### Task 14: Encoder + ActionIndexer
**Depends on:** Task 7 (independent of Phases 2–3 web work)

**Files:**
- Create: `carcassonne/nn/encode.py`, `carcassonne/nn/actions.py`
- Test: `tests/nn/test_encode.py`, `tests/nn/test_actions.py`

**ActionIndexer (exact):** window `B=31` (config), centre = board bounding-box
centre. Action id = `((wy * B + wx) * 4 + rot) * 14 + a` where
`a ∈ {0: none, 1–6: PlaceMeeple(f, MEEPLE) f=0..5, 7–12: PlaceMeeple(f, ABBOT), 13: RetrieveAbbot}`.
Total = 31·31·4·14 = 53,816. API: `encode_move(state, move) -> int`,
`decode_move(state, idx) -> Move`, `legal_mask(state) -> np.ndarray[bool]`.

**Encoder planes (float32, `C×31×31`, C = 33 + |TILE_TYPES|):** per-cell:
occupied(1); edge-kind one-hot × 4 sides (12); per-feature-kind presence (4);
shield(1); completed-feature flag(1); meeple owner one-hot (2); abbot owner (2);
monastery-neighbour-count/8 (1). Broadcast scalars: scores/50 (2), meeple
supply/7 (2), abbots (2), deck_remaining/71 (1), current-player (1), turn/72 (1);
current-tile one-hot (|TILE_TYPES| = 30 planes). All from the **current
player's perspective** (own planes first; value target likewise).

**Tests:** exhaustive round-trip `decode(encode(m)) == m` over every legal move
of 50 random positions; `legal_mask` count equals `len(legal_moves)`; D4
equivariance — `encode(rot90(state))` equals `np.rot90(encode(state))` on the
spatial planes (build rot90(state) by replaying rotated moves).

Commit: `git commit -am "feat(nn): state encoder + action indexer with D4 tests"`

---

### Task 15: Network + Checkpoint
**Depends on:** Task 14

**Files:**
- Create: `carcassonne/nn/model.py`, `carcassonne/nn/checkpoint.py`
- Test: `tests/nn/test_model.py`

**Model (exact):** conv3x3(C→64)+BN+ReLU → 5 residual blocks (64ch, two
conv3x3+BN, skip, ReLU) → policy head: conv1x1(64→14·4=56)? No — policy head:
conv1x1(64→14×4)… **spec:** conv1x1(64→56)+BN+ReLU reshaped so each cell emits
4·14 logits ⇒ `B·B·4·14 = 53,816` logits matching ActionIndexer layout; value
head: conv1x1(64→4)+BN+ReLU → flatten → linear(4·31·31→64) → ReLU →
linear(64→1) → tanh. `forward(x, mask) -> (masked_log_policy, value)` with
illegal logits set to `-inf` pre-softmax. ~1.1M params — Pi-sized.

`checkpoint.py`: `save(dir, step, model, optimizer, config) -> path`
(`step_000123.pt`, torch.save of state dicts + config json sidecar);
`load(path, device)`, `latest(dir)`, `CheckpointId = str` (filename stem).
Device helper `pick_device()` → cuda > mps > cpu.

**Tests:** forward shape on batch of 4; masked illegal logits are `-inf`;
save→load→identical outputs; runs on `pick_device()`.

Commit: `git commit -am "feat(nn): residual policy/value network + checkpoints"`

---

## Phase 5 — MCTS + RL microscope

### Task 16: MCTS + MctsAgent
**Depends on:** Tasks 15, 10

**Files:**
- Create: `carcassonne/agents/mcts.py`
- Test: `tests/agents/test_mcts.py`

**Algorithm (exact, AlphaZero-style):** tree over `GameState` (immutability =
free branching). Per node: `N, W, Q, P` arrays over legal moves. Selection:
PUCT `Q + c_puct · P · √ΣN / (1+N)`, `c_puct = 1.5`. Leaf: encode → net →
(masked policy priors, value); terminal leaf: value from `final_scores` sign.
Backup negates value each ply (two-player zero-sum). Root Dirichlet noise
(α=0.3, ε=0.25) when `self_play=True`. Move selection: sample ∝ N^(1/τ), τ=1
for turns < 12 else τ→0 (argmax). Config: `MctsConfig(sims=200, c_puct=1.5,
dirichlet_alpha=0.3, noise_frac=0.25, temp_turns=12)`.

`MctsAgent.choose` returns `(move, Annot(value=root_value, sims=sims,
top=[(move, prior, visits) top-8], think_ms))` — feeding the Replay `annot`
directly. Also exposes `visit_policy(state) -> dict[Move, float]` for training
targets and hints.

**Tests:** with an untrained (random-weight) net and 64 sims, MctsAgent beats
RandomAgent ≥ 7/10 seeded games (search alone adds strength); visit_policy sums
to 1; deterministic given seeded rng and τ=0.

Commit: `git commit -am "feat(agents): PUCT MCTS agent guided by the network"`

---

### Task 17: RL microscope wiring (hints + search panel live)
**Depends on:** Tasks 16, 13

**Files:**
- Modify: `web/app.py` (`ckpt:<id>` opponents; `/hint` runs a short MCTS
  (64 sims) and returns `{policy, value, top: [{move, prior, visits}]}`),
  `static/app.js` (hint overlay renders prior vs visit colour scales; sidebar
  shows value gauge), `static/replay.js` (panel already built — verify against
  real MCTS annotations)
- Test: `tests/web/test_hint.py` — hint endpoint returns normalised policy and
  top-k with priors *and* visits for a checkpoint opponent.

Definition of done: play vs an (untrained) checkpoint in the browser; every AI
move shows the search panel; hint overlay toggles.

Commit: `git commit -am "feat(web): live MCTS hints — the RL microscope"`

---

## Phase 6 — Training

*Re-read design §3 before starting this phase.*

### Task 18: Self-play worker + ReplayBuffer
**Depends on:** Tasks 16, 8

**Files:**
- Create: `carcassonne/training/selfplay.py`, `carcassonne/training/buffer.py`,
  `carcassonne/training/run.py` (TrainingRun dir layout + config)
- Test: `tests/training/test_selfplay.py`, `tests/training/test_buffer.py`

**TrainingRun dir:** `runs/<name>/{config.json, checkpoints/, replays/, buffer.sqlite, metrics.jsonl}`.

**Buffer (sqlite):** table `examples(game_id TEXT, move_n INT, policy JSON
sparse {action_idx: prob}, value REAL, PRIMARY KEY(game_id, move_n))` + table
`games(game_id, replay_path, n_moves, created_at)`. States are **not** stored —
sampling re-derives them by replaying the referenced game through the engine
(deterministic) with an LRU cache of decoded games (size 64). Window: keep
newest `window_games` (config, default 2000); older games deleted.

**Self-play worker:** `play_selfplay_game(net_path, seed, run_dir, mcts_cfg) ->
game_id` — MctsAgent vs itself (self_play=True), records replay with annot,
inserts examples: policy target = visit distribution (sparse over action ids),
value target = final outcome ±1 (0 draw) from each position's current player.
D4 augmentation happens at **sampling** time (Task 19), not storage.
Multiprocess pool orchestrated in Task 20.

**Tests:** one tiny self-play game (8 sims) populates buffer with `n_moves`
examples whose sparse policies sum to ~1; window eviction works; sampled
batch reconstructs encodable states.

Commit: `git commit -am "feat(training): self-play generation + sqlite replay buffer"`

---

### Task 19: Learner + resume
**Depends on:** Task 18

**Files:**
- Create: `carcassonne/training/learner.py`
- Test: `tests/training/test_learner.py`

**Spec:** Adam(lr=1e-3, weight_decay=1e-4), batch 256 (config; 64 on Pi).
Loss = CE(policy_target, log_policy) + MSE(value, value_target). Each sampled
example applies a uniformly random D4 transform to (planes, policy). Metrics
line per step to `metrics.jsonl`: `{step, loss, policy_loss, value_loss,
buffer_games, ts}`. Checkpoint every `ckpt_every` steps.
**Resume:** `Learner.from_run(run_dir)` restores model+optimizer from latest
checkpoint and continues step numbering — test kills a learner mid-run and
resumes to identical continued step count.

**Test:** 20 steps on a synthetic buffer overfits (loss strictly decreases over
the run); resume test as above.

Commit: `git commit -am "feat(training): learner with D4 augmentation and lossless resume"`

---

### Task 20: Arena gating + `train` CLI orchestration
**Depends on:** Task 19

**Files:**
- Create: `carcassonne/training/arena.py`, `carcassonne/training/orchestrate.py`;
  CLI `train` subcommand
- Test: `tests/training/test_train_smoke.py`

**Orchestration (single process owns everything):** loop —
`selfplay_games_per_iter` games via `multiprocessing.Pool(workers)` (workers
load the current-best checkpoint, CPU inference) → `learn_steps_per_iter`
learner steps → every `gate_every` iters run Arena: candidate vs best over
`gate_games` (seat-swapped, τ=0), promote if wins > 55%; always also report vs
GreedyAgent; append `{iter, candidate, promoted, wr_best, wr_greedy}` to
`metrics.jsonl`. SIGINT/SIGTERM → finish current step, flush, exit 0 (systemd-
friendly for the Pi). `carcassonne train --run runs/first --workers 3 ...` and
`--resume runs/first`.

**Smoke test (the harness proof):** tiny config (8 sims, 2 games/iter, 4 learn
steps, gate every iter, 4 gate games) runs 2 iterations in <5 min on CI/Mac,
produces checkpoints + metrics, then `--resume` runs 1 more iteration without
error. This test is the e2e gate for the whole training system.

Commit: `git commit -am "feat(training): arena gating + resumable train orchestration"`

---

### Task 21: Training view
**Depends on:** Tasks 20, 13

**Files:**
- Modify: `web/app.py` (`GET /api/runs`, `GET /api/runs/{name}` → parsed
  config + metrics + checkpoint list); Create: `static/training.html`,
  `static/training.js`
- Test: `tests/web/test_runs_api.py`

**Contract:** run picker; charts (canvas, no chart lib): loss curves
(total/policy/value vs step), arena win-rate vs best and vs greedy (with the
55% gate line), games generated over time; checkpoint timeline with promotion
markers; auto-refresh every 10s. Point it at a live Pi run over the LAN
(`carcassonne serve --runs-dir /path` reads any runs dir).

Commit: `git commit -am "feat(web): training run dashboard"`

---

### Task 22: Final verification (success bar)
**Depends on:** all

No new features. Execute and record results in `docs/plans/` as
`2026-XX-XX-iteration1-verification.md`:

1. `uv run pytest` — full suite green; `uv run ruff check .` and `uv run mypy carcassonne` clean.
2. Train on the Mac until candidate beats **GreedyAgent > 70%** over
   `carcassonne evaluate --p0 ckpt:latest --p1 greedy --games 100` (record the
   number; if not reached, record honestly and iterate on config — do not gate
   the codebase on training success, gate the *claim*).
3. Browser e2e: full game vs best checkpoint → replay file → open in Replay
   view with search panels (screenshot).
4. Pi: document setup (`docs/pi-setup.md`: install, `carcassonne train --resume`,
   systemd unit), run ≥ 2 hours, interrupt, resume, show metrics continuity.

---

## Dependency graph (parallelism for subagents)

```
T1 → T2 → T3 → T4 ┬→ T6 → T7 ┬→ T8 → T9 → T10 ┬→ T11 → T12 → T13 ┬→ T17 → (T21)
              T5 ┘            └→ T14 → T15 ──→ T16 ┴──────────────┘
                                                └→ T18 → T19 → T20 → T21 → T22
```
Parallel opportunities: T4 ∥ T5; Phase 3 (T11–13) ∥ Phase 4 (T14–15);
T17 ∥ T18–20.

## Review
- [ ] Code review requested
- [ ] All feedback addressed
- [ ] Final verification passed
