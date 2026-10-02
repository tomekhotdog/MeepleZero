# Engine Hot Path (M18b) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use tomek-superpowers:build to implement this plan task-by-task.

**Goal:** Lift single-process self-play from 2.39× to ≥5× the iteration-1 baseline (≥717 games/hour) by removing redundant rules-engine work, with zero behaviour change.
**Architecture:** Profiling (2026-10-01, scratchpad `profile/`) showed the engine, not the net, dominates self-play: `legal_moves` 41%, `apply` 23% (mostly `draw_playable` → `placements_for_tile`), net forward 17%, `encode_state` 11%. Three fixes, prototyped as runtime patches with byte-identical trajectories, measured 1.96× combined: **D** table-driven placement enumeration, **C** meeple-claim check via neighbour roots instead of a full board+index copy per placement, **B** carry the drawn tile's placements on `GameState` so `legal_moves` doesn't recompute them. A golden behaviour digest recorded on the pre-optimisation engine guards every task.
**Tech Stack:** Python 3.12 (pure-Python engine in `carcassonne/core`), pytest.
**Design:** profile report in the mega plan's M18 entry + user decision 2026-10-02 ("proceed with finding some optimisations"); parent: `docs/plans/2026-09-20-mega-plan.md` (M18b).

**Invariant for every task:** the golden digest test (Task 1) must pass unchanged. A digest change is a behaviour change, never "expected" — stop and investigate.

**Machine note:** the dev Mac sleeps with the lid closed and stalls long runs; wrap runs in `caffeinate -dims`, never run two pytest processes at once, and benchmark only on an idle machine.

---

### Task 1: Golden engine digest + seeded bench + "before" measurement ✅ DONE (c206c19 + 06aabb8; before median 210.0 g/h under load — Task 5 re-measures interleaved)
**Depends on:** none

**Files:**
- Create: `tests/core/test_engine_golden.py`
- Modify: `carcassonne/cli/main.py` (`_bench`: seed the net)

**Step 1: Write the golden test (digest placeholder)**
```python
"""Golden behaviour digest of the rules engine (M18b guard).

Hot-path optimisations must not change any game. The digest covers every
legal_moves() result -- content AND order, since agents pick by index -- and the
resulting scores/supplies/tile/discards, across seeded random games at 2 and 4
players. Recorded on the pre-optimisation engine; a changed digest is a
behaviour change, not a speed-up.
"""

from __future__ import annotations

import hashlib
import random

import pytest

from carcassonne.core import GameConfig, apply, is_terminal, legal_moves, new_game

GOLDEN = {
    2: "FILL_ME_STEP_2",
    4: "FILL_ME_STEP_2",
}


def _digest(num_players: int, seeds: range) -> str:
    h = hashlib.sha256()
    for seed in seeds:
        rng = random.Random(seed)
        state = new_game(seed, GameConfig(num_players=num_players))
        while not is_terminal(state):
            moves = legal_moves(state)
            h.update(repr(moves).encode())
            state = apply(state, rng.choice(moves))
            h.update(
                repr(
                    (state.scores, state.meeples, state.abbots, state.abbot_at,
                     state.current_tile, state.discarded, state.last_events)
                ).encode()
            )
    return h.hexdigest()


@pytest.mark.parametrize("num_players", [2, 4])
def test_engine_behaviour_is_unchanged(num_players: int) -> None:
    assert _digest(num_players, range(12)) == GOLDEN[num_players]
```
(Verify `carcassonne.core` exports these names; adjust imports to the real public API, not semantics.)

**Step 2: Record the digests on the CURRENT engine (before any optimisation)**
Run:
```bash
caffeinate -dims uv run python -c "from tests.core.test_engine_golden import _digest; print(_digest(2, range(12))); print(_digest(4, range(12)))"
```
Paste the two hex digests into `GOLDEN`. Then run twice to prove determinism:
`caffeinate -dims uv run pytest tests/core/test_engine_golden.py -q` → Expected: 2 passed (both runs).
Sanity mutation: temporarily swap two moves' order inside `legal_moves` (e.g. reverse the tuple) → test FAILS; restore exactly.

**Step 3: Seed the bench net.** In `carcassonne/cli/main.py` `_bench`, immediately before `net = CarcassonneNet(...)`:
```python
    torch.manual_seed(args.seed)  # same untrained net every run: bench compares code, not nets
```
(The profiler found the unseeded net made runs play different games: ±15% run-to-run.)

**Step 4: "Before" measurement** (idle machine, lid open):
```bash
for i in 1 2 3; do caffeinate -dims uv run python -m carcassonne.cli.main bench --games 6 --sims 16 --seed 0; done
```
Record all three lines + median in the dynamic-crop plan's Results section under a new "M18b engine hot path" heading as **before**.

**Step 5: Commit**
```bash
git add tests/core/test_engine_golden.py carcassonne/cli/main.py docs/plans/2026-09-19-dynamic-crop-nn-plan.md
git commit -m "test(core): golden engine digest; seeded bench (M18b baseline)"
```

---

### Task 2: D — table-driven `placements_for_tile` ✅ DONE (fac5ab7; precomputed _EDGES table + tuple frontier; equivalence vs fits()-reference on all tile types; golden digest unchanged; micro-bench 2495→379 µs/call)
**Depends on:** Task 1

**Files:**
- Modify: `carcassonne/core/placement.py`
- Test: `tests/core/test_placement.py` (add; create if absent)

**Step 1: Write the failing equivalence test** (reference = the current implementation, copied verbatim into the test):
```python
import random

from carcassonne.core import GameConfig, apply, is_terminal, legal_moves, new_game
from carcassonne.core.placement import fits, frontier, placements_for_tile
from carcassonne.core.tiles import TILE_TYPES
from carcassonne.core.types import Rotation


def _reference_placements(board, tt):
    return [(p, r) for p in sorted(frontier(board)) for r in Rotation if fits(board, tt, p, r)]


def test_placements_match_reference_for_every_tile_type_on_real_boards() -> None:
    for seed in range(4):
        rng = random.Random(seed)
        state = new_game(seed, GameConfig())
        n = 0
        while not is_terminal(state):
            if n % 6 == 0:  # sample boards across the whole game
                for tt in TILE_TYPES.values():
                    assert placements_for_tile(state.board, tt) == _reference_placements(state.board, tt)
            state = apply(state, rng.choice(legal_moves(state)))
            n += 1


def test_placement_table_is_used(monkeypatch) -> None:
    """The hot path must not fall back to per-rotation fits() calls."""
    import carcassonne.core.placement as pl
    monkeypatch.setattr(pl, "fits", lambda *a: (_ for _ in ()).throw(AssertionError("fits called")))
    state = new_game(0, GameConfig())
    assert placements_for_tile(state.board, TILE_TYPES[state.current_tile])
```

**Step 2: Run to verify** — `caffeinate -dims uv run pytest tests/core/test_placement.py -q` → Expected: first test PASSES (same impl), `test_placement_table_is_used` FAILS ("fits called").

**Step 3: Implement** — in `carcassonne/core/placement.py`, keep `frontier` and `fits` (used by `apply` validation and tests), replace `placements_for_tile`:
```python
_DELTAS = ((0, 1), (1, 0), (0, -1), (-1, 0))  # indexed by Side; must match Pos.neighbor
_ROTATIONS = tuple(Rotation)

# _EDGES[type_id][rotation][world_side] -> EdgeKind: world_edge() for every tile
# type and rotation, precomputed once (placement enumeration is the self-play
# hot path -- it runs on every MCTS expansion).
_EDGES = {
    tid: tuple(tuple(world_edge(tt, r, s) for s in Side) for r in Rotation)
    for tid, tt in TILE_TYPES.items()
}


def placements_for_tile(board: Board, tt: TileType) -> list[tuple[Pos, Rotation]]:
    """All legal placements, deterministically ordered by (Pos, Rotation).

    Same result as filtering frontier() x Rotation through fits(), but with one
    neighbour lookup per frontier cell and side, and edges read from _EDGES."""
    occupied = {(p.x, p.y): placed for p, placed in board.items()}
    front = {
        (x + dx, y + dy)
        for (x, y) in occupied
        for dx, dy in _DELTAS
        if (x + dx, y + dy) not in occupied
    }
    mine = _EDGES[tt.id]
    out: list[tuple[Pos, Rotation]] = []
    for x, y in sorted(front):  # (x, y) tuple order == Pos order (order=True, fields x, y)
        need = []
        for side, (dx, dy) in enumerate(_DELTAS):
            placed = occupied.get((x + dx, y + dy))
            if placed is not None:
                need.append((side, _EDGES[placed.type_id][placed.rotation][(side + 2) % 4]))
        pos = Pos(x, y)
        for r in _ROTATIONS:
            edges = mine[r]
            if all(edges[s] is kind for s, kind in need):
                out.append((pos, r))
    return out
```
(Verify `TileType.id` exists and keys `TILE_TYPES`; verify `Side` iteration order is N,E,S,W = 0..3 and `_DELTAS` matches `Pos.neighbor`.)

**Step 4: Verify** — `caffeinate -dims uv run pytest tests/core -q` → Expected: all pass, including `test_engine_golden.py` (digest unchanged).

**Step 5: Commit**
```bash
git add carcassonne/core/placement.py tests/core/test_placement.py
git commit -m "perf(core): table-driven placement enumeration (golden digest unchanged)"
```

---

### Task 3: C — meeple-claim check without integrating the tile ✅ DONE (_claimed_on_placement via neighbour roots, no _integrated in legal_moves; equals reference on 12 games incl. 3346 claimed-feature cases; sibling-bridge tile invariant pinned by test; golden digest unchanged; micro-bench 1587→563 µs/call)
**Depends on:** Task 2

**Files:**
- Modify: `carcassonne/core/engine.py` (`legal_moves`, new `_claimed_on_placement`)
- Test: `tests/core/test_engine.py` (add)

**Step 1: Write the failing tests** (reference = current `legal_moves`, copied into the test module as `_reference_legal_moves` using `engine._integrated` / `engine._meeple_legal`, which stay because `apply` uses them):
```python
import random

import carcassonne.core.engine as engine
from carcassonne.core import GameConfig, apply, is_terminal, legal_moves, new_game
from carcassonne.core.tiles import TILE_TYPES
from carcassonne.core.types import MeepleKind, Move, PlaceMeeple, RetrieveAbbot


def _reference_legal_moves(state):
    if state.current_tile is None:
        return ()
    tt = TILE_TYPES[state.current_tile]
    p = state.current_player
    any_piece = state.meeples[p] > 0 or state.abbots[p]
    can_retrieve = state.abbot_at[p] is not None
    moves = []
    for pos, rot in engine.placements_for_tile(state.board, tt):
        moves.append(Move(pos, rot, None))
        if any_piece:
            _, idx = engine._integrated(state.board, state.features, state.current_tile, pos, rot)
            for i in range(len(tt.features)):
                for kind in (MeepleKind.MEEPLE, MeepleKind.ABBOT):
                    if engine._meeple_legal(state, idx, tt, pos, i, kind):
                        moves.append(Move(pos, rot, PlaceMeeple(i, kind)))
        if can_retrieve:
            moves.append(Move(pos, rot, RetrieveAbbot()))
    return tuple(moves)


def _withheld_city_road_options(state, moves) -> int:
    """Placements where a city/road meeple option is absent although the player
    has meeples -- i.e. the feature would join one that is already claimed."""
    if state.current_tile is None or state.meeples[state.current_player] == 0:
        return 0
    tt = TILE_TYPES[state.current_tile]
    present = set(moves)
    return sum(
        1
        for m in moves
        if m.action is None
        for i, f in enumerate(tt.features)
        if f.kind in (FeatureKind.CITY, FeatureKind.ROAD)
        and Move(m.pos, m.rotation, PlaceMeeple(i, MeepleKind.MEEPLE)) not in present
    )


def test_legal_moves_match_reference_incl_claimed_features() -> None:
    withheld = 0
    for seed in range(6):
        for n in (2, 3):
            rng = random.Random(seed)
            state = new_game(seed, GameConfig(num_players=n))
            while not is_terminal(state):
                got, want = legal_moves(state), _reference_legal_moves(state)
                assert got == want
                withheld += _withheld_city_road_options(state, want)
                state = apply(state, rng.choice(got))
    # The fixture must exercise the claimed-feature path, or equality proves nothing.
    assert withheld > 50, f"only {withheld} claimed-feature cases; add seeds"


def test_legal_moves_never_copies_the_board(monkeypatch) -> None:
    monkeypatch.setattr(engine, "_integrated", lambda *a: (_ for _ in ()).throw(AssertionError("copied")))
    state = new_game(0, GameConfig())
    assert legal_moves(state)
```
(Add `from carcassonne.core.types import FeatureKind` to the imports. Verify `Move` field names (`pos`, `rotation`, `action`) and that `Move` is hashable; if the `> 50` floor isn't met on the first run, print the count and set the floor to about half of the observed value.)

**Step 2: Run** — `caffeinate -dims uv run pytest tests/core/test_engine.py -q -k "reference or copies"` → Expected: reference test PASSES (same impl), `never_copies` FAILS ("copied").

**Step 3: Implement** — replace `legal_moves` in `engine.py`, add the helper (import `rotated_feature_sides` from `carcassonne.core.tiles`):
```python
def legal_moves(state: GameState) -> tuple[Move, ...]:
    """All legal moves, deterministically ordered: (pos, rot) as produced by
    placements_for_tile; actions None, PlaceMeeple by (feature, MEEPLE < ABBOT),
    RetrieveAbbot last."""
    if state.current_tile is None:
        return ()
    tt = TILE_TYPES[state.current_tile]
    p = state.current_player
    has_meeple = state.meeples[p] > 0
    has_abbot = state.abbots[p]
    can_retrieve = state.abbot_at[p] is not None
    moves: list[Move] = []
    for pos, rot in placements_for_tile(state.board, tt):
        moves.append(Move(pos, rot, None))
        if has_meeple or has_abbot:
            for i, f in enumerate(tt.features):
                claimed: bool | None = None  # computed lazily, at most once per feature
                for kind in (MeepleKind.MEEPLE, MeepleKind.ABBOT):
                    if f.kind not in _ALLOWED_ON[kind]:
                        continue
                    if kind is MeepleKind.MEEPLE and not has_meeple:
                        continue
                    if kind is MeepleKind.ABBOT and not has_abbot:
                        continue
                    if claimed is None:
                        claimed = _claimed_on_placement(state.board, state.features, f, pos, rot)
                    if not claimed:
                        moves.append(Move(pos, rot, PlaceMeeple(i, kind)))
        if can_retrieve:
            moves.append(Move(pos, rot, RetrieveAbbot()))
    return tuple(moves)


def _claimed_on_placement(
    board: Board, features: FeatureIndex, f: TileFeature, pos: Pos, rot: Rotation
) -> bool:
    """Would feature `f` of a tile placed at (pos, rot) join a feature that already
    holds a meeple?

    Equivalent to integrating the tile and reading the merged root's meeples
    (`_meeple_legal` on `_integrated`'s index): FeatureIndex.with_tile unions a new
    city/road node only with the matching neighbour feature across each of its
    sides, and monasteries/gardens never merge -- so the merged root's meeples are
    exactly the union of those neighbours' roots' meeples. No board/index copy."""
    if f.kind not in (FeatureKind.CITY, FeatureKind.ROAD):
        return False
    for side in rotated_feature_sides(f, rot):
        npos = pos.neighbor(side)
        nplaced = board.get(npos)
        if nplaced is None:
            continue
        for j, nf in enumerate(TILE_TYPES[nplaced.type_id].features):
            if side.opposite in rotated_feature_sides(nf, nplaced.rotation):
                if features.root_data((npos, j)).meeples:
                    return True
                break
    return False
```
(Import `TileFeature` from `carcassonne.core.tiles`. `_integrated`/`_meeple_legal` stay — `apply` validates the chosen move through them.)

**Step 4: Verify** — `caffeinate -dims uv run pytest tests/core tests/agents -q` → Expected: all pass incl. golden digest.

**Step 5: Commit**
```bash
git add carcassonne/core/engine.py tests/core/test_engine.py
git commit -m "perf(core): meeple-claim check via neighbour roots, no per-placement board copy (digest unchanged)"
```

---

### Task 4: B — carry the drawn tile's placements on GameState (self-validating)
**Depends on:** Task 3

**Files:**
- Modify: `carcassonne/core/engine.py` (`GameState`, `draw_playable`, `apply`, `legal_moves`), `carcassonne/core/game.py` (`new_game`)
- Test: `tests/core/test_engine.py` (add)

**Hazard (why self-validating):** `learner.rotate_state_90` builds rotated states with `dataclasses.replace(state, board=..., features=..., abbot_at=...)`, and `GreedyAgent` uses `replace(state, deck=())`. A plain cached field would be copied onto a rotated state → wrong placements, silently. The cache therefore records the board object and tile it was computed for, and is ignored unless both still match.

**Step 1: Write the failing tests**
```python
import dataclasses

import carcassonne.core.engine as engine


def _counting(monkeypatch):
    calls = []
    orig = engine.placements_for_tile
    monkeypatch.setattr(engine, "placements_for_tile", lambda b, t: calls.append(1) or orig(b, t))
    return calls


def test_legal_moves_reuses_placements_computed_while_drawing(monkeypatch) -> None:
    s0 = new_game(0, GameConfig())
    s1 = apply(s0, legal_moves(s0)[0])
    calls = _counting(monkeypatch)
    legal_moves(s1)
    legal_moves(s0)  # new_game also carries them
    assert calls == []


def test_replaced_board_invalidates_the_carried_placements(monkeypatch) -> None:
    s0 = new_game(0, GameConfig())
    s1 = apply(s0, legal_moves(s0)[0])
    calls = _counting(monkeypatch)
    s2 = dataclasses.replace(s1, board=dict(s1.board))  # new board object, same content
    assert legal_moves(s2) == legal_moves(s1)
    assert len(calls) == 1  # s2 recomputed; s1 reused


def test_carried_placements_never_affect_equality() -> None:
    s0 = new_game(0, GameConfig())
    assert s0 == dataclasses.replace(s0, placements_cache=None)
```

**Step 2: Run** — `caffeinate -dims uv run pytest tests/core/test_engine.py -q -k "reuses or invalidates or equality"` → Expected: FAIL (`placements_cache` unknown / calls non-empty).

**Step 3: Implement**
- `engine.py` imports: `from dataclasses import dataclass, field`; `from collections.abc import Sequence`.
- `GameState`: add as the LAST field:
```python
    # Placements of current_tile on `board`, computed while drawing it (apply /
    # new_game) so legal_moves needn't recompute them. Self-validating: used only
    # while the recorded board object and tile id still match -- a
    # dataclasses.replace() with a new board (learner.rotate_state_90) leaves it
    # stale, and it is then ignored. compare=False: never part of state equality
    # (MCTS subtree reuse matches states with ==).
    placements_cache: tuple[Board, str, tuple[tuple[Pos, Rotation], ...]] | None = field(
        default=None, compare=False, repr=False
    )
```
- New helper, used by `legal_moves` in place of `placements_for_tile(state.board, tt)`:
```python
def _current_placements(state: GameState) -> Sequence[tuple[Pos, Rotation]]:
    """Placements of the current tile -- carried from the draw when still valid."""
    cache = state.placements_cache
    if cache is not None and cache[0] is state.board and cache[1] == state.current_tile:
        return cache[2]
    assert state.current_tile is not None  # callers handle the terminal state first
    return placements_for_tile(state.board, TILE_TYPES[state.current_tile])
```
- `draw_playable` returns the drawn tile's placements too (core-internal API):
```python
def draw_playable(
    board: Board, deck: tuple[str, ...]
) -> tuple[str | None, tuple[str, ...], tuple[str, ...], tuple[tuple[Pos, Rotation], ...]]:
    """Draw until a playable tile appears (core-internal, not part of the public API).

    Returns (drawn tile or None, remaining deck, unplaceable tiles set aside,
    the drawn tile's placements -- () when the deck ran out)."""
    rest = list(deck)
    set_aside: list[str] = []
    while rest:
        candidate = rest.pop(0)
        placements = placements_for_tile(board, TILE_TYPES[candidate])
        if placements:
            return candidate, tuple(rest), tuple(set_aside), tuple(placements)
        set_aside.append(candidate)
    return None, (), tuple(set_aside), ()
```
- `apply`: `current, deck, set_aside, placements = draw_playable(board, state.deck)` and pass `placements_cache=(board, current, placements) if current is not None else None` to `GameState(...)`.
- `game.new_game`: same unpacking and `placements_cache=(board, current, placements) if current is not None else None`.
- grep for every other `draw_playable(` caller (carcassonne/ and tests/) and update the unpacking.

**Step 4: Verify**
- `caffeinate -dims uv run pytest tests/core tests/agents tests/training/test_learner.py -q` → Expected: all pass, incl. golden digest AND the learner's rotation-bijection test (`test_augmented_target_consistent_with_engine` — it calls legal_moves on `rotate_state_90` states, i.e. the stale-cache path).
- Full suite once, background, logged: `caffeinate -dims uv run pytest -q > <scratchpad>/m18b-full.log 2>&1` → Expected: all pass, 0 skipped.
- `uv run ruff check . && uv run mypy carcassonne` → clean.

**Step 5: Commit**
```bash
git add carcassonne/core/engine.py carcassonne/core/game.py tests/core/test_engine.py
git commit -m "perf(core): carry the drawn tile's placements on GameState (self-validating; digest unchanged)"
```

---

### Task 5: "After" measurement + decision
**Depends on:** Task 4

**Files:**
- Modify: `docs/plans/2026-09-19-dynamic-crop-nn-plan.md` (Results → "M18b engine hot path")

**Steps:**
1. **Interleaved A/B bench** (Task 1's "before" was taken under load — Chrome ~60% of a core, one run 30% slow — so before/after must be measured in the same session). Export the pre-optimisation tree without touching git state: `mkdir -p <scratchpad>/before && git archive c206c19 | tar -x -C <scratchpad>/before`, then alternate 3× each: `(cd <scratchpad>/before && caffeinate -dims uv run python -m carcassonne.cli.main bench --games 6 --sims 16 --seed 0)` and the same in the repo (HEAD). Record all six lines, medians, and speedup = after/before; then vs the 143.4 baseline (bar ≥717 games/hour → PASS/FAIL stated plainly). Before starting, record `ps -Ao %cpu,command | sort -rn | head -5` to document machine load.
2. Worker scaling re-check (same command as M18: `train --sims 8 --selfplay-games 8 --learn-steps 1 --iterations 1 --workers 1` vs `--workers 4`, fresh scratch run dirs) → record.
3. If FAIL: do NOT optimise further in this task. Report the remaining profile split; the orchestrator plans the next step (candidates: incremental `encode_state` ≈1.2×, leaf batching ≈1.15×).
4. Commit:
```bash
git add docs/plans/2026-09-19-dynamic-crop-nn-plan.md
git commit -m "docs: M18b after-bench (engine hot path)"
```

---

## Review
- [ ] Code review requested (two-stage per task)
- [ ] All feedback addressed
- [ ] Final verification passed (golden digest unchanged; full suite green; after-bench recorded)
