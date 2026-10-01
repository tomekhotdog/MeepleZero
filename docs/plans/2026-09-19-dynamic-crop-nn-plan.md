# Dynamic-Crop NN Representation & Training Throughput — Implementation Plan

> **⚠️ ORCHESTRATED BY THE MEGA PLAN — do not execute this file standalone.**
> Task order, dependencies, and merge deltas live in
> `docs/plans/2026-09-20-mega-plan.md` (this plan's tasks are referenced there
> as DC-T1…DC-T10). The mega plan runs the multiplayer engine widening FIRST,
> so this plan's "targets the 2-player engine" caveat is resolved by its
> M8/M10 deltas (2-player guards folded into the encoder and MctsAgent).

> **For Claude:** REQUIRED SUB-SKILL: Use tomek-superpowers:build to implement this plan task-by-task.

**Goal:** Replace the fixed 31×31 window with a per-state dynamic Crop, shrink the net to a fully-convolutional ~190k params, key actions by world coordinates, and parallelise self-play — ≥5× single-process throughput, ~linear 4-worker scaling.
**Architecture:** `nn/actions.py` grows a `Crop` type and a world-coordinate `(x, y, rot, a)` Key codec; `nn/encode.py` emits `(52, H, W)` planes over the crop (incl. FeaturePlanes from `FeatureIndex`); `nn/model.py` becomes size-agnostic (MaskedGroupNorm, masked global-average-pool value head); the learner pads batches to the batch max; `training/pool.py` adds a spawn-based self-play worker pool; `agents/mcts.py` gains subtree reuse (sound because `apply` draws from the in-state deck — search is determinized).
**Tech Stack:** Python 3.12, numpy 1.26, torch 2.2 (CPU), sqlite, multiprocessing (spawn).
**Design:** `docs/plans/2026-09-19-dynamic-crop-nn-design.md`

**Ordering caveats:**
- This plan targets the **current 2-player engine**. The iteration-2 multiplayer-engine plan (`2026-08-23-multiplayer-engine-plan.md`) is designed but NOT yet implemented. If that plan executes first, re-review Tasks 3, 6, 7 (me/opp indexing, `scores` tuple width, `_winner`).
- Tasks 2–7 are one representation switchover: each task's own tests pass at that task, but the **full suite is only green again at the end of Task 7**. Commit per task regardless.
- Clean break: `runs/mac-demo` checkpoints/buffer become unloadable. Do not migrate; do not delete (user data).
- Arena/gate games stay sequential this iteration (self-play dominates: 20 games/iter every iter vs 40 gate games every 5). Noted as follow-up.

**Constants used throughout:** `ROTS=4`, `ACTIONS=14`, `ACTIONS_PER_CELL=56`. Flat index = `((gy*W + gx)*4 + rot)*14 + a` where `W` is the stride width (crop width, or padded batch width). Key = `(x, y, rot, a)` in world coords. New net defaults: `channels=48`, `n_blocks=4`, `NUM_PLANES=52` → 192,489 params (verify in Task 4; budget ≤250k).

---

### Task 1: Bench harness + 31×31 baseline
**Depends on:** none

**Files:**
- Modify: `carcassonne/cli/main.py`
- Test: `tests/cli/test_bench.py` (create)
- Modify (record baseline): this plan file

**Step 1: Write the failing test**

```python
# tests/cli/test_bench.py
"""`carcassonne bench` times self-play games — the before/after throughput yardstick."""
from carcassonne.cli.main import main


def test_bench_runs_and_reports(capsys) -> None:
    rc = main(["bench", "--games", "1", "--sims", "2", "--channels", "16", "--n-blocks", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "games/hour=" in out
    assert "s/game=" in out
    assert "params=" in out
```

**Step 2: Run test to verify it fails**
Run: `uv run pytest tests/cli/test_bench.py -q`
Expected: FAIL (`KeyError: 'bench'` or argparse error)

**Step 3: Implement**

In `_build_parser()` (after the `train` parser block):

```python
    be = sub.add_parser("bench", help="time self-play games (throughput yardstick)")
    be.add_argument("--games", type=int, default=4, help="games to play (default 4)")
    be.add_argument("--sims", type=int, default=32, help="MCTS sims per move (default 32)")
    be.add_argument("--channels", type=int, default=48, help="network channels")
    be.add_argument("--n-blocks", type=int, default=4, help="residual blocks")
    be.add_argument("--seed", type=int, default=0, help="seed of game 0; game i uses seed+i")
```

Register `"bench": _bench` in the `handlers` dict, and add:

```python
def _bench(args: argparse.Namespace) -> int:
    """Time sequential self-play with a fresh (untrained) net on CPU.

    Untrained-net throughput is representative: per-move cost is sims x one
    net evaluation regardless of weights. Uses a throwaway run dir for replays.
    """
    import tempfile
    import time

    import torch

    from carcassonne.agents.mcts import MctsConfig
    from carcassonne.nn.model import CarcassonneNet, param_count
    from carcassonne.training.run import TrainingRun
    from carcassonne.training.selfplay import play_selfplay_game

    net = CarcassonneNet(channels=args.channels, n_blocks=args.n_blocks)
    net.eval()
    device = torch.device("cpu")
    with tempfile.TemporaryDirectory() as td:
        run = TrainingRun.create(Path(td) / "bench", {"bench": True})
        t0 = time.perf_counter()
        for i in range(args.games):
            play_selfplay_game(
                net=net,
                device=device,
                seed=args.seed + i,
                run=run,
                mcts_config=MctsConfig(sims=args.sims),
                game_id=f"bench{i:04d}",
                started_at="bench",
                order_key=i,
                buffer=None,
            )
        dt = time.perf_counter() - t0
    print(f"params={param_count(net)}")
    print(
        f"games={args.games} sims={args.sims} channels={args.channels} "
        f"n_blocks={args.n_blocks} total={dt:.1f}s s/game={dt / args.games:.1f} "
        f"games/hour={args.games / dt * 3600:.1f}"
    )
    return 0
```

(Note: Task 6 changes `play_selfplay_game`'s signature; it updates this call — listed there.)

**Step 4: Run test to verify it passes**
Run: `uv run pytest tests/cli/test_bench.py -q`
Expected: PASS

**Step 5: Record the 31×31 baseline** (old net defaults, small sims so it finishes in minutes on the Mac):
Run: `uv run carcassonne bench --games 2 --sims 16 --channels 64 --n-blocks 5 --seed 0`
Paste the two output lines into the **Baseline** section at the bottom of this plan file.

**Step 6: Commit**
```bash
git add carcassonne/cli/main.py tests/cli/test_bench.py docs/plans/2026-09-19-dynamic-crop-nn-plan.md
git commit -m "feat(cli): bench command + 31x31 throughput baseline"
```

---

### Task 2: `nn/actions.py` — Crop + world-coordinate Key codec
**Depends on:** none

**Files:**
- Rewrite: `carcassonne/nn/actions.py`
- Rewrite: `tests/nn/test_actions.py`

Breaks importers (`model`, `checkpoint`, `mcts`, `selfplay`, `learner`) until Tasks 4–7; only run this task's test file.

**Step 1: Write the failing tests**

```python
# tests/nn/test_actions.py
"""Crop geometry and the world-coordinate Move codec.

Correctness backbone: for ANY reachable state, the crop contains every legal
move, and encode/decode is a bijection between legal moves and flat indices.
"""
from __future__ import annotations

import random

import pytest

from carcassonne.core import GameConfig, apply, is_terminal, legal_moves, new_game
from carcassonne.core.types import MeepleKind, Move, PlaceMeeple, Pos, Rotation, RulesError
from carcassonne.nn.actions import (
    ACTIONS_PER_CELL,
    Crop,
    decode_move,
    encode_move,
    flat_index,
    key_move,
    legal_mask,
    move_key,
    rotate_key,
)


def _random_states(seed: int, n_states: int = 30):
    """Yield states sampled from a random playthrough (fresh game per exhaustion)."""
    rng = random.Random(seed)
    state = new_game(rng.randrange(10_000), GameConfig())
    for _ in range(n_states):
        if is_terminal(state):
            state = new_game(rng.randrange(10_000), GameConfig())
        yield state
        state = apply(state, rng.choice(legal_moves(state)))


def test_crop_is_bbox_plus_one_border() -> None:
    state = new_game(0, GameConfig())  # single start tile at (0, 0)
    crop = Crop.of(state)
    assert (crop.ox, crop.oy, crop.h, crop.w) == (-1, -1, 3, 3)
    assert crop.n_actions == 3 * 3 * ACTIONS_PER_CELL


def test_crop_contains_every_legal_move() -> None:
    for state in _random_states(1, n_states=60):
        crop = Crop.of(state)
        for m in legal_moves(state):
            gx, gy = m.pos.x - crop.ox, m.pos.y - crop.oy
            assert 0 <= gx < crop.w and 0 <= gy < crop.h


def test_encode_decode_roundtrip_all_legal_moves() -> None:
    for state in _random_states(2, n_states=40):
        for m in legal_moves(state):
            assert decode_move(state, encode_move(state, m)) == m


def test_legal_mask_bijection() -> None:
    """mask True exactly at the encodings of legal moves — no drops, no ghosts."""
    for state in _random_states(3, n_states=40):
        crop = Crop.of(state)
        mask = legal_mask(state)
        assert mask.shape == (crop.n_actions,)
        expected = {encode_move(state, m) for m in legal_moves(state)}
        got = {int(i) for i in mask.nonzero()[0]}
        assert got == expected


def test_padded_stride_indexing() -> None:
    """flat_index with a wider stride matches manual arithmetic (learner padding)."""
    state = new_game(0, GameConfig())
    crop = Crop.of(state)  # 3x3
    m = legal_moves(state)[0]
    key = move_key(m)
    idx = flat_index(crop, key, stride_w=10)
    gx, gy = key[0] - crop.ox, key[1] - crop.oy
    assert idx == ((gy * 10 + gx) * 4 + key[2]) * 14 + key[3]
    padded = legal_mask(state, pad_hw=(10, 10))
    assert padded.shape == (10 * 10 * ACTIONS_PER_CELL,)
    assert padded[idx]


def test_move_key_roundtrip_and_rotation() -> None:
    m = Move(Pos(3, -2), Rotation.R90, PlaceMeeple(2, MeepleKind.ABBOT))
    assert key_move(move_key(m)) == m
    k = move_key(m)
    k4 = k
    for _ in range(4):
        k4 = rotate_key(k4)
    assert k4 == k  # four quarter-turns are the identity
    assert rotate_key(k)[:3] == (-2, -3, 2)  # (x,y)->(y,-x), rot+1


def test_flat_index_raises_outside_crop() -> None:
    state = new_game(0, GameConfig())
    with pytest.raises(RulesError):
        flat_index(Crop.of(state), (99, 99, 0, 0))
```

**Step 2: Run to verify fail**
Run: `uv run pytest tests/nn/test_actions.py -q`
Expected: FAIL (ImportError: `Crop`)

**Step 3: Rewrite `carcassonne/nn/actions.py`** (complete file):

```python
"""World-coordinate ``Move`` codec + the per-state dynamic **Crop**.

Iteration 3 removed the fixed 31x31 window. The board is encoded over a
per-state Crop: the bounding box of placed tiles plus a 1-cell border. A new
tile can only be placed adjacent to an existing one, so the border provably
contains every legal placement -- every legal move is always representable
(the old "silently dropped out-of-window move" case is impossible).

Two vocabularies:

* **Key** ``(x, y, rot, a)`` -- world coordinates, the engine's own vocabulary.
  Stable under any crop; this is what the replay buffer stores.
* **flat index** ``((gy*W + gx) * 4 + rot) * 14 + a`` with ``gx = x - crop.ox``,
  ``gy = y - crop.oy`` -- the policy head's per-state layout. ``W`` is the
  *stride width*: the crop's own width, or the padded width in a learner batch.

``a`` selects the meeple decision on the freshly placed tile::

    a == 0            -> action None (place tile, no piece)
    1 <= a <= 6       -> PlaceMeeple(feature = a - 1, MEEPLE)   feature 0..5
    7 <= a <= 12      -> PlaceMeeple(feature = a - 7, ABBOT)    feature 0..5
    a == 13           -> RetrieveAbbot()
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from carcassonne.core.engine import GameState, legal_moves
from carcassonne.core.types import (
    MeepleKind,
    Move,
    PlaceMeeple,
    Pos,
    RetrieveAbbot,
    Rotation,
    RulesError,
    TurnAction,
)

ROTS = 4
ACTIONS = 14  # None + 6 meeple slots + 6 abbot slots + retrieve
ACTIONS_PER_CELL = ROTS * ACTIONS  # 56 policy channels per cell
_MAX_FEATURE = 5  # feature index must fit in the 6 slots reserved per meeple kind

type Key = tuple[int, int, int, int]  # (x, y, rot, a) in world coordinates


class Crop(NamedTuple):
    """Per-state board window: bounding box of placed tiles + 1-cell border."""

    ox: int  # world x mapped to grid column 0
    oy: int  # world y mapped to grid row 0
    h: int
    w: int

    @staticmethod
    def of(state: GameState) -> Crop:
        xs = [p.x for p in state.board]
        ys = [p.y for p in state.board]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        return Crop(minx - 1, miny - 1, (maxy - miny) + 3, (maxx - minx) + 3)

    @property
    def n_actions(self) -> int:
        return self.h * self.w * ACTIONS_PER_CELL


def _action_index(action: TurnAction) -> int:
    if action is None:
        return 0
    if isinstance(action, PlaceMeeple):
        if not 0 <= action.feature <= _MAX_FEATURE:
            raise RulesError(f"feature index {action.feature} out of action range")
        return (1 if action.kind is MeepleKind.MEEPLE else 7) + action.feature
    if isinstance(action, RetrieveAbbot):
        return 13
    raise RulesError(f"unknown action {action!r}")


def _decode_action(a: int) -> TurnAction:
    if a == 0:
        return None
    if 1 <= a <= 6:
        return PlaceMeeple(a - 1, MeepleKind.MEEPLE)
    if 7 <= a <= 12:
        return PlaceMeeple(a - 7, MeepleKind.ABBOT)
    if a == 13:
        return RetrieveAbbot()
    raise RulesError(f"action sub-index {a} out of range")


def move_key(move: Move) -> Key:
    """``Move`` -> world-coordinate Key (the buffer's storage format)."""
    return (move.pos.x, move.pos.y, int(move.rotation), _action_index(move.action))


def key_move(key: Key) -> Move:
    """Key -> ``Move`` (inverse of :func:`move_key`)."""
    x, y, rot, a = key
    return Move(Pos(x, y), Rotation(rot), _decode_action(a))


def rotate_key(key: Key, k: int = 1) -> Key:
    """Rotate a Key 90 degrees clockwise in world space, ``k`` times.

    Matches ``learner.rotate_state_90``: cell ``(x, y) -> (y, -x)`` and the tile
    rotation advances one quarter-turn. The sub-action ``a`` names a tile-local
    feature index, invariant under board rotation.
    """
    x, y, rot, a = key
    for _ in range(k % 4):
        x, y, rot = y, -x, (rot + 1) % 4
    return (x, y, rot, a)


def flat_index(crop: Crop, key: Key, stride_w: int | None = None) -> int:
    """Key -> flat policy index for ``crop``, using ``stride_w`` as row stride.

    ``stride_w`` defaults to ``crop.w``; the learner passes the padded batch
    width. Raises :class:`RulesError` if the key's cell is outside the crop --
    impossible for a legal move, so this failing is loud, never silent.
    """
    x, y, rot, a = key
    gx, gy = x - crop.ox, y - crop.oy
    if not (0 <= gx < crop.w and 0 <= gy < crop.h):
        raise RulesError(f"cell ({x}, {y}) outside crop {crop}")
    w = crop.w if stride_w is None else stride_w
    return ((gy * w + gx) * ROTS + rot) * ACTIONS + a


def encode_move(state: GameState, move: Move) -> int:
    """``Move`` -> flat index in ``[0, Crop.of(state).n_actions)``."""
    return flat_index(Crop.of(state), move_key(move))


def decode_move(state: GameState, idx: int) -> Move:
    """Flat index -> ``Move`` (inverse of :func:`encode_move`)."""
    crop = Crop.of(state)
    if not 0 <= idx < crop.n_actions:
        raise RulesError(f"action id {idx} out of range for crop {crop}")
    a = idx % ACTIONS
    rest = idx // ACTIONS
    rot = rest % ROTS
    rest //= ROTS
    gx = rest % crop.w
    gy = rest // crop.w
    return Move(Pos(gx + crop.ox, gy + crop.oy), Rotation(rot), _decode_action(a))


def legal_mask(
    state: GameState, crop: Crop | None = None, pad_hw: tuple[int, int] | None = None
) -> NDArray[np.bool_]:
    """Boolean mask, True at exactly the flat indices of legal moves.

    Shape ``(ph * pw * 56,)`` where ``(ph, pw)`` is ``pad_hw`` (the learner's
    padded batch geometry) or the crop's own ``(h, w)``. Never all-False for a
    non-terminal state: the crop contains every legal move by construction.
    """
    c = crop or Crop.of(state)
    ph, pw = pad_hw or (c.h, c.w)
    assert ph >= c.h and pw >= c.w, f"padding {(ph, pw)} smaller than crop {(c.h, c.w)}"
    mask = np.zeros(ph * pw * ACTIONS_PER_CELL, dtype=np.bool_)
    for move in legal_moves(state):
        mask[flat_index(c, move_key(move), stride_w=pw)] = True
    return mask
```

**Step 4: Run to verify pass**
Run: `uv run pytest tests/nn/test_actions.py -q`
Expected: PASS (other nn/training tests are known-red until Task 7)

**Step 5: Commit**
```bash
git add carcassonne/nn/actions.py tests/nn/test_actions.py
git commit -m "feat(nn)!: dynamic Crop + world-coordinate action codec (replaces 31x31 window)"
```

---

### Task 3: `nn/encode.py` — dynamic-crop planes with FeaturePlanes
**Depends on:** Task 2

**Files:**
- Rewrite: `carcassonne/nn/encode.py`
- Rewrite: `tests/nn/test_encode.py`

**Plane layout (52 total), fixed offsets:**

| offset | count | name(s) | content |
|---|---|---|---|
| 0 | 1 | `occupied` | tile present |
| 1 | 12 | `edge_{N,E,S,W}_{city,road,field}` | world-side edge kinds (as before) |
| 13 | 2 | `feat_monastery`, `feat_garden` | monastic presence (city/road presence dropped — derivable from edges) |
| 15 | 1 | `shield` | as before |
| 16 | 4 | `meeple_me/opp`, `abbot_me/opp` | as before |
| 20 | 1 | `monfill` | as before |
| 21 | 8 | `fp_{city,road}_{size,open,mine,theirs}` | **FeaturePlanes**: per-cell max over that kind's features touching the tile — `min(len(tiles),10)/10`, `min(open_edges,6)/6`, `min(my_meeples,3)/3`, `min(opp_meeples,3)/3`. (`completed` plane dropped: completed ⇔ open==0.) |
| 29 | 12 | `cur_edge_{N,E,S,W}_{city,road,field}` | current tile's R0 edges (`derived_edges`), broadcast over the crop — replaces the 30-plane tile-ID one-hot |
| 41 | 3 | `cur_monastery`, `cur_garden`, `cur_shield` | current tile flags, broadcast |
| 44 | 8 | `s_*` scalars | as before (scores/50, meeples/7, abbots, deck/71, turn/72) |

Broadcast planes fill **only the crop region** (`[:crop.h, :crop.w]`) — padding stays exactly zero (required for the padded==unpadded invariant in Task 4).

**Step 1: Write the failing tests** — rewrite `tests/nn/test_encode.py`. Keep the existing `_rot_pos`/`rotate_state_90`-style helpers (copy from current file lines 73–105 if defined locally, or import from `carcassonne.training.learner`). New/adapted tests:

```python
# tests/nn/test_encode.py (key tests; keep existing helper machinery)
from __future__ import annotations

import numpy as np

from carcassonne.core import GameConfig, apply, legal_moves, new_game
from carcassonne.nn.actions import Crop
from carcassonne.nn.encode import (
    NUM_PLANES,
    PLANES,
    _CUR0,
    _EDGE0,
    encode_state,
    valid_mask,
)
from carcassonne.training.learner import rotate_state_90


def _mid_state(seed: int = 7, n_moves: int = 12):
    import random
    rng = random.Random(seed)
    state = new_game(seed, GameConfig())
    for _ in range(n_moves):
        state = apply(state, rng.choice(legal_moves(state)))
    return state


def test_shapes_track_crop() -> None:
    state = new_game(0, GameConfig())
    crop = Crop.of(state)
    planes = encode_state(state)
    assert planes.shape == (NUM_PLANES, crop.h, crop.w) == (52, 3, 3)
    mid = _mid_state()
    c2 = Crop.of(mid)
    assert encode_state(mid).shape == (52, c2.h, c2.w)


def test_padded_encoding_zero_outside_crop() -> None:
    state = _mid_state()
    crop = Crop.of(state)
    ph, pw = crop.h + 3, crop.w + 2
    planes = encode_state(state, pad_hw=(ph, pw))
    assert planes.shape == (NUM_PLANES, ph, pw)
    assert np.array_equal(planes[:, : crop.h, : crop.w], encode_state(state))
    assert not planes[:, crop.h :, :].any()
    assert not planes[:, :, crop.w :].any()
    vm = valid_mask(crop, pad_hw=(ph, pw))
    assert vm.shape == (ph, pw)
    assert vm[: crop.h, : crop.w].all() and vm.sum() == crop.h * crop.w


def test_broadcast_planes_cover_exactly_the_crop() -> None:
    state = _mid_state()
    planes = encode_state(state)
    turn_plane = planes[PLANES.index("s_turn")]
    assert (turn_plane == state.turn / 72.0).all()  # constant across the whole crop


def test_featureplanes_report_connected_feature() -> None:
    """After 1 move on the start tile's city, fp_city_size reflects the merged size."""
    state = new_game(0, GameConfig())
    # Not asserting a specific board: property — every occupied cell whose tile
    # touches a city has fp_city_size > 0, and open==0 implies a completed city.
    mid = _mid_state()
    planes = encode_state(mid)
    occ = planes[PLANES.index("occupied")].astype(bool)
    fp_size = planes[PLANES.index("fp_city_size")]
    city_edge = planes[_EDGE0 : _EDGE0 + 12 : 3].sum(axis=0) > 0  # any city edge
    assert ((fp_size > 0) == (occ & city_edge)).all()


def test_d4_rotation_equivariance() -> None:
    """encode(rot90(state)) == spatial-rot90 of encode(state), with the 12
    board-edge planes side-permuted (N<-W, E<-N, S<-E, W<-S); current-tile
    planes (offsets >= _CUR0) and scalars are rotation-INVARIANT (tile in hand
    does not rotate with the board)."""
    state = _mid_state()
    planes = encode_state(state)
    rplanes = encode_state(rotate_state_90(state))
    # Spatial rot90: world (x,y)->(y,-x) means grid rot90 with k=-1 on (gy,gx) axes.
    expected = np.rot90(planes, k=-1, axes=(1, 2)).copy()
    # Permute the 12 edge planes: new world side N was old W, E was N, S was E, W was S.
    perm = [3, 0, 1, 2]  # index by new side -> old side (N,E,S,W = 0,1,2,3)
    edges = expected[_EDGE0 : _EDGE0 + 12].copy()
    for new_s in range(4):
        expected[_EDGE0 + new_s * 3 : _EDGE0 + new_s * 3 + 3] = edges[
            perm[new_s] * 3 : perm[new_s] * 3 + 3
        ]
    assert np.allclose(rplanes, expected)
```

(If the existing file's equivariance test machinery differs, adapt to it — the invariant is what matters: spatial planes rotate, edge sides permute, `cur_*`/scalar planes are unchanged. Note the crop of a rotated board has swapped h/w — `np.rot90` handles rectangles.)

**Step 2: Run to verify fail**
Run: `uv run pytest tests/nn/test_encode.py -q` — Expected: FAIL (ImportError)

**Step 3: Rewrite `carcassonne/nn/encode.py`** (complete file):

```python
"""``GameState`` -> ``float32`` planes over the dynamic :class:`Crop`.

Output shape is ``(NUM_PLANES, H, W)`` where ``H x W`` is the per-state crop
(or ``pad_hw`` when the learner pads a batch; padding rows/cols are all-zero,
including broadcast planes -- the model relies on that for its padded==unpadded
invariant). Indexed ``planes[c, gy, gx]``. Perspective: "me" is
``state.current_player``.

Layout (52 planes; see PLANES for names in order):

Spatial (21): occupied; 12 edge-kind one-hots (world side x {city,road,field});
monastery/garden presence; shield; meeple/abbot owner (me/opp); monfill.

FeaturePlanes (8): per-cell max over the {city, road} features touching the
tile, from the engine's FeatureIndex: normalised size, normalised open-edge
count, my/opp meeple count. Hands the net long-range connectivity that convs
cannot infer across snaky features. (Subsumes the old ``completed`` plane:
completed <=> open == 0.)

Current tile (15, broadcast over the crop): the drawn tile's R0 edge kinds
(12) + monastery/garden/shield flags -- structural properties, not a tile-ID
one-hot, so structurally identical tiles share representation.

Scalars (8, broadcast): scores/50, meeples/7, abbot flags, deck/71, turn/72.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from carcassonne.core.engine import GameState, neighbours8
from carcassonne.core.tiles import TILE_TYPES, derived_edges, world_edge
from carcassonne.core.types import EdgeKind, FeatureKind, MeepleKind, Side
from carcassonne.nn.actions import Crop

_SIDES = (Side.N, Side.E, Side.S, Side.W)
_EDGE_KINDS = (EdgeKind.CITY, EdgeKind.ROAD, EdgeKind.FIELD)
_FP_KINDS = (FeatureKind.CITY, FeatureKind.ROAD)

_SPATIAL_PLANES: list[str] = (
    ["occupied"]
    + [f"edge_{s.name}_{k.value}" for s in _SIDES for k in _EDGE_KINDS]
    + ["feat_monastery", "feat_garden", "shield"]
    + ["meeple_me", "meeple_opp", "abbot_me", "abbot_opp", "monfill"]
)
_FP_PLANES: list[str] = [
    f"fp_{k.value}_{q}" for k in _FP_KINDS for q in ("size", "open", "mine", "theirs")
]
_CUR_PLANES: list[str] = [f"cur_edge_{s.name}_{k.value}" for s in _SIDES for k in _EDGE_KINDS] + [
    "cur_monastery",
    "cur_garden",
    "cur_shield",
]
_SCALAR_PLANES: list[str] = [
    "s_my_score",
    "s_opp_score",
    "s_my_meeples",
    "s_opp_meeples",
    "s_my_abbot",
    "s_opp_abbot",
    "s_deck_remaining",
    "s_turn",
]

PLANES: list[str] = _SPATIAL_PLANES + _FP_PLANES + _CUR_PLANES + _SCALAR_PLANES
NUM_PLANES = len(PLANES)

assert len(_SPATIAL_PLANES) == 21, len(_SPATIAL_PLANES)
assert len(_FP_PLANES) == 8 and len(_CUR_PLANES) == 15, (len(_FP_PLANES), len(_CUR_PLANES))
assert NUM_PLANES == 52, NUM_PLANES

# Fixed plane offsets (kept in sync with PLANES by the asserts above).
_OCC = 0
_EDGE0 = 1  # 12 planes: side * 3 + kind
_MONASTERY = 13
_GARDEN = 14
_SHIELD = 15
_MEEPLE_ME = 16
_MEEPLE_OPP = 17
_ABBOT_ME = 18
_ABBOT_OPP = 19
_MONFILL = 20
_FP0 = 21  # 8 planes: kind * 4 + {size, open, mine, theirs}
_CUR0 = 29  # 12 current-tile edge planes + 3 flags
_CUR_FLAGS = 41
_SCALAR0 = 44

# FeaturePlane normalisation caps (clipped): sizes rarely exceed 10 tiles,
# open edges 6, meeples-per-player-per-feature 3 in real base-game play.
_SIZE_CAP = 10.0
_OPEN_CAP = 6.0
_MEEPLE_CAP = 3.0


def valid_mask(crop: Crop, pad_hw: tuple[int, int] | None = None) -> NDArray[np.float32]:
    """``(ph, pw)`` float mask: 1 inside the crop, 0 in padding. The model's
    MaskedGroupNorm/value-pool consume this so padding never contaminates stats."""
    ph, pw = pad_hw or (crop.h, crop.w)
    m = np.zeros((ph, pw), dtype=np.float32)
    m[: crop.h, : crop.w] = 1.0
    return m


def encode_state(
    state: GameState, crop: Crop | None = None, pad_hw: tuple[int, int] | None = None
) -> NDArray[np.float32]:
    """Encode ``state`` as ``(NUM_PLANES, ph, pw)`` float32 planes.

    Every legal move's cell is inside the crop by construction (bbox + 1), so
    nothing is ever silently dropped. Padding (beyond ``crop.h/w``) is all-zero.
    """
    c = crop or Crop.of(state)
    ph, pw = pad_hw or (c.h, c.w)
    assert ph >= c.h and pw >= c.w, f"padding {(ph, pw)} smaller than crop {(c.h, c.w)}"
    planes = np.zeros((NUM_PLANES, ph, pw), dtype=np.float32)
    me = state.current_player
    opp = 1 - me
    board = state.board

    for pos, placed in board.items():
        gx, gy = pos.x - c.ox, pos.y - c.oy
        tt = TILE_TYPES[placed.type_id]
        planes[_OCC, gy, gx] = 1.0
        for si, side in enumerate(_SIDES):
            ki = _EDGE_KINDS.index(world_edge(tt, placed.rotation, side))
            planes[_EDGE0 + si * 3 + ki, gy, gx] = 1.0
        has_monastic = False
        for i, f in enumerate(tt.features):
            if f.shield:
                planes[_SHIELD, gy, gx] = 1.0
            if f.kind is FeatureKind.MONASTERY:
                planes[_MONASTERY, gy, gx] = 1.0
                has_monastic = True
            elif f.kind is FeatureKind.GARDEN:
                planes[_GARDEN, gy, gx] = 1.0
                has_monastic = True
            elif f.kind in _FP_KINDS:
                d = state.features.root_data((pos, i))
                base = _FP0 + _FP_KINDS.index(f.kind) * 4
                mine = sum(1 for m in d.meeples if m.player == me)
                theirs = sum(1 for m in d.meeples if m.player == opp)
                vals = (
                    min(len(d.tiles), _SIZE_CAP) / _SIZE_CAP,
                    min(d.open_edges, _OPEN_CAP) / _OPEN_CAP,
                    min(mine, _MEEPLE_CAP) / _MEEPLE_CAP,
                    min(theirs, _MEEPLE_CAP) / _MEEPLE_CAP,
                )
                for j, v in enumerate(vals):  # max-aggregate over same-kind features
                    planes[base + j, gy, gx] = max(planes[base + j, gy, gx], v)
        if has_monastic:
            occupied_n = sum(1 for q in neighbours8(pos) if q in board)
            planes[_MONFILL, gy, gx] = occupied_n / 8.0

    for d in state.features.data.values():
        for m in d.meeples:
            mpos, _feat = m.node
            gx, gy = mpos.x - c.ox, mpos.y - c.oy
            if m.kind is MeepleKind.MEEPLE:
                plane = _MEEPLE_ME if m.player == me else _MEEPLE_OPP
            else:
                plane = _ABBOT_ME if m.player == me else _ABBOT_OPP
            planes[plane, gy, gx] = 1.0

    # Broadcast planes fill ONLY the crop region: padding must stay exactly zero.
    region = (slice(None, c.h), slice(None, c.w))
    if state.current_tile is not None:
        tt = TILE_TYPES[state.current_tile]
        for si, kind in enumerate(derived_edges(tt)):
            planes[(_CUR0 + si * 3 + _EDGE_KINDS.index(kind), *region)] = 1.0
        for f in tt.features:
            if f.kind is FeatureKind.MONASTERY:
                planes[(_CUR_FLAGS, *region)] = 1.0
            elif f.kind is FeatureKind.GARDEN:
                planes[(_CUR_FLAGS + 1, *region)] = 1.0
            if f.shield:
                planes[(_CUR_FLAGS + 2, *region)] = 1.0

    scalars = (
        state.scores[me] / 50.0,
        state.scores[opp] / 50.0,
        state.meeples[me] / 7.0,
        state.meeples[opp] / 7.0,
        1.0 if state.abbots[me] else 0.0,
        1.0 if state.abbots[opp] else 0.0,
        len(state.deck) / 71.0,
        state.turn / 72.0,
    )
    for j, v in enumerate(scalars):
        planes[(_SCALAR0 + j, *region)] = v

    return planes
```

**Step 4: Run to verify pass**
Run: `uv run pytest tests/nn/test_encode.py tests/nn/test_actions.py -q`
Expected: PASS

**Step 5: Commit**
```bash
git add carcassonne/nn/encode.py tests/nn/test_encode.py
git commit -m "feat(nn)!: dynamic-crop encoder with FeaturePlanes (52 planes)"
```

---

### Task 4: `nn/model.py` — fully-convolutional net (MaskedGroupNorm, pooled value head) + `nn/checkpoint.py`
**Depends on:** Task 3

**Files:**
- Rewrite: `carcassonne/nn/model.py`
- Modify: `carcassonne/nn/checkpoint.py`
- Rewrite: `tests/nn/test_model.py`

**Why MaskedGroupNorm (the padding-correctness crux):** BatchNorm's stats and plain GroupNorm's per-sample stats both include padding cells, and GN's affine β would make padded cells non-zero, bleeding into real cells through the next 3×3 conv. MaskedGroupNorm (a) computes stats over valid cells only and (b) multiplies its output by the cell mask — so padded cells stay exactly 0 at every layer, a conv at a real border cell sees the same zeros as implicit zero-padding, and **padded forward == unpadded forward exactly**. With `cell_mask=None` (single-state inference) it's plain `F.group_norm`.

**Step 1: Write the failing tests**

```python
# tests/nn/test_model.py
from __future__ import annotations

import random

import numpy as np
import torch

from carcassonne.core import GameConfig, apply, legal_moves, new_game
from carcassonne.nn.actions import Crop, encode_move
from carcassonne.nn.encode import NUM_PLANES, encode_state, valid_mask
from carcassonne.nn.model import CarcassonneNet, evaluate_state, param_count


def _mid_state(seed: int = 5, n_moves: int = 10):
    rng = random.Random(seed)
    state = new_game(seed, GameConfig())
    for _ in range(n_moves):
        state = apply(state, rng.choice(legal_moves(state)))
    return state


def _tiny_net() -> CarcassonneNet:
    torch.manual_seed(0)
    return CarcassonneNet(channels=16, n_blocks=2).eval()


def test_param_budget() -> None:
    assert param_count(CarcassonneNet()) <= 250_000  # design bar; expect ~192k


def test_policy_head_matches_action_indexer() -> None:
    """Logit at flat index i must come from cell (gy,gx) channel rot*14+a."""
    net = _tiny_net()
    state = _mid_state()
    crop = Crop.of(state)
    x = torch.from_numpy(encode_state(state)).unsqueeze(0)
    logits, _ = net(x)
    assert logits.shape == (1, crop.n_actions)
    for m in legal_moves(state)[:5]:
        i = encode_move(state, m)
        assert torch.isfinite(logits[0, i])


def test_forward_accepts_any_size() -> None:
    net = _tiny_net()
    for h, w in ((3, 3), (5, 9), (12, 7)):
        x = torch.randn(2, NUM_PLANES, h, w)
        logits, value = net(x)
        assert logits.shape == (2, h * w * 56)
        assert value.shape == (2,)
        assert (value.abs() <= 1).all()


def test_padded_forward_equals_unpadded() -> None:
    """THE invariant: padding + cell_mask changes nothing for the real cells."""
    net = _tiny_net()
    state = _mid_state()
    crop = Crop.of(state)
    ph, pw = crop.h + 4, crop.w + 3

    x = torch.from_numpy(encode_state(state)).unsqueeze(0)
    logits, value = net(x)

    xp = torch.from_numpy(encode_state(state, pad_hw=(ph, pw))).unsqueeze(0)
    cm = torch.from_numpy(valid_mask(crop, pad_hw=(ph, pw))).unsqueeze(0)
    logits_p, value_p = net(xp, cell_mask=cm)

    # Compare per real cell: reshape to (gy, gx, 56) grids.
    grid = logits.reshape(crop.h, crop.w, 56)
    grid_p = logits_p.reshape(ph, pw, 56)[: crop.h, : crop.w]
    assert torch.allclose(grid, grid_p, atol=1e-5)
    assert torch.allclose(value, value_p, atol=1e-5)


def test_mask_sets_illegal_to_neg_inf() -> None:
    from carcassonne.nn.actions import legal_mask

    net = _tiny_net()
    state = new_game(0, GameConfig())
    x = torch.from_numpy(encode_state(state)).unsqueeze(0)
    mask = torch.from_numpy(legal_mask(state)).unsqueeze(0)
    logits, _ = net(x, mask=mask)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    assert float(probs[~mask.squeeze(0)].sum()) == 0.0
    assert abs(float(probs.sum()) - 1.0) < 1e-5


def test_evaluate_state_bridge() -> None:
    net = _tiny_net()
    state = _mid_state()
    probs, value = evaluate_state(net, state, torch.device("cpu"))
    assert probs.shape == (Crop.of(state).n_actions,)
    assert abs(float(probs.sum()) - 1.0) < 1e-4
    assert -1.0 <= value <= 1.0
    legal = {encode_move(state, m) for m in legal_moves(state)}
    assert set(np.nonzero(probs)[0]).issubset(legal)
```

**Step 2: Run to verify fail** — `uv run pytest tests/nn/test_model.py -q` — Expected: FAIL

**Step 3: Rewrite `carcassonne/nn/model.py`** (complete file):

```python
"""Fully-convolutional residual policy/value CNN -- size-agnostic (iteration 3).

Reads the ``(NUM_PLANES, H, W)`` dynamic-crop encoding and emits, per state:

* **policy** -- one logit per flat action index, layout ``((gy*W+gx)*4+rot)*14+a``
  (see :mod:`carcassonne.nn.actions`); shape ``(N, H*W*56)`` for whatever H, W
  the input has.
* **value** in ``[-1, 1]`` (tanh), current-player perspective, via a masked
  global-average pool -- no Linear tied to a board size anywhere.

Normalisation is :class:`MaskedGroupNorm`: GroupNorm statistics over *valid*
cells only, output re-zeroed on padding. Invariant (tested): a batch-padded
forward pass equals the unpadded forward pass exactly on the real cells --
padding cells stay 0 through every layer, so a 3x3 conv at a real border cell
sees the same zeros as implicit zero-padding.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import nn

from carcassonne.core.engine import GameState
from carcassonne.nn.actions import ACTIONS, ACTIONS_PER_CELL, ROTS, Crop
from carcassonne.nn.encode import NUM_PLANES, encode_state

_VALUE_CHANNELS = 8
_VALUE_HIDDEN = 32
_DEFAULT_CHANNELS = 48
_DEFAULT_N_BLOCKS = 4


def _conv3x3(in_ch: int, out_ch: int) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)


def _gn_groups(channels: int) -> int:
    return 8 if channels % 8 == 0 else math.gcd(8, channels)


class MaskedGroupNorm(nn.Module):
    """GroupNorm whose statistics ignore padding cells.

    ``forward(x, cell_mask)``: with ``cell_mask=None`` this is exactly
    ``F.group_norm`` (single-state inference, no padding). With a ``(N, 1, H, W)``
    0/1 mask, mean/var are computed over valid cells only and the output is
    multiplied by the mask -- keeping padded cells exactly 0 (the affine bias
    would otherwise leak into real cells through the next conv).
    """

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.groups = _gn_groups(channels)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor, cell_mask: torch.Tensor | None = None) -> torch.Tensor:
        if cell_mask is None:
            return F.group_norm(x, self.groups, self.weight, self.bias, self.eps)
        n, c, h, w = x.shape
        g = self.groups
        xg = x.reshape(n, g, c // g, h, w)
        mg = cell_mask.reshape(n, 1, 1, h, w)
        count = mg.sum(dim=(2, 3, 4), keepdim=True) * (c // g)
        mean = (xg * mg).sum(dim=(2, 3, 4), keepdim=True) / count
        var = ((xg - mean) ** 2 * mg).sum(dim=(2, 3, 4), keepdim=True) / count
        xg = (xg - mean) / torch.sqrt(var + self.eps)
        out = xg.reshape(n, c, h, w)
        out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out * cell_mask


class _ResBlock(nn.Module):
    """[conv3x3+MGN+ReLU, conv3x3+MGN] + skip, then ReLU. Mask-preserving."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = _conv3x3(channels, channels)
        self.gn1 = MaskedGroupNorm(channels)
        self.conv2 = _conv3x3(channels, channels)
        self.gn2 = MaskedGroupNorm(channels)

    def forward(self, x: torch.Tensor, cell_mask: torch.Tensor | None) -> torch.Tensor:
        out = torch.relu(self.gn1(self.conv1(x), cell_mask))
        out = self.gn2(self.conv2(out), cell_mask)
        return torch.relu(out + x)


def _policy_head_flatten(logits: torch.Tensor) -> torch.Tensor:
    """``(N, 56, H, W)`` conv output -> ``(N, H*W*56)`` in ActionIndexer order.

    Channels encode ``rot*14 + a``; target flat order (slow->fast) is
    ``gy, gx, rot, a`` -- split channels to (rot, a), permute, flatten.
    """
    n, _, h, w = logits.shape
    logits = logits.reshape(n, ROTS, ACTIONS, h, w)
    logits = logits.permute(0, 3, 4, 1, 2)
    return logits.reshape(n, h * w * ACTIONS_PER_CELL)


class CarcassonneNet(nn.Module):
    """Size-agnostic residual CNN: shared trunk, per-cell policy head, pooled
    value head.

    ``forward(x, mask=None, cell_mask=None) -> (policy_logits, value)``:

    * ``x`` -- ``(N, NUM_PLANES, H, W)`` for any H, W.
    * ``mask`` -- optional ``(N, H*W*56)`` bool; illegal logits -> ``-inf``.
    * ``cell_mask`` -- optional ``(N, H, W)`` or ``(N, 1, H, W)`` 0/1 float;
      REQUIRED whenever the batch is padded (learner), None for unpadded input.
    """

    def __init__(
        self,
        num_planes: int = NUM_PLANES,
        channels: int = _DEFAULT_CHANNELS,
        n_blocks: int = _DEFAULT_N_BLOCKS,
    ) -> None:
        super().__init__()
        self.num_planes = num_planes
        self.channels = channels
        self.n_blocks = n_blocks

        self.stem_conv = _conv3x3(num_planes, channels)
        self.stem_gn = MaskedGroupNorm(channels)
        self.blocks = nn.ModuleList(_ResBlock(channels) for _ in range(n_blocks))

        # Policy head: plain 1x1 conv (bias on, no norm) -- logits per cell.
        self.policy_conv = nn.Conv2d(channels, ACTIONS_PER_CELL, kernel_size=1)

        # Value head: 1x1 conv -> ReLU -> masked global-average pool -> MLP -> tanh.
        self.value_conv = nn.Conv2d(channels, _VALUE_CHANNELS, kernel_size=1)
        self.value_fc1 = nn.Linear(_VALUE_CHANNELS, _VALUE_HIDDEN)
        self.value_fc2 = nn.Linear(_VALUE_HIDDEN, 1)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        cell_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cm = None
        if cell_mask is not None:
            cm = cell_mask if cell_mask.dim() == 4 else cell_mask.unsqueeze(1)

        h = torch.relu(self.stem_gn(self.stem_conv(x), cm))
        for block in self.blocks:
            h = block(h, cm)

        policy_logits = _policy_head_flatten(self.policy_conv(h))
        if mask is not None:
            # Illegal (and padded) actions -> -inf; the learner's cross-entropy
            # must stay on the target's support (0 * -inf = NaN otherwise).
            policy_logits = policy_logits.masked_fill(~mask, float("-inf"))

        v = torch.relu(self.value_conv(h))
        if cm is None:
            pooled = v.mean(dim=(2, 3))
        else:
            pooled = (v * cm).sum(dim=(2, 3)) / cm.sum(dim=(2, 3))
        value = torch.tanh(self.value_fc2(torch.relu(self.value_fc1(pooled)))).squeeze(-1)

        return policy_logits, value


def param_count(net: nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def evaluate_state(
    net: CarcassonneNet, state: GameState, device: torch.device
) -> tuple[NDArray[np.float32], float]:
    """Encode one ``state``, run the net masked to legal moves, return (probs, value).

    ``probs`` has shape ``(Crop.of(state).n_actions,)``, sums to 1 over legal
    moves, exactly 0 elsewhere. Unpadded single state -> ``cell_mask=None``
    (fast ``F.group_norm`` path). The bridge MCTS calls; the model stays tensors.
    """
    from carcassonne.nn.actions import legal_mask  # local: avoid import cycle at module load

    planes = encode_state(state)
    x = torch.from_numpy(planes).unsqueeze(0).to(device)
    mask = torch.from_numpy(legal_mask(state)).unsqueeze(0).to(device)

    was_training = net.training
    net.eval()
    try:
        with torch.no_grad():
            policy_logits, value = net(x, mask)
            probs = torch.softmax(policy_logits, dim=1).squeeze(0)
    finally:
        net.train(was_training)

    return probs.cpu().numpy().astype(np.float32), float(value.item())
```

**Step 4: Modify `carcassonne/nn/checkpoint.py`** — delete line 17 (`from carcassonne.nn.actions import ACTION_SPACE`) and the `"action_space": ACTION_SPACE,` entry in `_net_arch` (the action space is per-state now; `num_planes`/`channels`/`n_blocks` fully describe the net).

**Step 5: Run to verify pass**
Run: `uv run pytest tests/nn/ -q`
Expected: PASS (all three nn test files)

**Step 6: Commit**
```bash
git add carcassonne/nn/model.py carcassonne/nn/checkpoint.py tests/nn/test_model.py
git commit -m "feat(nn)!: size-agnostic fully-conv net — MaskedGroupNorm, pooled value head (~192k params)"
```

---

### Task 5: MCTS on per-state indices (drop window pruning)
**Depends on:** Task 4

**Files:**
- Modify: `carcassonne/agents/mcts.py`
- Modify: `tests/agents/test_mcts.py` (adapt only if it referenced window pruning)

**Step 1/2 (TDD):** run `uv run pytest tests/agents/test_mcts.py -q` first — Expected: FAIL (import/shape errors from the old pruning path).

**Step 3: Implement.** In `_make_node` (`mcts.py:186-232`), the crop now contains every legal move, so the prune-on-`RulesError` loop is dead. Replace lines 202–221 with:

```python
        probs, value = evaluate_state(self._net, state, self._device)
        moves = legal_moves(state)
        prior = np.array([float(probs[encode_move(state, m)]) for m in moves], dtype=np.float64)
        total = prior.sum()
        # probs already sum to 1 over legal moves; renormalise for numerical safety,
        # falling back to uniform only in the degenerate all-zero case.
        prior = prior / total if total > 0 else np.full(len(moves), 1.0 / len(moves))
```

Also: remove `RulesError` from the `carcassonne.core` import list (no longer used here), delete the module-docstring sentence about pruning out-of-window moves (lines 202–204 comment included), and keep everything else identical.

**Step 4: Run to verify pass**
Run: `uv run pytest tests/agents/ tests/nn/ -q`
Expected: PASS

**Step 5: Commit**
```bash
git add carcassonne/agents/mcts.py tests/agents/test_mcts.py
git commit -m "refactor(mcts): every legal move is representable — drop window pruning"
```

---### Task 6: World-keyed policy targets — selfplay, buffer, orchestrate call-site, bench call-site
**Depends on:** Task 5

**Files:**
- Modify: `carcassonne/training/selfplay.py`, `carcassonne/training/buffer.py`, `carcassonne/training/orchestrate.py` (call-site only), `carcassonne/cli/main.py` (bench call-site)
- Modify: `tests/training/test_selfplay.py`, `tests/training/test_buffer.py`

**Step 1: Write/adapt failing tests.** In `tests/training/test_buffer.py` add:

```python
def test_policy_json_world_keys_roundtrip() -> None:
    from carcassonne.training.buffer import _policy_from_json, _policy_to_json

    policy = {(3, -2, 1, 0): 0.75, (0, 0, 3, 13): 0.25}
    assert _policy_from_json(_policy_to_json(policy)) == policy
```

In `tests/training/test_selfplay.py`, assert the new return shape and key format:

```python
    game_id, examples = play_selfplay_game(...)   # buffer/order_key params gone
    assert all(isinstance(k, tuple) and len(k) == 4 for ex in examples for k in ex.policy)
```

(Adapt each existing call site in these test files: `play_selfplay_game` no longer takes `order_key`/`buffer`; tests that asserted buffer contents now call `buffer.add_game(game_id, str(path), examples, order_key)` themselves.)

**Step 2: Run to verify fail** — `uv run pytest tests/training/test_selfplay.py tests/training/test_buffer.py -q` — Expected: FAIL

**Step 3: Implement.**

`buffer.py`: change `Example.policy` and `SampledExample.policy` types to `dict[Key, float]` (`from carcassonne.nn.actions import Key`); update the docstring line about `{action_id: prob}` to `{(x, y, rot, a): prob}` world keys; replace the JSON helpers:

```python
def _policy_to_json(policy: dict[Key, float]) -> str:
    return json.dumps(
        {",".join(map(str, k)): v for k, v in policy.items()}, separators=(",", ":")
    )


def _policy_from_json(raw: str) -> dict[Key, float]:
    out: dict[Key, float] = {}
    for ks, v in json.loads(raw).items():
        x, y, rot, a = (int(t) for t in ks.split(","))
        out[(x, y, rot, a)] = float(v)
    return out
```

`selfplay.py`: replace the `encode_move` import with `move_key`; signature and body:

```python
def play_selfplay_game(
    net: CarcassonneNet,
    device: torch.device,
    seed: int,
    run: TrainingRun,
    mcts_config: MctsConfig,
    game_id: str,
    started_at: str,
) -> tuple[str, list[Example]]:
    """Play one MctsAgent-vs-itself game; write its replay into ``run.replays_dir``
    and return ``(game_id, examples)``. The caller owns buffer insertion -- this
    is the single-game worker, usable from the multiprocess pool."""
```

Inside: `sparse = {move_key(m): p for m, p in policy.items()}`; delete the `buffer` insertion block at the end and `return game_id, examples`. Update the module docstring ("the caller inserts into the buffer; pool-safe").

`orchestrate.py` self-play block (lines 173–187) becomes:

```python
                for _ in range(cfg.selfplay_games_per_iter):
                    started_at = datetime.now(UTC).isoformat(timespec="seconds")
                    game_id = f"g{order_key:020d}"
                    _, examples = play_selfplay_game(
                        net=best_net,
                        device=device,
                        seed=cfg.seed + game_counter,
                        run=run,
                        mcts_config=gate_cfg,
                        game_id=game_id,
                        started_at=started_at,
                    )
                    buffer.add_game(
                        game_id, str(run.replays_dir / f"{game_id}.jsonl"), examples, order_key
                    )
                    order_key += 1
                    game_counter += 1
```

`cli/main.py` `_bench`: drop `order_key=i, buffer=None` from the call (now returns a tuple; ignore it).

**Step 4: Run to verify pass**
Run: `uv run pytest tests/training/test_selfplay.py tests/training/test_buffer.py tests/cli/test_bench.py -q`
Expected: PASS (test_learner/test_train_smoke still red until Task 7)

**Step 5: Commit**
```bash
git add carcassonne/training/selfplay.py carcassonne/training/buffer.py carcassonne/training/orchestrate.py carcassonne/cli/main.py tests/training/test_selfplay.py tests/training/test_buffer.py
git commit -m "feat(training)!: world-coordinate policy keys; selfplay returns examples (pool-ready)"
```

---

### Task 7: Learner — padded batches, arithmetic key rotation; full suite green
**Depends on:** Task 6

**Files:**
- Modify: `carcassonne/training/learner.py`
- Modify: `tests/training/test_learner.py`, `tests/training/test_train_smoke.py` (adapt to new shapes/signatures)

**Step 1: Write failing tests** (add to `tests/training/test_learner.py`):

```python
def test_build_batch_pads_to_batch_max(tmp_path) -> None:
    """Mixed crop sizes -> one padded tensor; masks/targets in padded geometry."""
    # Arrange a buffer with two games of different lengths (existing test
    # fixtures already build one; add a second with more moves), then:
    learner = ...  # existing fixture construction
    batch = learner._buffer.sample(8, random.Random(0))
    x, mask, cell_mask, policy_target, value_target = learner._build_batch(batch)
    n, p, ph, pw = x.shape
    assert p == NUM_PLANES
    assert mask.shape == (n, ph * pw * 56)
    assert cell_mask.shape == (n, ph, pw)
    assert policy_target.shape == (n, ph * pw * 56)
    # every target index must be a legal (masked-in) action
    assert bool(mask[policy_target > 0].all())


def test_augmented_target_consistent_with_engine(tmp_path) -> None:
    """rotate_key must agree with encode-on-rotated-state, for every k."""
    from carcassonne.nn.actions import flat_index, move_key, rotate_key, Crop
    from carcassonne.training.learner import rotate_state_90

    state = ...  # any mid-game state from the fixture game
    move = legal_moves(state)[0]
    rstate = state
    key = move_key(move)
    for k in (1, 2, 3):
        rstate = rotate_state_90(rstate)
        rkey = rotate_key(key, k)
        # the rotated key must be a legal move's key on the rotated state
        assert rkey in {move_key(m) for m in legal_moves(rstate)}
        flat_index(Crop.of(rstate), rkey)  # must not raise


def test_train_step_finite_loss(tmp_path) -> None:
    metrics = learner.train_step()
    assert math.isfinite(metrics["loss"])
```

(Wire these into the file's existing fixture style — it already builds a run + buffer from a real self-play game; follow it.)

**Step 2: Run to verify fail** — `uv run pytest tests/training/test_learner.py -q` — Expected: FAIL

**Step 3: Implement in `learner.py`.**

Imports: replace the `actions` import line with
`from carcassonne.nn.actions import Crop, Key, flat_index, legal_mask, rotate_key`
and add `valid_mask` to the encode import. Delete `_remap_policy` entirely (world keys rotate arithmetically). Update `_DEFAULT_CHANNELS = 48`, `_DEFAULT_N_BLOCKS = 4`.

Replace `_build_batch` + `_prepare` with:

```python
    def _build_batch(
        self, batch: list[SampledExample]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batch of variable-size crops -> tensors padded to the batch max.

        Rotation augmentation happens BEFORE padding (a rotated crop swaps h/w).
        Padding cells are all-zero planes, False mask, 0 cell_mask -- the model's
        MaskedGroupNorm guarantees they cannot influence real cells.
        """
        prepared: list[tuple[GameState, Crop, dict[Key, float]]] = []
        for ex in batch:
            state = self._buffer.state_for(ex.game_id, ex.move_n)
            k = self._rng.randrange(4) if self._config.augment else 0
            policy = ex.policy
            if k:
                for _ in range(k):
                    state = rotate_state_90(state)
                policy = {rotate_key(key, k): prob for key, prob in policy.items()}
            prepared.append((state, Crop.of(state), policy))

        ph = max(c.h for _, c, _ in prepared)
        pw = max(c.w for _, c, _ in prepared)
        n_actions = ph * pw * 56

        planes_list, mask_list, cell_list, policy_list, value_list = [], [], [], [], []
        for (state, crop, policy), ex in zip(prepared, batch, strict=True):
            planes_list.append(encode_state(state, crop, pad_hw=(ph, pw)))
            mask_list.append(legal_mask(state, crop, pad_hw=(ph, pw)))
            cell_list.append(valid_mask(crop, pad_hw=(ph, pw)))
            dense = np.zeros(n_actions, dtype=np.float32)
            for key, prob in policy.items():
                dense[flat_index(crop, key, stride_w=pw)] = prob
            policy_list.append(dense)
            value_list.append(ex.value)

        x = torch.from_numpy(np.stack(planes_list)).to(self._device)
        mask_t = torch.from_numpy(np.stack(mask_list)).to(self._device)
        cell_t = torch.from_numpy(np.stack(cell_list)).to(self._device)
        policy_target = torch.from_numpy(np.stack(policy_list)).to(self._device)
        value_target = torch.tensor(value_list, dtype=torch.float32, device=self._device)
        return x, mask_t, cell_t, policy_target, value_target
```

In `train_step`, unpack five tensors and call `self._net(x, mask, cell_mask=cell_mask)`. Note the old out-of-window `RulesError` fallback is GONE — a rotated crop always fits its own crop; if rotation ever raises, that's a real bug and must surface. Update the module docstring's augmentation paragraph accordingly (world keys rotate arithmetically via `rotate_key`; `rotate_state_90` is still the source of truth for planes). Keep `rotate_state_90`, `_rot_pos`, `_rot_rotation`, `_rotate_move` (tests use them).

**Step 4: Full suite must be green now.**
Run: `uv run pytest -q`
Expected: **ALL PASS**. Also run: `uv run ruff check . && uv run mypy carcassonne` — clean.

**Step 5: Commit**
```bash
git add carcassonne/training/learner.py tests/training/
git commit -m "feat(training)!: learner pads dynamic crops per batch; arithmetic key rotation"
```

---

### Task 8: MCTS subtree reuse
**Depends on:** Task 5 (parallel with Tasks 6–7)

**Files:**
- Modify: `carcassonne/agents/mcts.py`
- Test: `tests/agents/test_mcts.py` (add)

**Soundness note (resolved design question):** `apply` draws the next tile deterministically from the deck inside `GameState`, so the search tree is built over the *actual* future draws — the child reached by the committed move is exactly the next real state. Reuse is exact, not approximate. `GameState` is a frozen dataclass, so `==` is a full structural compare (cheap: once per move).

**Step 1: Write the failing tests** (add to `tests/agents/test_mcts.py`):

```python
def test_tree_reuse_carries_visits_across_moves() -> None:
    net = _tiny_net()  # reuse the file's existing net fixture/helper
    agent = MctsAgent(net, torch.device("cpu"), MctsConfig(sims=8, reuse_tree=True))
    state = new_game(0, GameConfig())
    ctx = TurnContext(rng=random.Random(0), annotate=False)
    move, _ = agent.choose(state, ctx)
    assert agent._stash is not None  # subtree kept for the next ply
    nxt = apply(state, move)
    if not is_terminal(nxt):
        root = agent._search(nxt, np.random.default_rng(0), add_noise=False)
        assert int(root.n.sum()) >= 8  # inherited visits + this search's sims


def test_tree_reuse_misses_on_unrelated_state() -> None:
    net = _tiny_net()
    agent = MctsAgent(net, torch.device("cpu"), MctsConfig(sims=4, reuse_tree=True))
    ctx = TurnContext(rng=random.Random(0), annotate=False)
    agent.choose(new_game(0, GameConfig()), ctx)
    # A different game must not match the stash — fresh root, no crash.
    move, _ = agent.choose(new_game(99, GameConfig()), ctx)
    assert move in legal_moves(new_game(99, GameConfig()))


def test_reuse_off_matches_old_behaviour() -> None:
    net = _tiny_net()
    cfg = MctsConfig(sims=8, reuse_tree=False)
    a1 = MctsAgent(net, torch.device("cpu"), cfg)
    a2 = MctsAgent(net, torch.device("cpu"), cfg)
    state = new_game(3, GameConfig())
    m1, _ = a1.choose(state, TurnContext(rng=random.Random(1), annotate=False))
    m2, _ = a2.choose(state, TurnContext(rng=random.Random(1), annotate=False))
    assert m1 == m2  # stateless determinism preserved when reuse is off
```

**Step 2: Run to verify fail** — Expected: FAIL (`reuse_tree` unknown field)

**Step 3: Implement.**
- `MctsConfig`: add `reuse_tree: bool = True`.
- `MctsAgent.__init__`: add `self._stash: _Node | None = None`.
- `_search(self, root_state, rng, add_noise, reuse: bool = True)`: build the root via

```python
        root = (self._take_stash(root_state) if reuse and self._config.reuse_tree else None)
        if root is None:
            root = self._make_node(root_state)
```

- New method:

```python
    def _take_stash(self, state: GameState) -> _Node | None:
        """Consume the stashed subtree if it matches ``state``.

        The stash is the child we committed last move. In self-play (one agent,
        both seats) it matches directly; in arena/web play the opponent moved
        once since, so also check the stash's children (one ply deeper). apply()
        draws from the in-state deck, so a matching state is the exact node.
        """
        stash, self._stash = self._stash, None
        if stash is None:
            return None
        if not stash.terminal and stash.state == state:
            return stash
        for child in stash.children.values():
            if not child.terminal and child.state == state:
                return child
        return None
```

- In `choose` and `choose_with_policy`, after computing `move_idx`/picking the move: `self._stash = root.children.get(move_idx)` (before returning).
- `visit_policy` passes `reuse=False` to `_search` (hints must not consume the play stash).
- Dirichlet noise: `_search` already noises whatever root it has — a reused root gets fresh noise, correct AlphaZero practice. Add one docstring sentence: reused roots carry inherited visit counts into the policy target, standard AlphaZero subtree reuse.

**Step 4: Run to verify pass**
Run: `uv run pytest tests/agents/test_mcts.py tests/training/ -q`
Expected: PASS

**Step 5: Commit**
```bash
git add carcassonne/agents/mcts.py tests/agents/test_mcts.py
git commit -m "feat(mcts): exact subtree reuse across moves (determinized deck makes it sound)"
```

---

### Task 9: Multiprocess self-play pool + orchestrator/CLI wiring
**Depends on:** Tasks 7, 8

**Files:**
- Create: `carcassonne/training/pool.py`
- Modify: `carcassonne/training/orchestrate.py`, `carcassonne/cli/main.py`
- Test: `tests/training/test_pool.py` (create)

**Step 1: Write the failing test**

```python
# tests/training/test_pool.py
"""Spawn-pool self-play: format-identical output, parent-owned buffer."""
from __future__ import annotations

import random

import torch

from carcassonne.agents.mcts import MctsConfig
from carcassonne.nn.model import CarcassonneNet
from carcassonne.training.buffer import ReplayBuffer
from carcassonne.training.pool import SelfplayPool
from carcassonne.training.run import TrainingRun


def test_pool_plays_games_and_parent_inserts(tmp_path) -> None:
    run = TrainingRun.create(tmp_path / "run", {"net_arch": {}})
    net = CarcassonneNet(channels=16, n_blocks=1)
    buffer = ReplayBuffer(run.buffer_path)
    pool = SelfplayPool(workers=2, net=net, mcts_config=MctsConfig(sims=2), run_root=run.root)
    try:
        jobs = [(seed, f"g{seed:04d}", "t0") for seed in (0, 1, 2)]
        for game_id, replay_path, examples in pool.play(jobs):
            buffer.add_game(game_id, replay_path, examples, order_key=int(game_id[1:]))
    finally:
        pool.close()
    assert buffer.n_games() == 3
    assert len(buffer) > 0
    # states re-derivable: buffer + replays are format-identical to sequential mode
    ex = buffer.sample(1, random.Random(0))[0]
    assert buffer.state_for(ex.game_id, ex.move_n) is not None
    buffer.close()
```

**Step 2: Run to verify fail** — `uv run pytest tests/training/test_pool.py -q` — Expected: FAIL (no module `pool`)

**Step 3: Create `carcassonne/training/pool.py`** (complete file):

```python
"""Multiprocess self-play: a spawn-based worker pool playing whole games.

Game-level parallelism only (design decision): each worker owns a private copy
of the best net (rebuilt from a state_dict at pool start), pins torch to one
intra-op thread (4 workers x default threads would oversubscribe the Pi), and
plays complete games via :func:`play_selfplay_game`. Workers write replay files
(distinct per game); the PARENT keeps sole ownership of the sqlite buffer.

Determinism: each game is fully determined by its seed, identical to sequential
mode; only buffer insertion order varies with completion order. The pool must
be recreated when the best net changes (promotion) -- the orchestrator does.

Spawn (not fork) context: fork + a threaded torch parent is unsafe, and spawn
behaves identically on macOS dev and the Pi.
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from carcassonne.agents.mcts import MctsConfig
from carcassonne.nn.model import CarcassonneNet
from carcassonne.training.buffer import Example
from carcassonne.training.run import TrainingRun
from carcassonne.training.selfplay import play_selfplay_game

type Job = tuple[int, str, str]  # (seed, game_id, started_at)
type GameResult = tuple[str, str, list[Example]]  # (game_id, replay_path, examples)

_worker_net: CarcassonneNet | None = None
_worker_cfg: MctsConfig | None = None
_worker_run: TrainingRun | None = None


def _init_worker(
    arch: dict[str, int], state_dict: dict[str, Any], mcts_cfg: MctsConfig, run_root: str
) -> None:
    import torch

    torch.set_num_threads(1)
    global _worker_net, _worker_cfg, _worker_run
    net = CarcassonneNet(**arch)
    net.load_state_dict(state_dict)
    net.eval()
    _worker_net = net
    _worker_cfg = mcts_cfg
    _worker_run = TrainingRun(Path(run_root))


def _play_one(job: Job) -> GameResult:
    import torch

    assert _worker_net is not None and _worker_cfg is not None and _worker_run is not None
    seed, game_id, started_at = job
    _, examples = play_selfplay_game(
        net=_worker_net,
        device=torch.device("cpu"),
        seed=seed,
        run=_worker_run,
        mcts_config=_worker_cfg,
        game_id=game_id,
        started_at=started_at,
    )
    return game_id, str(_worker_run.replays_dir / f"{game_id}.jsonl"), examples


class SelfplayPool:
    """A fixed-net worker pool. Recreate on promotion (the net is baked in at start)."""

    def __init__(
        self, workers: int, net: CarcassonneNet, mcts_config: MctsConfig, run_root: Path
    ) -> None:
        arch = {
            "num_planes": net.num_planes,
            "channels": net.channels,
            "n_blocks": net.n_blocks,
        }
        state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        self._executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
            initializer=_init_worker,
            initargs=(arch, state, mcts_config, str(run_root)),
        )

    def play(self, jobs: Iterable[Job]) -> Iterator[GameResult]:
        """Run all jobs; yield results as workers finish (order not guaranteed)."""
        yield from self._executor.map(_play_one, list(jobs))

    def close(self) -> None:
        self._executor.shutdown(wait=True)
```

**Step 4: Wire into `orchestrate.py`.**
- `TrainConfig`: add `workers: int = 1`. Add `"workers": cfg.workers` to `run_config_dict`'s `train` dict.
- In `train()`: track `pool: SelfplayPool | None = None`; replace the sequential self-play block with:

```python
                if cfg.workers > 1:
                    if pool is None:
                        pool = SelfplayPool(cfg.workers, best_net, gate_cfg, run.root)
                    jobs, keys = [], {}
                    for _ in range(cfg.selfplay_games_per_iter):
                        started_at = datetime.now(UTC).isoformat(timespec="seconds")
                        game_id = f"g{order_key:020d}"
                        jobs.append((cfg.seed + game_counter, game_id, started_at))
                        keys[game_id] = order_key
                        order_key += 1
                        game_counter += 1
                    for game_id, replay_path, examples in pool.play(jobs):
                        buffer.add_game(game_id, replay_path, examples, keys[game_id])
                else:
                    ... # the Task-6 sequential loop, unchanged
```

- On promotion (`_gate_and_maybe_promote` returns; check `promoted` where it's called or have it return the flag): after `best_net.load_state_dict(...)`, the pool is stale → in `train()`, close and null it: change `_gate_and_maybe_promote(...)` to return `promoted: bool`, and in the loop:

```python
                if gated and _gate_and_maybe_promote(...) and pool is not None:
                    pool.close()
                    pool = None  # recreated next iteration with the promoted net
```

- In the `finally:` block: `if pool is not None: pool.close()`.
- Update the module docstring paragraph "Self-play is sequential here…" to describe the pool (workers > 1, recreated on promotion, parent-owned buffer).

**Step 5: CLI.** In `main.py` `train` parser: `tr.add_argument("--workers", type=int, default=0, help="self-play worker processes (0 = one per CPU core)")`; in `_train`: `workers=args.workers or (os.cpu_count() or 1)` (add `import os` at top of `_train`), passed to `TrainConfig`.

**Step 6: Run to verify pass**
Run: `uv run pytest tests/training/test_pool.py tests/training/test_train_smoke.py -q`
Expected: PASS. Then full suite: `uv run pytest -q` — ALL PASS.

**Step 7: Commit**
```bash
git add carcassonne/training/pool.py carcassonne/training/orchestrate.py carcassonne/cli/main.py tests/training/test_pool.py
git commit -m "feat(training): multiprocess self-play pool (spawn, worker-pinned threads, parent-owned buffer)"
```

---

### Task 10: New defaults, docs, after-bench, final verification
**Depends on:** Tasks 1, 9

**Files:**
- Modify: `carcassonne/cli/main.py` (defaults), `carcassonne/training/orchestrate.py` (defaults), `docs/pi-setup.md`, this plan file (results)

**Steps:**
1. Defaults: `TrainConfig.channels = 48`, `TrainConfig.n_blocks = 4`; CLI `--channels` default 48, `--n-blocks` default 4 (help text updated). Confirm `learner._DEFAULT_CHANNELS/_DEFAULT_N_BLOCKS` already 48/4 (Task 7).
2. `docs/pi-setup.md`: update example commands to include `--workers 4`; replace the "CPU self-play dominates" paragraph with the new levers (crop-proportional compute, workers, tree reuse); note pre-iteration-3 checkpoints don't load and `runs/mac-demo` is retired.
3. **After-bench (success-bar evidence), same conditions as the Task 1 baseline:**
   - `uv run carcassonne bench --games 2 --sims 16 --channels 48 --n-blocks 4 --seed 0`
   - Record in **Results** below. Success bar: ≥5× the baseline games/hour (single-process).
   - Worker scaling (quick check on the Mac): time `uv run carcassonne train --run /tmp/scale-test --sims 8 --selfplay-games 8 --learn-steps 1 --iterations 1 --workers 1` vs `--workers 4` (fresh run dir each); self-play wall-clock should scale ~linearly. Record both.
4. Full verification (verify skill): `uv run pytest -q` (all pass, none skipped), `uv run ruff check .`, `uv run mypy carcassonne`, plus one end-to-end smoke: fresh 2-iteration train run with `--workers 2`, kill and `--resume`, confirm it continues.
5. Update `docs/features.md`: flip the iteration-3 items from *(planned)* to implemented.

**Commit**
```bash
git add -A && git commit -m "feat(training)!: iteration-3 defaults (48ch x 4 blocks, workers) + docs + benchmarks"
```

---

## Baseline (recorded at Task 1, 2026-09-20, Intel Mac CPU)

- `bench --games 2 --sims 16 --channels 64 --n-blocks 5` →
  `params=655865`
  `games=2 sims=16 channels=64 n_blocks=5 total=50.2s s/game=25.1 games/hour=143.4`
- ≥5× bar for Task 10's after-bench: **≥ 717 games/hour** at the same games/sims settings.

## Results (recorded at Task 10)

Recorded 2026-10-01 on the same Intel Mac (4 physical / 8 logical cores, 16 GB,
torch 2.2 CPU). Every run was wrapped in `caffeinate -dims` with no other load.
The bench is single-process, with torch's default intra-op threads, as in the
baseline.

- `bench --games 2 --sims 16 --channels 48 --n-blocks 4 --seed 0`, run 3 times:
  ```
  params=192673
  games=2 sims=16 channels=48 n_blocks=4 total=21.0s s/game=10.5 games/hour=343.4
  games=2 sims=16 channels=48 n_blocks=4 total=23.1s s/game=11.6 games/hour=311.6
  games=2 sims=16 channels=48 n_blocks=4 total=18.0s s/game=9.0 games/hour=400.2
  ```
  **Median 343.4 games/hour.**
- **Speedup vs baseline: 343.4 / 143.4 = 2.39× — FAIL against the ≥5× bar
  (≥717 games/hour).** The single-process target was not met. The run-to-run
  spread is about ±13%, which is nowhere near enough to close the gap.
- Attribution: `--channels 64 --n-blocks 5` re-run on the current code, 3 times:
  ```
  params=404481
  games=2 sims=16 channels=64 n_blocks=5 total=30.2s s/game=15.1 games/hour=238.5
  games=2 sims=16 channels=64 n_blocks=5 total=22.5s s/game=11.3 games/hour=319.5
  games=2 sims=16 channels=64 n_blocks=5 total=31.7s s/game=15.8 games/hour=227.4
  ```
  Median 238.5. (a) Representation, MCTS, and tree-reuse changes at 64×5:
  238.5 / 143.4 = **1.66×**. This 64×5 net is not the baseline's net: the
  heads are now fully convolutional, which cuts it from 655,865 to 404,481
  params, so (a) includes the head change. (b) Net-size change, 48×4 vs 64×5
  on the current code: 343.4 / 238.5 = **1.44×**. Together:
  1.66 × 1.44 ≈ 2.39×.
- `param_count` at defaults: **192,673** (bar ≤250k: PASS). This is pinned by
  `tests/cli/test_train_defaults.py` and `tests/nn/test_model.py::test_param_budget`.
- Worker scaling 1→4. The command was `train --run <fresh dir> --sims 8
  --selfplay-games 8 --learn-steps 1 --iterations 1 --device cpu` at 48×4. The
  self-play phase was timed on its own, from the end of `_establish_nets` to
  the first `evict_to`, so the 4-worker figure includes spawning the pool.
  | workers | self-play | total wall-clock |
  |---|---|---|
  | 1 | 69.2s | 104.5s |
  | 4 | 22.9s | 44.4s |

  Self-play speeds up **3.02×** with 4 workers on 4 physical cores, which is
  close to linear. Combined with the 2.39× single-process speedup, 4-worker
  self-play produces about 2.39 × 3.02 ≈ **7.2×** the baseline's games/hour on
  this Mac. That is aggregate throughput, not the single-process bar above.
- Deferred measurement: learner rotation cost (M12). Measured on the buffer
  from the 1-worker scaling run (568 examples / 8 games) at batch 256, 48×4,
  CPU, as the mean of 5 steps after a warm-up:
  ```
  augment=False: build_batch=1681ms/step  train_step=4999ms/step
  augment=True:  build_batch=2616ms/step (rotate_state_90 513ms = 20% of build_batch)
                 train_step=20343ms/step (rotation 547ms = 3%)
  ```
  Rotating the state itself is cheap: about 0.5s/step, 3% of a step, even
  though `rotate_state_90` rebuilds FeatureIndex for each 90° turn. **The real
  cost of augmentation is padding.** A rotated crop swaps h and w, so a mixed
  batch pads to max(h) × max(w) over both orientations:
  ```
  augment=False: padded cells/example=750  mean real cells=203  pad-efficiency=27%  (e.g. 14x54)
  augment=True:  padded cells/example=2798 mean real cells=205  pad-efficiency=7%   (e.g. 53x52)
  ```
  That is 3.7× more padded cells, which matches the roughly 4× slower
  `train_step`. Even without augmentation only 27% of cells are real. Not
  optimised here: recorded as a follow-up. Possible fixes are choosing the
  rotation so every crop is landscape, or bucketing batches by shape.
- Deferred measurement: subtree-reuse equality cost (M13). Two `MctsAgent`s
  (reuse on, `self_play=False`) alternated in an arena-style game at sims=16,
  48×4, with `_take_stash` and `choose` wrapped in `perf_counter`:
  ```
  seed=0: game=14.5s choose=14.3s take_stash=3ms = 0.02% of choose
  seed=1: game=14.3s choose=14.1s take_stash=2ms = 0.02% of choose
  ```
  This is negligible, so no action.
- End-to-end smoke test (`--workers 2 --sims 4 --selfplay-games 6
  --learn-steps 2`). A fresh 3-iteration run got SIGINT after 2 games of
  iteration 1. It finished that iteration and exited 0 (`games_generated=6`,
  checkpoint `step_000002`). `--resume --iterations 2` then continued to
  `games_generated=18` and checkpoint `step_000006`. The buffer holds 18 rows
  with 18 distinct `game_id`s and there are 18 replay files. `pgrep -fl
  spawn_main` showed no orphan workers after the stop or after the resume.

## Deferred (explicitly)

- Arena/gate games through the pool (self-play dominates; follow-up).
- MCTS leaf-eval batching, quantization/compile, GNN representation, checkpoint migration.

## Review
- [ ] Code review requested (final `code-reviewer` over all tasks)
- [ ] All feedback addressed
- [ ] Final verification passed (`verify` skill: full suite green, ruff, mypy, bench recorded)
