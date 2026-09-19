# Dynamic-Crop Representation & Training Throughput Design

Iteration 3: make training an order of magnitude faster and the network
smaller/easier to learn, by fitting the representation to Carcassonne's
actual structure (relative tile placement, feature-graph value) instead of a
fixed chess-like board.

## Research

Findings from the codebase-analyzer sweep (2026-09-19):

### Current representation (`carcassonne/nn/encode.py`, `actions.py`)

- Input: **62 planes × 31×31** (`WINDOW = 31`, `actions.py:38`;
  `NUM_PLANES = 62`, `encode.py:26`). Only 24 planes are spatial; **38 are
  constants broadcast across all 961 cells** — 8 normalized scalars (scores,
  supplies, deck, turn) and a **30-plane one-hot of the current tile's ID**.
- The window centres the occupied bounding box (`window_origin`,
  `actions.py:47-60`), so the net already sees only relative geometry — no
  absolute-coordinate leakage. Translation equivariance of convs is the right
  inductive bias for "relative tile placement"; the flaw is the *fixed
  oversized window*, not CNN-vs-something-else.
- Action space: `31·31·4·14 = 53,816` (`actions.py:43-44`) — window cell ×
  rotation × 14 sub-actions (tile-only / meeple on feature 0-5 / abbot on
  feature 0-5 / retrieve-abbot). **Legal moves outside the window are
  silently dropped** (`actions.py:147-150`, `mcts.py:208-211`) — a quiet
  rules deviation, rare at 31×31 but structurally possible.

### Current network (`carcassonne/nn/model.py`)

- Stem conv3×3(62→64) + 5 residual blocks (BatchNorm) + policy head
  (1×1 conv → 56 = 4 rot × 14 sub-action channels per cell, reshaped to the
  action-id formula) + value head (1×1 conv → 4, flatten,
  `Linear(4·31·31=3844 → 64)`, `Linear(64→1)`, tanh).
- **655,865 params at defaults; the value-head `Linear(3844→64)` alone is
  ~246k (37%)** and is the only thing welding the weights to 31×31.
- ~400M MACs per evaluation; compute scales linearly with window area.

### Current training loop (`carcassonne/training/`)

- AlphaZero-style: self-play (one MCTS search per move, visit distribution
  as policy target, ±1/0 value backfill) → sqlite ReplayBuffer → learner
  (policy CE + value MSE, Adam) → arena gating (promote >55% vs best;
  GreedyAgent yardstick).
- **Self-play is sequential, single-process, and net evaluation is unbatched**
  (one state per call, `evaluate_state`, `model.py:142-168`) — documented as
  the dominant wall-clock cost (`docs/pi-setup.md:43-46`). **No MCTS tree
  reuse between moves** — every move rebuilds the search from scratch
  (`mcts.py:173-184`).
- D4 augmentation (C4 rotations only; tiles are chiral so no reflections) is
  done correctness-by-construction at sample time: rotate the real GameState,
  re-encode, re-key the policy target (`learner.py:229-298`). Out-of-window
  rotations silently fall back to identity (`learner.py:206-213`).
- Targets: Pi 5 (4-core aarch64, CPU) for multi-day runs; Intel Mac (CPU,
  torch 2.2 pin) for prove-out. Success bar: >70% vs GreedyAgent over 100
  games. Only run on disk: `runs/mac-demo` (16 games — a smoke run).
- Coupling: the action-id formula is baked into the policy-head reshape,
  MCTS, self-play targets, D4 remap, and buffer JSON — **no versioning**, so
  a representation change invalidates old checkpoints/replays.

## Problem Statement

Training is CPU-bound on the Pi and pays for a 31×31 board regardless of the
actual board size (a game starts at 1×1 and typically ends far below 31×31).
The network is larger than the game needs — a third of its parameters exist
only to flatten the oversized window — and it must *infer* long-range feature
connectivity (snaky roads, sprawling cities) through conv layers, which is
exactly what convs are bad at. Separately, the fixed window silently drops
out-of-window legal moves, and self-play uses one core with no tree reuse.

## Solution

Make the network fully convolutional and size-agnostic, feed it a **dynamic
crop** (bounding box of placed tiles + 1-cell border — provably contains all
legal placements), hand it the feature graph as input planes, and parallelise
self-play across cores with MCTS tree reuse. Break checkpoint/replay
compatibility cleanly (only smoke runs exist; user-confirmed).

Decided in interview (2026-09-19): scope = representation **and** throughput;
clean compatibility break; dynamic exact crop (not fixed-smaller, not
bucket-padded); engineered feature planes in (sample efficiency over
from-scratch purity).

### 1. Encoding — dynamic crop with feature-graph planes

`encode_state` outputs `(P, H, W)` with per-state H×W = bounding box + 1
border. By the adjacency rule the border contains every legal placement, so
out-of-window drops become structurally impossible.

Planes (~48, exact list fixed at planning):

- **Kept spatial**: occupancy (1), edge-kind one-hot city/road/field × NESW
  (12), monastery/garden presence (2), shield (1), meeple/abbot ownership
  me/opp (4), monastery-fill (1).
- **New FeaturePlanes (~8)** from the existing `FeatureIndex`: for {city,
  road} touching each tile — normalized feature size, normalized open-edge
  count, my-meeple count, opp-meeple count. `_COMPLETED` is subsumed
  (completed ⇔ open edges 0) and dropped.
- **Current tile: ~15 property planes** (4 edges × 3 kinds + monastery +
  garden + shield) replacing the 30-plane tile-ID one-hot — same information
  the rules use; structurally identical tiles share representation.
- **Kept broadcast scalars (8)**: scores, supplies, deck, turn. Meeple supply
  genuinely conditions policy, so it stays in the trunk.

### 2. Network — fully convolutional, ~200k params

- Stem + residual trunk, new defaults **48 channels × 4 blocks** (still
  configurable). FeaturePlanes carry connectivity, so the trunk no longer
  needs depth to derive it.
- **Policy head unchanged in concept**: 1×1 conv → 56 channels (4 rot × 14
  sub-actions) per cell; per-state flatten to H·W·56 logits.
- **Value head**: 1×1 conv → **masked global-average pool** (mean over true
  crop cells only) → small MLP → tanh. Deletes the `Linear(3844→64)`.
- **BatchNorm → GroupNorm**: no running stats, size-independent — makes
  "padded forward == unpadded forward" a testable invariant (BN batch stats
  would be polluted by padding and misbehave at small batch sizes).

### 3. Action space — world-coordinate native

The global `ACTION_SPACE` constant and window-based action IDs disappear.
Actions are natively `(world_pos, rotation, sub-action)` — the engine's own
vocabulary. Buffer policy targets are keyed by world coordinates (stable
under any crop) and mapped to per-state grid indices at encode time.
Per-state action count = H·W·56. D4 augmentation keeps the
rotate-the-real-GameState approach; its out-of-window identity fallback is
deleted (a rotated crop always fits its own crop).

### 4. Throughput — multiprocess self-play + MCTS tree reuse

- **Worker pool** (default = core count; 4 on Pi): each worker plays complete
  games with its own net copy, torch intra-op threads pinned to 1. Parent
  owns buffer + replay writing. Game-level parallelism only — no shared
  inference server. Gate/yardstick games use the same primitive.
- **Tree reuse**: after a move commits, the matching child subtree becomes
  the next search root. Open question for planning: how the current MCTS
  models tile-draw chance — the subtree is reusable only where the realized
  draw matches the searched branch.

### 5. Expected impact

| Lever | Effect |
|---|---|
| Dynamic crop | ~6-8× less conv compute averaged over a game (area-proportional) |
| Smaller trunk + pooled value head | 656k → ~200k params |
| 4 self-play workers | ~3.5-4× wall-clock |
| Tree reuse | fewer fresh evaluations per move (magnitude TBD) |
| FeaturePlanes | fewer games to a given strength (sample efficiency) |

## User Stories

1. As the trainer, I want self-play on the Pi to be an order of magnitude
   faster, so multi-day runs reach a strong agent within my patience budget.
2. As an RL learner, I want the representation to match the game's real
   structure (relative placement, feature graph), so I can see *why* design
   choices matter, not just that they do.
3. As the engine's author, I want every legal move representable to the net,
   so the NN layer never silently deviates from the rules.

## Implementation Decisions

- Fully convolutional net + masked global-average pool = the deep-module win:
  `WINDOW` stops being a hyperparameter baked into weights; the net accepts
  any board size.
- GroupNorm over BatchNorm (padding correctness + small-batch behavior).
- Learner batches pad each batch to its largest member; padding cells are
  semantically "empty board" (occupancy 0) and masked out of value pooling.
- World-coordinate action keying in the buffer; grid indices are an
  encode-time detail. No action-id versioning/migration — clean break,
  `runs/mac-demo` invalidated.
- Exact crop, not size buckets — buckets only pay off if MCTS leaf batching
  lands later, and it's explicitly out of scope.
- Trade-off accepted: engineered FeaturePlanes give up "learned from raw
  board from scratch" purity for sample efficiency at Pi-scale compute.

## Testing Decisions

Carry over the correctness-by-construction style:

- Property test: crop always contains every `legal_moves` position
  (bounding-box+1 invariant).
- Mask ↔ `legal_moves` bijection on random states (no drops, no ghosts).
- Policy flatten round-trip (per-state H·W·56 indexing).
- D4 equivariance on dynamic crops (existing test adapted; FeaturePlanes are
  rotation-covariant like other spatial planes).
- **Padded forward == unpadded forward** (GroupNorm + masked-pool payoff).
- Multiprocess self-play: N workers produce byte-identical replay/buffer
  formats; determinism per worker seed.
- Throughput benchmark harness (games/hour at fixed sims) run before/after on
  the Mac — the measured basis for the success bar.

## Success Bar (iteration 3)

1. Full suite green, including the new invariants above.
2. `param_count` ≤ 250k at new defaults.
3. Measured self-play throughput ≥ 5× iteration-2 baseline on the Mac at
   equal sims (before multiprocessing), and ~linear scaling to 4 workers.
4. End-to-end smoke train run (mac-demo scale) completes and resumes.
5. Yardstick vs GreedyAgent at least matches the old net's trajectory at
   equal games generated (sample efficiency should improve, not regress).

## Out of Scope

- MCTS leaf-evaluation batching (deliberate deferral stands).
- Quantization / `torch.compile`.
- Multiplayer (3-5p) NN/training work (raises loudly, per iteration 2).
- Graph-neural-network representation of the feature graph.
- Checkpoint/replay migration from the 31×31 format.
