# ProjectCarcasonne — Product Spec

A Python Carcassonne simulator, browser visualisation, and AlphaZero-inspired
training system. Three deliverables built on one rules engine and one **Replay**
file format.

## The game

- Carcassonne **base game plus abbots**; **no farmers** (fields affect tile
  placement compatibility only, never scoring).
- **2-5 players** (parametric `num_players`, default 2); standard supplies
  per seat (7 meeples + 1 abbot).
- Full scoring: roads, cities (with shields), monasteries, gardens (abbot),
  including end-game partial scoring; abbot early-retrieval rule. Majority
  scoring awards all tied leaders; game outcome is a **Winners** tie set of
  top scorers.

## The engine

- One canonical, pure-Python rules engine (`carcassonne.core`) behind a
  five-function interface: `new_game`, `legal_moves`, `apply`, `is_terminal`,
  `final_scores`.
- Immutable **GameState**; a **Move** is one full turn decision (tile + optional
  meeple, or abbot retrieval). Per-player state is sized by `num_players`;
  turn order rotates `(player + 1) % num_players`.
- Deterministic: `(seed, moves)` exactly reproduces any game.

## Replays

- Every game — self-play, arena, human-vs-AI — is one JSONL **Replay** file
  (format v2): header (seed, config incl. `num_players`, agents/checkpoint
  ids) + one line per move; footer carries `final_scores` and `winners`.
  v1 (2-player) replays remain loadable.
- AI moves carry annotations: value estimate, top policy/visit distribution,
  simulation count, think time. Human moves carry none.

## Agents

- Common **Agent** protocol; implementations: **RandomAgent** (any player
  count), **GreedyAgent** (1-ply heuristic yardstick: my delta minus
  best-of-rest delta; any player count), **MctsAgent** (PUCT search guided by
  a policy/value network, AlphaZero-style; **2-player only, enforced**).
  MctsAgent reuses the search subtree across moves where the realized tile
  draw matches the searched branch.

## Neural network — dynamic crop, fully convolutional

- The **Encoder** maps a GameState to `(planes, H, W)` where H×W is the
  **Crop**: the bounding box of placed tiles plus a 1-cell border — by the
  adjacency rule the border contains every legal placement, so every legal
  move is always representable (no fixed window, no silent drops).
- ~48 input planes: spatial board planes (occupancy, edge kinds, meeples,
  shields, monastery fill), **FeaturePlanes** derived from the engine's
  feature graph (per-tile city/road size, open edges, meeple ownership),
  the current tile as ~15 structural property planes (edges + flags, not a
  tile-ID one-hot), and 8 broadcast scalars (scores, supplies, deck, turn).
- Fully convolutional residual trunk (GroupNorm; default 48 channels ×
  4 blocks, ~200k params) that accepts any crop size. Policy head: 1×1 conv
  → 4 rotations × 14 sub-actions per cell (H·W·56 logits per state). Value
  head: 1×1 conv → masked global-average pool over true crop cells → small
  MLP → tanh.
- Actions are natively `(world_pos, rotation, sub-action)`; buffer policy
  targets are keyed by world coordinates and mapped to crop indices at
  encode time. Learner batches pad to the batch max; padding is semantically
  empty board and excluded from value pooling (padded forward == unpadded
  forward is a tested invariant).

## Training — strictly 2-player

The RL stack (encoder, network, MCTS, self-play, learner, arena) assumes
2-player zero-sum play and **raises loudly** if handed a 3-5 player state —
it never fails silently. Multiplayer RL (max^n search, vector value heads) is
deferred research; groundwork in
`docs/plans/2026-08-23-multiplayer-3-5-research.md`.

- Single-machine loop via `train` CLI: **multiprocess self-play** (worker
  pool, default = core count, one game per worker, torch intra-op threads
  pinned to 1) → on-disk replay buffer → learner (policy CE + value MSE,
  C4-rotation augmentation) → **Arena** gating (promote candidate on >55% vs
  current best; GreedyAgent as absolute yardstick). Arena games use the same
  worker pool. MCTS leaf evaluation stays unbatched (deliberate deferral).
- A **TrainingRun** is a directory (config, checkpoints, replays, buffer,
  metrics); fully resumable with `--resume` — required for multi-day runs on a
  Raspberry Pi 5. Development/prove-out is CPU-only on this Intel Mac (no MPS —
  Apple-Silicon only); training on the Pi (aarch64 CPU). Network sized for Pi
  inference; per-evaluation compute scales with actual board area, not a
  fixed window.

## Web app — the RL microscope

FastAPI + static JS canvas, three views. A first-class goal is exposing the RL
machinery for learning purposes. The RL microscope (hints, search panel,
win-probability) is a **2-player feature**; multiplayer games play and replay
without it.

1. **Play** — one human vs 1-4 agents: procedural tile rendering, legal
   placements highlighted, click-to-place, meeple spots; per-seat score cards
   and colours; consecutive AI seats step automatically. In 2-player games vs
   a checkpoint: optional hint overlay showing the AI's policy/value heatmap.
   Checkpoint opponents are unavailable (cleanly rejected) at 3-5 players.
   Games saved as Replays.
2. **Replay** — step through any Replay (2-5 players): score deltas per seat,
   winners at the end; for 2-player AI games additionally the per-move search
   panel (network prior vs MCTS visits) and win-probability chart.
3. **Training** — live view of a TrainingRun: loss curves, arena win rates,
   games generated, checkpoint timeline.

## Success bar (iteration 3 — dynamic-crop representation & throughput)

1. Full suite green, including new invariants: crop contains all legal moves;
   mask ↔ `legal_moves` bijection; padded forward == unpadded forward; D4
   equivariance on dynamic crops.
2. `param_count` ≤ 250k at new defaults.
3. Measured self-play throughput ≥ 5× the 31×31 baseline on the Mac at equal
   sims (single process), ~linear scaling to 4 workers.
4. End-to-end smoke train run completes and resumes.
5. Yardstick vs GreedyAgent at least matches the old net's trajectory at
   equal games generated.

Iteration-1 bar (rules engine proven; trained agent beats GreedyAgent >70%;
browser play vs AI; resumable Pi run) and iteration-2 bar (engine parametric
over 2-5 players) remain in force. Note: iteration 3 is a clean break for
checkpoints and buffers — pre-iteration-3 checkpoints (31×31 format) no
longer load; there are no real training runs worth migrating.

## Explicitly out of scope

Farmers, expansions beyond the abbot, multiplayer RL training (max^n / vector
value heads), multi-human play, arena/evaluate for 3+ players, Gymnasium
adapter (provisioned but unbuilt), distributed training, MCTS leaf-eval
batching, quantization/torch.compile, GNN representations, checkpoint
migration from the 31×31 format, web auth.
