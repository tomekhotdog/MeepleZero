# ProjectCarcasonne — Product Spec

A Python Carcassonne simulator, browser visualisation, and AlphaZero-inspired
training system. Three deliverables built on one rules engine and one **Replay**
file format.

## The game

- Carcassonne **base game plus abbots**; **no farmers** (fields affect tile
  placement compatibility only, never scoring).
- **2 players**, zero-sum.
- Full scoring: roads, cities (with shields), monasteries, gardens (abbot),
  including end-game partial scoring; abbot early-retrieval rule.

## The engine

- One canonical, pure-Python rules engine (`carcassonne.core`) behind a
  five-function interface: `new_game`, `legal_moves`, `apply`, `is_terminal`,
  `final_scores`.
- Immutable **GameState**; a **Move** is one full turn decision (tile + optional
  meeple, or abbot retrieval).
- Deterministic: `(seed, moves)` exactly reproduces any game.

## Replays

- Every game — self-play, arena, human-vs-AI — is one JSONL **Replay** file:
  header (seed, config, agents/checkpoint ids) + one line per move.
- AI moves carry annotations: value estimate, top policy/visit distribution,
  simulation count, think time. Human moves carry none.

## Agents

- Common **Agent** protocol; implementations: **RandomAgent**, **GreedyAgent**
  (hand-written 1-ply heuristic, the evaluation yardstick), **MctsAgent**
  (PUCT search guided by a policy/value network, AlphaZero-style).

## Training

- Single-machine loop via `train` CLI: sequential self-play → on-disk replay
  buffer → learner (policy CE + value MSE) → **Arena** gating (promote candidate
  on >55% vs current best; GreedyAgent as absolute yardstick). (Self-play is
  sequential in iteration 1; a multiprocess worker pool is a planned follow-up.)
- A **TrainingRun** is a directory (config, checkpoints, replays, buffer,
  metrics); fully resumable with `--resume` — required for multi-day runs on a
  Raspberry Pi 5. Development/prove-out is CPU-only on this Intel Mac (no MPS —
  Apple-Silicon only); training on the Pi (aarch64 CPU). Network sized for Pi
  inference (small residual CNN).

## Web app — the RL microscope

FastAPI + static JS canvas, three views. A first-class goal is exposing the RL
machinery for learning purposes.

1. **Play** — human vs any agent/checkpoint: procedural tile rendering, legal
   placements highlighted, click-to-place, meeple spots; optional hint overlay
   showing the AI's policy/value heatmap for your move. Games saved as Replays.
2. **Replay** — step through any Replay: score deltas, per-move search panel
   (network prior vs MCTS visits, showing search revising the network), win-
   probability chart across the game.
3. **Training** — live view of a TrainingRun: loss curves, arena win rates,
   games generated, checkpoint timeline.

## Success bar (iteration 1)

1. Rules engine proven by unit + property tests (meeple conservation, replay
   determinism, legality invariants).
2. Trained agent beats GreedyAgent >70% over 100+ arena games.
3. Browser play vs AI works end-to-end, producing a viewable annotated Replay.
4. A resumable multi-hour training run demonstrated on the Raspberry Pi 5.

## Explicitly out of scope

Farmers, expansions beyond the abbot, 3+ players, Gymnasium adapter (provisioned
but unbuilt), distributed training, web auth.
