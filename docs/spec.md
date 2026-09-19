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

## Training — strictly 2-player

The RL stack (encoder, network, MCTS, self-play, learner, arena) assumes
2-player zero-sum play and **raises loudly** if handed a 3-5 player state —
it never fails silently. Multiplayer RL (max^n search, vector value heads) is
deferred research; groundwork in
`docs/plans/2026-08-23-multiplayer-3-5-research.md`.

- Single-machine loop via `train` CLI: sequential self-play → on-disk replay
  buffer → learner (policy CE + value MSE) → **Arena** gating (promote candidate
  on >55% vs current best; GreedyAgent as absolute yardstick). (Self-play is
  sequential; a multiprocess worker pool is a planned follow-up.)
- A **TrainingRun** is a directory (config, checkpoints, replays, buffer,
  metrics); fully resumable with `--resume` — required for multi-day runs on a
  Raspberry Pi 5. Development/prove-out is CPU-only on this Intel Mac (no MPS —
  Apple-Silicon only); training on the Pi (aarch64 CPU). Network sized for Pi
  inference (small residual CNN).

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

## Success bar (iteration 2 — multiplayer engine)

1. Engine property tests (meeple conservation, replay determinism, legality,
   turn rotation) pass parametrised over 2-5 players; full existing suite
   stays green.
2. `simulate` runs a 5-player greedy-vs-randoms game to completion and the
   Replay opens in the Replay view.
3. Browser Play: one human vs 3 agents end-to-end, correct per-seat scoring,
   game saved and reviewable.
4. Encoder and MctsAgent demonstrably raise on a 3-player state (tests);
   existing 2-player checkpoints load and play unchanged.

Iteration-1 bar (rules engine proven; trained agent beats GreedyAgent >70%;
browser play vs AI; resumable Pi run) remains in force — see
`docs/plans/2026-08-22-iteration1-verification.md`.

## Explicitly out of scope

Farmers, expansions beyond the abbot, multiplayer RL training (max^n / vector
value heads), multi-human play, arena/evaluate for 3+ players, Gymnasium
adapter (provisioned but unbuilt), distributed training, web auth.
