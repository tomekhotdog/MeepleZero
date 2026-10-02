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

## Success bars — status (2026-10-02)

Verified at M19 (HEAD `93cc951`): full suite 302 passed / 0 failed, `ruff check`
clean, `mypy carcassonne` clean.

Iteration 1:
- ✅ Rules engine proven — `tests/core/test_properties.py`
  (`test_random_game_invariants`, `test_replay_determinism`, now over 2-5
  players), `tests/core/test_engine_golden.py::test_engine_behaviour_is_unchanged`.
- ❌ *not yet run* — trained agent beats GreedyAgent >70%: needs a real
  training campaign (mechanism built: `evaluate`, arena yardstick).
- ✅ Browser play vs AI — M19 headless-Chrome game vs checkpoint `step_000004`
  played to the end with hints on (every `/hint` 200, value readout shown);
  all 35 AI moves in the saved Replay carry annotations (sims 100, prior/visits
  `top`); UI, saved and engine-recomputed scores agree.
- ❌ *not yet run* — multi-day resumable Pi run (needs the hardware; resume is
  proven in code, see iteration-3 item 4).

Iteration 2 (engine parametric over 2-5 players):
- ✅ `f254d75` (parametric GameState), `a43bb31` (Replay v2); tests
  `tests/core/test_game.py::test_new_game_rejects_bad_player_count`,
  `tests/game/test_match_multiplayer.py`, `tests/web/test_api.py::test_three_player_game_end_to_end`,
  `::test_five_player_cli_replay_serves_through_api` (`93cc951`). M19 browser
  4-player game (human vs greedy/random/greedy) ended with UI scores
  64·18·23·60 = saved Replay = engine recomputation.
- ✅ RL stack raises on 3+ players — `tests/nn/test_encode.py::test_encode_rejects_non_two_player_state`,
  `tests/agents/test_mcts.py::test_mcts_rejects_non_two_player_state` (+ the
  `choose_with_policy` / `visit_policy` variants),
  `tests/web/test_api.py::test_checkpoint_opponents_rejected_beyond_two_players`;
  live API: 3-player game with `ckpt:step_000004` → 422 "checkpoint agents are
  2-player only".

Iteration 3 (dynamic Crop & throughput):
1. ✅ Invariants — `tests/nn/test_actions.py::test_crop_contains_every_legal_move`,
   `::test_legal_mask_bijection`, `tests/nn/test_model.py::test_padded_forward_equals_unpadded`,
   `tests/nn/test_encode.py::test_d4_rotation_equivariance`.
2. ✅ `param_count` 192,673 ≤ 250k — `tests/nn/test_model.py::test_param_budget` (`89971b6`).
3. ❌ Single-process throughput 3.51× vs ≥5× (503 vs ≥717 games/hour) — recorded
   FAIL, accepted by the user (`f5967c5`). Worker scaling ≈2.5× on 4 workers
   (sub-linear); aggregate ≈8.8× the baseline.
4. ✅ Smoke train completes and resumes — `tests/training/test_train_smoke.py`;
   M19 manual run (`--workers 2`): SIGINT during iteration 2 → finished it, exit
   0, checkpoint `step_000004`, `games_generated=8`; `--resume --iterations 2`
   → `games_generated=16`, checkpoint `step_000008`; buffer 16 games / 16
   distinct `game_id`s; no orphan workers.
5. ❌ *not yet run* — yardstick vs GreedyAgent trajectory comparison needs a
   real training campaign.

## Explicitly out of scope

Farmers, expansions beyond the abbot, multiplayer RL training (max^n / vector
value heads), multi-human play, arena/evaluate for 3+ players, Gymnasium
adapter (provisioned but unbuilt), distributed training, MCTS leaf-eval
batching, quantization/torch.compile, GNN representations, checkpoint
migration from the 31×31 format, web auth.
