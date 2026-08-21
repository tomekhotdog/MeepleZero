# ProjectCarcasonne Design

## Research

Greenfield project — the directory was empty at design time. No codebase research
applied; all decisions below come from the design interview (2026-08-21).

Key interview outcomes:

- **Ruleset:** base game **plus abbots**, **no farmers** initially (fields exist only
  as edge-compatibility, never as scorable features).
- **Players:** 2 only. Zero-sum, standard AlphaZero setting.
- **Performance model:** one clean canonical engine, optimise later from profiles.
  No dual implementations.
- **UI stack:** FastAPI backend + one static vanilla-JS canvas frontend. No build step.
- **RL scope:** staged — baselines (random, greedy) prove the harness end-to-end,
  then MCTS + network drops in behind the same **Agent** interface.
- **Compute:** Mac (Apple Silicon, MPS) to prove out; then multi-day CPU-only training
  runs on a Raspberry Pi 5. Small network, resumability is a hard requirement.
- **Gymnasium:** not the core abstraction (MCTS needs state cloning + legal-move
  branching, which the gym API doesn't offer). Our core API is a superset; a
  ~50-line gym adapter can be added in `nn/` later if wanted. Out of scope now.
- **Learning goal:** the user is using this project to learn RL. The visualisation
  must expose the machinery (policy priors vs MCTS visits, value estimates,
  training curves) — a first-class requirement, not polish.

## Problem Statement

Build a Carcassonne simulator in Python that serves three masters at once:

1. A **correct, composable rules engine** — the single source of truth for the game.
2. An **attractive web visualisation** where a human can play, watch replays, and
   *see inside* the AI's decision process (the "RL microscope").
3. An **AlphaZero-inspired training loop** that runs on modest hardware (Mac to
   prove out, Raspberry Pi 5 for multi-day runs) and produces checkpoints a human
   can play against in the browser.

The connective tissue is a **Replay** file: every game — self-play, arena,
human-vs-AI — is recorded in one format that the trainer consumes and the UI renders.

## Solution

**One layered Python package with the Replay file as the spine** (Approach A from
the interview; Approaches B "separate packages" and C "gym-env-first" rejected —
B is overhead without benefit at solo scale, C trades domain clarity for
compatibility purchasable later for ~50 lines).

```
carcassonne/
  core/      # pure domain: tiles, geometry, features, rules, scoring. Zero deps, no I/O.
  game/      # turn orchestration, Replay recording/loading
  agents/    # Agent protocol + RandomAgent, GreedyAgent, MctsAgent
  nn/        # network, state→tensor Encoder, Move↔index ActionIndexer
  training/  # self-play workers, replay buffer, learner, Arena, checkpoints, TrainingRun
  web/       # FastAPI app + static JS canvas frontend
  cli/       # entry points: simulate, train, serve, evaluate
```

Layering is strictly one-way (top of list = bottom of stack). `core` imports
nothing from the project; `nn` may import `core`/`game`; `web` and `training`
sit on top. Enforced by import-lint in CI, not package boundaries.

## User Stories

1. As a player, I want to play Carcassonne in the browser against any agent
   (random, greedy, or a trained checkpoint), so that I can test the AI and enjoy the game.
2. As a learner, I want to see the AI's policy heatmap, value estimate, and
   MCTS visit counts per move, so that I understand how AlphaZero-style search works.
3. As a researcher, I want a `train` command that runs self-play + learning
   unattended for days on a Raspberry Pi 5 and survives interruption, so that I can
   train without babysitting.
4. As a researcher, I want every game stored as a Replay file, so that any game
   can be re-examined move by move with full analytics.
5. As a developer, I want the rules engine behind a five-function interface with
   property-tested invariants, so that everything above it can trust the rules.

## Implementation Decisions

### Core domain (`core/`)

- **Tiny types** (frozen dataclasses / NewTypes): `TileId`, `Rotation` (0/90/180/270),
  `Pos(x, y)`, `EdgeKind` (CITY|ROAD|FIELD), `FeatureKind` (CITY|ROAD|MONASTERY|GARDEN),
  `Player` (0|1), `MeepleKind` (MEEPLE|ABBOT).
- **Declarative deck:** every tile type declared once as data in `core/tiles.py` —
  4 edges + internal features + which edges each feature touches (+ shield flag,
  garden flag). The full 72-tile base distribution lives here. This is the most
  error-prone data in the project → dedicated verification tests.
- **Fields:** exist only as `FIELD` edges for placement compatibility. Never
  scorable, no meeples on them. Farmers are a future extension the tile data
  already anticipates (edges are recorded; field *features* are not).
- **Move = one full turn decision:**
  `PlaceTile(pos, rotation, meeple: FeaturePlacement | None) | RetrieveAbbot(pos)`.
  Single-ply turns keep the game tree at one node per turn — better for MCTS than
  a two-ply split. Accepted trade-off: larger per-node action space (mitigated:
  few legal meeple spots per placement).
- **`GameState` is immutable** (persistent-ish): board `dict[Pos, PlacedTile]`,
  deck as counts + RNG state, meeple supply, scores, current drawn tile.
  Connected cities/roads tracked **incrementally via union-find carried in the
  state**; monasteries/gardens by neighbour counts. `legal_moves`/scoring are
  O(small) per move, never board scans.
- **The deep module — core's entire public interface:**

  ```python
  new_game(seed, config) -> GameState
  legal_moves(state) -> tuple[Move, ...]   # never empty; forced-skip if tile unplayable
  apply(state, move) -> GameState           # pure; raises IllegalMove
  is_terminal(state) -> bool
  final_scores(state) -> tuple[int, int]    # includes end-game partial scoring
  ```

  Union-find, geometry, deck mechanics all hidden behind it.

### Agents (`agents/`)

- **Agent protocol:**

  ```python
  class Agent(Protocol):
      def choose(self, state: GameState, ctx: TurnContext) -> Move: ...
      def policy(self, state) -> dict[Move, float] | None  # optional, for visualisation
  ```

- `RandomAgent` — uniform over legal moves (harness smoke tests).
- `GreedyAgent` — hand-written 1-ply scorer (immediate points + weighted potential
  + meeple efficiency). The evaluation yardstick.
- `MctsAgent` — PUCT search, AlphaZero-style: no rollouts, leaf evaluation by
  network, Dirichlet root noise during self-play, temperature schedule
  (τ=1 early → τ→0), move chosen by visit counts.

### Network & encoding (`nn/`)

- **Encoder:** fixed `B×B` window centred on the board's bounding box (base game
  fits in 31×31; window size configurable). ~30 planes: per-side edge kinds,
  feature-completion progress, meeple ownership/kind, shields; scalar planes for
  scores, supply, deck-remaining, current-tile one-hot.
- **ActionIndexer:** deterministic `Move ↔ int` mapping
  (`window_pos × 4 rotations × meeple_slot` + abbot-retrieval actions) with a
  legal-move mask. Exhaustively round-trip tested.
- **Network:** small residual CNN — 4–6 res-blocks × 64 channels (sized for Pi 5
  CPU inference), policy head (masked logits) + value head (tanh, current-player
  perspective). PyTorch, device-agnostic (`mps`/`cpu`/`cuda`).
- **D4 symmetry** exploited as training-data augmentation (rotate/reflect
  state+policy), not baked into the architecture.
- Quantised/compiled inference on the Pi: a later, measured optimisation. Not designed-in.

### Training harness (`training/`)

Single-machine AlphaZero-inspired loop, one `train` CLI process owning everything:

1. **Self-play workers** (multiprocessing, N = cores−1) play MctsAgent-vs-itself
   with the latest network; each game written as Replay + policy targets (visit
   distributions) into the run directory.
2. **Replay buffer:** window of most recent K positions on disk (mmap or sqlite —
   decided at plan time), uniform sampling.
3. **Learner:** batches from buffer; loss = policy cross-entropy + value MSE + L2;
   periodic checkpoints.
4. **Arena gating:** every N steps, candidate vs current-best over M games
   (+ vs GreedyAgent as absolute yardstick); promote on >55% win rate.
   Gating kept deliberately — it is *the* AlphaGo Zero idea and the Training view
   visualises it (learning goal).
5. **TrainingRun = a directory:** `config.json`, `checkpoints/`, `replays/`,
   `buffer/`, `metrics.jsonl`. `train --resume <run-dir>` reconstructs all state
   from disk; a crash or power cut loses at most the current batch. This is the
   Pi requirement made structural.

### Replay format (`game/`)

One JSONL file per game. Header line: version, seed, config, players/agents,
checkpoint id if AI. Then one line per turn:

```json
{"n": 14, "move": {...}, "score_after": [12, 9],
 "annot": {"value": 0.31, "top_policy": [["<move>", 0.42, 161], ...], "sims": 200, "think_ms": 412}}
```

Engine determinism means `(seed, moves)` alone reproduces every state; `annot`
(null for human moves) preserves what the search *thought* without re-running it.
Written identically by self-play, arena, and the web app.

### Web app (`web/`) — the RL microscope

FastAPI + one static JS canvas page, three views:

1. **Play view** — play vs any agent/checkpoint. Procedurally drawn tiles, current
   tile preview + rotation, legal placements highlighted, click-to-place, meeple
   spots on placed tile. Toggleable **hint overlay**: the AI's policy/value heatmap
   for *your* move.
2. **Replay view** — step through any Replay. Per AI move, the **search panel**:
   top-k candidates as prior-vs-visits bars (showing MCTS revising the network's
   first instinct), plus a whole-game win-probability line chart.
3. **Training view** — reads a TrainingRun directory: loss curves, arena win
   rates, games generated, checkpoint timeline. Watch a Pi run from the Mac browser.

Sessions in memory keyed by id; everything durable is a Replay file or run dir.

### Error handling

- `core` raises typed `IllegalMove` / `RulesError`; nothing above it re-implements rules.
- Web maps domain errors to 4xx with human-readable explanations.
- Training treats worker crashes as recoverable: log, respawn, never silently drop games.

## Testing Decisions

- **Deck-data tests:** tile counts vs official base-game distribution; per-tile
  edge/feature consistency checks.
- **Rules unit tests:** curated tricky scenarios — multi-city merges through one
  tile, road loops, monastery completion, abbot placement/retrieval timing,
  end-game partial scoring.
- **Property tests (hypothesis):** random legal games to termination. Invariants:
  meeple conservation, scores never decrease, `apply` rejects anything not in
  `legal_moves`, replay determinism (same seed+moves ⇒ identical states).
- **Cross-checks:** GreedyAgent beats RandomAgent decisively; ActionIndexer
  round-trips exhaustively; Encoder respects D4 symmetry
  (encode(rot(s)) == rot(encode(s))).
- **Boundary:** core and nn get the deep test investment; web gets thin API tests;
  training gets an integration smoke test (tiny net, tiny run, resumes correctly).

## Success Criteria (verification bar for iteration 1)

1. Engine correctness proven (test suite above, green).
2. Trained agent beats GreedyAgent >70% over a 100+ game arena.
3. Human plays vs the AI in the browser; game saved as Replay; replay viewable
   with per-move analytics.
4. A multi-hour resumable training run demonstrated on the Raspberry Pi 5 with
   improving checkpoints.

## Out of Scope

- Farmers/field scoring (tile data anticipates it; rules do not implement it).
- Expansions beyond the abbot (rivers, inns & cathedrals, etc.).
- 3+ players; multi-agent value heads.
- Gymnasium/PettingZoo adapter (provisioned-for in `nn/`, trivially added later).
- Distributed training; GPU-cluster support (device-agnostic code only).
- Quantised/compiled Pi inference (later, measured optimisation).
- Authentication/multi-user for the web app (localhost tool).
