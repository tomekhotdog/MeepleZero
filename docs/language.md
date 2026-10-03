# Ubiquitous Language — ProjectCarcasonne

Canonical terms for code, docs, commits, and conversation. PascalCase signals a
defined term; in docs, bold on first use per section.

## Domain (game)

- **Tile** — one square card with four edges and internal features. A tile *type*
  is declared once as data; the deck holds counted copies.
- **Edge** — one of a Tile's four sides; its **EdgeKind** is CITY, ROAD, or FIELD.
  Placement requires matching EdgeKinds on touching edges.
- **Feature** — a scorable structure on/through tiles; **FeatureKind** is CITY,
  ROAD, MONASTERY, or GARDEN. (Fields are edges, not Features — no farmers.)
- **Meeple** — a follower placed on a Feature; **MeepleKind** is MEEPLE or ABBOT.
  The **Abbot** may only be placed on a MONASTERY or GARDEN and may be retrieved
  in lieu of placing a Meeple.
- **Move** — one full turn decision: `Move(pos, rotation, action)` where the
  action is `None`, `PlaceMeeple(feature, kind)`, or `RetrieveAbbot()` (a tile
  is always placed; abbot retrieval replaces the meeple step). One Move = one
  ply in the game tree.
- **GameState** — the immutable complete state of a game: board, deck, supplies,
  scores, current drawn tile, feature connectivity. Per-player state is sized
  by the game's player count (2-5).
- **Seat** — a player position, `0..num_players-1`; turn order rotates through
  seats. "Player" in code is a seat index.
- **Winners** — the tie set of top scorers at game end (singleton = outright
  win, larger = shared win). Replaces the old binary `winner`.
- **PlacedTile** — a Tile fixed on the board at a **Pos** (x, y grid coordinate)
  with a **Rotation** (0/90/180/270).

## Engine & recording

- **Engine** — the pure rules core (`carcassonne.core`); sole authority on
  legality and scoring, exposed as five functions.
- **Replay** — a JSONL file fully describing one game: header (seed, config
  incl. player count, agents) + one line per Move, with optional
  **Annotation** (the AI's value estimate, policy/visit distribution, sims,
  think time) per move; footer carries final scores and **Winners**.
  Format v2; v1 (2-player) files remain loadable.
- **GoldenDigest** — the sha256 of Engine behaviour over seeded random games at
  2 and 4 players: every `legal_moves` result (content *and* order) plus the
  resulting scores, supplies, current tile, discards, and score events (tile
  sets hashed sorted) (`tests/core/test_engine_golden.py`). Recorded on the
  pre-optimisation Engine; a changed digest is a behaviour change. It must
  never be re-recorded to make a change pass.

## Agents & search

- **Agent** — anything implementing `choose(state, ctx) -> Move`; optionally
  exposes a policy for visualisation. Implementations: **RandomAgent** (any
  player count), **GreedyAgent** (1-ply heuristic yardstick: my delta minus
  best-of-rest delta; any player count), **MctsAgent** (2-player only —
  raises on other player counts).
- **Policy** — a probability distribution over legal Moves (from the network's
  policy head or MCTS visit counts).
- **Value** — the network's estimated game outcome in [-1, 1] from the
  current player's perspective.
- **Prior** — the network policy's probability for a Move before search;
  contrasted with **Visits**, the MCTS visit count after search.
- **SubtreeReuse** — after a Move is played, MctsAgent keeps the played
  child's search subtree (the *stash*) and starts its next search from it if
  the new GameState equals the stash itself (self-play: one agent, both seats)
  or one of the stash's children (arena: the opponent moved in between). Exact,
  not approximate, because the deck lives in GameState, so an equal state is
  the same draw. The reused root gets fresh Dirichlet noise, and its inherited
  Visits flow into the Policy target. Disabled for interactive checkpoint
  agents (`reuse_tree=False`), because web hints call `choose` and a hint's
  stash would bias the AI's next reply.

## Neural network

- **Crop** — the per-state board window fed to the network: the bounding box
  of placed tiles plus a 1-cell border. By the adjacency rule it contains
  every legal placement; its size H×W varies per state (replaces the fixed
  31×31 window).
- **Key** — a Move in world coordinates, `(x, y, rotation, sub-action)`: the
  stable vocabulary the ReplayBuffer stores policy targets in. The policy head
  sees a per-state *flat index* over the Crop instead,
  `((gy*W + gx)*4 + rot)*14 + a` (grid cell, rotation, sub-action). It is
  derived from the Key at encode time and is never stored (`nn/actions.py`:
  `flat_index`, `move_indices`, `legal_mask`).
- **Encoder** — maps a GameState to `(planes, H, W)` tensor planes over the
  Crop; includes FeaturePlanes and the current tile as structural property
  planes.
- **FeaturePlanes** — input planes derived from the engine's feature graph
  (per-tile city/road size, open-edge count, meeple ownership); hands the
  net long-range connectivity instead of making convs infer it.
- *ActionIndexer* — **retired** (iteration ≤2): the fixed bijection between
  Moves and the 31×31 window's policy indices. It was replaced by the **Key**
  plus the per-state flat index over the Crop (see **Key**).
- **Checkpoint** — a saved network (weights + config + training step id);
  playable from the web app. Pre-iteration-3 (31×31) checkpoints do not load.

## Training

- **TrainingRun** — a directory holding everything about one training session
  (config, checkpoints, replays, buffer, metrics); resumable unit of work.
- **SelfPlay** — MctsAgent vs itself generating Replays + policy targets;
  runs in a SelfplayPool when `--workers` > 1, and in-process otherwise.
- **SelfplayPool** — the spawn-based pool of worker processes that play whole
  self-play games, each worker with its own copy of the best net. The parent
  process is the only one that writes to the ReplayBuffer. The pool is
  recreated when a new net is promoted (`training/pool.py`).
- **ReplayBuffer** — the on-disk window of recent positions the learner samples.
- **Arena** — head-to-head evaluation: candidate vs current-best (gating,
  promote on >55%) and vs GreedyAgent (absolute yardstick).

## Product

- **RL microscope** — the web app's stated purpose beyond play: making the
  AI's internals (Prior vs Visits, Value over time, training curves) visible
  so the project doubles as an RL learning tool.
