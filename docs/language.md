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
  scores, current drawn tile, feature connectivity.
- **PlacedTile** — a Tile fixed on the board at a **Pos** (x, y grid coordinate)
  with a **Rotation** (0/90/180/270).

## Engine & recording

- **Engine** — the pure rules core (`carcassonne.core`); sole authority on
  legality and scoring, exposed as five functions.
- **Replay** — a JSONL file fully describing one game: header (seed, config,
  agents) + one line per Move, with optional **Annotation** (the AI's value
  estimate, policy/visit distribution, sims, think time) per move.

## Agents & search

- **Agent** — anything implementing `choose(state, ctx) -> Move`; optionally
  exposes a policy for visualisation. Implementations: **RandomAgent**,
  **GreedyAgent** (1-ply heuristic yardstick), **MctsAgent**.
- **Policy** — a probability distribution over legal Moves (from the network's
  policy head or MCTS visit counts).
- **Value** — the network's estimated game outcome in [-1, 1] from the
  current player's perspective.
- **Prior** — the network policy's probability for a Move before search;
  contrasted with **Visits**, the MCTS visit count after search.

## Neural network

- **Encoder** — maps a GameState to fixed-size tensor planes (windowed board crop).
- **ActionIndexer** — the deterministic bijection between Moves and policy-head
  indices, with legal-move masking.
- **Checkpoint** — a saved network (weights + config + training step id);
  playable from the web app.

## Training

- **TrainingRun** — a directory holding everything about one training session
  (config, checkpoints, replays, buffer, metrics); resumable unit of work.
- **SelfPlay** — MctsAgent vs itself generating Replays + policy targets.
- **ReplayBuffer** — the on-disk window of recent positions the learner samples.
- **Arena** — head-to-head evaluation: candidate vs current-best (gating,
  promote on >55%) and vs GreedyAgent (absolute yardstick).

## Product

- **RL microscope** — the web app's stated purpose beyond play: making the
  AI's internals (Prior vs Visits, Value over time, training curves) visible
  so the project doubles as an RL learning tool.
