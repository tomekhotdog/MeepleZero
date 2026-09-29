# meeplezero

AlphaZero-style self-play reinforcement learning for Carcassonne — small enough to train on a Raspberry Pi 5, instrumented so you can watch it think.

A pure, deterministic rules **Engine** (base game + abbots, no farmer scoring; 2–5 players) feeds an AlphaGo-Zero-style loop: a policy/value residual CNN guides PUCT **MCTS**, self-play games fill a replay buffer, a learner trains on search visit distributions and ±1 game outcomes, and candidates are only promoted after beating the incumbent in a seat-swapped arena (win rate > 0.55). The RL stack plays the 2-player game and raises loudly if handed anything else. A FastAPI web app doubles as an **RL microscope**: play against checkpoints, inspect Prior vs. Visits per move, replay games, and watch training live.

## Quickstart

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --all-extras          # ml extra = torch (pinned 2.2.x for Intel macOS)
uv run pytest                 # full suite

# agent-vs-agent match, no training needed
uv run carcassonne evaluate --p0 greedy --p1 random --seed 0 --games 20

# a 4-player game (engine, agents, replays are 2-5 player; training is 2-player)
uv run carcassonne simulate --agents greedy,random,random,random --seed 5 --out replays

# start a resumable training run (arena-gates vs best + greedy yardstick built in)
uv run carcassonne train --run runs/first --channels 48 --n-blocks 4 --sims 100 --selfplay-games 20
uv run carcassonne train --run runs/first --resume   # continue after any interruption

# play in the browser — vs. trained checkpoints, with replay viewer + training dashboard
uv run carcassonne serve --checkpoints runs/first/checkpoints
```

## How it fits together

**Choosing a move** — the inference path every agent turn takes:

```mermaid
flowchart LR
    GS["GameState<br>core/ — immutable,<br>deterministic"] --> ENC["Encoder<br>52 planes × H×W<br>(dynamic Crop)"]
    ENC --> NET["CarcassonneNet<br>policy H·W·56 · value [−1,1]"]
    NET --> MCTS["MctsAgent<br>PUCT search"]
    GS -- "legal_moves · apply" --> MCTS
    MCTS -- "Move" --> GS
```

**Training** — the AlphaZero loop, one iteration per cycle:

```mermaid
flowchart LR
    SP["SelfPlay<br>best net, both seats"] --> BUF[("buffer.sqlite<br>targets only — states<br>re-derived by replay")]
    BUF --> LRN["Learner<br>policy CE + value MSE"]
    LRN --> GATE{"arena gate<br>win rate &gt; 0.55?"}
    GATE -- "promote: best ← learning" --> SP
    GATE -- "reject: best unchanged" --> SP
```

**The RL microscope** — every artifact the loop writes has a view in the web app:

```mermaid
flowchart LR
    RPL["replays/ (JSONL)"] -.-> WORK["Replay workbench"]
    CKPT["checkpoints/ (*.pt)"] -.-> PLAY["Play vs checkpoints<br>hints · prior vs visits"]
    MET["metrics.jsonl"] -.-> DASH["Training dashboard<br>loss · gate win rates"]
```

The dependency rule matches the arrows: `core → game → agents/nn → training/web`, enforced one-way — the engine knows nothing about the network, and the RL stack consumes the engine only through its five-function API.

## The RL setup at a glance

| Piece | Design |
|---|---|
| Observation | 52 planes over the **Crop** — the board's bounding box + 1-cell border, so shape is H×W per state and every legal placement is representable by construction. 21 spatial planes, 8 **FeaturePlanes** (each city/road's size, open edges, and meeple counts, read off the engine's feature graph — long-range connectivity convolutions can't infer), 15 current-tile structural planes, 8 broadcast scalars; current-player perspective (`nn/encode.py`) |
| Actions | world-coordinate keys `(x, y, rotation, sub-action)` — the engine's own vocabulary, stable across crops; laid out per state as H·W·56 logits (4 rotations × 14 sub-actions per cell), illegal moves masked to −∞ (`nn/actions.py`) |
| Network | fully-convolutional residual CNN, 48 channels × 4 blocks, ~193k params — accepts any board size. GroupNorm with masked statistics and a masked global-average-pool value head make a padded batch bit-identical to unpadded forwards; single-state inference costs scale with actual board area (~2 ms on an empty board → ~10 ms end-game on one CPU core) (`nn/model.py`) |
| Search | PUCT MCTS, no rollouts; Dirichlet root noise + opening temperature in self-play (`agents/mcts.py`) |
| Reward | terminal only: +1 win / −1 loss / 0 draw — no shaping (`training/selfplay.py`) |
| Buffer | sliding window of 2,000 games in SQLite; stores world-keyed policy targets only, re-derives states by deterministic replay (`training/buffer.py`) |
| Gating | best net generates data; learner promoted only on arena win rate > 0.55, plus a greedy-agent yardstick (`training/arena.py`) |

A `TrainingRun` is just a directory (`config.json`, `checkpoints/`, `replays/`, `buffer.sqlite`, `metrics.jsonl`) — SIGTERM-safe and resumable, so a run started on a laptop continues on the Pi under systemd.

## Layout

```
carcassonne/
  core/       pure rules engine: five functions over an immutable GameState
  game/       match orchestration, arena, JSONL replays
  agents/     random · greedy · MCTS, one choose(state, ctx) protocol
  nn/         encoder, action indexer, network, checkpoints
  training/   self-play, buffer, learner, gating, resumable runs
  web/        FastAPI app: play, replay workbench, training dashboard
docs/         spec, ubiquitous language, plans, Pi setup
```

## Status

Working today: the 2–5 player engine, agents, replays, and CLI; the dynamic-crop NN stack and training loop (2-player, arena-gated, resumable); browser play/replay/training views for 2-player games. In progress (see `docs/plans/2026-09-20-mega-plan.md`): MCTS subtree reuse, multiprocess self-play, 2–5 player browser views. Not built: multiplayer RL (max^n search), the first multi-day Pi training campaign.

---

Carcassonne is a trademark of Hans im Glück. This is an unaffiliated research/learning project; it contains no assets from the published game.
