# Features — ProjectCarcasonne

High-level user journeys. Status: journeys 1-6 **implemented**; remaining
*(planned)* items — web multiplayer views, MCTS tree reuse, multiprocess
self-play — are tracked in `docs/plans/2026-09-20-mega-plan.md`.
Two success criteria are runtime outcomes rather than code — training an
agent to beat greedy >70%, and a multi-day run on the Pi — see
`docs/plans/2026-08-22-iteration1-verification.md`.

## 1. Simulate a game (CLI)

Run `simulate` with agents and a seed; the game plays to completion and is
written as an annotated Replay file. 2-5 agents via
`--agents random,greedy,greedy` (or `--p0/--p1` as 2-player shorthand);
checkpoint agents are 2-player-only.

## 2. Play in the browser

Open the Play view, pick an opponent (any Agent or Checkpoint). See the drawn
tile, rotate it, see all legal placements highlighted, click to place, choose a
meeple spot. Optionally toggle the hint overlay to see the AI's policy/value
heatmap for your own move. Finished games are saved as Replays.
*(planned — web multiplayer)*: one human vs 1-4 agents with per-**Seat** score
cards and colours; consecutive AI seats step automatically. Checkpoint
opponents and the hint overlay stay 2-player-only (cleanly rejected/hidden
otherwise).

## 3. Review a replay

Open any Replay in the Replay view and step through it move by move: board
evolution, score deltas, and for AI moves the search panel — network Prior vs
MCTS Visits per candidate move, and a win-probability chart across the game.
*(planned — web multiplayer)*: 2-5 player Replays render with per-Seat score
chips and **Winners** at the end; the search panel and win-probability chart
remain 2-player features.

## 4. Train an agent

Run `train` (Mac to prove out, Raspberry Pi 5 for multi-day runs). Self-play
generates games, the learner updates the network, the Arena gates promotions.
Interrupt at any time; `train --resume <run-dir>` continues losslessly.
Training is **strictly 2-player**; the RL stack raises loudly on other player
counts rather than failing silently. The network sees a dynamic **Crop** (board
bounding box + 1) with **FeaturePlanes** from the feature graph — a ~193k-param
fully-convolutional net whose evaluation cost scales with actual board area;
policy targets are keyed by world coordinates.
*(planned)*: multiprocess self-play worker pool and MCTS tree reuse.

## 5. Watch training

Open the Training view pointed at a TrainingRun directory (e.g. the Pi's, from
the Mac's browser): loss curves, arena win rates vs best/greedy, games
generated, checkpoint timeline. (Unchanged by multiplayer.)

## 6. Evaluate agents

Run `evaluate` to pit any two agents/checkpoints over N games and get win rates
plus Replays — used both by the harness (gating) and manually (is my new
checkpoint actually better?). (Stays strictly two-agent head-to-head.)
