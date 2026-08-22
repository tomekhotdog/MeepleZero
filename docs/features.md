# Features — ProjectCarcasonne

High-level user journeys. Status: all **implemented** (iteration 1 complete,
2026-08-22; 174 tests green). Two success criteria are runtime outcomes rather
than code — training an agent to beat greedy >70%, and a multi-day run on the
Pi — see `docs/plans/2026-08-22-iteration1-verification.md`.

## 1. Simulate a game (CLI)

Run `simulate` with two agents (random/greedy/checkpoint) and a seed; the game
plays to completion and is written as an annotated Replay file.

## 2. Play in the browser

Open the Play view, pick an opponent (any Agent or Checkpoint). See the drawn
tile, rotate it, see all legal placements highlighted, click to place, choose a
meeple spot. Optionally toggle the hint overlay to see the AI's policy/value
heatmap for your own move. Finished games are saved as Replays.

## 3. Review a replay

Open any Replay in the Replay view and step through it move by move: board
evolution, score deltas, and for AI moves the search panel — network Prior vs
MCTS Visits per candidate move, and a win-probability chart across the game.

## 4. Train an agent

Run `train` (Mac to prove out, Raspberry Pi 5 for multi-day runs). Self-play
workers generate games, the learner updates the network, the Arena gates
promotions. Interrupt at any time; `train --resume <run-dir>` continues losslessly.

## 5. Watch training

Open the Training view pointed at a TrainingRun directory (e.g. the Pi's, from
the Mac's browser): loss curves, arena win rates vs best/greedy, games
generated, checkpoint timeline.

## 6. Evaluate agents

Run `evaluate` to pit any two agents/checkpoints over N games and get win rates
plus Replays — used both by the harness (gating) and manually (is my new
checkpoint actually better?).
