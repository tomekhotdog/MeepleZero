# Replay Analysis Workbench — Design

Date: 2026-08-23. Source: playtest feedback (`docs/feedback/tomek-2026-08-23.md`)
plus a follow-up brainstorm to expose far more of what a replay contains.

## Problem

The Replay view currently shows a board scrubber, a "this move" line, a
prior-vs-visits search panel, and a win-probability chart in one cramped 300px
sidebar. A replay actually holds much more — the exact move played, the
alternatives MCTS weighed, running score, tiles played/remaining, meeple supply,
per-feature values — but almost none of it is surfaced, and there's no room to.
Goal: turn the Replay view into an **analysis workbench** with total visibility
into a recorded game. Separately, the Play tab gets a small tile counter.

## Layout — 3-column analysis workbench (approved)

```
┌ nav ──────────────────────────────────────────────┐
├───────────┬────────────────────────┬───────────────┤
│ move list │        BOARD           │  THIS MOVE    │
│  #0       │   (overlays: current-  │   search-read │
│  #1  ◀    │    move highlight,     │   alternatives│
│  #2       │    alternative ghosts, │   score event │
│  ...      │    meeple hover)       ├───────────────┤
│ (scrolls) │                        │  THIS GAME    │
│           ├────────────────────────┤   score+chart │
│           │ ◀◀ ◀ ▶ ▶▶ [===|====]   │   win-prob    │
│           │ scrubber + key markers │   deck tracker│
│           │                        │   meeple supp │
└───────────┴────────────────────────┴───────────────┘
```

- **Left rail (~180px):** the move list — always visible for navigation.
- **Centre (flex):** board with overlays on top; scrubber + key-moment markers
  directly beneath it.
- **Right sidebar (~320px):** two stacked, labelled sections — **This move**
  (updates as you scrub) and **This game** (game-wide). Each section's panels
  stack; the sidebar scrolls if needed.
- Existing replay picker moves to the top of the left rail (above the move list).
- Reuses the shared `#app` flex shell + `board.js`; the top nav is unchanged.
- 1024px is the min supported width (tightest but workable); no horizontal
  page scroll — inner columns scroll.

## Data sources (almost all already present)

`GET /api/replays/{name}` returns `header`, `moves[]` (each: `n, player, tile,
move{x,y,rot,action}, score_after, annot{value,sims,think_ms,top:[{move,prior,
visits}]}|null`), `states[]` (a full StateView per position, incl. terminal),
`final_scores`, `winner`. Each StateView tile's meeples already carry
`feature_tiles/score_now/score_potential/feature_kind/complete` (from the Play
F3 work), so meeple hover and per-feature values are **free** in replay.

**Only backend change:** add per-type `count` to each `/api/tiledefs` entry
(`TileType.count` already exists) so the deck tracker knows the full 72-tile
distribution. No new endpoints.

## Features

### Board overlays (centre canvas; reuse Play F1/F3 patterns + board.js)
- **① Current-move highlight.** Ring the tile placed at the current step in the
  mover's colour (`--p0`/`--p1`); if the move placed a meeple, mark it; dim the
  rest of the board slightly so the eye lands on the move. (For step 0 / the
  start tile, nothing to highlight.)
- **② Alternative-move ghosts.** From the current move's `annot.top`, draw each
  candidate placement as a translucent ghost tile at its `(x,y,rot)`, opacity ∝
  visit share; the played move is solid + ringed (①). Hover a ghost → a small
  tag with prior% and visit count. **Off by default**, toggled by a "Show
  alternatives" control in the This-move section. Only AI moves with real search
  have `top` with visits; greedy moves show priors-only (visits 0, note it);
  human/`annot==null` moves show a "no search data" state and the toggle is
  disabled.
- **④ Score-event spotlight.** The events for move `i` are on
  `states[i+1].last_events` (each `{player, points, kind, tiles:[[x,y]...]}`).
  When the current move produced events, highlight the exact tiles of each
  scored feature and float a "+N" near them. Reuses the F2 tile-highlight
  drawing. (No events → nothing shown.)
- **Meeple hover (replay).** Reuse the Play F3 tooltip + feature-tile highlight,
  driven by the already-present `feature_report` fields on each state's meeples.

### Left rail
- **⑨ Move list.** One row per move: `n · player swatch · tile id · score Δ ·
  event dots`. Current move highlighted; click to jump; auto-scrolls to keep the
  current row in view. Its own internally-scrolling column.

### Centre, under the board
- **⑩ Key-moment markers.** Ticks on the scrubber timeline for: win-prob lead
  changes (crossing 0.5), large score swings, and the first meeple placement.
  Click a tick to jump. Derived client-side from `states`/`annots`.

### Right sidebar — "This move" section
- **⑧ Enriched search-read** (extends the existing panel): value as a win% with
  Δ vs the previous move ("AI 62% ▲ +8%"), sims + think-time, and a flag —
  "search agreed with its top prior" vs "search overruled it" (compare argmax
  prior vs argmax visits). Keeps the prior-vs-visits bars.
- Move detail (who/tile/placement) — the existing "this move" line, kept here.
- **② toggle** ("Show alternatives") lives here.

### Right sidebar — "This game" section
- **③ Running score.** Big `P0 : P1` readout at the current step + a dual-line
  canvas chart of actual points vs move number (separate from win-prob).
- Win-probability chart — the existing chart, moved here.
- **⑤ Deck tracker.** "N/72 placed · M remaining" + a compact grid of tile types
  with placed/remaining counts (from `/api/tiledefs` counts vs placed tiles in
  the current state); the current drawn tile flagged.
- **⑦ Meeple supply.** Each player's meeple/abbot pips at the current step
  (reuse Play's pip UI), from the state's `meeples`/`abbots`.

### Play tab
- **⑥ Tile counter.** A compact "played X · remaining Y" line in the Play
  sidebar. `played = tiles.length`; `remaining = 72 − played`. (No deck grid on
  Play — counts only, per the ask.)

## Reuse / shared code

- `board.js` (camera + `drawTiles`/`drawFrontier`) — already shared by Play and
  Replay. Board overlays add draw passes on top, mirroring Play's `drawLastMove`
  and the F2/F3 highlight/tooltip code. Where a drawing helper is genuinely
  identical to Play's, factor it into `board.js` (or a small `overlays.js`)
  rather than copying; keep it simple.
- `tiles.js` (`drawTile`, `drawMeepleGlyph`, `featureAnchor`) — for ghosts and
  meeple hit-testing.
- `feature_report` fields — already in every replay StateView meeple.

## Defaults chosen
- Alternative ghosts **off by default** (toggle in This-move).
- Move list lives in the **left rail** (not a sidebar panel).

## Build phasing (two reviewable passes)

- **Pass A — restructure + per-move insight:** rebuild `replay.html`/`replay.js`
  into the 3-column workbench; move list (⑨); board overlays ① ② ④ + replay
  meeple-hover; enriched search-read (⑧); Play tile counter (⑥, tiny, rides
  along). Ship the win-prob chart and existing "this move" info relocated into
  the new columns so nothing regresses.
- **Pass B — game-wide visibility:** running score + chart (③), deck tracker (⑤,
  incl. the `/api/tiledefs` count backend change), meeple supply (⑦), key-moment
  scrubber markers (⑩).

## Testing

- Backend: web test asserting `/api/tiledefs` entries carry a positive `count`
  summing to 72.
- Frontend: `node --check` on the JS; the web suite stays green (update any test
  that asserts the old replay DOM structure); a headless Playwright pass per
  feature — drive a **checkpoint** replay (real MCTS) so ② has genuine data,
  plus a greedy/human replay to exercise the "no search data" states; inspect
  screenshots.
- Regression guard: the existing win-prob chart and replay verification
  (`load_replay`/`replay_states`) must still work after the restructure.

## Out of scope (candidate iteration 3)
Per-player placement heatmap; "what greedy would do here" overlay; exporting an
annotated game summary. Noted, not built.
