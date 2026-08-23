# Replay Analysis Workbench Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use tomek-superpowers:build to implement this plan task-by-task.

**Goal:** Rebuild the Replay view into a 3-column analysis workbench exposing 10
new content features + replay meeple-hover; add a tile counter to the Play tab.

**Architecture:** Almost entirely frontend (`web/static/replay.{html,js}`,
`app.js`, `style.css`) reusing `board.js`/`tiles.js` and the data already in
`GET /api/replays/{name}` (states carry per-meeple `feature_report`; AI moves
carry `annot.top` prior+visits+value; `score_after`/`last_events` per move). One
tiny backend change: add per-type `count` to `/api/tiledefs`.

**Tech stack:** vanilla JS canvas (no build), FastAPI, pytest, Playwright (headless
verify via `npx --no-install playwright`).

**Design:** `docs/superpowers/specs/2026-08-23-replay-analysis-workbench-design.md`

**Environment:** run everything with `PYTHONPATH="$PWD"` prefix (editable-install
`.pth` is unreliable under iCloud). After each commit run `git log -1`/`git status`
to confirm it stuck. Do NOT run git-mutating agents concurrently with build work
(see docs/lessons.md). Keep `PYTHONPATH="$PWD" uv run {pytest,ruff check .,mypy carcassonne tests}` green.

**Frontend granularity note:** backend + test code is given complete; frontend
tasks give exact behavioural contracts + verification (implementation within the
contract, headlessly verified) — matching how the Play view was built.

---

## Pass A — restructure + per-move insight

### Task 1: Backend — per-type tile count on /api/tiledefs ✅ DONE (23b25bd)
**Depends on:** none (independent; different files from the replay work)

**Files:** Modify `carcassonne/web/views.py` (`tiledefs_view`). Test: `tests/web/test_api.py`.

**Step 1: failing test** (add to test_api.py):
```python
def test_tiledefs_carries_counts_summing_to_72(client: TestClient) -> None:
    tiles = client.get("/api/tiledefs").json()["tiles"]
    assert all(t["count"] >= 1 for t in tiles.values())
    assert sum(t["count"] for t in tiles.values()) == 72
```
**Step 2:** `PYTHONPATH="$PWD" uv run pytest tests/web/test_api.py::test_tiledefs_carries_counts_summing_to_72` → FAIL (no `count`).
**Step 3:** in `tiledefs_view`, add `"count": tt.count,` to each tile dict.
**Step 4:** run → PASS; full `tests/web` green; ruff+mypy clean.
**Step 5:** commit `feat(web): expose per-type tile count on /api/tiledefs` (+ Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>).

### Task 2: Replay 3-column workbench layout (restructure, no new content yet) ✅ DONE (443a069)
**Depends on:** none

Rebuild `replay.html` + the layout half of `replay.js` + `style.css` into the
approved 3-column shell WITHOUT regressing existing behaviour. Relocate the
existing panels into the new columns; add empty labelled containers for the
new panels (filled by later tasks).

**Layout contract** (see design for the wireframe):
- Body flex column: `#topnav` (unchanged) then `#app`.
- `#app` = 3 columns: left rail `#move-rail` (~180px, its own scroll), centre
  `#stage` (flex: board canvas + scrubber beneath), right `#analysis` (~320px,
  scrolls) containing two labelled sections `#this-move` and `#this-game`.
- Move the existing **replay picker** to the top of `#move-rail`.
- Move the existing **scrubber** (`#scrubber`) beneath the board in `#stage`.
- Move **move-info**, **search-panel** into `#this-move`; **win-prob** into
  `#this-game`. Add empty placeholder containers (hidden until filled):
  `#move-list` (in rail), `#score-panel`, `#deck-panel`, `#supply-panel` (in
  this-game), and an alternatives toggle slot in this-move.
- Reuse `board.js` camera/`resizeToDisplay`; board must still resize correctly
  in the narrower centre column. Design tokens unchanged; `--ai` stays reserved
  for AI-analysis (search-read/value only).

**Verification:** `node --check` on replay.js; extend `tests/web/test_static.py`
if it asserts replay DOM ids (update to new structure); headless: load `/replay`
with an existing replay, confirm the 3 columns render, the board + scrubber +
win-prob + search panel still work (scrub through, no console errors), no
horizontal page scroll at 1024px. Screenshot + inspect.
**Commit:** `refactor(web): replay view as 3-column analysis workbench`.

### Task 3: Move-list rail (⑨) ✅ DONE (211ffe4 + 1-based move number)
**Depends on:** Task 2

Populate `#move-list` in the left rail: one row per move — `n · player swatch ·
tile id · score Δ · event dots`. Current move highlighted; click a row jumps the
scrubber to that move; list auto-scrolls to keep the current row visible.
Rows keyboard-activatable (button/role) + focus-visible. Score Δ from
`states[i+1].scores` vs `states[i].scores` (for the mover). Event dot when
`states[i+1].last_events` non-empty.
**Verify:** headless — rows render for every move; clicking row N sets position N;
current row highlighted + scrolled into view. Screenshot.
**Commit:** `feat(web): replay move-list rail with jump-to-move`.

### Task 4: Board overlays — current-move highlight (①), score-event spotlight (④), meeple hover ✅ DONE (c3df249 + empty-anchor guard ab2ecbb; shared overlays.js)
**Depends on:** Task 2

On the replay board canvas, per current step:
- **①** ring the tile placed this step (`moves[i].move` pos) in the mover's
  colour; mark its placed meeple if any; dim the rest subtly. (Step 0 = start
  tile, nothing to ring.)
- **④** if `states[i+1].last_events` non-empty, highlight each event's `tiles`
  and float a small `+points` near them (reuse the Play F2 highlight drawing).
- **Meeple hover:** reuse the Play F3 pattern (hit-test meeples drawn this frame;
  tooltip with `feature_kind`, `score now`, `if completed`/`complete`; highlight
  `feature_tiles`) — data already on each state's meeples.
Reuse/port the Play helpers; where a drawing helper is identical, factor into
`board.js` rather than copying.
**Verify:** headless on a real replay — after scrubbing to a scoring move, the
scored tiles highlight + "+N" shows; the placed tile is ringed; hovering a meeple
shows the tooltip + feature highlight. Screenshot each.
**Commit:** `feat(web): replay board overlays — move highlight, score spotlight, meeple hover`.

### Task 5: Alternative-move ghosts (②) + toggle ✅ DONE (3748d9c)
**Depends on:** Task 2, Task 4 (chosen-move ring already drawn)

Add a "Show alternatives" toggle in `#this-move` (off by default). When on and
the current move has `annot.top` with visits>0: draw each candidate `top[k].move`
as a translucent ghost tile at its `(x,y,rot)`, opacity ∝ `visits / sum(visits)`;
the played move stays solid+ringed (from ①). Hover a ghost → a small tag with
`prior%` and `visits`. States where `annot` is null → toggle disabled + "no search
data"; greedy annots (visits all 0) → show priors-only with a note.
**Verify:** headless — needs a CHECKPOINT replay (real MCTS). Generate one:
`PYTHONPATH="$PWD" uv run python -m carcassonne.cli.main` … actually produce it
via the web API against `ckpt:latest` (runs/mac-demo/checkpoints), or reuse an
existing annotated replay. Toggle on → ghosts appear at candidate cells; toggle
off → gone; on a human/greedy replay → disabled/no-data state. Screenshot.
**Commit:** `feat(web): replay alternative-move ghosts (prior vs visits on the board)`.

### Task 6: Enriched search-read panel (⑧) ✅ DONE (66f457f)
**Depends on:** Task 2

Extend the existing prior-vs-visits panel in `#this-move`: show `annot.value` as
a player-0-perspective win% with Δ vs the previous annotated move
("AI 62% ▲ +8%"); show `sims` + `think_ms`; add a flag comparing argmax-prior vs
argmax-visits ("search agreed with its top prior" / "search overruled it"). Keep
the existing bars. Null-annot moves show the existing "no search data".
**Verify:** headless on a checkpoint replay — value%, Δ, sims/think shown; the
agreed/overruled flag matches a hand-check on one move. Screenshot.
**Commit:** `feat(web): enriched replay search-read (value %, delta, agree/overrule)`.

### Task 7: Play tab tile counter (⑥) ✅ DONE (dfd5c70)
**Depends on:** none (Play view files only — parallel with all replay work)

Add a compact "tiles played X · remaining Y" line to the Play sidebar
(`index.html` + `app.js` + `style.css`). `played = S.view.tiles.length`;
`remaining = 72 - played`. Updates every render. mono, `--bone-dim`.
**Verify:** headless on `/` — the counter shows and increments after a move.
Screenshot. `node --check` app.js.
**Commit:** `feat(web): tiles played/remaining counter on the Play tab`.

---

## Pass B — game-wide visibility

### Task 8: Running score + score-over-time chart (③)
**Depends on:** Task 2

In `#score-panel` (this-game): a big `P0 : P1` readout at the current step, plus a
dual-line canvas chart of actual points vs move number (from each `states[i]`
scores), current-move marker, distinct colours per player (`--p0`/`--p1`, NOT
`--ai`). Reuse the win-prob chart's canvas idiom + `resizeToDisplay`.
**Verify:** headless — readout matches `states[i].scores`; chart draws two lines +
marker; clicking is not required. Screenshot.
**Commit:** `feat(web): replay running score + score-over-time chart`.

### Task 9: Deck tracker (⑤)
**Depends on:** Task 2, Task 1

In `#deck-panel`: header "N/72 placed · M remaining"; a compact grid of tile
types (from `/api/tiledefs` `count`) showing placed vs remaining per type at the
current step (placed = count of that type among `states[i].tiles`); flag the
current drawn tile (`states[i].current_tile`). Fetch tiledefs once (already
fetched for drawing) — reuse.
**Verify:** headless — totals sum to 72; remaining decreases as you scrub
forward; current tile flagged. Screenshot.
**Commit:** `feat(web): replay deck tracker (tiles played/remaining by type)`.

### Task 10: Meeple supply (⑦)
**Depends on:** Task 2

In `#supply-panel`: each player's meeple pips + abbot glyph at the current step
(from `states[i].meeples`/`abbots`), reusing the Play pip UI. Player-coloured.
**Verify:** headless — pips reflect `states[i]` supply; deplete/return as you
scrub. Screenshot.
**Commit:** `feat(web): replay meeple-supply panel`.

### Task 11: Key-moment scrubber markers (⑩)
**Depends on:** Task 2, Task 8 (win-prob/score series available)

Add ticks on the scrubber timeline for: win-prob lead changes (annot.value sign
crossing, player-0 perspective, crossing 0.5), large score swings (Δ ≥ a
threshold, e.g. 6), and the first meeple placement. Click a tick → jump to that
move. Derive client-side from `states`/`annots`. Tooltip on hover naming the
moment.
**Verify:** headless on a checkpoint replay — ticks appear at computed moves;
clicking jumps. Screenshot.
**Commit:** `feat(web): replay scrubber key-moment markers`.

### Task 12: Final verification
**Depends on:** all

No new features. Headless full pass on TWO replays: (a) a checkpoint replay (real
MCTS — exercises ② ⑧ ⑩ fully), (b) the existing human-vs-greedy replay (exercises
the "no search data" states). Verify every feature renders, zero console errors,
no horizontal scroll at 1024px, and REGRESSION: win-prob chart still works and
`load_replay`/`replay_states` still verify. Run full `PYTHONPATH="$PWD" uv run
pytest` (expect green), ruff, mypy. Record a short verification note + browser
checklist. Commit any test updates.

## Dependency graph
```
T1 ─┐                         (independent)
T7 ─┼─ parallel               (Play view only)
T2 ─┴─→ T3, T4, T6, T8, T10, T11   (all edit replay.js/html/css → SEQUENTIAL)
        T4 → T5
        T1 + T2 → T9
        T8 → T11
   all → T12
```
Note: T3–T6, T8–T11 all edit the same replay files, so build them sequentially
(one implementer at a time). T1 and T7 touch different files and may go anytime.

## Review
- [ ] Code review requested
- [ ] All feedback addressed
- [ ] Final verification passed
