# Tiles Reference Tab — Design

A standalone "Tiles" tab in the web app: a reference gallery of every tile type
in the deck, showing its simulator letter, copy count, rendered art, and a plain
scoring note — a learning page for the deck. Answers "what tiles exist, what are
they called, and what do they do" at a glance (including where all 8 gardens live).

## Goal

Make the deck learnable. One page, purely additive, no gameplay coupling. Serves
the project's stated **RL microscope** learning purpose beyond play.

## Scope

- **In:** a new `/tiles` page; the four feature groups; per-tile card (art,
  letter, count, structural note, garden/shield badges); click-to-rotate; a deck
  summary + scoring legend header; nav link on every page; static + e2e tests.
- **Out (YAGNI):** enlarge/modal detail view, live legal-placement demo,
  filtering/search, any backend data change.

## Architecture — zero new backend logic

All data and rendering already exist:

- `GET /api/tiledefs` (`views.tiledefs_view`) already returns, per tile id:
  `count`, `edges` (4 EdgeKind strings), and `features` (each with `kind`,
  `edges`, `shield`). This is the complete data source — **no backend change**.
- `static/tiles.js` already exports `drawTile(ctx, def, rot, size, opts)` and
  `TOKENS`. The atlas reuses `drawTile` as the single source of truth for tile
  art — no duplicated rendering.
- Rotation is `0..3` quarter-turns (`core.types.Rotation`); `drawTile`'s `rot`
  arg takes the same. Click-to-rotate cycles `rot = (rot + 1) % 4` and redraws.

New/changed files:

- **`carcassonne/web/app.py`** — add one route:
  `@app.get("/tiles")` → `FileResponse(_STATIC_DIR / "tiles.html")`
  (mirrors the existing `/replay` and `/training` routes).
- **`carcassonne/web/static/tiles.html`** (new) — nav + page shell (header mount,
  four group section mounts), imports `atlas.js` as a module.
- **`carcassonne/web/static/atlas.js`** (new) — the page controller: fetch
  `/api/tiledefs`, classify tiles, render cards, wire click-to-rotate, build the
  header summary + legend. Named `atlas.js` to avoid colliding with the existing
  `tiles.js` renderer module.
- **`carcassonne/web/static/style.css`** — a `.atlas`-scoped block (group
  headings, card grid, card, canvas, letter badge, count, note, badges) using
  existing design tokens.
- **Nav** — add `<a href="/tiles">Tiles</a>` to the `#topnav` in all four static
  pages: `index.html`, `replay.html`, `training.html`, `tiles.html`
  (persistent-nav rule). Placement: after "Play/Replays/Training", so order is
  Play · Replays · Training · Tiles.

## Grouping — four sections

Each tile is classified **exactly once** from its `features`, in this priority:

1. **Monasteries & Gardens** — has any `MONASTERY` or `GARDEN` feature. This is
   the abbot-target group and collects all 8 garden tiles in one place.
   Members: `A, B, E_G, H_G, N_G, P_G, U_G, V_G`.
2. **Cities & Roads** — (not group 1) has both a `CITY` and a `ROAD` feature.
   Members: `D, J, K, L, O, P, S, T`.
3. **Cities** — (not 1/2) has a `CITY` feature. Members: `C, E, F, G, H, I, M, N, Q, R`.
4. **Roads** — everything else (has a `ROAD` feature). Members: `U, V, W, X`.

Classification is derived from feature kinds only (client-side), so it stays
correct if the deck changes. Within a section, tiles render in deck order. One
card is drawn **per deck entry**, so garden variants get their own card — `E`
(×4) and `E_G` (×1) both appear, which makes visible that one of the five E tiles
carries a garden. The deck has 30 entries total (24 letters + the 6 `_G`
variants). Each section heading shows its entry count and total copies, e.g.
"Cities — 10 tiles · 20 copies". Group sizes: Monasteries & Gardens (8 entries,
14 copies), Cities & Roads (8 entries, 20 copies), Cities (10 entries, 20
copies), Roads (4 entries, 18 copies) — 30 entries, 72 copies total.

## Card contents

- **Art:** a small `<canvas>` (e.g. 96px) drawn once via `drawTile(ctx, def, 0, size)`.
- **Letter ID:** the tile id shown prominently. Garden variants keep their id
  (`E_G`) but display the base letter large with a "_G" affordance / garden badge.
- **Count:** `×N` from `def.count`.
- **Structural note:** a concise human description derived from features, e.g.
  "City edge", "City edge + road", "Straight road", "Road bend", "T-junction",
  "Crossroads", "Monastery", "Monastery + road", "City edge + garden". Generated
  by a small pure `noteFor(def)` helper in `atlas.js`.
- **Badges:** a **garden** badge on `_G` tiles; a **shield** badge when any city
  feature has `shield: true`.
- **Interaction:** clicking the card rotates its tile 90° (`rot = (rot+1)%4`) and
  redraws its canvas — teaches the four orientations. Keyboard: card is a
  `<button>` so Enter/Space also rotate; `aria-label` names the tile.

## Page header — the learning layer

- **Deck summary** line computed from tiledefs: total tiles (72), letters (24),
  monasteries (6), gardens (8). All derived client-side by summing counts and
  counting distinct base letters (id before `_`) — no hardcoded numbers, so it
  self-corrects if the deck changes.
- **Scoring legend** stated once (so per-card notes stay short):
  - City: 2 points/tile + 2/shield (halved if unfinished at game end).
  - Road: 1 point/tile.
  - Monastery / Garden: up to 9 (its tile + 8 neighbours) when enclosed.

## Colour discipline

Uses `--stage`, `--bone`, `--serif`, `--mono`, and neutral card surfaces.
**Does not use `--ai` cyan** (reserved for AI-analysis surfaces only). Player
tints `--p0`/`--p1` are not used on this page (no per-player content).

## Testing

- **`tests/web/test_static.py`** (extend): `GET /tiles` returns 200; the page body
  contains the four nav links (Play/Replays/Training/Tiles), an `atlas.js` module
  import, and the section mount points.
- **e2e (headless Playwright)** — load `/tiles` and assert:
  1. All four group sections render with non-empty card grids.
  2. All 30 tile cards (one per deck entry) draw non-blank canvases.
  3. Clicking a card changes its canvas pixels (rotation works).
  4. The deck summary shows 72 tiles / 8 gardens / 6 monasteries.
  5. Nav is present with the Tiles link; zero console errors.

No new Python logic beyond the route, so the unit surface stays tiny; correctness
lives in the classification/note helpers (pure functions in `atlas.js`) and the
e2e check.

## Success criteria

1. `/tiles` is reachable from the nav on every page and renders the four groups.
2. Every deck entry (30 = 24 letters + 6 garden variants) appears exactly once,
   in the correct group, with correct count, art, note, and badges.
3. All 8 gardens appear together in Monasteries & Gardens; the summary reads
   72 tiles · 8 gardens · 6 monasteries.
4. Clicking a tile rotates it; keyboard-accessible.
5. Full test suite green; ruff/mypy clean; zero console errors on the page.
