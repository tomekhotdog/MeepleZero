# Tiles Reference Tab — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use tomek-superpowers:build to implement this plan task-by-task.

**Goal:** Add a `/tiles` reference tab — a grouped gallery of every deck entry (letter, count, art, note, badges) with click-to-rotate — to make the deck learnable.

**Architecture:** Purely additive front-end. A new FastAPI route serves `tiles.html`; a new `atlas.js` module fetches the existing `/api/tiledefs`, classifies each tile into four feature groups, and renders one card per deck entry using the *existing* `drawTile()` from `tiles.js` (single source of truth for tile art). No backend data change. Nav link added to all four pages.

**Tech Stack:** FastAPI (route), vanilla ES-module JS + `<canvas>`, CSS design tokens, pytest (static tests), headless Playwright (behavioural verification, mirroring the replay-workbench build).

**Design:** `docs/superpowers/specs/2026-08-23-tiles-reference-tab-design.md`

**Key facts locked by the spec:** 72 copies · 30 deck entries · 24 letters · 6 monasteries · 8 gardens. One card per deck entry, so `E` (×4) and `E_G` (×1) are separate cards. Group sizes: Monasteries & Gardens (8 entries / 14 copies), Cities & Roads (8 / 20), Cities (10 / 20), Roads (4 / 18). Classification priority: monastery/garden → city+road → city → road. `--ai` cyan is reserved and MUST NOT be used on this page.

---

### Task 1: Backbone — route, page shell, nav on all pages, static tests

**Depends on:** none

Delivers a reachable `/tiles` page with persistent nav and a minimal `atlas.js` that fetches tiledefs and renders a placeholder, plus committed static tests. End-to-end tracer: server serves the page, nav links everywhere, module loads without console error.

**Files:**
- Modify: `carcassonne/web/app.py` (add route)
- Create: `carcassonne/web/static/tiles.html`
- Create: `carcassonne/web/static/atlas.js` (minimal, replaced in Task 2)
- Modify: `carcassonne/web/static/index.html`, `replay.html`, `training.html` (nav link)
- Test: `tests/web/test_static.py`

**Step 1: Write the failing tests** — edit `tests/web/test_static.py`:

Add a page-served test after `test_replay_page_served`:
```python
def test_tiles_page_served(client: TestClient) -> None:
    r = client.get("/tiles")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert 'id="app"' in r.text
    assert "atlas.js" in r.text
    # persistent nav: all four views linked
    for href in ('href="/"', 'href="/replay"', 'href="/training"', 'href="/tiles"'):
        assert href in r.text, f"missing nav link {href}"
```

Add `("tiles.html", "text/html")` and `("atlas.js", "javascript")` to the `test_static_assets_served` parametrize list.

Extend `test_page_references_only_existing_files`: change the parametrize to
`["index.html", "replay.html", "training.html", "tiles.html"]` and change
`app_routes = {"/", "/replay", "/training"}` to `{"/", "/replay", "/training", "/tiles"}`.

Add `"atlas.js"` to the `test_js_syntax` parametrize list.

**Step 2: Run tests to verify they fail**
Run: `PYTHONPATH="$PWD" uv run pytest tests/web/test_static.py -q`
Expected: FAIL (route 404 / missing files).

**Step 3: Implement**

`carcassonne/web/app.py` — add after the `/training` route (lines ~68-70):
```python
    @app.get("/tiles")
    def tiles_page() -> FileResponse:
        return FileResponse(_STATIC_DIR / "tiles.html")
```

`carcassonne/web/static/tiles.html` (new):
```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Carcassonne — tiles</title>
    <link rel="icon" href="data:," />
    <link rel="stylesheet" href="/static/style.css" />
  </head>
  <body>
    <nav id="topnav">
      <span class="brand">Carcassonne</span>
      <a href="/">Play</a>
      <a href="/replay">Replays</a>
      <a href="/training">Training</a>
      <a href="/tiles" aria-current="page">Tiles</a>
    </nav>
    <div id="app" class="atlas">
      <header id="atlas-head">
        <h1>The deck</h1>
        <div id="summary" class="summary">Loading…</div>
        <div class="legend">
          <span class="legend-title">Scoring</span>
          <ul>
            <li><b>City</b> — 2 / tile + 2 / shield (halved if unfinished at game end)</li>
            <li><b>Road</b> — 1 / tile</li>
            <li><b>Monastery / Garden</b> — up to 9 (its tile + 8 neighbours) when enclosed</li>
          </ul>
        </div>
        <div id="atlas-error" class="error" role="alert"></div>
      </header>
      <div id="groups"></div>
    </div>
    <script type="module" src="/static/atlas.js"></script>
  </body>
</html>
```

`carcassonne/web/static/atlas.js` (new, minimal — Task 2 replaces the body):
```js
// atlas.js — the Tiles reference tab (see Task 2 for the full renderer).
async function main() {
  const res = await fetch("/api/tiledefs");
  const { tiles } = await res.json();
  document.getElementById("summary").textContent =
    `${Object.keys(tiles).length} tile types loaded`;
}
main().catch((e) => {
  document.getElementById("atlas-error").textContent = String(e);
});
```

Add `<a href="/tiles">Tiles</a>` to `#topnav` in `index.html`, `replay.html`, and
`training.html`, immediately after the `<a href="/training">…</a>` link (no
`aria-current` on those three).

**Step 4: Run tests to verify they pass**
Run: `PYTHONPATH="$PWD" uv run pytest tests/web/test_static.py -q`
Expected: PASS (all static tests green).

**Step 5: Commit**
```bash
git add carcassonne/web/app.py carcassonne/web/static/tiles.html \
  carcassonne/web/static/atlas.js carcassonne/web/static/index.html \
  carcassonne/web/static/replay.html carcassonne/web/static/training.html \
  tests/web/test_static.py
git commit -m "feat(web): /tiles route, page shell, persistent nav link"
```

---

### Task 2: Full atlas — classify, cards, badges, header summary

**Depends on:** Task 1

Replaces `atlas.js` with the real renderer: classification into four groups, one
`drawTile` card per deck entry with letter/count/note/badges, and the computed
deck-summary header. (Click-to-rotate + final CSS polish land in Task 3, but the
click handler is included here since it is one line.)

**Files:**
- Modify: `carcassonne/web/static/atlas.js` (full body)
- Modify: `carcassonne/web/static/style.css` (append `.atlas` block)
- Test: `tests/web/test_static.py` (already covers syntax + refs; no new pytest)

**Step 1: Write the failing check** — the behavioural assertions are the headless
verification in Task 3. Here the committed guard is `test_js_syntax[atlas.js]`
(added in Task 1) — it will fail if the new module has a syntax error.
Run: `PYTHONPATH="$PWD" uv run pytest tests/web/test_static.py -q` → PASS baseline before editing.

**Step 2: Implement** — replace `carcassonne/web/static/atlas.js` entirely:
```js
// atlas.js — the Tiles reference tab. A static gallery of every deck entry,
// grouped by feature and rendered via the shared drawTile renderer. Click a
// card to rotate its tile. Reads /api/tiledefs once; holds no gameplay state.

import { drawTile } from "/static/tiles.js";

// Groups are tested in order; first match wins. Order encodes the priority:
// monastery/garden (abbot targets, holds all gardens) → city+road → city → road.
const GROUPS = [
  {
    key: "mon",
    title: "Monasteries & Gardens",
    blurb: "Abbot targets — every garden in the deck lives here.",
    test: (f) => f.some((x) => x.kind === "monastery" || x.kind === "garden"),
  },
  {
    key: "cityroad",
    title: "Cities & Roads",
    blurb: "A city edge and a road on the same tile.",
    test: (f) => f.some((x) => x.kind === "city") && f.some((x) => x.kind === "road"),
  },
  {
    key: "city",
    title: "Cities",
    blurb: "City edges, no road.",
    test: (f) => f.some((x) => x.kind === "city"),
  },
  { key: "road", title: "Roads", blurb: "Roads only.", test: () => true },
];

function classify(features) {
  for (const g of GROUPS) if (g.test(features)) return g.key;
  return "road";
}

// A concise, human note for a tile derived from its features.
function noteFor(def) {
  const cities = def.features.filter((f) => f.kind === "city");
  const roads = def.features.filter((f) => f.kind === "road");
  const parts = [];
  if (def.features.some((f) => f.kind === "monastery")) parts.push("Monastery");
  if (def.features.some((f) => f.kind === "garden")) parts.push("Garden");
  if (cities.length > 1) {
    parts.push(`${cities.length} separate cities`);
  } else if (cities.length === 1) {
    const n = cities[0].edges.length;
    parts.push(n === 1 ? "City edge" : n === 4 ? "City (all sides)" : `City (${n} sides)`);
  }
  if (roads.length >= 3) {
    parts.push(roads.length === 4 ? "Crossroads" : "T-junction");
  } else if (roads.length === 2) {
    parts.push("Roads");
  } else if (roads.length === 1) {
    const e = roads[0].edges;
    const straight = e.length === 2 && Math.abs(e[0] - e[1]) === 2;
    parts.push(straight ? "Straight road" : e.length === 2 ? "Road bend" : "Road end");
  }
  return parts.join(" + ") || "Field";
}

function summarize(tiles) {
  let copies = 0;
  let monasteries = 0;
  let gardens = 0;
  const letters = new Set();
  for (const [id, def] of Object.entries(tiles)) {
    copies += def.count;
    letters.add(id.split("_")[0]);
    if (def.features.some((f) => f.kind === "monastery")) monasteries += def.count;
    if (def.features.some((f) => f.kind === "garden")) gardens += def.count;
  }
  return { copies, letters: letters.size, monasteries, gardens };
}

const CARD_PX = 96;

function chip(kind, text) {
  const s = document.createElement("span");
  s.className = `chip chip-${kind}`;
  s.textContent = text;
  return s;
}

function makeCard(id, def) {
  const btn = document.createElement("button");
  btn.className = "tile-card";
  btn.type = "button";
  const note = noteFor(def);
  btn.setAttribute("aria-label", `Tile ${id}, ${def.count} copies, ${note}. Click to rotate.`);

  const cv = document.createElement("canvas");
  const dpr = window.devicePixelRatio || 1;
  cv.width = CARD_PX * dpr;
  cv.height = CARD_PX * dpr;
  cv.style.width = `${CARD_PX}px`;
  cv.style.height = `${CARD_PX}px`;
  const ctx = cv.getContext("2d");
  ctx.scale(dpr, dpr);
  let rot = 0;
  const draw = () => {
    ctx.clearRect(0, 0, CARD_PX, CARD_PX);
    drawTile(ctx, def, rot, CARD_PX);
  };
  draw();
  btn.addEventListener("click", () => {
    rot = (rot + 1) % 4;
    draw();
  });

  const badges = document.createElement("div");
  badges.className = "badges";
  if (def.features.some((f) => f.kind === "garden")) badges.append(chip("garden", "Garden"));
  if (def.features.some((f) => f.kind === "city" && f.shield)) badges.append(chip("shield", "Shield"));

  const label = document.createElement("div");
  label.className = "label";
  const idEl = document.createElement("span");
  idEl.className = "id";
  idEl.textContent = id;
  const countEl = document.createElement("span");
  countEl.className = "count";
  countEl.textContent = `×${def.count}`;
  label.append(idEl, countEl);

  const noteEl = document.createElement("div");
  noteEl.className = "note";
  noteEl.textContent = note;

  btn.append(cv, badges, label, noteEl);
  return btn;
}

async function main() {
  const res = await fetch("/api/tiledefs");
  if (!res.ok) throw new Error(`tiledefs ${res.status}`);
  const { tiles } = await res.json();

  const s = summarize(tiles);
  document.getElementById("summary").textContent =
    `${s.copies} tiles · ${s.letters} letters · ${s.monasteries} monasteries · ${s.gardens} gardens`;

  const buckets = { mon: [], cityroad: [], city: [], road: [] };
  for (const [id, def] of Object.entries(tiles)) buckets[classify(def.features)].push([id, def]);

  const root = document.getElementById("groups");
  for (const g of GROUPS) {
    const entries = buckets[g.key];
    const copies = entries.reduce((n, [, d]) => n + d.count, 0);
    const section = document.createElement("section");
    section.className = "group";
    section.dataset.group = g.key;
    const h = document.createElement("h2");
    h.textContent = g.title;
    const meta = document.createElement("span");
    meta.className = "meta";
    meta.textContent = `${entries.length} tiles · ${copies} copies`;
    h.append(" ", meta);
    const blurb = document.createElement("p");
    blurb.className = "blurb";
    blurb.textContent = g.blurb;
    const grid = document.createElement("div");
    grid.className = "grid";
    for (const [id, def] of entries) grid.append(makeCard(id, def));
    section.append(h, blurb, grid);
    root.append(section);
  }
}

main().catch((e) => {
  const el = document.getElementById("atlas-error");
  if (el) el.textContent = `Failed to load tiles: ${e.message ?? e}`;
});
```

**Step 3: Append the `.atlas` CSS block** to `carcassonne/web/static/style.css`:
```css
/* --- Tiles reference (atlas) ------------------------------------------------ */
/* Reserved: does NOT use --ai (AI-analysis surfaces only). */
.atlas { display: block; max-width: 1100px; margin: 0 auto; padding: 24px 20px 64px; }
.atlas #atlas-head { margin-bottom: 8px; }
.atlas h1 { font-family: var(--serif); font-size: 28px; margin: 0 0 8px; }
.atlas .summary { font-family: var(--mono); color: var(--bone); opacity: 0.85; margin-bottom: 16px; }
.atlas .legend { background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 12px 16px; font-size: 13px; }
.atlas .legend-title { font-family: var(--mono); text-transform: uppercase; letter-spacing: 0.08em; font-size: 11px; opacity: 0.7; }
.atlas .legend ul { margin: 6px 0 0; padding-left: 18px; }
.atlas .legend li { margin: 2px 0; }
.atlas .error { color: var(--p1); margin-top: 8px; }
.atlas .group { margin-top: 32px; }
.atlas .group h2 { font-family: var(--serif); font-size: 20px; margin: 0 0 2px; }
.atlas .group h2 .meta { font-family: var(--mono); font-size: 12px; opacity: 0.6; font-weight: normal; }
.atlas .group .blurb { margin: 0 0 14px; font-size: 13px; opacity: 0.7; }
.atlas .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 14px; }
.atlas .tile-card { position: relative; background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.09); border-radius: 10px; padding: 10px; cursor: pointer; display: flex; flex-direction: column; align-items: center; gap: 6px; color: inherit; font: inherit; transition: border-color 0.12s, transform 0.06s; }
.atlas .tile-card:hover { border-color: rgba(255, 255, 255, 0.25); }
.atlas .tile-card:active { transform: scale(0.97); }
.atlas .tile-card canvas { border-radius: 4px; }
.atlas .tile-card .label { display: flex; align-items: baseline; gap: 6px; }
.atlas .tile-card .id { font-family: var(--serif); font-size: 18px; font-weight: 600; }
.atlas .tile-card .count { font-family: var(--mono); font-size: 12px; opacity: 0.7; }
.atlas .tile-card .note { font-size: 11px; opacity: 0.65; text-align: center; line-height: 1.3; }
.atlas .badges { position: absolute; top: 8px; right: 8px; display: flex; gap: 4px; }
.atlas .chip { font-family: var(--mono); font-size: 9px; text-transform: uppercase; letter-spacing: 0.05em; padding: 2px 5px; border-radius: 4px; }
.atlas .chip-garden { background: rgba(122, 155, 94, 0.25); color: #b6d29a; }
.atlas .chip-shield { background: rgba(201, 162, 94, 0.25); color: #e0c48a; }
```

**Step 4: Run tests**
Run: `PYTHONPATH="$PWD" uv run pytest tests/web/test_static.py -q`
Expected: PASS (js syntax + refs still green).

**Step 5: Commit**
```bash
git add carcassonne/web/static/atlas.js carcassonne/web/static/style.css
git commit -m "feat(web): tiles atlas — grouped cards, notes, badges, deck summary"
```

---

### Task 3: Behavioural verification (headless)

**Depends on:** Task 2

No committed Playwright suite exists (matching the replay-workbench build); verify
behaviour headlessly against a running server and record the result. Fix any defect
found, then this task is done.

**Files:**
- Create (scratchpad, not committed): a headless verification script
- No production code unless a defect is found.

**Step 1: Start a server** (test port, isolated dirs):
```bash
PYTHONPATH="$PWD" uv run python -m carcassonne.cli.main serve --port 8210 \
  --replays replays --runs-dir runs --checkpoints runs/mac-demo/checkpoints &
```
Wait for "server up on 8210".

**Step 2: Write and run the headless check** — create
`<scratchpad>/verify_tiles.mjs`:
```js
import { chromium } from "playwright";
const errors = [];
const browser = await chromium.launch();
const page = await browser.newPage();
page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
page.on("pageerror", (e) => errors.push(String(e)));
await page.goto("http://127.0.0.1:8210/tiles", { waitUntil: "networkidle" });

const groups = await page.$$eval(".group", (els) => els.map((e) => e.dataset.group));
const cards = await page.$$(".tile-card");
const summary = await page.textContent("#summary");

// Card canvas is non-blank, and rotation changes pixels.
const before = await page.$eval(".tile-card canvas", (c) => c.toDataURL());
await page.click(".tile-card");
const after = await page.$eval(".tile-card canvas", (c) => c.toDataURL());

const navLinks = await page.$$eval("#topnav a", (as) => as.map((a) => a.getAttribute("href")));

const ok =
  groups.length === 4 &&
  ["mon", "cityroad", "city", "road"].every((k) => groups.includes(k)) &&
  cards.length === 30 &&
  summary.includes("72 tiles") &&
  summary.includes("8 gardens") &&
  summary.includes("6 monasteries") &&
  before !== after &&
  ["/", "/replay", "/training", "/tiles"].every((h) => navLinks.includes(h)) &&
  errors.length === 0;

console.log(JSON.stringify({ groups, cards: cards.length, summary, rotated: before !== after, navLinks, errors, ok }, null, 2));
await browser.close();
process.exit(ok ? 0 : 1);
```
Run: `cd carcassonne/web/static && npx --no-install playwright test 2>/dev/null; node <scratchpad>/verify_tiles.mjs`
(Playwright is invoked via the repo's existing install; if `node` cannot resolve
`playwright`, run the script from a dir where it is installed, as in the
replay-workbench verification.)
Expected output: `"ok": true` — 4 groups, 30 cards, summary reads
`72 tiles · 24 letters · 6 monasteries · 8 gardens`, rotation flips pixels, all
four nav links present, zero console errors. Exit code 0.

**Step 3: Stop the server**
```bash
kill %1 2>/dev/null || true
```

**Step 4: Commit** (only if a production fix was needed)
```bash
git add -A && git commit -m "fix(web): tiles atlas — <defect> from headless verification"
```

---

## Review
- [ ] Code review requested (final `code-reviewer` over all three tasks)
- [ ] All feedback addressed
- [ ] Final verification passed (`verify` skill: full suite green, ruff, mypy, headless `ok:true`)
