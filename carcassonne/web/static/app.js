// app.js — the play view state machine.
//
// Shape: fetch state -> render -> await input. One shared render path; the
// board canvas redraws fully every animation frame (boards are tiny). All
// server contracts are consumed exactly as produced by web/views.py.

import { TOKENS, drawTile, drawMeepleGlyph, featureAnchor } from "./tiles.js";
import {
  cellPx as boardCellPx,
  worldToScreen as boardWorldToScreen,
  screenToCell as boardScreenToCell,
  screenToCellFloat as boardScreenToCellFloat,
  resizeToDisplay,
  drawTiles,
  drawFrontier,
} from "./board.js";

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

// --- application state -------------------------------------------------------

const S = {
  tiledefs: null, // {id: {edges, features}} fetched once at startup
  gameId: null,
  view: null, // last StateView
  legal: [], // legal moves list; move.idx indexes THIS list for POST /move
  rotation: 0, // current lens rotation, 0..3 clockwise
  lensDeg: 0, // cumulative CSS rotation so 270->360 animates forward
  humanPlayer: 0,
  phase: "idle", // idle | placing | picking | awaiting | over
  pending: null, // {x, y, moves} while the radial picker is open
  hintOn: false,
  hint: null, // {policy: [{x, y, rot, prob}], value}
  hintEpoch: 0, // discard stale hint responses
  cam: { x: 0, y: 0, zoom: 1 },
  hover: null, // hovered board cell {x, y}
  aiPulse: null, // {x, y, start} settle pulse on the AI's placed tile
  drag: null,
  error: null, // fatal fetch failure message; overrides the status line until a new game
};

// --- DOM ----------------------------------------------------------------------

const $ = (id) => document.getElementById(id);
const board = $("board");
const bctx = board.getContext("2d");
const lensCanvas = $("lens-canvas");
const picker = $("picker");
const dialog = $("dialog");
const statusEl = $("status");
const logEl = $("log");

// --- server calls ---------------------------------------------------------------

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    const err = new Error(`HTTP ${r.status}`);
    err.status = r.status;
    err.body = await r.json().catch(() => null);
    throw err;
  }
  return r.json();
}

const postJSON = (path, body) =>
  api(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });

async function refreshLegal() {
  const data = await api(`/api/games/${S.gameId}/legal`);
  S.legal = data.moves;
}

async function refreshHint() {
  if (!S.hintOn || S.phase !== "placing") return;
  const epoch = ++S.hintEpoch;
  try {
    const hint = await api(`/api/games/${S.gameId}/hint`);
    if (epoch === S.hintEpoch) {
      S.hint = hint;
      updateValueReadout();
    }
  } catch {
    /* a finished or evicted game has no hint; leave the overlay empty */
  }
}

// --- game flow -------------------------------------------------------------------

async function newGame(opponent, seat, seed) {
  const body = { opponent, human_player: seat };
  S.error = null;
  if (seed !== null) body.seed = seed;
  const data = await postJSON("/api/games", body);
  S.gameId = data.game_id;
  S.view = data.state;
  S.humanPlayer = seat;
  S.pending = null;
  S.hint = null;
  S.aiPulse = null;
  S.cam = { x: 0, y: 0, zoom: 1 };
  setRotation(0, true);
  logEl.replaceChildren();
  logMeta(`New game vs ${opponent} — you are ${seat === 0 ? "blue" : "red"}`);
  $("banner").classList.remove("show");
  await refreshLegal();
  S.phase = S.view.terminal ? "over" : "placing";
  updateSidebar();
  refreshHint();
}

async function postMove(idx) {
  closePicker();
  S.phase = "awaiting";
  S.hint = null;
  updateSidebar();
  const gameId = S.gameId; // guard: a "New game" during this await must not stomp the new state
  let data;
  try {
    data = await postJSON(`/api/games/${gameId}/move`, { idx });
  } catch (err) {
    if (S.gameId !== gameId) return;
    if (err.status === 409) {
      // Stale idx: the legal list moved under us. Resync and let the human retry.
      logMeta("Move was stale — board resynced, try again");
      const g = await api(`/api/games/${gameId}`);
      if (S.gameId !== gameId) return;
      S.view = g.state;
      await refreshLegal();
      S.phase = S.view.terminal ? "over" : "placing";
      updateSidebar();
      refreshHint();
      return;
    }
    S.error = `Move failed — ${err.message || err}. Start a new game.`;
    S.phase = "over";
    updateSidebar();
    return;
  }
  if (S.gameId !== gameId) return;
  S.view = data.state;
  for (const e of data.events) logScore(e);
  if (data.ai_move && !reducedMotion.matches) {
    S.aiPulse = {
      x: data.ai_move.move.x,
      y: data.ai_move.move.y,
      start: performance.now(),
    };
  }
  setRotation(0, true); // fresh tile drawn: reset the lens
  if (S.view.terminal) {
    S.legal = [];
    S.phase = "over";
    showBanner(data.replay);
  } else {
    await refreshLegal();
    S.phase = "placing";
    refreshHint();
  }
  updateSidebar();
}

function showBanner(replayName) {
  const fs = S.view.final_scores;
  const you = fs[S.humanPlayer];
  const ai = fs[1 - S.humanPlayer];
  const verdict = you > ai ? "You win" : you < ai ? "AI wins" : "Draw";
  $("banner-scores").textContent = `${verdict} — you ${you}, AI ${ai}`;
  $("banner-saved").textContent = replayName ? `Saved as ${replayName}` : "";
  $("banner").classList.add("show");
  logMeta(`Game over — ${you}:${ai}`);
}

// --- rotation / the lens -----------------------------------------------------------

function setRotation(rot, snap = false) {
  const delta = snap ? 0 : 90;
  S.rotation = ((rot % 4) + 4) % 4;
  if (snap) {
    lensCanvas.style.transition = "none";
    S.lensDeg = S.rotation * 90;
  } else {
    S.lensDeg += delta;
  }
  lensCanvas.style.transform = `rotate(${S.lensDeg}deg)`;
  if (snap) {
    void lensCanvas.offsetWidth; // flush so the next rotation animates again
    lensCanvas.style.transition = "";
  }
}

function rotate() {
  if (S.phase !== "placing" && S.phase !== "picking") return;
  if (S.phase === "picking") closePicker();
  setRotation(S.rotation + 1);
}

function drawLens() {
  const dpr = window.devicePixelRatio || 1;
  const px = 128 * dpr;
  if (lensCanvas.width !== px) {
    lensCanvas.width = px;
    lensCanvas.height = px;
  }
  const ctx = lensCanvas.getContext("2d");
  ctx.clearRect(0, 0, px, px);
  const id = S.view && S.view.current_tile;
  if (id && S.tiledefs[id]) {
    // Drawn unrotated; the visible rotation is the CSS transform on the canvas.
    drawTile(ctx, S.tiledefs[id], 0, px);
  }
}

// --- board camera + geometry -------------------------------------------------------

// Thin adapters over the shared board module, bound to this page's canvas + cam
// so the many call sites below stay terse. The renderer itself lives in board.js.
const cellPx = () => boardCellPx(S.cam);
const worldToScreen = (x, y) => boardWorldToScreen(board, S.cam, x, y);
const screenToCell = (px, py) => boardScreenToCell(board, S.cam, px, py);

// --- board rendering ------------------------------------------------------------------

function render(now) {
  const { w, h } = resizeToDisplay(board, bctx);
  bctx.clearRect(0, 0, w, h);
  if (!S.view || !S.tiledefs) {
    requestAnimationFrame(render);
    return;
  }
  const c = drawTiles(bctx, board, S.view, S.tiledefs, S.cam);
  drawFrontier(bctx, board, S.view, S.cam);

  const placing = S.phase === "placing";
  const legalCells = new Map(); // cells legal at the CURRENT rotation
  if (placing) {
    for (const m of S.legal) {
      if (m.rot === S.rotation) legalCells.set(`${m.x},${m.y}`, m);
    }
  }

  // Hint overlay: --ai glow, opacity proportional to policy prob (current rotation).
  if (placing && S.hintOn && S.hint) {
    const entries = S.hint.policy.filter((p) => p.rot === S.rotation);
    const maxProb = Math.max(...entries.map((p) => p.prob), 1e-9);
    for (const p of entries) {
      const { sx, sy } = worldToScreen(p.x, p.y);
      const a = 0.08 + 0.5 * (p.prob / maxProb);
      bctx.fillStyle = `rgba(95, 212, 196, ${a.toFixed(3)})`;
      bctx.fillRect(sx - c / 2 + 2, sy - c / 2 + 2, c - 4, c - 4);
    }
  }

  // Legal placements: dashed bone-dim outlines, gently pulsing.
  if (placing) {
    const pulse = reducedMotion.matches
      ? 1
      : 0.75 + 0.25 * Math.sin((now / 2000) * 2 * Math.PI);
    bctx.setLineDash([5, 4]);
    bctx.lineWidth = 1.5;
    bctx.strokeStyle = `rgba(155, 163, 173, ${(0.7 * pulse).toFixed(3)})`;
    for (const key of legalCells.keys()) {
      const [x, y] = key.split(",").map(Number);
      const { sx, sy } = worldToScreen(x, y);
      bctx.strokeRect(sx - c / 2 + 3, sy - c / 2 + 3, c - 6, c - 6);
    }
    bctx.setLineDash([]);

    // Hover: solid outline + ghost tile at 40%.
    if (S.hover && legalCells.has(`${S.hover.x},${S.hover.y}`)) {
      const { sx, sy } = worldToScreen(S.hover.x, S.hover.y);
      bctx.save();
      bctx.translate(sx - c / 2, sy - c / 2);
      drawTile(bctx, S.tiledefs[S.view.current_tile], S.rotation, c, { alpha: 0.4 });
      bctx.restore();
      bctx.strokeStyle = TOKENS.bone;
      bctx.lineWidth = 2;
      bctx.strokeRect(sx - c / 2 + 1.5, sy - c / 2 + 1.5, c - 3, c - 3);
    }
  }

  // Pending placement while the picker is open: tile shown committed-looking.
  if (S.phase === "picking" && S.pending) {
    const { sx, sy } = worldToScreen(S.pending.x, S.pending.y);
    bctx.save();
    bctx.translate(sx - c / 2, sy - c / 2);
    drawTile(bctx, S.tiledefs[S.view.current_tile], S.rotation, c);
    bctx.restore();
    bctx.strokeStyle = TOKENS.bone;
    bctx.lineWidth = 2;
    bctx.strokeRect(sx - c / 2 + 1.5, sy - c / 2 + 1.5, c - 3, c - 3);
    positionPicker(sx, sy);
  }

  // AI settle pulse: one brief --ai outline on the AI's placed tile.
  if (S.aiPulse) {
    const age = now - S.aiPulse.start;
    if (age > 400) {
      S.aiPulse = null;
    } else {
      const { sx, sy } = worldToScreen(S.aiPulse.x, S.aiPulse.y);
      bctx.strokeStyle = `rgba(95, 212, 196, ${(1 - age / 400).toFixed(3)})`;
      bctx.lineWidth = 3;
      bctx.strokeRect(sx - c / 2 + 1.5, sy - c / 2 + 1.5, c - 3, c - 3);
    }
  }

  drawLens();
  requestAnimationFrame(render);
}

// --- radial meeple picker ---------------------------------------------------------------

function openPicker(x, y, moves) {
  S.pending = { x, y, moves };
  S.phase = "picking";
  picker.replaceChildren();
  const def = S.tiledefs[S.view.current_tile];
  const box = 148;

  // One spot per meeple action, at the feature's approximate position.
  const meepleMoves = moves.filter((m) => m.action && m.action.type === "meeple");
  const byFeature = new Map();
  for (const m of meepleMoves) {
    if (!byFeature.has(m.action.feature)) byFeature.set(m.action.feature, []);
    byFeature.get(m.action.feature).push(m);
  }
  for (const [feature, fmoves] of byFeature) {
    const a = featureAnchor(def, feature, S.rotation);
    let px = a.x * box;
    let py = a.y * box;
    if (Math.hypot(px - box / 2, py - box / 2) < 22) py -= 28; // clear the skip button
    fmoves.forEach((m, i) => {
      const spot = document.createElement("button");
      spot.className = "spot";
      spot.type = "button";
      const kind = m.action.kind;
      const fkind = def.features[feature].kind;
      spot.title = kind === "abbot" ? `Abbot on ${fkind}` : `Meeple on ${fkind}`;
      const offset = (i - (fmoves.length - 1) / 2) * 26;
      spot.style.left = `${px + offset}px`;
      spot.style.top = `${py}px`;
      const cv = document.createElement("canvas");
      cv.width = 56;
      cv.height = 56;
      drawMeepleGlyph(cv.getContext("2d"), 56, S.humanPlayer, kind);
      spot.append(cv);
      spot.addEventListener("click", () => postMove(m.idx));
      picker.append(spot);
    });
  }

  const skipMove = moves.find((m) => m.action === null);
  if (skipMove) {
    const skip = document.createElement("button");
    skip.className = "skip";
    skip.type = "button";
    skip.textContent = "Skip";
    skip.addEventListener("click", () => postMove(skipMove.idx));
    picker.append(skip);
  }

  const retrieveMove = moves.find((m) => m.action && m.action.type === "retrieve_abbot");
  if (retrieveMove) {
    const chip = document.createElement("button");
    chip.className = "retrieve";
    chip.type = "button";
    chip.textContent = "Retrieve abbot";
    chip.addEventListener("click", () => postMove(retrieveMove.idx));
    picker.append(chip);
  }

  picker.classList.add("show");
  updateSidebar();
  const first = picker.querySelector("button");
  if (first) first.focus();
}

function positionPicker(sx, sy) {
  picker.style.left = `${sx - 74}px`;
  picker.style.top = `${sy - 74}px`;
}

function closePicker() {
  picker.classList.remove("show");
  picker.replaceChildren();
  if (S.phase === "picking") S.phase = "placing";
  S.pending = null;
  updateSidebar();
}

// --- board input -----------------------------------------------------------------------

function cellClick(px, py) {
  if (S.phase !== "placing") return;
  const { x, y } = screenToCell(px, py);
  const candidates = S.legal.filter((m) => m.x === x && m.y === y && m.rot === S.rotation);
  if (candidates.length === 0) return;
  if (candidates.length === 1 && candidates[0].action === null) {
    postMove(candidates[0].idx); // nothing to pick: place immediately
  } else {
    openPicker(x, y, candidates);
  }
}

board.addEventListener("pointerdown", (e) => {
  S.drag = { px: e.clientX, py: e.clientY, moved: false };
  board.setPointerCapture(e.pointerId);
});

board.addEventListener("pointermove", (e) => {
  const rect = board.getBoundingClientRect();
  S.hover = null;
  if (S.drag) {
    const dx = e.clientX - S.drag.px;
    const dy = e.clientY - S.drag.py;
    if (S.drag.moved || Math.hypot(dx, dy) > 4) {
      S.drag.moved = true;
      board.classList.add("dragging");
      const c = cellPx();
      S.cam.x -= dx / c;
      S.cam.y += dy / c; // screen y down = world south
      S.drag.px = e.clientX;
      S.drag.py = e.clientY;
    }
  } else {
    S.hover = screenToCell(e.clientX - rect.left, e.clientY - rect.top);
  }
});

board.addEventListener("pointerup", (e) => {
  const wasDrag = S.drag && S.drag.moved;
  S.drag = null;
  board.classList.remove("dragging");
  if (!wasDrag) {
    const rect = board.getBoundingClientRect();
    cellClick(e.clientX - rect.left, e.clientY - rect.top);
  }
});

board.addEventListener("pointerleave", () => {
  S.hover = null;
});

board.addEventListener(
  "wheel",
  (e) => {
    e.preventDefault();
    const rect = board.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const before = screenToCellFloat(px, py);
    S.cam.zoom = Math.min(2.5, Math.max(0.5, S.cam.zoom * Math.exp(-e.deltaY * 0.0012)));
    const after = screenToCellFloat(px, py);
    S.cam.x += before.x - after.x; // keep the point under the cursor fixed
    S.cam.y += before.y - after.y;
  },
  { passive: false }
);

const screenToCellFloat = (px, py) => boardScreenToCellFloat(board, S.cam, px, py);

// --- keyboard ------------------------------------------------------------------------------

document.addEventListener("keydown", (e) => {
  if (dialog.open || e.target.matches("input, select, textarea")) return;
  if (e.key === "r" || e.key === "R") {
    rotate();
  } else if (e.key === "Escape" && S.phase === "picking") {
    closePicker();
  }
});

// --- sidebar -----------------------------------------------------------------------------------

const STATUS_TEXT = {
  idle: "No game — press New game",
  placing: "Your turn — place the tile",
  picking: "Your turn — meeple, skip, or Escape to move the tile",
  awaiting: "AI is thinking…",
  over: "Game over",
};

function updateSidebar() {
  statusEl.textContent = S.error ?? STATUS_TEXT[S.phase];
  updateValueReadout();
  if (!S.view) return;
  const scores = S.view.final_scores ?? S.view.scores; // end bonuses once terminal
  for (const seat of [0, 1]) {
    const card = $(`card-${seat}`);
    card.style.setProperty("--edge", seat === 0 ? "var(--p0)" : "var(--p1)");
    card.querySelector(".who").textContent = seat === S.humanPlayer ? "You" : "AI";
    card.querySelector(".points").textContent = String(scores[seat]);
    const supply = card.querySelector(".supply");
    supply.replaceChildren();
    for (let i = 0; i < 7; i++) {
      supply.append(pip(seat, "meeple", i < S.view.meeples[seat]));
    }
    supply.append(pip(seat, "abbot", S.view.abbots[seat]));
  }
}

function pip(seat, kind, filled) {
  const cv = document.createElement("canvas");
  const size = kind === "abbot" ? 16 : 12;
  cv.width = size * 2;
  cv.height = size * 2;
  cv.style.width = `${size}px`;
  cv.style.height = `${size}px`;
  const ctx = cv.getContext("2d");
  ctx.scale(2, 2);
  drawMeepleGlyph(ctx, size, seat, kind, filled);
  return cv;
}

function updateValueReadout() {
  const el = $("value-readout");
  if (S.hintOn && S.hint && S.phase === "placing") {
    const v = S.hint.value;
    el.textContent = `AI thinks: ${v >= 0 ? "+" : ""}${v.toFixed(2)}`;
  } else {
    el.textContent = "";
  }
}

// --- event log ------------------------------------------------------------------------------------

function logLine(nodes, cls) {
  const div = document.createElement("div");
  if (cls) div.className = cls;
  div.append(...nodes);
  logEl.append(div);
  logEl.scrollTop = logEl.scrollHeight;
}

function logMeta(text) {
  logLine([text], "meta");
}

function logScore(e) {
  const mine = e.player === S.humanPlayer;
  const who = mine ? "You" : "AI";
  const n = e.tiles.length;
  const text = `${who} scored ${e.points} — ${e.kind} (${n} tile${n === 1 ? "" : "s"})`;
  if (mine) {
    logLine([text]);
  } else {
    const dot = document.createElement("span");
    dot.className = "ai-dot";
    logLine([dot, text]);
  }
}

// --- new game dialog ----------------------------------------------------------------------------------

$("new-game").addEventListener("click", () => dialog.showModal());
$("banner-again").addEventListener("click", () => dialog.showModal());
$("dialog-cancel").addEventListener("click", () => dialog.close());
$("dialog-start").addEventListener("click", async () => {
  const opponent = $("opt-opponent").value;
  const seat = Number($("opt-seat").value);
  const seedRaw = $("opt-seed").value.trim();
  const seed = seedRaw === "" ? null : Number(seedRaw);
  dialog.close();
  try {
    await newGame(opponent, seat, Number.isFinite(seed) ? seed : null);
  } catch (err) {
    S.error = `Could not start the game — ${err.message || err}`;
    S.phase = "idle";
    updateSidebar();
    dialog.showModal();
  }
});

// --- hint toggle -------------------------------------------------------------------------------------------

$("hint-toggle").addEventListener("change", (e) => {
  S.hintOn = e.target.checked;
  if (S.hintOn) {
    refreshHint();
  } else {
    S.hint = null;
  }
  updateValueReadout();
});

// --- lens interactions ------------------------------------------------------------------------------------------

$("lens").addEventListener("click", rotate);

// --- boot ---------------------------------------------------------------------------------------------------------

async function populateCheckpoints() {
  // Offer each saved checkpoint as a `ckpt:<id>` opponent. If none are configured
  // the dialog keeps just greedy/random.
  try {
    const { checkpoints } = await api("/api/checkpoints");
    const select = $("opt-opponent");
    for (const ckpt of checkpoints) {
      const opt = document.createElement("option");
      opt.value = `ckpt:${ckpt.id}`;
      opt.textContent = `Checkpoint ${ckpt.step}`;
      select.appendChild(opt);
    }
  } catch {
    /* checkpoints are optional; ignore a missing/errored endpoint */
  }
}

async function boot() {
  const defs = await api("/api/tiledefs");
  S.tiledefs = defs.tiles;
  await populateCheckpoints();
  updateSidebar();
  requestAnimationFrame(render);
  // Start a default game straight away so the board is playable and the top nav
  // stays reachable. A modal dialog on load would sit in the browser top layer
  // and block the nav links until dismissed. "New game" reconfigures any time.
  try {
    await newGame("greedy", 0, null);
  } catch {
    dialog.showModal(); // couldn't auto-start (e.g. server error) — let the user choose
  }
}

boot();
