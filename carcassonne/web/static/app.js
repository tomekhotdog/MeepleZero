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
  lastHumanMove: null, // {x, y} of the human's most recent tile placement
  lastAiMove: null, // {x, y} of the AI's most recent tile placement
  highlightTiles: new Set(), // "x,y" keys highlighted by a clicked score row (F2)
  highlightRow: null, // the currently-active score row button, or null
  highlightColor: null, // scoring player's colour for the active highlight
  meepleHits: [], // per-frame [{cx, cy, r, meeple}] for meeple hover hit-testing (F3)
  hoverMeeple: null, // the meeple dict under the cursor, or null
};

// --- DOM ----------------------------------------------------------------------

const $ = (id) => document.getElementById(id);
const board = $("board");
const bctx = board.getContext("2d");
const stage = $("stage");
const lensCanvas = $("lens-canvas");
const picker = $("picker");
const dialog = $("dialog");
const statusEl = $("status");
const logEl = $("log");

// Board-corner "reset view" control (F4) and meeple-hover tooltip (F3), built
// here so the whole Play view stays in app.js + style.css.
const resetViewBtn = document.createElement("button");
resetViewBtn.id = "reset-view";
resetViewBtn.type = "button";
resetViewBtn.textContent = "⟳";
resetViewBtn.setAttribute("aria-label", "Reset view — recentre the board");
resetViewBtn.title = "Reset view";
resetViewBtn.addEventListener("click", resetView);
stage.append(resetViewBtn);

const meepleTip = document.createElement("div");
meepleTip.id = "meeple-tip";
meepleTip.setAttribute("role", "tooltip");
stage.append(meepleTip);

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
  S.lastHumanMove = null;
  S.lastAiMove = null;
  S.hoverMeeple = null;
  clearScoreHighlight();
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
  const chosen = S.legal.find((m) => m.idx === idx);
  if (chosen) S.lastHumanMove = { x: chosen.x, y: chosen.y }; // F1: mark the human placement
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
  if (data.ai_move) {
    S.lastAiMove = { x: data.ai_move.move.x, y: data.ai_move.move.y }; // F1: mark the AI placement
    if (!reducedMotion.matches) {
      S.aiPulse = {
        x: data.ai_move.move.x,
        y: data.ai_move.move.y,
        start: performance.now(),
      };
    }
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

  // Record every drawn meeple's screen disc so pointermove can hit-test them (F3).
  // Anchors are defined unrotated; drawTile rotates the box, so featureAnchor with
  // the tile's rotation reproduces the on-screen position.
  S.meepleHits = [];
  for (const t of S.view.tiles) {
    const def = S.tiledefs[t.type];
    const { sx, sy } = worldToScreen(t.x, t.y);
    for (const m of t.meeples ?? []) {
      const a = featureAnchor(def, m.feature, t.rot);
      S.meepleHits.push({ cx: sx - c / 2 + a.x * c, cy: sy - c / 2 + a.y * c, r: c * 0.1, meeple: m });
    }
  }

  // (F2) Score-row highlight: fill + outline the clicked event's feature tiles in
  // the scoring player's colour — distinct from the neutral meeple-hover outline.
  if (S.highlightTiles.size > 0) {
    const col = S.highlightColor ?? TOKENS.bone;
    for (const key of S.highlightTiles) {
      const [x, y] = key.split(",").map(Number);
      const { sx, sy } = worldToScreen(x, y);
      bctx.save();
      bctx.globalAlpha = 0.18;
      bctx.fillStyle = col;
      bctx.fillRect(sx - c / 2 + 2, sy - c / 2 + 2, c - 4, c - 4);
      bctx.restore();
      bctx.strokeStyle = col;
      bctx.lineWidth = 2.5;
      bctx.strokeRect(sx - c / 2 + 2, sy - c / 2 + 2, c - 4, c - 4);
    }
  }

  // (F3) Meeple-hover: neutral --bone outline around the hovered meeple's feature.
  if (S.hoverMeeple) {
    bctx.strokeStyle = TOKENS.bone;
    bctx.lineWidth = 2;
    bctx.setLineDash([2, 3]);
    for (const [x, y] of S.hoverMeeple.feature_tiles) {
      const { sx, sy } = worldToScreen(x, y);
      bctx.strokeRect(sx - c / 2 + 2.5, sy - c / 2 + 2.5, c - 5, c - 5);
    }
    bctx.setLineDash([]);
  }

  // (F1) Persistent last-move markers: a solid ring in the mover's colour plus a
  // labelled corner chip, so "your last move" vs "AI's last move" is unambiguous.
  drawLastMove(S.lastHumanMove, playerToken(S.humanPlayer), "You", c);
  drawLastMove(S.lastAiMove, playerToken(1 - S.humanPlayer), "AI", c);

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

const playerToken = (p) => (p === 0 ? TOKENS.p0 : TOKENS.p1);

// (F1) One persistent last-move marker: a solid ring in the mover's colour and a
// small labelled chip in the tile's top-left corner ("You" / "AI").
function drawLastMove(pos, color, label, c) {
  if (!pos) return;
  const { sx, sy } = worldToScreen(pos.x, pos.y);
  const left = sx - c / 2;
  const top = sy - c / 2;
  bctx.save();
  bctx.strokeStyle = color;
  bctx.lineWidth = 3;
  bctx.strokeRect(left + 1.5, top + 1.5, c - 3, c - 3);
  // Corner chip with the mover's label.
  bctx.font = `600 ${Math.max(8, c * 0.16)}px ${getComputedStyle(document.body).getPropertyValue("--ui") || "sans-serif"}`;
  const pad = Math.max(2, c * 0.04);
  const tw = bctx.measureText(label).width;
  const chipH = Math.max(11, c * 0.22);
  const chipW = tw + pad * 2;
  bctx.fillStyle = color;
  bctx.fillRect(left + 1.5, top + 1.5, chipW, chipH);
  bctx.fillStyle = TOKENS.ink;
  bctx.textBaseline = "middle";
  bctx.textAlign = "left";
  bctx.fillText(label, left + 1.5 + pad, top + 1.5 + chipH / 2 + 0.5);
  bctx.restore();
}

// (F4) Recentre the camera on the board's bounding-box centre at zoom 1.
function resetView() {
  if (!S.view || S.view.tiles.length === 0) {
    S.cam = { x: 0, y: 0, zoom: 1 };
    return;
  }
  const xs = S.view.tiles.map((t) => t.x);
  const ys = S.view.tiles.map((t) => t.y);
  S.cam = {
    x: (Math.min(...xs) + Math.max(...xs)) / 2,
    y: (Math.min(...ys) + Math.max(...ys)) / 2,
    zoom: 1,
  };
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
      clearMeepleHover();
      const c = cellPx();
      S.cam.x -= dx / c;
      S.cam.y += dy / c; // screen y down = world south
      S.drag.px = e.clientX;
      S.drag.py = e.clientY;
    }
  } else {
    S.hover = screenToCell(e.clientX - rect.left, e.clientY - rect.top);
    updateMeepleHover(e.clientX - rect.left, e.clientY - rect.top);
  }
});

board.addEventListener("pointerup", (e) => {
  const wasDrag = S.drag && S.drag.moved;
  S.drag = null;
  board.classList.remove("dragging");
  if (!wasDrag) {
    clearScoreHighlight(); // clicking the board clears any score-row highlight (F2)
    const rect = board.getBoundingClientRect();
    cellClick(e.clientX - rect.left, e.clientY - rect.top);
  }
});

board.addEventListener("pointerleave", () => {
  S.hover = null;
  clearMeepleHover();
});

// (F3) Find the placed meeple under the cursor (if any) and drive the tooltip.
function updateMeepleHover(px, py) {
  let found = null;
  for (const hit of S.meepleHits) {
    if (Math.hypot(px - hit.cx, py - hit.cy) <= hit.r + 3) {
      found = hit;
      break;
    }
  }
  if (!found) {
    clearMeepleHover();
    return;
  }
  S.hoverMeeple = found.meeple;
  showMeepleTip(found.meeple, px, py);
}

function clearMeepleHover() {
  S.hoverMeeple = null;
  meepleTip.classList.remove("show");
}

function showMeepleTip(m, px, py) {
  const kind = document.createElement("div");
  kind.className = "tip-kind";
  kind.textContent = m.feature_kind;
  const now = document.createElement("div");
  now.textContent = `score now: ${m.score_now}`;
  const rest = document.createElement("div");
  rest.textContent = m.complete ? "complete" : `if completed: ${m.score_potential}`;
  meepleTip.replaceChildren(kind, now, rest);
  meepleTip.classList.add("show");
  // Offset from the cursor, clamped inside the stage so nothing gets clipped.
  const sw = stage.clientWidth;
  const sh = stage.clientHeight;
  const tw = meepleTip.offsetWidth;
  const th = meepleTip.offsetHeight;
  let left = px + 16;
  let top = py + 16;
  if (left + tw > sw) left = px - tw - 16;
  if (top + th > sh) top = py - th - 16;
  meepleTip.style.left = `${Math.max(4, left)}px`;
  meepleTip.style.top = `${Math.max(4, top)}px`;
}

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

// (F2) Each score event is a clickable/keyboard-activatable row. Clicking it
// highlights the event's feature tiles on the board; clicking it again clears.
function logScore(e) {
  const mine = e.player === S.humanPlayer;
  const who = mine ? "You" : "AI";
  const n = e.tiles.length;

  const row = document.createElement("button");
  row.type = "button";
  row.className = `score-row ${mine ? "p0" : "p1"}`;
  row.style.setProperty("--tint", mine ? "var(--p0)" : "var(--p1)");

  const pts = document.createElement("span");
  pts.className = "pts";
  pts.textContent = `+${e.points}`;

  const lbl = document.createElement("span");
  lbl.className = "lbl";
  lbl.textContent = `${who} · ${e.kind} · ${n} tile${n === 1 ? "" : "s"}`;

  row.append(pts, lbl);
  const color = mine ? TOKENS.p0 : TOKENS.p1;
  const tiles = e.tiles;
  row.addEventListener("click", () => toggleScoreHighlight(row, tiles, color));
  logEl.append(row);
  logEl.scrollTop = logEl.scrollHeight;
}

function toggleScoreHighlight(row, tiles, color) {
  if (S.highlightRow === row) {
    clearScoreHighlight();
    return;
  }
  clearScoreHighlight();
  S.highlightRow = row;
  S.highlightColor = color;
  S.highlightTiles = new Set(tiles.map(([x, y]) => `${x},${y}`));
  row.classList.add("active");
}

function clearScoreHighlight() {
  if (S.highlightRow) S.highlightRow.classList.remove("active");
  S.highlightRow = null;
  S.highlightColor = null;
  S.highlightTiles = new Set();
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
