// app.js — the play view state machine.
//
// Shape: fetch state -> render -> await input. One shared render path; the
// board canvas redraws fully every animation frame (boards are tiny). All
// server contracts are consumed exactly as produced by web/views.py.

import { TOKENS, drawTile, drawMeepleGlyph, featureAnchor, playerColor } from "./tiles.js";
import {
  cellPx as boardCellPx,
  worldToScreen as boardWorldToScreen,
  screenToCell as boardScreenToCell,
  screenToCellFloat as boardScreenToCellFloat,
  resizeToDisplay,
  drawTiles,
  drawFrontier,
} from "./board.js";
import {
  drawMoveRing,
  drawTileHighlight,
  drawFeatureOutline,
  collectMeepleHits,
  hitTestMeeple,
  createMeepleTip,
  showMeepleTip as showTip,
  hideMeepleTip,
} from "./overlays.js";

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

const DECK_SIZE = 72; // full base deck; the start tile counts as played

// --- application state -------------------------------------------------------

const S = {
  tiledefs: null, // {id: {edges, features}} fetched once at startup
  gameId: null,
  view: null, // last StateView
  legal: [], // legal moves list; move.idx indexes THIS list for POST /move
  rotation: 0, // current lens rotation, 0..3 clockwise
  lensDeg: 0, // cumulative CSS rotation so 270->360 animates forward
  humanPlayer: 0,
  seats: [], // per-seat names from the server ("human" at humanPlayer); length = player count
  checkpoints: [], // [{id, step}] checkpoint opponents, offered only for 2-player games
  phase: "idle", // idle | placing | picking | awaiting | over
  pending: null, // {x, y, moves} while the radial picker is open
  hintOn: false,
  hint: null, // {policy: [{x, y, rot, prob}], value}
  hintEpoch: 0, // discard stale hint responses
  cam: { x: 0, y: 0, zoom: 1 },
  hover: null, // hovered board cell {x, y}
  aiPulse: null, // {x, y, start} settle pulse on the most recent AI placement
  drag: null,
  error: null, // fatal fetch failure message; overrides the status line until a new game
  lastHumanMove: null, // {x, y} of the human's most recent tile placement
  lastAiMove: null, // {x, y} of the most recent AI tile placement (any AI seat)
  lastAiSeat: null, // seat that made lastAiMove; colours its ring
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

const meepleTip = createMeepleTip(stage);

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

// Hints are a 2-player feature: the server 409s them for 3-5 players, and the
// hint row is hidden then, so never issue the request at all.
const hintsAvailable = () => S.seats.length === 2;

async function refreshHint() {
  if (!S.hintOn || S.phase !== "placing" || !hintsAvailable()) return;
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

// Drop the current hint AND invalidate any in-flight /hint request, so a
// response for an earlier position (or an earlier game) can never paint.
function clearHint() {
  S.hint = null;
  S.hintEpoch++;
}

// --- game flow -------------------------------------------------------------------

const SEAT_COLOR_NAMES = ["blue", "red", "purple", "pink", "yellow"];

// `opponents` are the AI seats' agent specs in seat order (the human's seat is
// skipped), so the table has opponents.length + 1 players.
async function newGame(opponents, seat, seed) {
  const body = { opponents, num_players: opponents.length + 1, human_player: seat };
  S.error = null;
  if (seed !== null) body.seed = seed;
  const data = await postJSON("/api/games", body);
  S.gameId = data.game_id;
  S.view = data.state;
  S.seats = data.seats;
  S.humanPlayer = seat;
  S.pending = null;
  clearHint();
  S.aiPulse = null;
  S.lastHumanMove = null;
  S.lastAiMove = null;
  S.lastAiSeat = null;
  S.hoverMeeple = null;
  clearScoreHighlight();
  S.cam = { x: 0, y: 0, zoom: 1 };
  setRotation(0, true);
  logEl.replaceChildren();
  $("cards").replaceChildren(); // rebuilt for this table's seat count
  const hideHints = !hintsAvailable();
  $("hint-row").hidden = hideHints;
  $("value-readout").hidden = hideHints;
  logMeta(`New game vs ${opponents.join(", ")} — you are ${SEAT_COLOR_NAMES[seat]}`);
  // AI seats before the human's open the game; `events` are their score events
  // in play order.
  for (const e of data.events) logScore(e);
  applyAiMoves(data.ai_moves);
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
  clearHint();
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
  // `events` is the human's move then every AI move's, in play order.
  for (const e of data.events) logScore(e);
  applyAiMoves(data.ai_moves);
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

// (F1) Mark the most recent of a turn's consecutive AI moves: its ring (in that
// seat's colour) and a brief settle pulse. Earlier AI moves in the same batch are
// visible on the board and in the score log, but only the latest gets a ring.
function applyAiMoves(aiMoves) {
  const last = aiMoves.at(-1);
  if (!last) return;
  S.lastAiMove = { x: last.move.x, y: last.move.y };
  S.lastAiSeat = last.seat;
  if (!reducedMotion.matches) {
    S.aiPulse = { x: last.move.x, y: last.move.y, start: performance.now() };
  }
}

// Display name for a seat: "You" for the human, else the agent name. Repeated
// agent names (two greedy seats) get a P<n> suffix so they stay distinct. Seat
// numbers shown to the user are 1-indexed, matching the dialog's 1st..5th.
function seatName(seat) {
  if (seat === S.humanPlayer) return "You";
  const name = S.seats[seat] ?? `P${seat + 1}`;
  return S.seats.filter((n) => n === name).length > 1 ? `${name} P${seat + 1}` : name;
}

function showBanner(replayName) {
  const fs = S.view.final_scores;
  const best = Math.max(...fs);
  const winners = fs.flatMap((pts, seat) => (pts === best ? [seat] : []));
  let verdict;
  if (winners.length === fs.length) {
    verdict = "Draw";
  } else if (winners.length === 1) {
    verdict = winners[0] === S.humanPlayer ? "You win" : `${seatName(winners[0])} wins`;
  } else {
    verdict = `Tie: ${winners.map(seatName).join(" & ")}`;
  }
  // "You win — You 34 · random 28 · greedy 41", winners bolded.
  const line = $("banner-scores");
  line.replaceChildren(`${verdict} — `);
  fs.forEach((pts, seat) => {
    if (seat > 0) line.append(" · ");
    const text = `${seatName(seat)} ${pts}`;
    if (winners.includes(seat)) {
      const b = document.createElement("strong");
      b.textContent = text;
      line.append(b);
    } else {
      line.append(text);
    }
  });
  $("banner-saved").textContent = replayName ? `Saved as ${replayName}` : "";
  $("banner").classList.add("show");
  logMeta(`Game over — ${fs.join(":")}`);
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
  S.meepleHits = collectMeepleHits(S.view, S.tiledefs, board, S.cam);

  // (F2) Score-row highlight: fill + outline the clicked event's feature tiles in
  // the scoring player's colour — distinct from the neutral meeple-hover outline.
  if (S.highlightTiles.size > 0) {
    drawTileHighlight(bctx, board, S.cam, S.highlightTiles, S.highlightColor);
  }

  // (F3) Meeple-hover: neutral --bone outline around the hovered meeple's feature.
  if (S.hoverMeeple) {
    drawFeatureOutline(bctx, board, S.cam, S.hoverMeeple.feature_tiles);
  }

  // (F1) Persistent last-move markers: a solid ring in the mover's colour plus a
  // labelled corner chip, so "your last move" vs "AI's last move" is unambiguous.
  // With several AI seats only the most recent AI move is ringed, named by seat.
  drawMoveRing(bctx, board, S.cam, S.lastHumanMove, playerColor(S.humanPlayer), "You");
  if (S.lastAiMove) {
    const aiLabel = S.seats.length === 2 ? "AI" : seatName(S.lastAiSeat);
    drawMoveRing(bctx, board, S.cam, S.lastAiMove, playerColor(S.lastAiSeat), aiLabel);
  }

  const placing = S.phase === "placing";
  const legalCells = new Map(); // cells legal at the CURRENT rotation
  if (placing) {
    for (const m of S.legal) {
      if (m.rot === S.rotation) legalCells.set(`${m.x},${m.y}`, m);
    }
  }

  // Hint overlay: --ai glow, opacity proportional to policy prob (current rotation).
  if (placing && S.hintOn && S.hint && hintsAvailable()) {
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
  const found = hitTestMeeple(S.meepleHits, px, py);
  if (!found) {
    clearMeepleHover();
    return;
  }
  S.hoverMeeple = found;
  showTip(meepleTip, stage, found, px, py);
}

function clearMeepleHover() {
  S.hoverMeeple = null;
  hideMeepleTip(meepleTip);
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
  const cards = $("cards");
  if (cards.children.length !== scores.length) {
    cards.replaceChildren(...scores.map((_, seat) => scoreCard(seat)));
    cards.classList.toggle("many", scores.length > 2);
  }
  for (let seat = 0; seat < scores.length; seat++) {
    const card = $(`card-${seat}`);
    card.querySelector(".who .name").textContent = seatName(seat);
    card.querySelector(".points").textContent = String(scores[seat]);
    const supply = card.querySelector(".supply");
    supply.replaceChildren();
    for (let i = 0; i < 7; i++) {
      supply.append(pip(seat, "meeple", i < S.view.meeples[seat]));
    }
    supply.append(pip(seat, "abbot", S.view.abbots[seat]));
  }
  const played = S.view.tiles.length;
  $("tile-count").textContent = `tiles played ${played} · remaining ${DECK_SIZE - played}`;
}

function scoreCard(seat) {
  const card = document.createElement("div");
  card.className = "score-card";
  card.id = `card-${seat}`;
  card.style.setProperty("--edge", playerColor(seat));
  for (const cls of ["who", "points", "supply"]) {
    const div = document.createElement("div");
    div.className = cls;
    card.append(div);
  }
  // Colour swatch beside the name, so seat colour is tied to the name in the
  // text line and not only to the card's edge (colour-blind players).
  const swatch = document.createElement("span");
  swatch.className = `swatch p${seat}`;
  swatch.title = SEAT_COLOR_NAMES[seat];
  swatch.setAttribute("role", "img");
  swatch.setAttribute("aria-label", SEAT_COLOR_NAMES[seat]);
  const name = document.createElement("span");
  name.className = "name";
  card.querySelector(".who").append(swatch, name);
  return card;
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
  const n = e.tiles.length;
  const color = playerColor(e.player);

  const row = document.createElement("button");
  row.type = "button";
  row.className = "score-row";
  row.style.setProperty("--tint", color);

  const pts = document.createElement("span");
  pts.className = "pts";
  pts.textContent = `+${e.points}`;

  const lbl = document.createElement("span");
  lbl.className = "lbl";
  lbl.textContent = `${seatName(e.player)} · ${e.kind} · ${n} tile${n === 1 ? "" : "s"}`;

  row.append(pts, lbl);
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

const ORDINALS = ["1st", "2nd", "3rd", "4th", "5th"];

function option(value, text) {
  const opt = document.createElement("option");
  opt.value = value;
  opt.textContent = text;
  return opt;
}

// One opponent <select> per AI seat (n-1 of them). Checkpoint agents are
// 2-player only (the server 422s otherwise), so they're offered only when n === 2;
// a checkpoint picked at 2 players falls back to greedy when the table grows.
function renderOpponentSelects(n) {
  const wrap = $("opt-opponents");
  const prev = [...wrap.querySelectorAll("select")].map((sel) => sel.value);
  wrap.replaceChildren();
  for (let i = 0; i < n - 1; i++) {
    const id = `opt-opponent-${i}`;
    const row = document.createElement("div");
    row.className = "row";
    const label = document.createElement("label");
    label.htmlFor = id;
    label.textContent = n === 2 ? "Opponent" : `Opponent ${i + 1}`;
    const sel = document.createElement("select");
    sel.id = id;
    sel.append(option("greedy", "Greedy"), option("random", "Random"));
    if (n === 2) {
      for (const ckpt of S.checkpoints) {
        sel.append(option(`ckpt:${ckpt.id}`, `Checkpoint ${ckpt.step}`));
      }
    }
    const keep = prev[i] ?? prev[0];
    sel.value = [...sel.options].some((o) => o.value === keep) ? keep : "greedy";
    row.append(label, sel);
    wrap.append(row);
  }
}

// Seats 0..n-1 labelled by turn order; keeps the current pick while it fits.
function renderSeatOptions(n) {
  const sel = $("opt-seat");
  const prev = Number(sel.value);
  sel.replaceChildren();
  for (let s = 0; s < n; s++) {
    const text = s === 0 ? "You start" : n === 2 ? "AI starts" : `You play ${ORDINALS[s]}`;
    sel.append(option(String(s), text));
  }
  sel.value = String(Number.isInteger(prev) && prev < n ? prev : 0);
}

function renderDialogOptions() {
  const n = Number($("opt-players").value);
  renderOpponentSelects(n);
  renderSeatOptions(n);
}

function openDialog() {
  $("dialog-error").textContent = "";
  dialog.showModal();
}

$("opt-players").addEventListener("change", renderDialogOptions);
$("new-game").addEventListener("click", openDialog);
$("banner-again").addEventListener("click", openDialog);
$("dialog-cancel").addEventListener("click", () => dialog.close());
$("dialog-start").addEventListener("click", async () => {
  const opponents = [...$("opt-opponents").querySelectorAll("select")].map((sel) => sel.value);
  const seat = Number($("opt-seat").value);
  const seedRaw = $("opt-seed").value.trim();
  const seed = seedRaw === "" ? null : Number(seedRaw);
  dialog.close();
  try {
    await newGame(opponents, seat, Number.isFinite(seed) ? seed : null);
  } catch (err) {
    // Config errors carry {"error", "explanation"}; prefer the explanation.
    const why = err.body?.explanation ?? err.message ?? String(err);
    S.error = `Could not start the game — ${why}`;
    S.phase = "idle";
    updateSidebar();
    openDialog();
    $("dialog-error").textContent = why;
  }
});

// --- hint toggle -------------------------------------------------------------------------------------------

$("hint-toggle").addEventListener("change", (e) => {
  S.hintOn = e.target.checked;
  if (S.hintOn) {
    refreshHint();
  } else {
    clearHint();
  }
  updateValueReadout();
});

// --- lens interactions ------------------------------------------------------------------------------------------

$("lens").addEventListener("click", rotate);

// --- boot ---------------------------------------------------------------------------------------------------------

async function populateCheckpoints() {
  // Offer each saved checkpoint as a `ckpt:<id>` opponent (2-player only). If
  // none are configured the dialog keeps just greedy/random.
  try {
    const { checkpoints } = await api("/api/checkpoints");
    S.checkpoints = checkpoints;
  } catch {
    /* checkpoints are optional; ignore a missing/errored endpoint */
  }
  renderDialogOptions();
}

renderDialogOptions(); // greedy/random immediately; checkpoints join once fetched

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
    await newGame(["greedy"], 0, null);
  } catch {
    openDialog(); // couldn't auto-start (e.g. server error) — let the user choose
  }
}

boot();
