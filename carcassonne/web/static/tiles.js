// tiles.js — procedural tile renderer, printed-boardgame look.
//
// Everything is drawn from /api/tiledefs data (feature kind + edges + shield);
// there are NO per-tile-id special cases. Flat colours + thin outlines only.
//
// Coordinate conventions:
//   - a tile is drawn in a size x size box with its top-left at (0, 0);
//   - sides are N=0 (top), E=1 (right), S=2 (bottom), W=3 (left);
//   - `rot` is clockwise quarter-turns: feature side s sits at world side (s+rot)%4.
//
// TOKENS mirrors the CSS custom properties in style.css (canvas cannot read
// CSS variables cheaply; keep the two in sync).

export const TOKENS = {
  bone: "#E8E2D4",
  boneDim: "#9BA3AD",
  field: "#7A9B5E",
  fieldLine: "#5C7A45",
  city: "#C9A25E",
  cityWall: "#8F6E3C",
  road: "#D8CFBB",
  roadCase: "#6E6654",
  monastery: "#8E4A3E",
  p0: "#4C7DD0",
  p1: "#C05B4D",
  ai: "#5FD4C4",
  ink: "#14181E", // dark outline for meeples / glyph details
};

// Edge midpoints and corner runs, unit coords (y down, N at top).
const MID = [
  [0.5, 0],
  [1, 0.5],
  [0.5, 1],
  [0, 0.5],
];
// Corner at the start of each side walking clockwise: N starts at NW, etc.
const CORNER = [
  [0, 0],
  [1, 0],
  [1, 1],
  [0, 1],
];

const playerColor = (p) => (p === 0 ? TOKENS.p0 : TOKENS.p1);

// --- geometry helpers -------------------------------------------------------

/** Anchor (unit coords, rotation applied) for feature i of a tile def:
 *  where a meeple stands and where the radial picker puts its glyph. */
export function featureAnchor(def, featureIndex, rot) {
  const f = def.features[featureIndex];
  let ax, ay;
  if (!f.edges || f.edges.length === 0) {
    [ax, ay] = [0.5, 0.5]; // monastery / garden: tile centre
  } else {
    let mx = 0;
    let my = 0;
    for (const s of f.edges) {
      mx += MID[s][0];
      my += MID[s][1];
    }
    mx /= f.edges.length;
    my /= f.edges.length;
    // Pull toward centre so the marker sits inside the feature body.
    ax = mx + (0.5 - mx) * 0.45;
    ay = my + (0.5 - my) * 0.45;
  }
  return rotatePoint(ax, ay, rot);
}

function rotatePoint(x, y, rot) {
  let dx = x - 0.5;
  let dy = y - 0.5;
  for (let i = 0; i < ((rot % 4) + 4) % 4; i++) {
    [dx, dy] = [-dy, dx]; // 90° clockwise in screen coords (y down)
  }
  return { x: 0.5 + dx, y: 0.5 + dy };
}

// --- tile drawing -----------------------------------------------------------

/**
 * Draw one tile with its top-left at the current origin.
 *   ctx      CanvasRenderingContext2D
 *   def      tiledef: {edges: [...], features: [{kind, edges, shield}]}
 *   rot      0..3 clockwise quarter-turns
 *   size     box size in px
 *   opts     {meeples: [{player, kind, feature}], alpha: 0..1}
 */
export function drawTile(ctx, def, rot, size, opts = {}) {
  ctx.save();
  if (opts.alpha !== undefined) ctx.globalAlpha = opts.alpha;
  // Rotate the whole drawing; features are defined unrotated.
  ctx.translate(size / 2, size / 2);
  ctx.rotate((((rot % 4) + 4) % 4) * (Math.PI / 2));
  ctx.translate(-size / 2, -size / 2);

  const s = size;
  const lw = (1.5 * s) / 64;
  const roadW = (6 * s) / 64;

  // 1. Field base.
  ctx.fillStyle = TOKENS.field;
  ctx.fillRect(0, 0, s, s);
  ctx.strokeStyle = TOKENS.fieldLine;
  ctx.lineWidth = lw;
  ctx.strokeRect(lw / 2, lw / 2, s - lw, s - lw);

  const roads = def.features.filter((f) => f.kind === "road");
  const cities = def.features.filter((f) => f.kind === "city");
  const hasMonastery = def.features.some((f) => f.kind === "monastery");
  const hasGarden = def.features.some((f) => f.kind === "garden");

  // 2. Roads: casing pass then fill pass, so crossings merge cleanly.
  if (roads.length > 0) {
    const paths = roads.map((f) => roadPath(f, s));
    ctx.lineCap = "butt";
    ctx.lineJoin = "round";
    for (const [style, width] of [
      [TOKENS.roadCase, roadW + 2 * lw],
      [TOKENS.road, roadW],
    ]) {
      ctx.strokeStyle = style;
      ctx.lineWidth = width;
      for (const p of paths) ctx.stroke(p);
    }
    // Junction: several distinct road features all end at the centre.
    if (roads.length >= 2) {
      ctx.beginPath();
      ctx.arc(s / 2, s / 2, roadW * 0.9, 0, 2 * Math.PI);
      ctx.fillStyle = TOKENS.road;
      ctx.fill();
      ctx.strokeStyle = TOKENS.roadCase;
      ctx.lineWidth = lw;
      ctx.stroke();
    }
  }

  // 3. Cities bulging toward the centre, over their edge(s).
  for (const f of cities) {
    const path = cityPath(f.edges, s);
    ctx.fillStyle = TOKENS.city;
    ctx.fill(path);
    ctx.strokeStyle = TOKENS.cityWall;
    ctx.lineWidth = lw;
    ctx.stroke(path);
    if (f.shield) drawShield(ctx, f, s, lw);
  }

  // 4. Monastery house glyph.
  if (hasMonastery) drawMonastery(ctx, s, lw);

  // 5. Garden: ring of six dots around the centre.
  if (hasGarden) drawGarden(ctx, s);

  // 6. Meeples (anchors are in the unrotated frame; ctx is already rotated).
  for (const m of opts.meeples ?? []) {
    const a = featureAnchor(def, m.feature, 0);
    drawMeeple(ctx, a.x * s, a.y * s, s * 0.1, m.player, m.kind, lw);
  }

  ctx.restore();
}

function roadPath(f, s) {
  const p = new Path2D();
  const e = f.edges;
  const c = s / 2;
  if (e.length === 1) {
    p.moveTo(MID[e[0]][0] * s, MID[e[0]][1] * s);
    p.lineTo(c, c);
  } else {
    const [a, b] = e;
    p.moveTo(MID[a][0] * s, MID[a][1] * s);
    if ((b - a) % 2 === 0) {
      p.lineTo(MID[b][0] * s, MID[b][1] * s); // straight across
    } else {
      p.quadraticCurveTo(c, c, MID[b][0] * s, MID[b][1] * s); // corner curve
    }
  }
  return p;
}

function cityPath(edges, s) {
  const set = new Set(edges);
  const p = new Path2D();
  if (set.size === 4) {
    p.rect(0, 0, s, s);
    return p;
  }
  // Opposite pair: a waisted band across the tile.
  if (set.size === 2 && (Math.abs(edges[0] - edges[1]) === 2)) {
    const [a, b] = [Math.min(...edges), Math.max(...edges)];
    const [a0, a1] = [CORNER[a], CORNER[(a + 1) % 4]];
    const [b0, b1] = [CORNER[b], CORNER[(b + 1) % 4]];
    const waist = 0.2 * s;
    p.moveTo(a0[0] * s, a0[1] * s);
    p.lineTo(a1[0] * s, a1[1] * s);
    p.quadraticCurveTo(...pullToCentre(a1, b0, waist, s), b0[0] * s, b0[1] * s);
    p.lineTo(b1[0] * s, b1[1] * s);
    p.quadraticCurveTo(...pullToCentre(b1, a0, waist, s), a0[0] * s, a0[1] * s);
    p.closePath();
    return p;
  }
  // Contiguous run of 1..3 edges: walk the boundary, then one inward curve back.
  const start = runStart(set);
  const first = CORNER[start];
  p.moveTo(first[0] * s, first[1] * s);
  let side = start;
  while (set.has(side)) {
    const end = CORNER[(side + 1) % 4];
    p.lineTo(end[0] * s, end[1] * s);
    side = (side + 1) % 4;
  }
  const last = CORNER[side]; // free endpoint reached
  // Curve back to the start, bulging toward (or just past) the centre.
  const avg = [(last[0] + first[0]) / 2, (last[1] + first[1]) / 2];
  let m; // point the curve passes through at its midpoint
  const dx = 0.5 - avg[0];
  const dy = 0.5 - avg[1];
  const d = Math.hypot(dx, dy);
  if (d < 1e-6) {
    // Two adjacent edges: chord is the diagonal; bulge just past the centre.
    const shared = CORNER[(start + 1) % 4]; // corner between the two edges
    m = [0.5 + (0.5 - shared[0]) * 0.12, 0.5 + (0.5 - shared[1]) * 0.12];
  } else {
    m = [avg[0] + (dx / d) * 0.35, avg[1] + (dy / d) * 0.35];
  }
  const cpx = 2 * m[0] - avg[0];
  const cpy = 2 * m[1] - avg[1];
  p.quadraticCurveTo(cpx * s, cpy * s, first[0] * s, first[1] * s);
  p.closePath();
  return p;
}

/** First side of the contiguous run (the side whose predecessor is not in the set). */
function runStart(set) {
  for (const s of set) {
    if (!set.has((s + 3) % 4)) return s;
  }
  return set.values().next().value; // unreachable for valid defs
}

/** Control point args for a quadratic that sags toward the centre by `waist`. */
function pullToCentre(from, to, waist, s) {
  const ax = (from[0] + to[0]) / 2;
  const ay = (from[1] + to[1]) / 2;
  const dx = 0.5 * s - ax * s;
  const dy = 0.5 * s - ay * s;
  const d = Math.hypot(dx, dy) || 1;
  const mx = ax * s + (dx / d) * waist;
  const my = ay * s + (dy / d) * waist;
  return [2 * mx - ax * s, 2 * my - ay * s];
}

function drawShield(ctx, f, s, lw) {
  // Badge centred a bit inside the city body.
  let mx = 0;
  let my = 0;
  for (const e of f.edges) {
    mx += MID[e][0];
    my += MID[e][1];
  }
  mx /= f.edges.length;
  my /= f.edges.length;
  const cx = (mx + (0.5 - mx) * 0.3) * s;
  const cy = (my + (0.5 - my) * 0.3) * s;
  const w = 0.13 * s;
  const h = 0.17 * s;
  ctx.beginPath();
  ctx.moveTo(cx - w / 2, cy - h / 2);
  ctx.lineTo(cx + w / 2, cy - h / 2);
  ctx.lineTo(cx + w / 2, cy + h * 0.1);
  ctx.lineTo(cx, cy + h / 2); // the point
  ctx.lineTo(cx - w / 2, cy + h * 0.1);
  ctx.closePath();
  ctx.fillStyle = TOKENS.cityWall;
  ctx.fill();
  ctx.strokeStyle = "rgba(0, 0, 0, 0.35)";
  ctx.lineWidth = lw;
  ctx.stroke();
}

function drawMonastery(ctx, s, lw) {
  const p = new Path2D();
  p.rect(0.38 * s, 0.46 * s, 0.24 * s, 0.18 * s); // body
  p.moveTo(0.35 * s, 0.46 * s); // roof
  p.lineTo(0.5 * s, 0.32 * s);
  p.lineTo(0.65 * s, 0.46 * s);
  p.closePath();
  ctx.fillStyle = TOKENS.monastery;
  ctx.fill(p);
  ctx.strokeStyle = "rgba(0, 0, 0, 0.35)";
  ctx.lineWidth = lw;
  ctx.stroke(p);
}

function drawGarden(ctx, s) {
  ctx.fillStyle = TOKENS.fieldLine;
  for (let i = 0; i < 6; i++) {
    const a = (i / 6) * 2 * Math.PI - Math.PI / 2;
    ctx.beginPath();
    ctx.arc(
      s / 2 + Math.cos(a) * 0.16 * s,
      s / 2 + Math.sin(a) * 0.16 * s,
      0.028 * s,
      0,
      2 * Math.PI
    );
    ctx.fill();
  }
}

function drawMeeple(ctx, cx, cy, r, player, kind, lw) {
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, 2 * Math.PI);
  ctx.fillStyle = playerColor(player);
  ctx.fill();
  ctx.strokeStyle = TOKENS.ink;
  ctx.lineWidth = lw;
  ctx.stroke();
  if (kind === "abbot") {
    ctx.beginPath();
    ctx.moveTo(cx - r * 0.55, cy);
    ctx.lineTo(cx + r * 0.55, cy);
    ctx.moveTo(cx, cy - r * 0.55);
    ctx.lineTo(cx, cy + r * 0.55);
    ctx.strokeStyle = TOKENS.ink;
    ctx.lineWidth = lw * 1.2;
    ctx.stroke();
  }
}

/** Standalone meeple glyph for sidebar pips and the radial picker. */
export function drawMeepleGlyph(ctx, size, player, kind, filled = true) {
  ctx.clearRect(0, 0, size, size);
  const r = size * 0.36;
  ctx.beginPath();
  ctx.arc(size / 2, size / 2, r, 0, 2 * Math.PI);
  if (filled) {
    ctx.fillStyle = playerColor(player);
    ctx.fill();
  }
  ctx.strokeStyle = filled ? TOKENS.ink : "rgba(155, 163, 173, 0.6)";
  ctx.lineWidth = 1.5;
  ctx.stroke();
  if (kind === "abbot") {
    ctx.beginPath();
    ctx.moveTo(size / 2 - r * 0.55, size / 2);
    ctx.lineTo(size / 2 + r * 0.55, size / 2);
    ctx.moveTo(size / 2, size / 2 - r * 0.55);
    ctx.lineTo(size / 2, size / 2 + r * 0.55);
    ctx.stroke();
  }
}
