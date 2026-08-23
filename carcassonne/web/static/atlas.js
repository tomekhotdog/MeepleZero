// atlas.js — the Tiles reference tab. A static gallery of every deck entry,
// grouped by feature and rendered via the shared drawTile renderer. Click a
// card to rotate its tile. Reads /api/tiledefs once; holds no gameplay state.
//
// Named atlas.js (not tiles.js) deliberately: tiles.js is the shared procedural
// tile renderer imported below and by app.js/replay.js; this is the page module.

import { drawTile } from "./tiles.js";

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
  // Shield doubles a city's score; surface it in the note so it also reaches the
  // card's aria-label (the visual badge alone is invisible to screen readers).
  if (cities.some((f) => f.shield)) parts.push("Shield");
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
