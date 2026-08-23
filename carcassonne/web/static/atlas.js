// atlas.js — the Tiles reference tab (the full renderer lands in a later task).
async function main() {
  const res = await fetch("/api/tiledefs");
  const { tiles } = await res.json();
  document.getElementById("summary").textContent =
    `${Object.keys(tiles).length} tile types loaded`;
}
main().catch((e) => {
  document.getElementById("atlas-error").textContent = String(e);
});
