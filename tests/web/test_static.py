"""Static frontend assets: served correctly, syntactically valid, self-consistent.

The JS behaviour itself gets its visual pass in a browser; here we pin what a
test can pin: the files are served with the right content types, the HTML only
references assets that exist, and the ES modules at least parse.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from carcassonne.web.app import create_app

STATIC_DIR = Path(__file__).resolve().parents[2] / "carcassonne" / "web" / "static"


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(tmp_path / "replays"))


def test_index_served_at_root(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert 'id="app"' in r.text  # the app root the JS mounts into
    assert "app.js" in r.text


def test_replay_page_served(client: TestClient) -> None:
    r = client.get("/replay")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert 'id="app"' in r.text
    assert "replay.js" in r.text


def test_tiles_page_served(client: TestClient) -> None:
    r = client.get("/tiles")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert 'id="app"' in r.text
    assert "atlas.js" in r.text
    # persistent nav: all four views linked
    for href in ('href="/"', 'href="/replay"', 'href="/training"', 'href="/tiles"'):
        assert href in r.text, f"missing nav link {href}"


@pytest.mark.parametrize(
    ("name", "content_type"),
    [
        ("app.js", "javascript"),
        ("board.js", "javascript"),
        ("overlays.js", "javascript"),
        ("replay.js", "javascript"),
        ("training.js", "javascript"),
        ("tiles.js", "javascript"),
        ("atlas.js", "javascript"),
        ("style.css", "text/css"),
        ("index.html", "text/html"),
        ("replay.html", "text/html"),
        ("training.html", "text/html"),
        ("tiles.html", "text/html"),
    ],
)
def test_static_assets_served(client: TestClient, name: str, content_type: str) -> None:
    r = client.get(f"/static/{name}")
    assert r.status_code == 200
    assert content_type in r.headers["content-type"]


@pytest.mark.parametrize("page", ["index.html", "replay.html", "training.html", "tiles.html"])
def test_page_references_only_existing_files(page: str) -> None:
    html = (STATIC_DIR / page).read_text(encoding="utf-8")
    app_routes = {"/", "/replay", "/training", "/tiles"}  # nav links to other views, not assets
    refs = re.findall(r'(?:src|href)="([^"]+)"', html)
    assert refs, f"{page} should reference its assets"
    for ref in refs:
        if ref.startswith("data:") or ref in app_routes:
            continue  # inline data URI (favicon) or an in-app nav route
        assert ref.startswith("/static/"), f"non-local reference: {ref}"
        assert (STATIC_DIR / ref.removeprefix("/static/")).is_file(), f"missing: {ref}"


@pytest.mark.parametrize(
    "name",
    ["app.js", "board.js", "overlays.js", "replay.js", "training.js", "tiles.js", "atlas.js"],
)
def test_js_syntax(name: str, tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not on PATH: cannot syntax-check JS")
    # Copy to .mjs so every node version parses it as an ES module.
    mjs = tmp_path / f"{name}.mjs"
    mjs.write_text((STATIC_DIR / name).read_text(encoding="utf-8"), encoding="utf-8")
    result = subprocess.run([node, "--check", str(mjs)], capture_output=True, text=True)
    assert result.returncode == 0, f"{name} failed node --check:\n{result.stderr}"
