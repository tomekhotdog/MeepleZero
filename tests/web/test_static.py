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


@pytest.mark.parametrize(
    ("name", "content_type"),
    [
        ("app.js", "javascript"),
        ("tiles.js", "javascript"),
        ("style.css", "text/css"),
        ("index.html", "text/html"),
    ],
)
def test_static_assets_served(client: TestClient, name: str, content_type: str) -> None:
    r = client.get(f"/static/{name}")
    assert r.status_code == 200
    assert content_type in r.headers["content-type"]


def test_index_references_only_existing_files() -> None:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    refs = re.findall(r'(?:src|href)="([^"]+)"', html)
    assert refs, "index.html should reference its assets"
    for ref in refs:
        if ref.startswith("data:"):
            continue  # inline data URI (favicon): local by definition
        assert ref.startswith("/static/"), f"non-local reference: {ref}"
        assert (STATIC_DIR / ref.removeprefix("/static/")).is_file(), f"missing: {ref}"


@pytest.mark.parametrize("name", ["app.js", "tiles.js"])
def test_js_syntax(name: str, tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not on PATH: cannot syntax-check JS")
    # Copy to .mjs so every node version parses it as an ES module.
    mjs = tmp_path / f"{name}.mjs"
    mjs.write_text((STATIC_DIR / name).read_text(encoding="utf-8"), encoding="utf-8")
    result = subprocess.run([node, "--check", str(mjs)], capture_output=True, text=True)
    assert result.returncode == 0, f"{name} failed node --check:\n{result.stderr}"
