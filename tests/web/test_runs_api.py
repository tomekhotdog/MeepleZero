"""The training-run dashboard API: /api/runs, /api/runs/{name}, /training.

A synthetic TrainingRun is laid out in a tmp dir -- a config, a handful of fake
metrics.jsonl lines (both learner steps and arena gates), and two empty
checkpoint files. No real training happens: the endpoints only read the run
directory, so hand-written fixtures exercise the exact parse the dashboard
consumes. We assert the *shape and provenance* of the served data.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from carcassonne.training.orchestrate import TrainConfig, run_config_dict
from carcassonne.training.run import TrainingRun
from carcassonne.web.app import create_app

STATIC_DIR = Path(__file__).resolve().parents[2] / "carcassonne" / "web" / "static"

# Two learner steps and two gates (one promoted) -- enough for both series.
_LEARN_LINES = [
    {"step": 1, "loss": 3.5, "policy_loss": 3.0, "value_loss": 0.5,
     "buffer_games": 2, "buffer_examples": 40},
    {"step": 2, "loss": 3.1, "policy_loss": 2.7, "value_loss": 0.4,
     "buffer_games": 4, "buffer_examples": 80},
]
_GATE_LINES = [
    {"kind": "gate", "iter": 1, "candidate_step": 2, "promoted": False,
     "wr_best": 0.40, "wr_greedy": 0.30},
    {"kind": "gate", "iter": 2, "candidate_step": 10, "promoted": True,
     "wr_best": 0.60, "wr_greedy": 0.55},
]


@pytest.fixture()
def runs_dir(tmp_path: Path) -> Path:
    """A runs dir holding one synthetic run ("first") plus a non-run dir."""
    root = tmp_path / "runs"
    run = TrainingRun.create(root / "first", run_config_dict(TrainConfig(channels=8, n_blocks=1)))
    with run.metrics_path.open("w", encoding="utf-8") as fh:
        # Interleaved, as the orchestrator writes them: learner steps then a gate.
        for line in (_LEARN_LINES[0], _LEARN_LINES[1], _GATE_LINES[0], _GATE_LINES[1]):
            fh.write(json.dumps(line) + "\n")
    (run.checkpoints_dir / "step_000010.pt").touch()  # out of order on purpose
    (run.checkpoints_dir / "step_000000.pt").touch()

    # A sibling directory that is NOT a training run (no config.json): must be skipped.
    (root / "not_a_run").mkdir()
    (root / "not_a_run" / "readme.txt").write_text("noise", encoding="utf-8")
    return root


@pytest.fixture()
def client(tmp_path: Path, runs_dir: Path) -> TestClient:
    return TestClient(create_app(tmp_path / "replays", None, runs_dir))


def test_list_runs_summarises_only_real_runs(client: TestClient) -> None:
    runs = client.get("/api/runs").json()["runs"]
    assert [r["name"] for r in runs] == ["first"]  # not_a_run skipped
    (run,) = runs
    assert run["n_checkpoints"] == 2
    assert run["last_step"] == 2
    assert run["iterations_or_steps"] == 2
    assert run["last_gate"]["iter"] == 2
    assert run["last_gate"]["promoted"] is True
    assert run["config"]["net_arch"]["channels"] == 8


def test_runs_endpoint_empty_without_dir(tmp_path: Path) -> None:
    empty = TestClient(create_app(tmp_path / "replays"))  # no runs_dir
    assert empty.get("/api/runs").json() == {"runs": []}


def test_run_detail_splits_metrics_and_sorts_checkpoints(client: TestClient) -> None:
    body = client.get("/api/runs/first").json()

    assert body["config"]["net_arch"]["n_blocks"] == 1
    assert body["checkpoints"] == [
        {"step": 0, "file": "step_000000.pt"},
        {"step": 10, "file": "step_000010.pt"},
    ]

    learn = body["metrics"]["learn"]
    gate = body["metrics"]["gate"]
    assert [r["step"] for r in learn] == [1, 2]
    assert set(learn[0]) == {
        "step", "loss", "policy_loss", "value_loss", "buffer_games", "buffer_examples"
    }
    assert [r["iter"] for r in gate] == [1, 2]
    assert set(gate[0]) == {
        "kind", "iter", "candidate_step", "promoted", "wr_best", "wr_greedy"
    }
    assert all(r["kind"] == "gate" for r in gate)  # tagging preserved


def test_run_detail_tolerates_a_half_written_final_line(tmp_path: Path, runs_dir: Path) -> None:
    """A killed learner can leave a truncated JSON line; it must be skipped, not 500."""
    with (runs_dir / "first" / "metrics.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"step": 3, "loss": 2.9')  # no newline, no closing brace
    client = TestClient(create_app(tmp_path / "replays", None, runs_dir))
    learn = client.get("/api/runs/first").json()["metrics"]["learn"]
    assert [r["step"] for r in learn] == [1, 2]  # the garbled line is dropped


@pytest.mark.parametrize(
    "name",
    ["../secret", "..", "../../etc/passwd", r"..\\secret", ".hidden"],
)
def test_path_traversal_is_rejected(client: TestClient, name: str) -> None:
    r = client.get(f"/api/runs/{name}")
    assert r.status_code == 404, name


def test_unknown_run_is_404(client: TestClient) -> None:
    r = client.get("/api/runs/nope")
    assert r.status_code == 404
    assert r.json()["error"] == "not_found"


def test_training_page_served(client: TestClient) -> None:
    r = client.get("/training")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert 'id="app"' in r.text
    assert "training.js" in r.text


def test_training_page_references_only_existing_files() -> None:
    html = (STATIC_DIR / "training.html").read_text(encoding="utf-8")
    app_routes = {"/", "/replay", "/training", "/tiles"}  # top-nav links to other views
    refs = re.findall(r'(?:src|href)="([^"]+)"', html)
    for ref in refs:
        if ref.startswith("data:") or ref in app_routes:
            continue
        assert ref.startswith("/static/"), f"non-local reference: {ref}"
        assert (STATIC_DIR / ref.removeprefix("/static/")).is_file(), f"missing: {ref}"


def test_training_js_syntax(tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not on PATH: cannot syntax-check JS")
    mjs = tmp_path / "training.js.mjs"
    mjs.write_text((STATIC_DIR / "training.js").read_text(encoding="utf-8"), encoding="utf-8")
    result = subprocess.run([node, "--check", str(mjs)], capture_output=True, text=True)
    assert result.returncode == 0, f"training.js failed node --check:\n{result.stderr}"
