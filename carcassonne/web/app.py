"""FastAPI app factory for the play/replay API.

Thin routing layer: game flow lives in sessions.py, response shapes in
views.py. Domain errors surface as typed exceptions and are mapped to HTTP
status codes by registered handlers, never by per-route try/except. Runs in a
single uvicorn worker: sessions are in-process state.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from carcassonne.agents import list_checkpoints, set_checkpoints_dir
from carcassonne.core import IllegalMove, RulesError
from carcassonne.game.replay import load_replay, replay_states
from carcassonne.training.run import TrainingRun
from carcassonne.web import views
from carcassonne.web.sessions import NotFound, SessionStore, UnknownOpponent

_REPLAY_NAME = re.compile(r"[A-Za-z0-9._-]+\.jsonl")
# A run name is a directory basename: a leading dot is disallowed so "." and
# ".." can never match, and no path separators are permitted -- traversal-safe.
_RUN_NAME = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]*")
_CKPT_PREFIX = "step_"
_CKPT_SUFFIX = ".pt"
_STATIC_DIR = Path(__file__).parent / "static"


class CreateGameBody(BaseModel):
    opponent: str
    human_player: Literal[0, 1] = 0
    seed: int | None = None


class MoveBody(BaseModel):
    idx: int


def create_app(
    replays_dir: Path,
    checkpoints_dir: Path | None = None,
    runs_dir: Path | None = None,
) -> FastAPI:
    app = FastAPI(title="carcassonne")
    store = SessionStore(replays_dir)
    # The agent registry is process-global; the checkpoints dir is per-app. Single
    # uvicorn worker, so setting it here (as sessions get replays_dir) is safe.
    set_checkpoints_dir(checkpoints_dir)
    _register_error_handlers(app)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    @app.get("/replay")
    def replay_page() -> FileResponse:
        return FileResponse(_STATIC_DIR / "replay.html")

    @app.get("/training")
    def training_page() -> FileResponse:
        return FileResponse(_STATIC_DIR / "training.html")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True}

    @app.post("/api/games")
    def create_game(body: CreateGameBody) -> dict[str, Any]:
        seed = body.seed if body.seed is not None else random.randrange(2**31)
        session = store.create(body.opponent, body.human_player, seed)
        if body.human_player == 1:
            session.ai_step()  # the AI holds seat 0 and opens the game
        return {"game_id": session.id, "state": views.state_view(session.state)}

    @app.get("/api/games/{game_id}")
    def get_game(game_id: str) -> dict[str, Any]:
        return {"state": views.state_view(store.get(game_id).state)}

    @app.get("/api/games/{game_id}/legal")
    def get_legal(game_id: str) -> dict[str, Any]:
        return views.legal_moves_view(store.get(game_id).legal_cache)

    @app.post("/api/games/{game_id}/move")
    def post_move(game_id: str, body: MoveBody) -> dict[str, Any]:
        session = store.get(game_id)
        events = list(session.apply_human(body.idx))
        ai_move: dict[str, Any] | None = None
        if not session.terminal:
            move, annot, ai_events = session.ai_step()
            events.extend(ai_events)
            ai_move = views.ai_move_view(move, annot)
        response: dict[str, Any] = {
            "state": views.state_view(session.state),
            "ai_move": ai_move,
            "events": [views.score_event_view(e) for e in events],
        }
        if session.terminal:
            response["replay"] = session.replay_name
        return response

    @app.get("/api/games/{game_id}/hint")
    def get_hint(game_id: str) -> dict[str, Any]:
        session = store.get(game_id)
        if session.terminal:
            raise NotFound(f"game {game_id} is over: no hint for a finished game")
        return views.hint_view(session.legal_cache, session.hint_annot())

    @app.get("/api/checkpoints")
    def checkpoints() -> dict[str, Any]:
        """Available checkpoint opponents ([{id, step}]); empty if none configured.
        The new-game dialog offers each as a ``ckpt:<id>`` opponent."""
        return {"checkpoints": list_checkpoints()}

    @app.get("/api/tiledefs")
    def tiledefs() -> dict[str, Any]:
        return views.tiledefs_view()

    @app.get("/api/replays")
    def list_replays() -> dict[str, Any]:
        return {"replays": _list_replays(replays_dir)}

    @app.get("/api/replays/{name}")
    def get_replay(name: str) -> dict[str, Any]:
        replay = load_replay(_replay_file(replays_dir, name))
        # replay_states re-derives every position through the rules core and
        # cross-checks each record; a RulesError here maps to a 400.
        states = [views.state_view(s) for s in replay_states(replay)]
        return {
            "header": views.replay_header_view(replay.header),
            "moves": [views.move_record_view(r) for r in replay.moves],
            "final_scores": list(replay.final_scores),
            "winner": replay.winner,
            "states": states,
        }

    @app.get("/api/runs")
    def list_runs() -> dict[str, Any]:
        """Summaries of every TrainingRun under ``runs_dir`` (dirs with a
        config.json). Cheap: config + a checkpoint count + the last metric lines."""
        return {"runs": _list_runs(runs_dir)}

    @app.get("/api/runs/{name}")
    def get_run(name: str) -> dict[str, Any]:
        """Full detail for one run: config, checkpoint timeline, and the two
        metric series (learner steps + arena gates) split apart."""
        return _run_detail(_run_dir(runs_dir, name))

    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
    return app


def _register_error_handlers(app: FastAPI) -> None:
    def register(exc_type: type[Exception], status: int, error: str) -> None:
        def handle(request: Request, exc: Exception) -> JSONResponse:
            return JSONResponse(
                status_code=status, content={"error": error, "explanation": str(exc)}
            )

        app.add_exception_handler(exc_type, handle)

    register(IllegalMove, 409, "illegal_move")
    register(NotFound, 404, "not_found")
    register(RulesError, 400, "replay_verification_failed")
    register(UnknownOpponent, 422, "unknown_opponent")


def _list_replays(replays_dir: Path) -> list[dict[str, Any]]:
    """Cheap listing: parse only the first and last lines of each file; skip
    anything unfinished or malformed (crash artifacts stay invisible)."""
    out: list[dict[str, Any]] = []
    for path in sorted(replays_dir.glob("*.jsonl")) if replays_dir.is_dir() else []:
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) < 2:
            continue
        try:
            header, end = json.loads(lines[0]), json.loads(lines[-1])
        except json.JSONDecodeError:
            continue
        if not (isinstance(header, dict) and isinstance(end, dict) and end.get("end") is True):
            continue
        out.append(
            {
                "name": path.name,
                "agents": header.get("agents"),
                "started_at": header.get("started_at"),
                "final_scores": end.get("final_scores"),
                "winner": end.get("winner"),
                "turns": len(lines) - 2,
            }
        )
    return out


def _replay_file(replays_dir: Path, name: str) -> Path:
    path = replays_dir / name
    if _REPLAY_NAME.fullmatch(name) is None or not path.is_file():
        raise NotFound(f"no such replay: {name}")
    return path


# --- training runs ---------------------------------------------------------


def _is_run_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file()


def _run_dir(runs_dir: Path | None, name: str) -> Path:
    """Resolve a run name to its directory, or raise ``NotFound``.

    The name regex forbids separators and a leading dot, so ``..`` and absolute
    paths can never match -- the join stays inside ``runs_dir``."""
    if runs_dir is None or _RUN_NAME.fullmatch(name) is None:
        raise NotFound(f"no such training run: {name}")
    path = runs_dir / name
    if not _is_run_dir(path):
        raise NotFound(f"no such training run: {name}")
    return path


def _checkpoint_steps(checkpoints_dir: Path) -> list[dict[str, Any]]:
    """Checkpoint timeline: ``[{step, file}]`` sorted ascending by step."""
    out: list[dict[str, Any]] = []
    if checkpoints_dir.is_dir():
        for path in checkpoints_dir.glob(f"{_CKPT_PREFIX}*{_CKPT_SUFFIX}"):
            try:
                step = int(path.stem[len(_CKPT_PREFIX) :])
            except ValueError:
                continue  # a stray file that isn't a real checkpoint
            out.append({"step": step, "file": path.name})
    out.sort(key=lambda c: c["step"])
    return out


def _read_metrics(metrics_path: Path) -> list[dict[str, Any]]:
    """Parse ``metrics.jsonl`` into a list of dicts; skip blank/garbled lines
    (a killed learner can leave a half-written final line)."""
    if not metrics_path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _split_metrics(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Split raw metric rows into the two series the dashboard charts.

    A ``"kind": "gate"`` row is an arena result (orchestrator); any other row is
    an untagged learner step. See training/orchestrate.py's metrics-tagging note."""
    learn: list[dict[str, Any]] = []
    gate: list[dict[str, Any]] = []
    for row in rows:
        (gate if row.get("kind") == "gate" else learn).append(row)
    return {"learn": learn, "gate": gate}


def _run_summary(path: Path) -> dict[str, Any]:
    rows = _read_metrics(TrainingRun(path).metrics_path)
    series = _split_metrics(rows)
    last_step = series["learn"][-1]["step"] if series["learn"] else None
    last_gate = series["gate"][-1] if series["gate"] else None
    checkpoints = _checkpoint_steps(TrainingRun(path).checkpoints_dir)
    # A single "how far has this run got" number for the picker line.
    iterations_or_steps = last_step
    if iterations_or_steps is None and last_gate is not None:
        iterations_or_steps = last_gate.get("iter")
    return {
        "name": path.name,
        "config": TrainingRun.open(path).config(),
        "iterations_or_steps": iterations_or_steps,
        "n_checkpoints": len(checkpoints),
        "last_step": last_step,
        "last_gate": last_gate,
    }


def _list_runs(runs_dir: Path | None) -> list[dict[str, Any]]:
    if runs_dir is None or not runs_dir.is_dir():
        return []
    return [_run_summary(p) for p in sorted(runs_dir.iterdir()) if _is_run_dir(p)]


def _run_detail(path: Path) -> dict[str, Any]:
    run = TrainingRun.open(path)
    return {
        "config": run.config(),
        "checkpoints": _checkpoint_steps(run.checkpoints_dir),
        "metrics": _split_metrics(_read_metrics(run.metrics_path)),
    }
