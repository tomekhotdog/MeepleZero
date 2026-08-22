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
from carcassonne.web import views
from carcassonne.web.sessions import NotFound, SessionStore, UnknownOpponent

_REPLAY_NAME = re.compile(r"[A-Za-z0-9._-]+\.jsonl")
_STATIC_DIR = Path(__file__).parent / "static"


class CreateGameBody(BaseModel):
    opponent: str
    human_player: Literal[0, 1] = 0
    seed: int | None = None


class MoveBody(BaseModel):
    idx: int


def create_app(replays_dir: Path, checkpoints_dir: Path | None = None) -> FastAPI:
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
