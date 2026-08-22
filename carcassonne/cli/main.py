"""Command-line interface. `carcassonne simulate` plays agent-vs-agent games;
`carcassonne evaluate` runs a seat-swapped head-to-head match and prints a table;
`carcassonne serve` runs the web app."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from carcassonne.agents import make_agent
from carcassonne.game.arena import run_match
from carcassonne.game.match import play_game


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handlers: dict[str, Callable[[argparse.Namespace], int]] = {
        "simulate": _simulate,
        "evaluate": _evaluate,
        "serve": _serve,
    }
    return handlers[args.command](args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="carcassonne", description="Carcassonne engine CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate", help="play agent-vs-agent games, writing replays")
    sim.add_argument("--p0", required=True, help="agent spec for player 0 (e.g. 'random')")
    sim.add_argument("--p1", required=True, help="agent spec for player 1 (e.g. 'random')")
    sim.add_argument("--seed", type=int, required=True, help="seed of game 0; game i uses seed+i")
    sim.add_argument("--out", type=Path, default=None, help="directory for replay files")
    sim.add_argument("--games", type=int, default=1, help="number of games (default 1)")
    sim.add_argument("--no-replay", action="store_true", help="do not write replay files")

    ev = sub.add_parser("evaluate", help="head-to-head match between two agents")
    ev.add_argument("--p0", required=True, help="agent spec for side p0 (e.g. 'greedy')")
    ev.add_argument("--p1", required=True, help="agent spec for side p1 (e.g. 'random')")
    ev.add_argument("--seed", type=int, required=True, help="seed of game 0; game i uses seed+i")
    ev.add_argument("--games", type=int, default=10, help="number of games (default 10)")
    ev.add_argument("--out", type=Path, default=None, help="directory for replay files (optional)")
    ev.add_argument(
        "--no-swap", action="store_true", help="p0 keeps seat 0 in every game (default: alternate)"
    )

    srv = sub.add_parser("serve", help="run the web app (play + replay API)")
    srv.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    srv.add_argument("--port", type=int, default=8000, help="port (default 8000)")
    srv.add_argument(
        "--replays", type=Path, default=Path("replays"), help="replay directory (default replays/)"
    )
    srv.add_argument(
        "--checkpoints",
        type=Path,
        default=None,
        help="checkpoints directory: enables ckpt:<id> opponents (default: none)",
    )
    return parser


def _simulate(args: argparse.Namespace) -> int:
    try:
        agents = (make_agent(args.p0), make_agent(args.p1))
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    replay_dir: Path | None = None if args.no_replay else args.out
    if replay_dir is None and not args.no_replay:
        print("error: --out is required unless --no-replay is given", file=sys.stderr)
        return 2

    wins = [0, 0]
    draws = 0
    for i in range(args.games):
        seed = args.seed + i
        result = play_game(agents, seed, replay_dir=replay_dir)
        line = (
            f"game {i}: seed={seed} scores={result.final_scores}"
            f" winner={result.winner} turns={result.turns}"
        )
        if result.replay_path is not None:
            line += f" replay={result.replay_path}"
        print(line)
        if result.winner is None:
            draws += 1
        else:
            wins[result.winner] += 1
    print(
        f"summary: p0[{agents[0].name}] wins={wins[0]}"
        f" p1[{agents[1].name}] wins={wins[1]} draws={draws}"
    )
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    try:
        a, b = make_agent(args.p0), make_agent(args.p1)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    r = run_match(a, b, args.games, args.seed, replay_dir=args.out, swap_seats=not args.no_swap)
    # Per-game lines and the table are in (p0, p1) order; replay files record seating.
    for i, g in enumerate(r.results):
        line = (
            f"game {i}: seed={args.seed + i} scores={g.final_scores}"
            f" winner={g.winner} turns={g.turns}"
        )
        if g.replay_path is not None:
            line += f" replay={g.replay_path}"
        print(line)
    rate = r.wins[0] / r.games if r.games else 0.0
    print(f"result: p0[{a.name}] vs p1[{b.name}] over {r.games} games")
    print(f"  wins: p0={r.wins[0]} p1={r.wins[1]} draws={r.draws}")
    print(f"  mean scores: p0={r.mean_scores[0]:.1f} p1={r.mean_scores[1]:.1f}")
    print(f"  p0 win rate: {rate:.1%}")
    return 0


def _serve(args: argparse.Namespace) -> int:
    # Imported lazily so simulate/evaluate never pay the fastapi import cost.
    import uvicorn

    from carcassonne.web.app import create_app

    # Single worker on purpose: game sessions are in-process state (see web.sessions).
    uvicorn.run(create_app(args.replays, args.checkpoints), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
