"""Command-line interface. `carcassonne simulate` plays agent-vs-agent games."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from carcassonne.agents import make_agent
from carcassonne.game.match import play_game


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handlers: dict[str, Callable[[argparse.Namespace], int]] = {"simulate": _simulate}
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


if __name__ == "__main__":
    sys.exit(main())
