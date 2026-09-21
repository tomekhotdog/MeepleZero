"""Command-line interface. `carcassonne simulate` plays agent-vs-agent games;
`carcassonne evaluate` runs a seat-swapped head-to-head match and prints a table;
`carcassonne serve` runs the web app; `carcassonne train` runs the AlphaZero loop."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from carcassonne.agents import make_agent
from carcassonne.core import GameConfig
from carcassonne.game.arena import run_match
from carcassonne.game.match import play_game


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handlers: dict[str, Callable[[argparse.Namespace], int]] = {
        "simulate": _simulate,
        "evaluate": _evaluate,
        "serve": _serve,
        "train": _train,
        "bench": _bench,
    }
    return handlers[args.command](args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="carcassonne", description="Carcassonne engine CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate", help="play agent-vs-agent games, writing replays")
    sim.add_argument("--p0", help="agent spec for player 0 (2-player shorthand)")
    sim.add_argument("--p1", help="agent spec for player 1 (2-player shorthand)")
    sim.add_argument(
        "--agents",
        help="comma-separated agent specs for all seats, 2-5 (e.g. 'greedy,random,random')",
    )
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

    tr = sub.add_parser("train", help="run the resumable AlphaZero training loop")
    tr.add_argument("--run", type=Path, required=True, help="training run directory")
    tr.add_argument("--resume", action="store_true", help="continue an existing run")
    tr.add_argument("--iterations", type=int, default=None, help="max iterations this invocation")
    tr.add_argument("--sims", type=int, default=100, help="MCTS sims for self-play + gating")
    tr.add_argument("--channels", type=int, default=64, help="network channels")
    tr.add_argument("--n-blocks", type=int, default=5, help="residual blocks")
    tr.add_argument("--selfplay-games", type=int, default=20, help="self-play games per iteration")
    tr.add_argument("--learn-steps", type=int, default=40, help="gradient steps per iteration")
    tr.add_argument("--gate-every", type=int, default=5, help="iterations between arena gates")
    tr.add_argument("--gate-games", type=int, default=20, help="games per arena gate")
    tr.add_argument("--window-games", type=int, default=2000, help="replay-buffer eviction window")
    tr.add_argument("--seed", type=int, default=0, help="base seed")
    tr.add_argument(
        "--device", choices=["auto", "cpu"], default="auto", help="compute device (default auto)"
    )

    be = sub.add_parser("bench", help="time self-play games (throughput yardstick)")
    be.add_argument("--games", type=int, default=4, help="games to play (default 4)")
    be.add_argument("--sims", type=int, default=32, help="MCTS sims per move (default 32)")
    be.add_argument("--channels", type=int, default=48, help="network channels")
    be.add_argument("--n-blocks", type=int, default=4, help="residual blocks")
    be.add_argument("--seed", type=int, default=0, help="seed of game 0; game i uses seed+i")

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
    srv.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="directory of training runs: enables the /training dashboard (default: none)",
    )
    return parser


def _simulate(args: argparse.Namespace) -> int:
    if args.agents and (args.p0 or args.p1):
        print("error: use either --agents or --p0/--p1, not both", file=sys.stderr)
        return 2
    if args.agents:
        specs = [s.strip() for s in args.agents.split(",")]
    elif args.p0 and args.p1:
        specs = [args.p0, args.p1]
    else:
        print("error: pass --agents, or both --p0 and --p1", file=sys.stderr)
        return 2
    if not 2 <= len(specs) <= 5:
        print(f"error: need 2-5 agents, got {len(specs)}", file=sys.stderr)
        return 2
    try:
        agents = tuple(make_agent(s) for s in specs)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    config = GameConfig(num_players=len(agents))
    replay_dir: Path | None = None if args.no_replay else args.out
    if replay_dir is None and not args.no_replay:
        print("error: --out is required unless --no-replay is given", file=sys.stderr)
        return 2

    wins = [0] * len(agents)
    draws = 0
    for i in range(args.games):
        seed = args.seed + i
        result = play_game(agents, seed, replay_dir=replay_dir, config=config)
        line = (
            f"game {i}: seed={seed} scores={result.final_scores}"
            f" winners={list(result.winners)} turns={result.turns}"
        )
        if result.replay_path is not None:
            line += f" replay={result.replay_path}"
        print(line)
        if len(result.winners) > 1:
            draws += 1
        else:
            wins[result.winners[0]] += 1
    print(
        "summary: "
        + " ".join(f"p{i}[{a.name}] wins={wins[i]}" for i, a in enumerate(agents))
        + f" shared={draws}"
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
            f" winners={list(g.winners)} turns={g.turns}"
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


def _train(args: argparse.Namespace) -> int:
    import torch

    from carcassonne.nn.checkpoint import pick_device
    from carcassonne.training.orchestrate import TrainConfig, run_config_dict, train
    from carcassonne.training.run import TrainingRun

    cfg = TrainConfig(
        selfplay_games_per_iter=args.selfplay_games,
        learn_steps_per_iter=args.learn_steps,
        gate_every=args.gate_every,
        gate_games=args.gate_games,
        window_games=args.window_games,
        mcts_sims=args.sims,
        channels=args.channels,
        n_blocks=args.n_blocks,
        seed=args.seed,
    )

    run_dir: Path = args.run
    config_exists = (run_dir / "config.json").exists()
    if config_exists and not args.resume:
        print(
            f"error: run already exists at {run_dir}; pass --resume to continue it",
            file=sys.stderr,
        )
        return 2
    if args.resume and not config_exists:
        print(f"error: no run to resume at {run_dir} (config.json missing)", file=sys.stderr)
        return 2
    run = TrainingRun.open(run_dir) if config_exists else TrainingRun.create(
        run_dir, run_config_dict(cfg)
    )

    device = torch.device("cpu") if args.device == "cpu" else pick_device()
    print(f"training run={run_dir} device={device} (Ctrl-C to stop cleanly)")
    try:
        train(run, cfg, device=device, max_iterations=args.iterations)
    except KeyboardInterrupt:
        # The signal handler set the stop flag; train() flushed a checkpoint and
        # returned. This only fires if a second Ctrl-C races the handler removal.
        print("\ninterrupted; latest checkpoint is intact", file=sys.stderr)
    return 0


def _bench(args: argparse.Namespace) -> int:
    """Time sequential self-play with a fresh (untrained) net on CPU.

    Untrained-net throughput is representative: per-move cost is sims x one
    net evaluation regardless of weights. Uses a throwaway run dir for replays.
    """
    import tempfile
    import time

    import torch

    from carcassonne.agents.mcts import MctsConfig
    from carcassonne.nn.model import CarcassonneNet, param_count
    from carcassonne.training.run import TrainingRun
    from carcassonne.training.selfplay import play_selfplay_game

    if args.games < 1:
        print("error: --games must be >= 1", file=sys.stderr)
        return 2
    net = CarcassonneNet(channels=args.channels, n_blocks=args.n_blocks)
    net.eval()
    device = torch.device("cpu")
    with tempfile.TemporaryDirectory() as td:
        run = TrainingRun.create(Path(td) / "bench", {"bench": True})
        t0 = time.perf_counter()
        for i in range(args.games):
            # returned (game_id, examples) ignored: bench only times generation.
            play_selfplay_game(
                net=net,
                device=device,
                seed=args.seed + i,
                run=run,
                mcts_config=MctsConfig(sims=args.sims),
                game_id=f"bench{i:04d}",
                started_at="bench",
            )
        dt = time.perf_counter() - t0
    print(f"params={param_count(net)}")
    print(
        f"games={args.games} sims={args.sims} channels={args.channels} "
        f"n_blocks={args.n_blocks} total={dt:.1f}s s/game={dt / args.games:.1f} "
        f"games/hour={args.games / dt * 3600:.1f}"
    )
    return 0


def _serve(args: argparse.Namespace) -> int:
    # Imported lazily so simulate/evaluate never pay the fastapi import cost.
    import uvicorn

    from carcassonne.web.app import create_app

    # Single worker on purpose: game sessions are in-process state (see web.sessions).
    uvicorn.run(
        create_app(args.replays, args.checkpoints, args.runs_dir),
        host=args.host,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
