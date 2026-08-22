# Training ProjectCarcasonne on a Raspberry Pi 5

The rules engine, self-play, learner, and arena gating are all pure-Python +
PyTorch and device-agnostic, so a training run started on the Mac continues
unchanged on the Pi. The Pi is the intended box for **multi-day** runs; the Mac
is for proving the loop out at small scale.

## Why the Pi (and not this Mac) for real training

This dev Mac is **Intel (x86_64)**, which pins us to `torch 2.2.x` (the last line
with Intel-macOS wheels) — CPU only, no MPS (MPS is Apple-Silicon only). The
Raspberry Pi 5 is **aarch64 Linux**, which has current PyTorch CPU wheels and 4
cores that run self-play in parallel-friendly Python. See
`docs/lessons.md` for the full pin rationale.

## One-time Pi setup

```bash
# Raspberry Pi OS (64-bit) / Ubuntu 24.04 aarch64, Python 3.12
sudo apt-get update && sudo apt-get install -y python3.12 python3.12-venv git
curl -LsSf https://astral.sh/uv/install.sh | sh    # uv (or use python -m venv + pip)

git clone <your-remote>/Carcasonne && cd Carcasonne
uv venv && uv sync --extra dev --extra ml          # ml extra pulls torch
uv run pytest -q                                   # sanity: full suite green
```

If `uv sync` can't resolve torch on aarch64, install the CPU wheel explicitly
(`uv pip install torch --index-url https://download.pytorch.org/whl/cpu`) and
relax the `ml` pin in `pyproject.toml` — the Pi is free of the Intel-Mac
constraint, so a newer torch (with numpy 2) is fine there; keep numpy and torch
mutually compatible.

## Start a run

```bash
uv run carcassonne train --run runs/pi-001 \
  --channels 64 --n-blocks 5 --sims 100 \
  --selfplay-games 20 --learn-steps 40 \
  --gate-every 5 --gate-games 20 --seed 0
```

Tune for the Pi's throughput: CPU self-play dominates wall-clock (each MCTS move
is `sims` sequential network evaluations). Start smaller (`--sims 50`,
`--selfplay-games 8`) and scale up once you see the loss falling and the
`wr_greedy` yardstick climbing on the dashboard.

## Run it for days, unattended (systemd)

A **TrainingRun is a directory** (`runs/pi-001/`: `config.json`, `checkpoints/`,
`replays/`, `buffer.sqlite`, `metrics.jsonl`, `train_state.json`). The loop is
SIGINT/SIGTERM-safe — it finishes the current iteration, flushes a checkpoint,
and exits 0 — so it survives power cuts and restarts losing at most one
iteration's un-checkpointed steps. On resume it reloads the **last promoted**
net as `best` (gating discipline survives restarts) and continues the step count.

`/etc/systemd/system/carcassonne-train.service`:

```ini
[Unit]
Description=ProjectCarcasonne AlphaZero training
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/Carcasonne
ExecStart=/home/pi/.local/bin/uv run carcassonne train --run runs/pi-001 --resume \
  --channels 64 --n-blocks 5 --sims 100 --selfplay-games 20 --learn-steps 40 \
  --gate-every 5 --gate-games 20
Restart=always
RestartSec=10
KillSignal=SIGTERM
TimeoutStopSec=120

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now carcassonne-train
journalctl -u carcassonne-train -f          # watch progress
```

`--resume` is safe on both first boot (creates the run) and every restart
(continues it). `Restart=always` + the clean-stop handler means the run
self-heals across reboots.

## Watch it from your Mac's browser

Serve the Pi's runs directory over the LAN and open the Training view:

```bash
# on the Pi
uv run carcassonne serve --host 0.0.0.0 --port 8000 \
  --runs-dir runs --replays replays --checkpoints runs/pi-001/checkpoints
```

Then browse to `http://<pi-ip>:8000/training` from the Mac: loss curves, arena
win-rates with the 0.55 gate line, games generated, and the checkpoint timeline,
auto-refreshing every 10s. **Security note:** `--host 0.0.0.0` exposes the app on
your LAN; it has no auth and loads checkpoints via `torch.load` (pickle). Only do
this on a trusted home network. Checkpoint ids are sanitised against path
traversal, but treat the server as localhost-trust.

## Play against a trained checkpoint

Once checkpoints exist, play them from the Play view (pick `Checkpoint <step>` as
the opponent) or evaluate headless:

```bash
uv run carcassonne evaluate --p0 ckpt:latest --p1 greedy --games 100 --seed 1
```

The iteration-1 success bar is `wr(ckpt:latest vs greedy) > 70%` over 100 games.
An **untrained** net's MCTS is weak — measured at 3/10 vs *RandomAgent* at 64
sims (search only amplifies the value head — see the T16 finding in the plan);
it has not been measured against greedy but is expected to lose to it too. The
win rate is a *training* outcome, so let the run climb the `wr_greedy` yardstick
on the dashboard before expecting it.
