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
  --channels 48 --n-blocks 4 --sims 100 \
  --selfplay-games 20 --learn-steps 40 \
  --gate-every 5 --gate-games 20 --workers 4 --seed 0
```

(48 channels × 4 blocks is the default — ~193k params, sized for Pi inference;
pass the flags explicitly so the run's config is self-describing. `--workers 4`
is one self-play process per Pi 5 core; `--workers 0`, the default, means "one
per CPU core" too.)

**Which flags count on `--resume`.** `--channels` and `--n-blocks` only matter
when a run is *created*: they are written to `config.json` and used to build the
step-0 net. On `--resume` the nets are loaded from the checkpoints, which carry
their own architecture, so arch flags passed then are ignored. Every other train
flag (`--sims`, `--selfplay-games`, `--learn-steps`, `--gate-every`,
`--gate-games`, `--window-games`, `--workers`, `--seed`, and the built-in
learner settings such as the learning rate) is taken from the command line on
*each* start, never from `config.json`, which is only a record of the creating
command. A restart with different flags therefore continues the run with the
new settings; omitted flags fall back to the CLI defaults, not to the values
the run was created with.

**Start from a fresh run directory.** Checkpoints are stamped with a format
marker, and anything written before iteration 3 (the dynamic-crop net) is
rejected with a clear `ValueError` ("…is from a pre-iteration-3 run… start a
new run directory"). There is no migration. The old `runs/mac-demo` smoke run
is retired: it stays on disk, but it can't be resumed or loaded as `ckpt:`.

**Training is 2-player.** The engine, `simulate`, and the web Play and Replay
views support 2 to 5 players, but the RL stack does not. The encoder,
`MctsAgent`, and self-play all raise on any other player count, and a
checkpoint agent can't be seated in a 3 to 5 player web game. Train and gate on
2-player games only; use the larger player counts for play and analysis against
`greedy`/`random`.

Throughput levers, roughly in order of how much they matter:

- **Compute follows the board (automatic).** The encoder crops to the placed
  tiles' bounding box plus a 1-cell margin, and the fully-convolutional net
  runs on exactly that crop. Early-game evaluations run on tiny grids and cost
  grows with the board, so a whole game costs far less than a fixed full-board
  window would.
- **Workers (`--workers`).** Self-play games run in a pool of spawned worker
  processes, each playing whole games with its own copy of the best net. Self-play
  scales with physical cores, so use 4 on a Pi 5. The learner and arena gating
  still run in the main process.
- **SubtreeReuse (automatic).** After each move, MCTS keeps the searched subtree
  below the move it played. The next search starts from that subtree, or from
  the matching node one ply deeper when an opponent moved in between, so it
  keeps the visits it already made.
- **`--sims` / `--selfplay-games`.** Every MCTS move is `sims` sequential
  network evaluations. Start smaller (`--sims 50`, `--selfplay-games 8`) and
  scale up once the loss is falling and the `wr_greedy` yardstick is climbing on
  the dashboard.

### Pi-only checks (can't be measured on the Mac)

The self-play pool **stays resident** between self-play phases: its workers sit
idle while the learner trains and the arena gates. They are recreated only when
a new net is promoted. Peak memory is therefore 4 workers × (torch + net + MCTS
trees) plus the main process's learner, best net, and replay batch, all at the
same time. During the first few iterations, measure resident memory while the
run is in its learn or gate phase:

```bash
free -m                                              # whole-box view
# per-process RSS (KiB): the main process runs as the `carcassonne` console
# script and the spawned self-play workers show up as `...spawn_main...`
ps -eo pid,rss,args --sort=-rss | grep -E 'carcassonne|spawn_main' | grep -v grep
```

(Not yet verified on the Pi: these commands can't be checked on the Mac. If the
filter misses the workers, `pgrep -af python` lists every Python process.)

If the 4-worker total pushes the Pi into swap, drop to `--workers 3`.

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
  --sims 100 --selfplay-games 20 --learn-steps 40 \
  --gate-every 5 --gate-games 20 --workers 4 --seed 0
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

The unit's `ExecStart` has no `--channels`/`--n-blocks`: on `--resume` the
architecture comes from the checkpoints (see "Which flags count on `--resume`"
above). The other flags are repeated because they are read from the command
line on every restart; keep them in step with the run's intended settings.

Before enabling the unit, create the run once. The unit can't create it itself:
`--resume` refuses a directory with no `config.json`. Run the "Start a run"
command above with `--iterations 0` added:

```bash
uv run carcassonne train --run runs/pi-001 \
  --channels 48 --n-blocks 4 --sims 100 \
  --selfplay-games 20 --learn-steps 40 \
  --gate-every 5 --gate-games 20 --workers 4 --seed 0 --iterations 0
```

This writes `config.json`, `train_state.json`, an empty `buffer.sqlite`, and the
step-0 checkpoint (the initial `best`), plays no games, and exits 0. Once the run
exists, every restart continues it, and `Restart=always` plus the clean-stop
handler let the run recover by itself after reboots.

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

The training success bar is `wr(ckpt:latest vs greedy) > 70%` over 100 games.
An **untrained** net's MCTS is weak — search only amplifies the value head, and
a random value head amplifies noise — so it loses even to simple opponents. The
win rate is a *training* outcome, so let the run climb the `wr_greedy` yardstick
on the dashboard before expecting it.
