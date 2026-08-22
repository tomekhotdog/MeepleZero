"""Learner tests: NaN-safe loss, overfit sanity, D4 augmentation consistency,
and lossless resume. Everything is kept tiny (channels=8, n_blocks=1, small
batches, few steps) so the whole module runs fast on CPU."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from carcassonne.agents import RandomAgent
from carcassonne.core.engine import legal_moves
from carcassonne.game.match import play_game
from carcassonne.game.replay import load_replay, replay_states
from carcassonne.nn.actions import encode_move
from carcassonne.nn.encode import NUM_PLANES, encode_state
from carcassonne.nn.model import CarcassonneNet
from carcassonne.training.buffer import Example, ReplayBuffer
from carcassonne.training.learner import Learner, LearnerConfig, rotate_state_90
from carcassonne.training.run import TrainingRun

_NET_ARCH = {"num_planes": NUM_PLANES, "channels": 8, "n_blocks": 1}


def _tiny_net() -> CarcassonneNet:
    return CarcassonneNet(num_planes=NUM_PLANES, channels=8, n_blocks=1)


def _real_buffer(tmp_path: Path, seed: int = 7) -> tuple[ReplayBuffer, int]:
    """A buffer backed by one real (net-free) random-agent replay, with policy
    targets that put mass on genuine legal actions of each position."""
    replay_dir = tmp_path / "replays"
    result = play_game((RandomAgent(), RandomAgent()), seed=seed, replay_dir=replay_dir)
    assert result.replay_path is not None
    replay = load_replay(result.replay_path)
    states = list(replay_states(replay))
    n_moves = len(replay.moves)

    buf = ReplayBuffer(tmp_path / "b.sqlite")
    examples: list[Example] = []
    for k in range(n_moves):
        state = states[k]
        chosen = legal_moves(state)[:3]  # up to 3 genuinely-legal actions
        ids = [encode_move(state, m) for m in chosen]
        prob = 1.0 / len(ids)
        policy = {i: prob for i in ids}
        examples.append(Example(move_n=k, policy=policy, value=float((-1, 0, 1)[k % 3])))
    buf.add_game("g", str(result.replay_path), examples, order_key=0)
    return buf, n_moves


def _run(tmp_path: Path) -> TrainingRun:
    return TrainingRun.create(tmp_path / "run", {"net_arch": _NET_ARCH})


def _cfg(**kw: object) -> LearnerConfig:
    base = dict(
        lr=1e-3,
        batch_size=6,
        ckpt_every=1000,
        device=torch.device("cpu"),
        augment=False,
    )
    base.update(kw)
    return LearnerConfig(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------


def test_train_step_runs_and_updates_params(tmp_path: Path) -> None:
    torch.manual_seed(0)
    buf, _ = _real_buffer(tmp_path)
    run = _run(tmp_path)
    learner = Learner(_tiny_net(), buf, run, _cfg(), random.Random(0))

    before = [p.detach().clone() for p in learner.net.parameters()]
    metrics = learner.train_step()

    assert metrics["step"] == 1
    for key in ("loss", "policy_loss", "value_loss", "buffer_games", "buffer_examples"):
        assert key in metrics
        assert math.isfinite(metrics[key])
    after = list(learner.net.parameters())
    assert any(
        not torch.equal(b, a) for b, a in zip(before, after, strict=True)
    ), "params unchanged"
    buf.close()


def test_policy_loss_is_nan_safe(tmp_path: Path) -> None:
    """The overwhelming majority of the ~53.8k actions are illegal (masked to
    -inf); the policy loss must still be finite (no 0 * -inf = NaN)."""
    torch.manual_seed(0)
    buf, _ = _real_buffer(tmp_path)
    run = _run(tmp_path)
    learner = Learner(_tiny_net(), buf, run, _cfg(batch_size=8), random.Random(1))

    for _ in range(3):
        m = learner.train_step()
        assert math.isfinite(m["policy_loss"])
        assert math.isfinite(m["value_loss"])
    for p in learner.net.parameters():
        assert torch.isfinite(p).all()
    buf.close()


def test_overfits_tiny_buffer(tmp_path: Path) -> None:
    """~20 steps on a small fixed buffer: loss over the last few steps is clearly
    below the first few (the net memorises the tiny target set)."""
    torch.manual_seed(0)
    buf, _ = _real_buffer(tmp_path)
    run = _run(tmp_path)
    learner = Learner(
        _tiny_net(), buf, run, _cfg(lr=1e-2, batch_size=8), random.Random(0)
    )

    losses = [learner.train_step()["loss"] for _ in range(20)]
    first = sum(losses[:3]) / 3
    last = sum(losses[-3:]) / 3
    assert last < first, f"loss did not decrease: first={first:.3f} last={last:.3f}"
    buf.close()


def test_rotate_state_matches_encode_equivariance(tmp_path: Path) -> None:
    """The learner's rotate_state_90 produces a state whose encoding is the
    np.rot90 (plus edge-side permutation) of the original -- the D4 property the
    augmentation relies on."""
    buf, n_moves = _real_buffer(tmp_path)
    state = buf.state_for("g", min(10, n_moves - 1))
    e = encode_state(state)
    er = encode_state(rotate_state_90(state))

    from carcassonne.nn.actions import window_origin
    from carcassonne.nn.encode import PLANES

    def crop(planes: NDArray[np.float32], st: object) -> NDArray[np.float32]:
        ox, oy = window_origin(st)  # type: ignore[arg-type]
        xs = [p.x for p in st.board]  # type: ignore[attr-defined]
        ys = [p.y for p in st.board]  # type: ignore[attr-defined]
        return planes[:, min(ys) - oy : max(ys) - oy + 1, min(xs) - ox : max(xs) - ox + 1]

    co = crop(e, state)
    cr = crop(er, rotate_state_90(state))
    # occupied plane is a plain rot90 (no side permutation involved).
    assert np.array_equal(cr[PLANES.index("occupied")], np.rot90(co[PLANES.index("occupied")], 1))
    buf.close()


def test_d4_augmentation_policy_mass_on_legal(tmp_path: Path) -> None:
    """With augment=True the (planes, policy) pair the learner builds is
    self-consistent: every non-zero policy id is a legal action of the same
    (rotated) state the planes encode."""
    torch.manual_seed(0)
    buf, _ = _real_buffer(tmp_path)
    run = _run(tmp_path)
    learner = Learner(
        _tiny_net(), buf, run, _cfg(augment=True), random.Random(3)
    )

    batch = buf.sample(12, random.Random(3))
    seen_rotation = False
    for ex in batch:
        planes, mask, dense = learner._prepare(ex)  # noqa: SLF001 (white-box on purpose)
        nonzero = np.nonzero(dense)[0]
        assert nonzero.size > 0
        # Policy mass sits exactly on legal actions of the encoded state.
        assert mask[nonzero].all(), "policy mass on an illegal action after augmentation"
        assert math.isclose(float(dense.sum()), 1.0, rel_tol=1e-5)
        # Confirm augmentation actually rotated at least one example: its planes
        # differ from the un-rotated encoding of the source state.
        src = buf.state_for(ex.game_id, ex.move_n)
        if not np.array_equal(planes, encode_state(src)):
            seen_rotation = True
    assert seen_rotation, "augmentation never rotated any example (rng/coverage)"
    buf.close()


def test_resume_continues_step_losslessly(tmp_path: Path) -> None:
    torch.manual_seed(0)
    buf, _ = _real_buffer(tmp_path)
    run = _run(tmp_path)

    learner = Learner.from_run(run, buf, _cfg(ckpt_every=2), random.Random(0))
    assert learner.step == 0
    learner.train(4)  # steps 1..4; checkpoints written at 2 and 4
    assert learner.step == 4

    # A brand-new Learner (fresh net + optimizer) restores from the latest
    # checkpoint and continues the step numbering without a gap.
    resumed = Learner.from_run(run, buf, _cfg(ckpt_every=2), random.Random(0))
    assert resumed.step == 4, "did not resume from the latest checkpoint"

    m = resumed.train_step()
    assert resumed.step == 5
    assert m["step"] == 5
    assert math.isfinite(m["loss"])
    buf.close()


def test_metrics_jsonl_one_line_per_step(tmp_path: Path) -> None:
    torch.manual_seed(0)
    buf, _ = _real_buffer(tmp_path)
    run = _run(tmp_path)
    learner = Learner(_tiny_net(), buf, run, _cfg(), random.Random(0))

    learner.train(3)
    lines = run.metrics_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    for i, line in enumerate(lines, start=1):
        rec = json.loads(line)
        assert rec["step"] == i
        assert set(rec) == {
            "step",
            "loss",
            "policy_loss",
            "value_loss",
            "buffer_games",
            "buffer_examples",
        }
    buf.close()
