"""Learner tests: NaN-safe loss, overfit sanity, padded dynamic-crop batches,
D4 augmentation consistency, and lossless resume. Everything is kept tiny
(channels=8, n_blocks=1, small batches, few steps) so the whole module runs
fast on CPU."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from carcassonne.agents import RandomAgent
from carcassonne.core.engine import legal_moves
from carcassonne.game.match import play_game
from carcassonne.game.replay import load_replay, replay_states
from carcassonne.nn.actions import Crop, flat_index, move_key, rotate_key
from carcassonne.nn.encode import NUM_PLANES, PLANES, encode_state
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
        prob = 1.0 / len(chosen)
        policy = {move_key(m): prob for m in chosen}
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
    """Almost every flat action index is illegal (masked to -inf) — even more so
    in a padded batch, where whole padding cells are masked out; the policy loss
    must still be finite (no 0 * -inf = NaN)."""
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


def test_build_batch_pads_to_batch_max(tmp_path: Path) -> None:
    """Mixed crop sizes -> one padded tensor; masks/targets in padded geometry."""
    torch.manual_seed(0)
    buf, _ = _real_buffer(tmp_path)
    run = _run(tmp_path)
    learner = Learner(_tiny_net(), buf, run, _cfg(batch_size=8), random.Random(0))

    batch = buf.sample(8, random.Random(0))
    x, mask, cell_mask, policy_target, value_target = learner._build_batch(batch)  # noqa: SLF001
    n, p, ph, pw = x.shape
    assert n == 8 and p == NUM_PLANES
    assert mask.shape == (n, ph * pw * 56)
    assert cell_mask.shape == (n, ph, pw)
    assert policy_target.shape == (n, ph * pw * 56)
    assert value_target.shape == (n,)
    # The batch genuinely mixes crop sizes (early vs late positions of one game).
    areas = cell_mask.sum(dim=(1, 2))
    assert len(set(areas.tolist())) > 1, "batch has uniform crop sizes; fixture too weak"
    # Every target index must be a legal (masked-in) action.
    assert bool(mask[policy_target > 0].all())
    # Planes are exactly zero on padding cells.
    assert bool((x * (1.0 - cell_mask.unsqueeze(1))).abs().sum() == 0)
    buf.close()


def test_rotate_state_matches_encode_equivariance(tmp_path: Path) -> None:
    """The learner's rotate_state_90 produces a state whose encoding is the
    np.rot90 of the original on spatial planes -- the D4 property the
    augmentation relies on. (The crop re-anchors, so no manual cropping.)"""
    buf, n_moves = _real_buffer(tmp_path)
    state = buf.state_for("g", min(10, n_moves - 1))
    e = encode_state(state)
    er = encode_state(rotate_state_90(state))

    occ = PLANES.index("occupied")
    assert np.array_equal(er[occ], np.rot90(e[occ], 1))
    buf.close()


def test_augmented_target_consistent_with_engine(tmp_path: Path) -> None:
    """rotate_key must agree with the engine: for every k in {1, 2, 3} it is a
    bijection from the legal Keys of a state onto the legal Keys of the rotated
    state -- over ALL legal moves, including meeple / abbot placements and
    RetrieveAbbot, whose tile-local sub-action must survive rotation unchanged.
    Set equality (not membership) catches a wrong rotation direction that
    happens to land on some other legal key of a symmetric tile. flat_index on
    the rotated crop must never raise."""
    buf, n_moves = _real_buffer(tmp_path)
    # Seed-7 positions chosen for sub-action coverage: move 2 offers abbot
    # placements, move 9 has an abbot on the board (RetrieveAbbot legal).
    subs_seen: set[int] = set()
    for move_n in (2, 9):
        assert move_n < n_moves
        state = buf.state_for("g", move_n)
        keys = {move_key(m) for m in legal_moves(state)}
        subs_seen |= {a for (_x, _y, _r, a) in keys}
        rstate = state
        for k in (1, 2, 3):
            rstate = rotate_state_90(rstate)
            rotated = {rotate_key(key, k) for key in keys}
            assert rotated == {move_key(m) for m in legal_moves(rstate)}, f"k={k}"
            crop = Crop.of(rstate)
            for rkey in rotated:
                flat_index(crop, rkey)  # must not raise
    # The chosen states must actually exercise non-trivial sub-actions.
    assert any(1 <= a <= 6 for a in subs_seen), "no meeple placements covered"
    assert any(7 <= a <= 12 for a in subs_seen), "no abbot placements covered"
    assert 13 in subs_seen, "no RetrieveAbbot covered"
    buf.close()


def test_d4_augmentation_policy_mass_on_legal(tmp_path: Path) -> None:
    """With augment=True the (planes, policy) pair the learner builds is
    self-consistent: every non-zero policy index is a legal action of the same
    (rotated) state the planes encode."""
    torch.manual_seed(0)
    buf, _ = _real_buffer(tmp_path)
    run = _run(tmp_path)
    learner = Learner(_tiny_net(), buf, run, _cfg(augment=True), random.Random(3))

    batch = buf.sample(12, random.Random(3))
    x, mask, _cell, policy_target, _ = learner._build_batch(batch)  # noqa: SLF001
    _, _, ph, pw = x.shape
    # Policy mass sits exactly on legal actions of the rotated states the planes encode.
    assert bool(mask[policy_target > 0].all()), "policy mass on an illegal action"
    sums = policy_target.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), rtol=1e-5)
    # Confirm augmentation actually rotated at least one example: its planes
    # differ from the un-rotated encoding of its source state.
    seen_rotation = False
    for i, ex in enumerate(batch):
        src = buf.state_for(ex.game_id, ex.move_n)
        crop = Crop.of(src)
        if crop.h > ph or crop.w > pw:
            # The unrotated crop cannot even fit the padded batch: this example
            # was necessarily rotated (its own rotated crop fits by construction).
            seen_rotation = True
            continue
        unrotated = torch.from_numpy(encode_state(src, pad_hw=(ph, pw)))
        if not torch.equal(x[i], unrotated):
            seen_rotation = True
    assert seen_rotation, "augmentation never rotated any example (rng/coverage)"
    buf.close()


def test_train_step_finite_loss_with_augmentation(tmp_path: Path) -> None:
    """A full augmented train step over a padded, rotated batch stays finite."""
    torch.manual_seed(0)
    buf, _ = _real_buffer(tmp_path)
    run = _run(tmp_path)
    learner = Learner(
        _tiny_net(), buf, run, _cfg(augment=True, batch_size=8), random.Random(5)
    )
    metrics = learner.train_step()
    assert math.isfinite(metrics["loss"])
    assert math.isfinite(metrics["policy_loss"])
    assert math.isfinite(metrics["value_loss"])
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


def test_resume_from_pre_iteration3_run_fails_clearly(tmp_path: Path) -> None:
    """Resuming an old run surfaces the checkpoint's clean-break error, before
    the learner ever samples the (equally old) buffer."""
    buf, _ = _real_buffer(tmp_path)
    run = _run(tmp_path)
    torch.save(
        {"step": 9, "model_state": {"policy_fc.weight": torch.zeros(1)}, "net_arch": _NET_ARCH},
        run.checkpoints_dir / "step_000009.pt",
    )
    with pytest.raises(ValueError, match="pre-iteration-3"):
        Learner.from_run(run, buf, _cfg(), random.Random(0))
    buf.close()
