"""The learner: one gradient step of the AlphaZero training loop.

Each :meth:`Learner.train_step` samples a batch of (policy, value) targets from
the :class:`ReplayBuffer`, re-derives each position's state, encodes it, runs the
net, and takes an Adam step on ``policy_cross_entropy + value_mse``. Metrics are
appended to ``run.metrics_path`` and the net/optimizer are checkpointed every
``ckpt_every`` steps, so a killed run resumes losslessly via :meth:`from_run`
(the Raspberry Pi requirement).

Two correctness cruxes live here:

* **NaN-safe policy loss.** :meth:`CarcassonneNet.forward` masks illegal actions
  to ``-inf`` so their softmax probability is exactly 0. Computing the cross
  entropy naively then does ``target(0) * log_softmax(-inf) = 0 * -inf = NaN``.
  We compute ``log_softmax`` (finite on legal actions, ``-inf`` on illegal),
  multiply by the target, and *replace* the contribution of non-target actions
  with 0 via :func:`torch.where`. The target is non-zero only on legal,
  MCTS-visited actions -- which are never masked -- so this is both safe and
  exactly the cross entropy over the target's support.

* **D4 policy augmentation, correctness by construction.** Rather than deriving
  action-id index math by hand, we *resample the target through the engine*: for
  a randomly chosen number of 90-degree rotations ``k in {0,1,2,3}`` we build the
  genuinely rotated :class:`GameState` (:func:`rotate_state_90`), encode *that*,
  and remap every visited action id via ``decode_move(state) -> rotate the Move
  -> encode_move(rotated_state)``. The value target is rotation-invariant. This
  guarantees the augmented (planes, policy, mask) triple is self-consistent.
  Reflections are intentionally excluded: base-game tiles are chiral, so a
  mirrored board is generally not a legal Carcassonne position (see module note).
"""

from __future__ import annotations

import dataclasses
import json
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from carcassonne.core.engine import GameState
from carcassonne.core.features import FeatureIndex, Meeple
from carcassonne.core.state import Board, PlacedTile
from carcassonne.core.types import Move, Pos, Rotation, RulesError
from carcassonne.nn import checkpoint
from carcassonne.nn.actions import ACTION_SPACE, decode_move, encode_move, legal_mask
from carcassonne.nn.encode import NUM_PLANES, encode_state
from carcassonne.nn.model import CarcassonneNet
from carcassonne.training.buffer import ReplayBuffer, SampledExample
from carcassonne.training.run import TrainingRun

_DEFAULT_CHANNELS = 64
_DEFAULT_N_BLOCKS = 5


@dataclass
class LearnerConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    ckpt_every: int = 100
    device: torch.device | None = None
    augment: bool = True  # D4 (rotation-only) augmentation on/off


class Learner:
    """Owns the net, optimizer, and step counter for one training run."""

    def __init__(
        self,
        net: CarcassonneNet,
        buffer: ReplayBuffer,
        run: TrainingRun,
        config: LearnerConfig,
        rng: random.Random,
    ) -> None:
        self._device = config.device or checkpoint.pick_device()
        self._net = net.to(self._device)
        self._buffer = buffer
        self._run = run
        self._config = config
        self._rng = rng
        self._optimizer = torch.optim.Adam(
            net.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self._step = 0

    @classmethod
    def from_run(
        cls,
        run: TrainingRun,
        buffer: ReplayBuffer,
        config: LearnerConfig,
        rng: random.Random,
    ) -> Learner:
        """Build a learner, restoring net + optimizer + step from the latest
        checkpoint if one exists; otherwise a fresh net from ``run.config()``."""
        device = config.device or checkpoint.pick_device()
        latest = checkpoint.latest(run.checkpoints_dir)
        if latest is not None:
            payload = checkpoint.load(latest, device)
            learner = cls(payload["net"], buffer, run, config, rng)
            opt_state = payload.get("optimizer_state")
            if opt_state is not None:
                learner._optimizer.load_state_dict(opt_state)
            learner._step = int(payload["step"])
            return learner
        return cls(_net_from_config(run.config()), buffer, run, config, rng)

    @property
    def step(self) -> int:
        return self._step

    @property
    def net(self) -> CarcassonneNet:
        return self._net

    def train_step(self) -> dict[str, float]:
        batch = self._buffer.sample(self._config.batch_size, self._rng)
        x, mask, policy_target, value_target = self._build_batch(batch)

        self._net.train()
        policy_logits, value = self._net(x, mask)

        # NaN-safe policy cross entropy over the target's (always-legal) support.
        logp = torch.log_softmax(policy_logits, dim=1)
        term = policy_target * logp  # NaN where target==0 and logp==-inf
        term = torch.where(policy_target > 0, term, torch.zeros_like(term))
        policy_loss = -term.sum(dim=1).mean()

        value_loss = torch.nn.functional.mse_loss(value, value_target)
        loss = policy_loss + value_loss  # L2 handled by Adam weight_decay

        self._optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        self._optimizer.step()
        self._step += 1

        metrics: dict[str, float] = {
            "step": self._step,
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "buffer_games": self._buffer.n_games(),
            "buffer_examples": len(self._buffer),
        }
        self._append_metrics(metrics)
        if self._step % self._config.ckpt_every == 0:
            self.checkpoint()
        return metrics

    def checkpoint(self) -> None:
        """Persist net + optimizer + step at the current step.

        Called automatically every ``ckpt_every`` steps by :meth:`train_step`, and
        on demand by the orchestrator (initial step-0 snapshot, promotion, and the
        clean-stop flush) so a killed run always resumes from a valid checkpoint.
        """
        checkpoint.save(
            self._run.checkpoints_dir,
            self._step,
            self._net,
            self._optimizer,
            self._run.config(),
        )

    def train(self, n_steps: int) -> None:
        for _ in range(n_steps):
            self.train_step()

    # -- batch construction -------------------------------------------------

    def _build_batch(
        self, batch: list[SampledExample]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        planes_list: list[NDArray[np.float32]] = []
        mask_list: list[NDArray[np.bool_]] = []
        policy_list: list[NDArray[np.float32]] = []
        value_list: list[float] = []
        for ex in batch:
            planes, mask, dense = self._prepare(ex)
            planes_list.append(planes)
            mask_list.append(mask)
            policy_list.append(dense)
            value_list.append(ex.value)
        x = torch.from_numpy(np.stack(planes_list)).to(self._device)
        mask_t = torch.from_numpy(np.stack(mask_list)).to(self._device)
        policy_target = torch.from_numpy(np.stack(policy_list)).to(self._device)
        value_target = torch.tensor(value_list, dtype=torch.float32, device=self._device)
        return x, mask_t, policy_target, value_target

    def _prepare(
        self, ex: SampledExample
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_], NDArray[np.float32]]:
        """One example -> (planes, legal mask, dense policy target), D4-augmented."""
        state = self._buffer.state_for(ex.game_id, ex.move_n)
        k = self._rng.randrange(4) if self._config.augment else 0
        rstate, policy = state, ex.policy
        if k:
            rotated = state
            for _ in range(k):
                rotated = rotate_state_90(rotated)
            try:
                remapped = _remap_policy(state, rotated, ex.policy, k)
            except RulesError:
                # A rotated move fell outside the fixed window; skip augmentation
                # for this example rather than dropping target mass.
                pass
            else:
                rstate, policy = rotated, remapped

        planes = encode_state(rstate)
        mask = legal_mask(rstate)
        dense = np.zeros(ACTION_SPACE, dtype=np.float32)
        for action_id, prob in policy.items():
            dense[action_id] = prob
        return planes, mask, dense

    def _append_metrics(self, metrics: dict[str, float]) -> None:
        line = json.dumps(metrics, separators=(",", ":"))
        with self._run.metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


# ---------------------------------------------------------------------------
# D4 (rotation) helpers -- correctness by construction via the rules engine.
#
# ``rotate_state_90`` mirrors the transform verified D4-equivariant against
# ``encode_state`` in tests/nn/test_encode.py::test_d4_rotation_equivariance.


def _rot_pos(p: Pos) -> Pos:
    """World 90-degree clockwise rotation of a cell: (x, y) -> (y, -x)."""
    return Pos(p.y, -p.x)


def _rot_rotation(r: Rotation) -> Rotation:
    return Rotation((int(r) + 1) % 4)


def _rotate_move(move: Move) -> Move:
    """Rotate a whole placement (cell + tile rotation) 90 degrees clockwise.

    The meeple sub-action is unchanged: it names a *tile-local* feature index,
    which is invariant under rotating the board.
    """
    return Move(_rot_pos(move.pos), _rot_rotation(move.rotation), move.action)


def rotate_state_90(state: GameState) -> GameState:
    """A genuine 90-degree-clockwise rotation of the whole game state.

    Rebuilds the feature index from the rotated board (union-find is order
    independent), re-places every meeple on its rotated node, and rotates the
    abbot bookkeeping, so the result is a fully valid ``GameState`` equivalent to
    the original under a board rotation.
    """
    rotated = {
        _rot_pos(p): PlacedTile(pt.type_id, _rot_rotation(pt.rotation))
        for p, pt in state.board.items()
    }
    board: Board = {}
    idx = FeatureIndex.empty()
    for pos in sorted(rotated):
        board[pos] = rotated[pos]
        idx = idx.with_tile(board, pos)
    for d in state.features.data.values():
        for m in d.meeples:
            mpos, feat = m.node
            node = (_rot_pos(mpos), feat)
            idx = idx.with_meeple(node, Meeple(m.player, m.kind, node))

    def rot_node(n: tuple[Pos, int] | None) -> tuple[Pos, int] | None:
        return (_rot_pos(n[0]), n[1]) if n is not None else None

    abbot_at = (rot_node(state.abbot_at[0]), rot_node(state.abbot_at[1]))
    return dataclasses.replace(state, board=board, features=idx, abbot_at=abbot_at)


def _remap_policy(
    state: GameState, rotated: GameState, policy: dict[int, float], k: int
) -> dict[int, float]:
    """Map a sparse policy from ``state`` onto its ``k``-times-rotated ``rotated``.

    Each action id is decoded to a ``Move`` on ``state``, rotated ``k`` times, and
    re-encoded on ``rotated``. Raises :class:`RulesError` (propagated) if any
    rotated move falls outside the encoding window.
    """
    remapped: dict[int, float] = {}
    for action_id, prob in policy.items():
        move = decode_move(state, action_id)
        for _ in range(k):
            move = _rotate_move(move)
        remapped[encode_move(rotated, move)] = prob
    return remapped


def _net_from_config(cfg: dict[str, Any]) -> CarcassonneNet:
    arch: dict[str, Any] = cfg.get("net_arch", {})
    return CarcassonneNet(
        num_planes=int(arch.get("num_planes", NUM_PLANES)),
        channels=int(arch.get("channels", _DEFAULT_CHANNELS)),
        n_blocks=int(arch.get("n_blocks", _DEFAULT_N_BLOCKS)),
    )
