"""PUCT Monte Carlo Tree Search guided by the network -- the AlphaZero heart.

This is the project's centerpiece algorithm. Given a state, we grow a search tree
whose leaves are evaluated by :func:`carcassonne.nn.model.evaluate_state` (policy
priors + a value), *never* by random rollouts, and choose a move from the visit
distribution the search produces.

Perspective / sign conventions (the subtle correctness points -- unit-tested):

* ``evaluate_state`` returns a value in ``[-1, 1]`` from the **current player's**
  perspective. ``apply`` flips ``current_player`` every turn, so a child node's
  perspective is always the *opposite* of its parent's.
* A leaf's value (net estimate, or a grounded terminal outcome) is from that
  leaf node's perspective. During BACKUP we **negate the value at each ply** as we
  walk up the path, so every node records values in *its own* perspective. Hence
  ``Q[i]`` at a node, and the root's overall Q, are read from that node's mover.
* Terminal value: ``winner = argmax(final_scores)``; ``+1`` if the terminal node's
  ``current_player`` is the winner, ``-1`` if the loser, ``0`` on a draw. No one
  moves at a terminal state, but the backup negation keeps this consistent with
  the parent that reached it.

Determinism: all randomness (Dirichlet root noise, temperature sampling) flows
through a single numpy ``Generator`` seeded from ``ctx.rng``, so a seeded game is
fully reproducible.

Performance note: ``evaluate_state`` is called one state at a time. Batching leaf
evaluations is a worthwhile later optimisation; it is deliberately not built here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.typing import NDArray

from carcassonne.core import (
    GameState,
    Move,
    RulesError,
    apply,
    final_scores,
    is_terminal,
    legal_moves,
)
from carcassonne.game.agent import TurnContext
from carcassonne.game.serde import Annot
from carcassonne.nn.actions import encode_move
from carcassonne.nn.model import CarcassonneNet, evaluate_state

_TOP_K = 8


@dataclass(frozen=True, slots=True)
class MctsConfig:
    """Search hyper-parameters (frozen so an agent's config can't drift mid-run)."""

    sims: int = 200
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    noise_frac: float = 0.25
    temp_turns: int = 12  # sample with temperature while turn < this, then argmax
    temp: float = 1.0


_DEFAULT_CONFIG = MctsConfig()  # shared immutable default (frozen dataclass)


@dataclass(slots=True)
class _Node:
    """One search-tree node: a state plus per-legal-move PUCT statistics.

    Arrays are parallel to ``moves``. ``leaf_value`` is this node's own-perspective
    value estimate (net value, or grounded terminal outcome), used the first time
    the node is reached. ``children`` maps a move index to its expanded child.
    """

    state: GameState
    terminal: bool
    leaf_value: float
    moves: tuple[Move, ...]
    prior: NDArray[np.float64]  # P: net priors over legal moves (sums to 1)
    n: NDArray[np.int64]  # N: visit counts
    w: NDArray[np.float64]  # W: total backed-up value
    q: NDArray[np.float64]  # Q: mean value (W / N, 0 when unvisited)
    children: dict[int, _Node] = field(default_factory=dict)


def _terminal_value(state: GameState) -> float:
    """Game outcome from the perspective of the player to move at this terminal
    state: +1 win, -1 loss, 0 draw. See the module docstring on why this is
    correct despite no one actually moving at a terminal node."""
    s0, s1 = final_scores(state)
    if s0 == s1:
        return 0.0
    winner = 0 if s0 > s1 else 1
    return 1.0 if winner == state.current_player else -1.0


class MctsAgent:
    """AlphaZero-style PUCT agent: search guided by ``net``, move by visit counts.

    ``self_play`` toggles the two training-time behaviours: Dirichlet noise on the
    root priors (exploration) and temperature sampling of the move for early turns.
    With ``self_play=False`` (evaluation / UI) the search is noise-free and the move
    is always the visit-argmax.
    """

    def __init__(
        self,
        net: CarcassonneNet,
        device: torch.device,
        config: MctsConfig = _DEFAULT_CONFIG,
        self_play: bool = False,
        name: str = "mcts",
    ) -> None:
        self._net = net
        self._device = device
        self._config = config
        self._self_play = self_play
        self.name = name

    # -- Agent protocol -----------------------------------------------------

    def choose(self, state: GameState, ctx: TurnContext) -> tuple[Move, Annot | None]:
        start = time.perf_counter()
        rng = np.random.default_rng(ctx.rng.getrandbits(64))
        root = self._search(state, rng, add_noise=self._self_play)

        move_idx = self._pick_move(root, state.turn, rng)
        move = root.moves[move_idx]
        if not ctx.annotate:
            return move, None

        think_ms = int((time.perf_counter() - start) * 1000)
        return move, self._annotate(root, think_ms)

    def choose_with_policy(
        self, state: GameState, ctx: TurnContext
    ) -> tuple[Move, Annot, dict[Move, float]]:
        """Single search -> (chosen move, annotation, visit distribution).

        The self-play primitive: one search yields the move to play, its annotation,
        AND the normalised visit distribution used as the policy target -- so a
        self-play game pays for exactly one search per move (not one for the move
        and another for the target). Honours ``self_play`` (root Dirichlet noise +
        temperature sampling) identically to :meth:`choose`; always annotates."""
        start = time.perf_counter()
        rng = np.random.default_rng(ctx.rng.getrandbits(64))
        root = self._search(state, rng, add_noise=self._self_play)
        move = root.moves[self._pick_move(root, state.turn, rng)]
        think_ms = int((time.perf_counter() - start) * 1000)
        return move, self._annotate(root, think_ms), self._visit_dist(root)

    # -- training / UI helper ----------------------------------------------

    def visit_policy(self, state: GameState) -> dict[Move, float]:
        """Run a fresh search and return the normalised visit distribution
        ``{move: N[i] / sum(N)}`` over legal moves -- the self-play training target
        and the ``/hint`` source. Noise-free and deterministic (search uses no
        randomness of its own), so it needs no rng."""
        return self._visit_dist(self._search(state, rng=None, add_noise=False))

    @staticmethod
    def _visit_dist(root: _Node) -> dict[Move, float]:
        total = int(root.n.sum())
        return {move: int(count) / total for move, count in zip(root.moves, root.n, strict=True)}

    # -- internals ----------------------------------------------------------

    def _search(
        self, root_state: GameState, rng: np.random.Generator | None, add_noise: bool
    ) -> _Node:
        root = self._make_node(root_state)
        if root.terminal:
            raise ValueError("cannot search a terminal state: no legal moves")
        if add_noise:
            assert rng is not None  # add_noise only ever set with a real rng
            self._add_dirichlet_noise(root, rng)
        for _ in range(self._config.sims):
            self._simulate(root)
        return root

    def _make_node(self, state: GameState) -> _Node:
        """Create (and evaluate) a node. Terminal states are grounded; others get
        net priors renormalised over their legal moves."""
        if is_terminal(state):
            empty_f = np.zeros(0, dtype=np.float64)
            return _Node(
                state=state,
                terminal=True,
                leaf_value=_terminal_value(state),
                moves=(),
                prior=empty_f,
                n=np.zeros(0, dtype=np.int64),
                w=empty_f.copy(),
                q=empty_f.copy(),
            )
        probs, value = evaluate_state(self._net, state, self._device)
        # Keep only moves the encoder can represent. On boards wider than the fixed
        # window a few legal placements fall outside it (the net has no slot for
        # them); we prune them from the search, exactly as legal_mask masks them.
        moves_list: list[Move] = []
        prior_list: list[float] = []
        for m in legal_moves(state):
            try:
                idx = encode_move(state, m)
            except RulesError:
                continue
            moves_list.append(m)
            prior_list.append(float(probs[idx]))
        if not moves_list:  # pathological: no legal move fits the window
            raise RulesError("no legal move fits the encoder window")
        moves = tuple(moves_list)
        prior = np.array(prior_list, dtype=np.float64)
        total = prior.sum()
        # probs already sum to 1 over legal moves; renormalise for numerical safety,
        # falling back to uniform only in the degenerate all-zero case.
        prior = prior / total if total > 0 else np.full(len(moves), 1.0 / len(moves))
        n = len(moves)
        return _Node(
            state=state,
            terminal=False,
            leaf_value=value,
            moves=moves,
            prior=prior,
            n=np.zeros(n, dtype=np.int64),
            w=np.zeros(n, dtype=np.float64),
            q=np.zeros(n, dtype=np.float64),
        )

    def _add_dirichlet_noise(self, root: _Node, rng: np.random.Generator) -> None:
        alpha = self._config.dirichlet_alpha
        frac = self._config.noise_frac
        noise = rng.dirichlet(np.full(len(root.moves), alpha))
        root.prior = (1.0 - frac) * root.prior + frac * noise

    def _select(self, node: _Node) -> int:
        """PUCT: argmax over legal moves of ``Q + c_puct * P * sqrt(sum N)/(1+N)``."""
        sqrt_total = float(np.sqrt(node.n.sum()))
        u = node.q + self._config.c_puct * node.prior * sqrt_total / (1 + node.n)
        return int(np.argmax(u))

    def _simulate(self, root: _Node) -> None:
        """One selection-expansion-backup pass from the root."""
        path: list[tuple[_Node, int]] = []
        node = root
        while True:
            if node.terminal:
                value = node.leaf_value  # already this node's perspective
                break
            i = self._select(node)
            path.append((node, i))
            child = node.children.get(i)
            if child is None:  # EXPAND
                child = self._make_node(apply(node.state, node.moves[i]))
                node.children[i] = child
                value = child.leaf_value  # child's perspective
                break
            node = child

        # BACKUP: negate at each ply so every node sees the value from its own
        # perspective. ``value`` starts from the deepest (leaf/terminal) node.
        for parent, i in reversed(path):
            value = -value
            parent.n[i] += 1
            parent.w[i] += value
            parent.q[i] = parent.w[i] / parent.n[i]

    def _pick_move(self, root: _Node, turn: int, rng: np.random.Generator) -> int:
        """Sample proportional to ``N^(1/temp)`` during early self-play turns;
        otherwise pick the visit-argmax (deterministic)."""
        if self._self_play and turn < self._config.temp_turns and self._config.temp > 0:
            counts = root.n.astype(np.float64)
            weights = counts ** (1.0 / self._config.temp)
            probs = weights / weights.sum()
            return int(rng.choice(len(root.moves), p=probs))
        return int(np.argmax(root.n))

    def _annotate(self, root: _Node, think_ms: int) -> Annot:
        total = int(root.n.sum())
        # Root value = visit-weighted mean Q = sum(N*Q)/sum(N): the search's overall
        # assessment from the current player's perspective. Falls back to the raw net
        # value only if nothing was visited (sims == 0).
        value = float((root.n * root.q).sum() / total) if total > 0 else root.leaf_value
        order = sorted(
            range(len(root.moves)), key=lambda i: (-int(root.n[i]), i)
        )[:_TOP_K]
        top = tuple(
            (root.moves[i], float(root.prior[i]), int(root.n[i])) for i in order
        )
        return Annot(value=value, sims=self._config.sims, think_ms=think_ms, top=top)
