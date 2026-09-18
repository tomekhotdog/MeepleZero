# State of the art: training multiplayer game agents via RL

Status: research note (web survey, 2026-09-18). Companion to
`2026-08-23-multiplayer-3-5-research.md`, which analyses *this codebase's*
path to 3-5 players; this note surveys the *external* prior art and what it
implies for the deferred Tier-3 (multiplayer RL) work.

## TL;DR

Multiplayer (N>2) RL for board games is a thin, fragmented literature next to
the two-player self-play mountain. Three findings shape our roadmap:

1. The design sketched in our research doc — vector value head + max^n MCTS
   backup — **is** the published textbook approach ("Multiplayer AlphaZero",
   Petosa & Balch 2019), but it has only ever been validated on toy domains.
2. Every *landmark* multiplayer result (poker, mahjong, Diplomacy, Dou Dizhu)
   **abandoned the AlphaZero recipe rather than extending it**, each inventing
   bespoke machinery. The reason is theoretical, not incidental.
3. Carcassonne-specific prior art is search-without-learning and almost
   entirely two-player. Our current 2-player AlphaZero loop is already at or
   past the published frontier *for this game*.

Net: keeping the RL stack strictly 2-player (with loud `num_players != 2`
guards) matches how the field itself has behaved. Tier 3 is a research
project, not an engineering task.

## Why two-player is special (the theory)

Self-play converging toward optimal play is a *guarantee* only in two-player
zero-sum games: there, a Nash equilibrium strategy is unexploitable and
minimax value is well defined — one scalar carries both players' outcomes,
which is exactly the convention our value head, negamax backup, ±1 targets,
and scalar MSE keep in lockstep.

For N>2 all of that evaporates:

- **No convergence guarantee.** Nash equilibria still exist but self-play need
  not find one, and playing "an" equilibrium is no longer unexploitable when
  opponents don't play their halves of the same equilibrium.
- **Kingmaker dynamics.** A player who cannot win can still decide who does;
  value backup must represent "A's loss may be B's gain, not mine".
- **Implicit collusion.** Nothing in the objective prevents two self-play
  seats from learning mutually beneficial (against the third) behaviour.
- **Sparser credit assignment.** Fixed-length games divided among more seats
  give each seat fewer decisions per game (in Carcassonne: ~35 turns each at
  2 players, ~14 at 5), so per-game learning signal shrinks as N grows.

Classical (pre-deep-RL) game search already reflects this: max^n (Luckhardt &
Irani 1986) generalises minimax with per-player payoff vectors but has far
weaker pruning and guarantees; "paranoid" search and best-reply search are
coping strategies, not solutions. MCTS versions of these were studied
extensively by Sturtevant and the Maastricht group on Chinese Checkers,
Hearts, and similar domains.

## Strand 1 — direct AlphaZero extensions (our Tier-3 blueprint)

- **Multiplayer AlphaZero — Petosa & Balch 2019**
  (https://arxiv.org/abs/1910.13012). The canonical reference for the exact
  design our research doc arrived at independently: terminal *score vectors*
  (e.g. 3-player win = [1, -1, -1]), a value head predicting an N-vector of
  expected utilities, MCTS backing up the whole vector unchanged, and each
  node selecting via PUCT on *its own mover's* component (previously described
  in the literature as MCTS-max^n). Validated on 3-player Tic-Tac-Toe and
  Connect-4 only.
- **Deep RL for 5×5 multiplayer Go, 2024** (https://arxiv.org/abs/2405.14265)
  — same family of ideas on a small multiplayer Go variant.
- Hobby/OSS attempts exist (e.g. an AlphaZero-style 4-player Catan,
  https://github.com/Jaime888888/jaime-catan) but no result of AlphaZero's
  stature has been produced with this recipe on a full-scale game.

Implication for us: the blueprint is published and simple to state; what is
unproven is whether it *trains well* on a real game at hobby compute. That is
the open question Tier 3 would actually be answering.

## Strand 2 — landmark multiplayer systems (which all left the recipe)

| System | Game (seats) | Core method | Note |
|---|---|---|---|
| Pluribus (Brown & Sandholm, Science 2019) | No-limit poker (6) | Counterfactual regret minimisation + depth-limited search | Explicitly argues equilibrium play is no longer the right target for N>2 |
| Suphx (Li et al. 2020, arXiv:2003.13590) | Riichi mahjong (4) | Policy gradients + "oracle" distillation + global reward shaping | Not search-based self-play in the AZ sense |
| DouZero (Zha et al., ICML 2021, arXiv:2106.06135) | Dou Dizhu (3) | Deep Monte-Carlo, no MCTS, no policy head | Beat prior SOTA with deliberately simple machinery |
| Cicero (Meta FAIR, Science 2022) | Diplomacy (7) | Planning + language model for negotiation | Multiplayer *and* natural-language cooperation |
| Hanabi challenge (Bard et al. 2019, arXiv:1902.00506) | Hanabi (2-5, cooperative) | Benchmark; theory-of-mind focus | The cooperative face of the same N-player difficulty |

The pattern to internalise: each team that crossed the two-player line found
the AlphaZero recipe did not survive the crossing intact, and the winning
systems were bespoke per game. There is no "AlphaZero of multiplayer" — no
single recipe that transfers across N-player games the way AZ transfers
across two-player perfect-information games.

## Strand 3 — frameworks, benchmarks, and the Catan lineage

- **OpenSpiel** (DeepMind, arXiv:1908.09453) — N-player games plus reference
  implementations of max^n-family algorithms; the natural place to sanity-
  check a Tier-3 backup implementation against known-good code.
- **PyTAG** (arXiv:2307.09905) — positions modern multiplayer tabletop games
  as an underexplored RL challenge (partial observability, large/parametric
  action spaces, multi-seat); useful framing and baselines.
- **PettingZoo / RLCard** — multi-agent env APIs; RLCard covers multiplayer
  card games.
- **Settlers of Catan** has the richest euro-game lineage: hierarchical RL
  self-play (Pfeiffer), the decade-old rule-based JSettlers still serving as
  the de-facto champion baseline, and recent deep-RL efforts
  (https://settlers-rl.github.io/, https://akrishna77.github.io/QSettlers/)
  with honest, mixed results. No learned Catan agent dominates the way AZ
  dominates chess — a sober calibration point for expectations at hobby scale.

## Carcassonne-specific prior art

All search-without-learning, essentially all two-player:

- Heyden 2009 (Maastricht): MCTS and minimax variants for 2-player
  Carcassonne — the standard early reference.
- Müller 2014 (Maastricht MSc): MCTS in the domain of Carcassonne
  (https://project.dke.maastrichtuniversity.nl/games/listMsc.htm).
- Jappert 2022 (Basel bachelor thesis): optimising MCTS variants for
  2-player Carcassonne
  (https://ai.dmi.unibas.ch/papers/theses/jappert-bachelor-22.pdf).

No published AlphaZero-style *learned* Carcassonne agent was found, and no
multiplayer Carcassonne agent of any kind. Completing our 2-player training
outcome (>70% vs GreedyAgent) would be at or past the published frontier for
this game; a trained multiplayer agent would be genuinely novel.

## Implications for this project

1. **The 2-player-only RL boundary is well placed.** The field's own history
   says extending self-play past two players is research, not widening.
   Keep the loud `num_players != 2` guards non-negotiable.
2. **If/when Tier 3 starts, the path is**: Petosa & Balch as the blueprint;
   OpenSpiel as the reference implementation to test our max^n backup
   against; rank-linear terminal payoff vectors (win-only is too sparse at
   4-5 seats); expect to redesign gating (head-to-head win rate has no
   N-player analogue — promote on expected rank vs the 1/N baseline).
3. **Set expectations from Catan, not from Go.** The closest published
   analogues at comparable ambition report mixed results against decent
   heuristics. The GreedyAgent yardstick generalised to "my delta minus
   best-of-rest" remains the honest multiplayer ruler.
4. **The interesting open questions are cheap to study here** once Tier 3
   exists: does self-play collude? do kingmaker positions confuse the vector
   value head? does the 14-turns-per-seat signal at 5 players train at all on
   Pi-scale compute? The RL microscope (per-seat value lines, Prior vs
   Visits) is exactly the instrument for observing these.

## Sources

- Petosa & Balch, "Multiplayer AlphaZero" — https://arxiv.org/abs/1910.13012
- "Deep RL for 5×5 Multiplayer Go" — https://arxiv.org/abs/2405.14265
- Brown & Sandholm, "Superhuman AI for multiplayer poker" (Pluribus), Science 2019
- Li et al., "Suphx: Mastering Mahjong" — https://arxiv.org/abs/2003.13590
- Zha et al., "DouZero" — https://arxiv.org/abs/2106.06135
- Meta FAIR, "Cicero: human-level Diplomacy", Science 2022
- Bard et al., "The Hanabi Challenge" — https://arxiv.org/abs/1902.00506
- OpenSpiel — https://arxiv.org/abs/1908.09453
- PyTAG — https://arxiv.org/abs/2307.09905
- Luckhardt & Irani, "An algorithmic solution of N-person games", AAAI 1986 (max^n)
- Maastricht Games & AI theses — https://project.dke.maastrichtuniversity.nl/games/listMsc.htm
- Jappert, "MCTS for Carcassonne" — https://ai.dmi.unibas.ch/papers/theses/jappert-bachelor-22.pdf
- settlers-rl — https://settlers-rl.github.io/ · QSettlers — https://akrishna77.github.io/QSettlers/
