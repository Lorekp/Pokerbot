"""Human-realistic tournament self-play — the conflict-free Python foundation.

This package implements the **buildable-now** Python pieces of the unified design in
``artifacts/tourney_selfplay_design.md`` (Maia-style behavioral clone + piKL strength dial +
AlphaStar-style league + a tournament-self-play env interface). It deliberately touches NONE of the
owned native crate ``tools/battery_native``; the fast Rust MTT sim is SPEC'd (see
:mod:`pokerbot.training.tourney_selfplay.league_match` — the ``LeagueMatchBackend`` protocol the Rust
sim will implement) but only a *reference* Python backend (over the existing ``mtt_lifecycle``) is
provided here.

The five pillars and their modules:

* **BC clone (the realism anchor τ)** — :mod:`human_clone` (``HumanCloneNet``, skill-conditioned),
  :mod:`clone_loss` (masked next-action CE + belief + aux), :mod:`clone_data` (skill-bin scoring +
  the in-engine trajectory builder smoke), :mod:`human_anchor` (BC trainer + frozen-anchor loader).
* **piKL strength dial** — :mod:`pikl` (the closed-form λ-regularized operator + DiL mixture),
  :mod:`pikl_lift` (``LiftedAgent`` = frozen clone tier + piKL + a Q source).
* **League / exploiters** — :mod:`league` (AlphaStar roles, PFSP matchmaking, add/prune, frozen past,
  diversity gate — wraps ``scapt_manager``).
* **Tournament env interface** — :mod:`league_match` (the ``LeagueMatchBackend`` protocol + a
  reference Python impl over ``mtt_lifecycle`` + ``eval_score``); :mod:`tourney_env` (the
  ``TourneySelfPlayEnv`` boundary the hero PPO trainer consumes, with the Rust sim pluggable).

Nothing here is committed; every module is additive and ``py_compile``-clean.
"""

from __future__ import annotations

__all__ = [
    "human_clone",
    "clone_loss",
    "clone_data",
    "human_anchor",
    "pikl",
    "pikl_lift",
    "league",
    "league_match",
    "tourney_env",
]
