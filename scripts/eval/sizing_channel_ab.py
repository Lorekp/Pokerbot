#!/usr/bin/env python3
"""Discriminating SIZING-CHANNEL A/B gate (Workstream B acceptance #7).

WHY a NEW gate (vs runs/bet_menu_ab "finer buys nothing"): the prior menu A/B used the hs_* eval
base, which never bluffs/thin-bets, so the VALUE-SIZING channel reads ~0 there regardless of the
menu — a non-discriminating null. This gate measures the value-sizing channel vs the IRC loose-passive
STATION mix specifically, where the deployed value_sizing overlay (value_sizing.py, +11.9 bb/100 vs
stations, population-hud-analysis) and the off-tree re-solve trigger actually express value.

WHAT it measures — three arms, hero = Pluribus depth-limited search on the DCFR+VR tabular blueprint,
the SAME bot under test as run_rotated_ladder_parallel, with the validated root-mixer exploitation ON:
  (A) baseline   = value-sizing overlay OFF + bluff-suppression OFF + off-tree OFF
  (B) overlay ON = value_sizing + bluff_suppression ON, off-tree OFF
  (C) overlay+off-tree = (B) + POKERBOT_OFFTREE_RESOLVE
All three play the SAME seat-rotated duplicate decks with COMMON RANDOM NUMBERS (villain streams,
deck, hero per-hand seed identical across arms), so the only thing that moves between arms is the
hero's overlay/off-tree config. This is the gold-standard variance control for the channel delta
(eval-rotated-duplicate memory); ALWAYS --no-aivat-leafq equivalent (AIVAT gives NEGATIVE variance
reduction on the search path, so we never use it here).

DISCRIMINATION REQUIREMENT (acceptance for the gate ITSELF): arm B vs arm A (overlay-ON minus
overlay-OFF), paired per deck, must be CI-EXCLUDING-0 vs the station mix. The overlay's WSD value-cut
guard fails CLOSED without showdown-weakness evidence, so we seed the station's whole-history server
HUD (exploitation_battery.population_hud("calling_station"): wtsd high / wsd low) onto the pooled
villain profile — exactly the live capture path. Without that seed the overlay correctly refuses to
escalate and the channel reads ~0 (the bet_menu_ab null); with it, the channel expresses. If B vs A
does not separate, the gate is non-discriminating — INCREASE --decks until it does.

RAKE-ADJUSTED net-EV (advances the D "rake net-EV column"): each arm also reports a net_of_rake
bb/100 column = chip-EV minus the hero's per-hand net cash cost (pokerbot.economics.rake.net_cost_bb:
5%-flat capped rake, contributed attribution, CoinRewards rakeback, splash fee) for a chosen stake.
The SIZE escalation grows the pot, so the rake-adjusted column is exactly where a bigger-bet channel
either pays for itself (above the BB cap, rake is flat) or eats its own value (below the cap).

NON-NEGOTIABLE: measurement/eval ONLY. No engine/menu/golden/firewall change. Overlay toggled via its
existing env kill-switches (POKERBOT_VALUE_SIZING / POKERBOT_BLUFF_SUPPRESS), off-tree via
pluribus_search.set_offtree_resolve. The hero, villains, and rotated-duplicate math are all REUSED
from pokerbot.eval.exploitation_battery / irc_villains / production_hero verbatim.

Pilot (discrimination) example, ~tens of minutes on one core (FIXED iterations for CRN determinism):
  python scripts/eval/sizing_channel_ab.py --decks 120 --iterations 25 \
      --blueprint-snapshot checkpoints/bp6max_lossless2_iter200.pkl \
      --stake 0.10/0.25 --out runs/sizing_channel_ab/pilot.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np

# Blueprint serve keys must match the trained table — the encoder bakes these at import, so set them
# BEFORE importing pokerbot (mirrors scripts/eval_exploitation_battery.py). Without POSTFLOP_BUCKETS the
# pluribus hero gets ~0% postflop blueprint coverage (silent uniform fallback).
os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_BUCKETS", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_MODE", "fast")
os.environ.setdefault("POKERBOT_POSTFLOP_NUM_BUCKETS", "200")
# Match the deployed lossless-preflop + search-preflop serve config the ladder uses.
os.environ.setdefault("POKERBOT_PREFLOP_LOSSLESS", "1")
os.environ.setdefault("POKERBOT_SEARCH_PREFLOP", "1")
# Native (Rust) eval fast path is statistically equivalent + byte-identical for the GTO base; leave it
# on by default (opt out with POKERBOT_NATIVE_SUBGAME=0). The exploit ON arm stays Python (firewall).
os.environ.setdefault("POKERBOT_NATIVE_SUBGAME", "1")
os.environ.setdefault("POKERBOT_NATIVE_ROLLOUT", "1")

from pokerbot.engine.game import GameState, play_hand               # noqa: E402
from pokerbot.eval.exploitation_battery import (                    # noqa: E402
    _allin_ev_settle,
    population_hud,
)
from pokerbot.eval.irc_villains import build_irc_villain            # noqa: E402
from pokerbot.economics.rake import STAKES, net_cost_bb             # noqa: E402
from pokerbot.opponent.stats import OpponentStats                   # noqa: E402
from pokerbot.training.self_play import _legal_mask                 # noqa: E402

N_SEATS = 6
BUTTON_SEAT = 3


# --- arm definitions: (value_sizing, bluff_suppress, offtree) toggles ------------------------------
ARMS = {
    "A_baseline": {"value_sizing": False, "bluff_suppress": False, "offtree": False},
    "B_overlay": {"value_sizing": True, "bluff_suppress": True, "offtree": False},
    "C_overlay_offtree": {"value_sizing": True, "bluff_suppress": True, "offtree": True},
}
ARM_ORDER = ["A_baseline", "B_overlay", "C_overlay_offtree"]

# Per-hand bet_shift accumulator. The patched live seam (install_live_serve_seam) captures the overlay's
# bet_shift (change in the hero's expected bet SIZE in pot fractions, root_mixer._exp_size) into this for
# the CURRENT arm/hand — the LOW-VARIANCE channel metric the value_sizing overlay was validated on
# (hud_sizing_acceptance). Chip-EV of a size change is high variance; the size SHIFT itself is the direct,
# tight measurement that the channel is expressing, so it is the gate's discrimination statistic.
_SHIFT_SINK: dict = {"sum": 0.0, "n": 0}


def _reset_shift_sink() -> None:
    _SHIFT_SINK["sum"] = 0.0
    _SHIFT_SINK["n"] = 0


def _drain_shift_mean() -> float:
    """Mean |bet_shift| over the decisions since the last reset (0 if the hero never reached a bet node)."""
    n = _SHIFT_SINK["n"]
    return (_SHIFT_SINK["sum"] / n) if n else 0.0


def install_live_serve_seam() -> None:
    """Reproduce the LIVE serve seam in the eval path WITHOUT touching any engine/search source.

    The deployed value-sizing channel lives in ``apply_live_value_sizing`` (root_mixer.py:457), which the
    LIVE bridge (live_advisor_bridge.py:456) applies to the hero's policy AFTER ``apply_root_exploit_mixer``
    — `search -> root mixer -> live value-sizing -> sample`. The eval ``SearchAgent`` only does the root
    mixer (search_agent.py:252), so the value-sizing overlay is INERT on the eval path (this is exactly
    why the channel reads ~0 there — the bet_menu_ab null, in a different guise). To measure the SAME
    channel the live serve uses, we MONKEYPATCH ``apply_root_exploit_mixer`` (in this eval process only)
    to compose the second live step: run the real mixer, then apply_live_value_sizing on its output,
    re-mask + renormalize exactly as the live bridge does. No source file changes; the patch is local to
    this measurement script. value_sizing / bluff_suppress are toggled per-arm via their env kill-
    switches (read at call time inside the overlay), so arm A (overlay OFF) makes this step a no-op and
    the patched path is byte-identical to the un-patched eval path for arm A."""
    import numpy as _np

    import pokerbot.opponent.root_mixer as rm

    if getattr(rm, "_SIZING_AB_PATCHED", False):
        return
    _real_mixer = rm.apply_root_exploit_mixer

    def _patched(state, seat, probs, profiles, *, unopened=False, solver_best_responds=False):
        mixed = _real_mixer(state, seat, probs, profiles, unopened=unopened,
                            solver_best_responds=solver_best_responds)
        # The live serve step: HUD value-SIZING / bluff-suppression overlay on the post-mixer policy.
        # Capture the overlay's bet_shift (size change in pot fractions) into the per-arm sink — the
        # low-variance channel metric. Only accumulate at genuine bet decisions where the overlay fired.
        _cap: dict = {}
        mixed = rm.apply_live_value_sizing(state, seat, mixed, profiles, out=_cap)
        if _cap.get("fired"):
            _SHIFT_SINK["sum"] += abs(float(_cap.get("bet_shift", 0.0) or 0.0))
            _SHIFT_SINK["n"] += 1
        legal = _np.asarray(_legal_mask(state, seat), dtype=bool)
        mixed = mixed * legal
        s = float(mixed.sum())
        return mixed / s if s > 1e-12 else _real_mixer(state, seat, probs, profiles,
                                                        unopened=unopened,
                                                        solver_best_responds=solver_best_responds)

    rm.apply_root_exploit_mixer = _patched
    rm._SIZING_AB_PATCHED = True
    print("[sizing-ab] installed live serve seam (root mixer -> apply_live_value_sizing)", flush=True)


def _set_arm(cfg: dict) -> None:
    """Configure the overlay/off-tree toggles for one arm. value_sizing / bluff_suppress are read at
    CALL TIME from os.environ (value_sizing._enabled / _bluff_suppress_enabled), so setting the env var
    here takes effect for the next decision. POKERBOT_LIVE_VALUE_SIZING stays ON (the seam itself);
    arm A turns OFF the component overlays so the seam is a no-op. off-tree is captured into a module
    global at import, so it must be flipped through the provided setter, never os.environ."""
    from pokerbot.search.pluribus_search import set_offtree_resolve

    os.environ["POKERBOT_LIVE_VALUE_SIZING"] = "1"
    os.environ["POKERBOT_VALUE_SIZING"] = "1" if cfg["value_sizing"] else "0"
    os.environ["POKERBOT_BLUFF_SUPPRESS"] = "1" if cfg["bluff_suppress"] else "0"
    # River thin-value stays OFF (default; it is the one overlay with value-cut risk, flagged off).
    os.environ.setdefault("POKERBOT_RIVER_THIN_VALUE", "0")
    # The rake-aware SIZE discount reads POKERBOT_CASH_STAKE for its stake context; align it with the
    # reporting stake so the live overlay's rake-discount matches the rake column we subtract.
    set_offtree_resolve(cfg["offtree"])


def _build_pluribus_factory(snapshot_path: str, *, iterations: int, time_budget_ms: int,
                            leaf_select: str, seed: int):
    """ProductionHero(Pluribus SearchAgent) factory — VERBATIM the eval_exploitation_battery wiring
    (north-star in-distribution hero on the un-poisoned tabular blueprint, root-mixer exploitation,
    no neural value net). exploit=False => opponent_model None => pure GTO base (the shared OFF arm)."""
    import pickle

    from pokerbot.eval.production_hero import ProductionHero
    from pokerbot.eval.search_agent import SearchAgent

    with open(snapshot_path, "rb") as f:
        snap_full = pickle.load(f)
    snap = snap_full["strat_sum"] if isinstance(snap_full, dict) and "strat_sum" in snap_full else snap_full
    print(f"blueprint snapshot: {len(snap)} infosets from {snapshot_path}", flush=True)

    def factory(*, exploit: bool):
        agent = SearchAgent(
            use_pluribus_search=True,
            blueprint_snapshot=snap,
            pluribus_leaf_select=leaf_select,
            time_budget_ms=time_budget_ms,
            transformer_checkpoint=None,
            hero_seat=0,
            root_exploit_mixer=True,          # the validated read->mixer_logits->safe_mix exploitation
            root_mixer_unopened=True,         # tilt our own value-bets from the response read too
            aivat_leaf_q=False,               # ALWAYS off for rotated (AIVAT broken on search path)
        )
        if iterations:
            # FIXED CFR iterations => each solve is a pure fn of (public state, opponent model, config),
            # so the CRN arms solve the SAME decision identically (a wall-clock budget would make the
            # iteration count load-dependent and break ON/ON arm determinism).
            agent.solver.config.max_iterations = iterations
            agent.solver.config.time_budget_ms = 10 ** 9
        return ProductionHero(agent, random.Random(seed), exploit=exploit, hero_seat=0, n_seats=N_SEATS)

    return factory


def _reseed(hero, s, hero_seat):
    """CRN per-hand reseed — see exploitation_battery._reseed: tell a search hero its seat BEFORE
    on_new_hand() or it plays seat-0's cards from every physical seat (5/6 rotations garbage)."""
    if hasattr(hero, "hero_seat"):
        hero.hero_seat = hero_seat
    if hasattr(hero, "reseed"):
        hero.reseed(s)
    else:
        hero.rng = random.Random(s)
    if hasattr(hero, "on_new_hand"):
        hero.on_new_hand()


def _play_capture(hero_seat, hero, bot_fn, state, settle):
    """Play one hand; return (hero_chip_delta_bb, net_cost_bb-inputs). Captures the final pot, the
    hero's committed total, and whether the hand ended preflop so the caller can subtract rake."""
    policies = {s: (hero if s == hero_seat else bot_fn) for s in range(N_SEATS)}
    final, deltas = play_hand(policies, state=state, allin_settle=settle)
    pot_bb = float(getattr(final, "pot_bb", 0.0) or 0.0)
    hero_committed = float(final.players[hero_seat].committed_total_bb)
    ended_preflop = len(final.board) == 0
    return float(deltas.get(hero_seat, 0.0)), pot_bb, hero_committed, ended_preflop


def _rake_cost(pot_bb, hero_committed, ended_preflop, stake_key, rakeback_mode):
    if stake_key is None:
        return 0.0
    return float(net_cost_bb(pot_bb, hero_committed, N_SEATS, ended_preflop, stake_key,
                             rakeback_mode=rakeback_mode))


def run(decks, seed, snapshot_path, *, iterations, time_budget_ms, leaf_select,
        stake_key, rakeback_mode, progress_every):
    """Seat-rotated duplicate sizing-channel A/B. Per deck d, per seat hs, ONE shared deck (pure fn of
    deck_seed) is played by:
      * the OFF hero (no profile, overlay irrelevant) ONCE -> the shared GTO/blueprint base, and
      * the ON hero under EACH arm's overlay config, with identical villain streams + hero per-hand
        seed (CRN), so the per-deck seat-mean DIFFERENCE between arms isolates the overlay channel.
    The station's whole-history server HUD is seeded onto the pooled villain profile so the value-cut-
    guarded overlay FIRES (else it correctly reads ~0 = the bet_menu_ab null)."""
    install_live_serve_seam()
    # The live overlay's rake-aware SIZE discount reads POKERBOT_CASH_STAKE; align it with the reporting
    # stake so the channel we measure matches the rake regime of the net column we subtract.
    if stake_key is not None:
        os.environ["POKERBOT_CASH_STAKE"] = stake_key
    factory = _build_pluribus_factory(snapshot_path, iterations=iterations,
                                       time_budget_ms=time_budget_ms, leaf_select=leaf_select,
                                       seed=seed)
    villain = ("irc_station", None)

    hero_off = factory(exploit=False)
    hero_on = factory(exploit=True)
    # One pooled profile for the homogeneous station field. Seed the population station HUD so the
    # value-sizing overlay's WSD value-cut guard has the showdown-weakness evidence it requires.
    bot_stats = OpponentStats()
    bot_stats.hud = population_hud("calling_station")
    hero_on.set_profile(bot_stats)

    # Per-arm per-deck seat-mean accumulators (chip-EV, rake-adjusted net, and the low-variance
    # bet_shift channel metric = mean |size change| the overlay applied at the hero's bet decisions).
    off_deck: list[float] = []
    off_deck_net: list[float] = []
    on_deck = {a: [] for a in ARM_ORDER}
    on_deck_net = {a: [] for a in ARM_ORDER}
    on_shift = {a: [] for a in ARM_ORDER}

    def villain_bot(deck_seed, hs):
        return {s: build_irc_villain(villain[0], random.Random(5000 + deck_seed + 17 * s),
                                     params=villain[1])
                for s in range(N_SEATS) if s != hs}

    for d in range(decks):
        if progress_every and d and d % progress_every == 0:
            print(f"[sizing-ab] deck {d}/{decks}", flush=True)
        deck_seed = 1000 + seed * 1_000_000 + d
        settle = lambda st, nb, _s=deck_seed: _allin_ev_settle(st, nb, 7777 + _s)

        seat_off, seat_off_net = [], []
        seat_on = {a: [] for a in ARM_ORDER}
        seat_on_net = {a: [] for a in ARM_ORDER}
        seat_shift = {a: [] for a in ARM_ORDER}

        for hs in range(N_SEATS):
            # OFF (shared GTO base): no profile => overlay/off-tree irrelevant; compute once per seat.
            _set_arm({"value_sizing": False, "bluff_suppress": False, "offtree": False})
            _reseed(hero_off, deck_seed, hs)
            st = GameState.new_hand(button_seat=BUTTON_SEAT, rng=random.Random(deck_seed))
            r, pot, comm, pre = _play_capture(hs, hero_off,
                                              lambda state, _b=villain_bot(deck_seed, hs): _b[state.action_seat](state),
                                              st, settle)
            seat_off.append(r)
            seat_off_net.append(r - _rake_cost(pot, comm, pre, stake_key, rakeback_mode))

            # ON arms: identical deck + identical villain streams + identical hero per-hand seed; the
            # ONLY difference across arms is the overlay/off-tree config => tight CRN on the channel.
            for arm in ARM_ORDER:
                _set_arm(ARMS[arm])
                _reset_shift_sink()
                _reseed(hero_on, deck_seed, hs)
                st = GameState.new_hand(button_seat=BUTTON_SEAT, rng=random.Random(deck_seed))
                bots = villain_bot(deck_seed, hs)
                r, pot, comm, pre = _play_capture(hs, hero_on,
                                                  lambda state, _b=bots: _b[state.action_seat](state),
                                                  st, settle)
                seat_on[arm].append(r)
                seat_on_net[arm].append(r - _rake_cost(pot, comm, pre, stake_key, rakeback_mode))
                seat_shift[arm].append(_drain_shift_mean())

        off_deck.append(float(np.mean(seat_off)))
        off_deck_net.append(float(np.mean(seat_off_net)))
        for arm in ARM_ORDER:
            on_deck[arm].append(float(np.mean(seat_on[arm])))
            on_deck_net[arm].append(float(np.mean(seat_on_net[arm])))
            on_shift[arm].append(float(np.mean(seat_shift[arm])))

    return _summarize(off_deck, off_deck_net, on_deck, on_deck_net, on_shift,
                      stake_key, rakeback_mode, villain[0], decks, seed)


def _stat(vals):
    a = np.asarray(vals, dtype=np.float64)
    n = len(a)
    m = float(a.mean()) if n else 0.0
    sd = float(a.std(ddof=1)) if n > 1 else 0.0
    se = float(sd / np.sqrt(n)) if n > 1 else 0.0
    return {"decks": n, "mean_bb": m, "se": se,
            "bb100": 100 * m, "ci95_bb100": [100 * (m - 1.96 * se), 100 * (m + 1.96 * se)]}


def _paired(on_vals, off_vals):
    """Paired (CRN) per-deck delta on_vals - off_vals: the uplift over the shared base, with its own
    (much tighter) CI since the dominant deck variance cancels deck-by-deck."""
    diff = np.asarray(on_vals, dtype=np.float64) - np.asarray(off_vals, dtype=np.float64)
    return _stat(diff)


def _stat_raw(vals):
    """Mean/SE/CI in RAW units (no x100), for the bet_shift channel which is already in pot fractions."""
    a = np.asarray(vals, dtype=np.float64)
    n = len(a)
    m = float(a.mean()) if n else 0.0
    sd = float(a.std(ddof=1)) if n > 1 else 0.0
    se = float(sd / np.sqrt(n)) if n > 1 else 0.0
    return {"decks": n, "mean": m, "se": se, "ci95": [m - 1.96 * se, m + 1.96 * se]}


def _paired_raw(on_vals, off_vals):
    diff = np.asarray(on_vals, dtype=np.float64) - np.asarray(off_vals, dtype=np.float64)
    return _stat_raw(diff)


def _summarize(off_deck, off_deck_net, on_deck, on_deck_net, on_shift, stake_key, rakeback_mode,
               villain_name, decks, seed):
    out = {
        "field": [villain_name] * (N_SEATS - 1),
        "villain": villain_name,
        "method": "seat_rotated_duplicate_crn_sizing_channel",
        "decks": len(off_deck),
        "seed": seed,
        "stake": stake_key,
        "rakeback_mode": rakeback_mode,
        "hud_seeded": True,
        "off_base": {"chip": _stat(off_deck), "net_of_rake": _stat(off_deck_net)},
        "arms": {},
    }
    for arm in ARM_ORDER:
        out["arms"][arm] = {
            "config": ARMS[arm],
            "chip": _stat(on_deck[arm]),
            "net_of_rake": _stat(on_deck_net[arm]),
            "bet_shift": _stat_raw(on_shift[arm]),     # mean |size change| (pot fractions) per deck
            "uplift_vs_base_chip": _paired(on_deck[arm], off_deck),
            "uplift_vs_base_net": _paired(on_deck_net[arm], off_deck_net),
        }
    # CHIP-EV channel deltas: arm B (overlay ON) minus arm A (overlay OFF), paired per deck. Headline EV
    # consequence, but high variance (a size change pays off / gets coolered with large dispersion).
    out["channel_B_minus_A_chip"] = _paired(on_deck["B_overlay"], on_deck["A_baseline"])
    out["channel_B_minus_A_net"] = _paired(on_deck_net["B_overlay"], on_deck_net["A_baseline"])
    out["channel_C_minus_B_chip"] = _paired(on_deck["C_overlay_offtree"], on_deck["B_overlay"])
    out["channel_C_minus_A_chip"] = _paired(on_deck["C_overlay_offtree"], on_deck["A_baseline"])
    # DISCRIMINATION metric: the bet_shift channel B - A (mean |size change| the overlay applies),
    # paired per deck. This is the LOW-VARIANCE direct measurement that the value-sizing channel is
    # EXPRESSING on the search base (the same bet_shift the live acceptance used). A>0 baseline shift is
    # only the residual root-mixer sizing; B-A isolates the value_sizing/bluff-suppression overlay.
    out["channel_B_minus_A_betshift"] = _paired_raw(on_shift["B_overlay"], on_shift["A_baseline"])
    bms = out["channel_B_minus_A_betshift"]
    lo, hi = bms["ci95"]
    out["overlay_detected"] = bool(lo > 0.0 or hi < 0.0)   # bet_shift CI excludes 0 (channel expresses)
    bma = out["channel_B_minus_A_chip"]
    clo, chi = bma["ci95_bb100"]
    out["chip_ev_detected"] = bool(clo > 0.0 or chi < 0.0)  # secondary: chip-EV CI excludes 0 (needs more N)
    return out


def _fmt_ci(ci):
    return f"[{ci[0]:+.1f}, {ci[1]:+.1f}]"


def _report(r):
    print("\n=== SIZING-CHANNEL A/B (IRC station mix, seat-rotated duplicate + CRN) ===", flush=True)
    print(f"villain={r['villain']}  decks={r['decks']}  seed={r['seed']}  "
          f"stake={r['stake']}  rakeback={r['rakeback_mode']}  HUD-seeded={r['hud_seeded']}", flush=True)
    print(f"\n{'arm':20s} {'chip bb/100':>12s} {'chip 95% CI':>20s} "
          f"{'net bb/100':>12s} {'net 95% CI':>20s}", flush=True)
    ob = r["off_base"]
    print(f"{'OFF base (GTO)':20s} {ob['chip']['bb100']:>+12.1f} {_fmt_ci(ob['chip']['ci95_bb100']):>20s} "
          f"{ob['net_of_rake']['bb100']:>+12.1f} {_fmt_ci(ob['net_of_rake']['ci95_bb100']):>20s}", flush=True)
    for arm in ARM_ORDER:
        a = r["arms"][arm]
        print(f"{arm:20s} {a['chip']['bb100']:>+12.1f} {_fmt_ci(a['chip']['ci95_bb100']):>20s} "
              f"{a['net_of_rake']['bb100']:>+12.1f} {_fmt_ci(a['net_of_rake']['ci95_bb100']):>20s} "
              f"  betshift={a['bet_shift']['mean']:+.3f}", flush=True)
    print(f"\n--- DISCRIMINATION metric: bet_shift channel (low variance, the gate's trust check) ---",
          flush=True)
    bms = r["channel_B_minus_A_betshift"]
    print(f"B - A (overlay bet_shift, pot frac)  {bms['mean']:>+8.4f}  "
          f"CI [{bms['ci95'][0]:+.4f}, {bms['ci95'][1]:+.4f}]", flush=True)
    print(f"OVERLAY EFFECT DETECTED (bet_shift B-A CI excludes 0): {r['overlay_detected']}", flush=True)
    if not r["overlay_detected"]:
        print("  -> NON-DISCRIMINATING at this hand count. Increase --decks until the bet_shift B-A CI "
              "separates, or diagnose why the overlay isn't expressing (HUD/fish-gate/value frequency).",
              flush=True)

    print(f"\n--- paired CRN chip-EV / net-rake channel deltas (headline EV, higher variance) ---",
          flush=True)
    bma = r["channel_B_minus_A_chip"]
    bman = r["channel_B_minus_A_net"]
    cmb = r["channel_C_minus_B_chip"]
    cma = r["channel_C_minus_A_chip"]
    print(f"B - A (overlay channel, chip)      {bma['bb100']:>+8.1f} bb/100  CI {_fmt_ci(bma['ci95_bb100'])}",
          flush=True)
    print(f"B - A (overlay channel, net-rake)  {bman['bb100']:>+8.1f} bb/100  CI {_fmt_ci(bman['ci95_bb100'])}",
          flush=True)
    print(f"C - B (off-tree increment, chip)   {cmb['bb100']:>+8.1f} bb/100  CI {_fmt_ci(cmb['ci95_bb100'])}",
          flush=True)
    print(f"C - A (overlay+off-tree, chip)     {cma['bb100']:>+8.1f} bb/100  CI {_fmt_ci(cma['ci95_bb100'])}",
          flush=True)
    print(f"chip-EV B-A CI excludes 0: {r['chip_ev_detected']} (needs large N; bet_shift is the gate)",
          flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decks", type=int, default=120, help="seat-rotated duplicate deals (each = "
                    "N_SEATS x (1 OFF + 3 ON) hands)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blueprint-snapshot",
                    default=r"checkpoints/bp6max_lossless2_iter200.pkl")
    ap.add_argument("--iterations", type=int, default=25,
                    help="FIXED CFR iterations per solve (CRN determinism). 0 => wall-clock budget.")
    ap.add_argument("--time-budget-ms", type=int, default=300)
    ap.add_argument("--leaf-select", default="uniform", choices=("uniform", "gadget"))
    ap.add_argument("--stake", default="0.10/0.25",
                    help=f"rake stake key for the net-of-rake column (one of {list(STAKES)}); "
                         "'none' to skip the rake column")
    ap.add_argument("--rakeback-mode", default="modeled", choices=("floor", "modeled"))
    ap.add_argument("--progress-every", type=int, default=10)
    ap.add_argument("--out", default="runs/sizing_channel_ab/sizing_channel_ab.json")
    args = ap.parse_args()

    stake_key = None if args.stake.lower() == "none" else args.stake
    if stake_key is not None and stake_key not in STAKES:
        raise SystemExit(f"unknown stake '{stake_key}'; known: {list(STAKES)}")

    r = run(args.decks, args.seed, args.blueprint_snapshot,
            iterations=args.iterations, time_budget_ms=args.time_budget_ms,
            leaf_select=args.leaf_select, stake_key=stake_key,
            rakeback_mode=args.rakeback_mode, progress_every=args.progress_every)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=1), encoding="utf-8")
    _report(r)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
