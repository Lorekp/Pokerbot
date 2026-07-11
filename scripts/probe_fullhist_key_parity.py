#!/usr/bin/env python3
"""PARITY GATE for the FULL action-history key (POKERBOT_FULLHIST_KEY=1, blueprint-v4 Stage 0).

Design of record: artifacts/blueprint_pluribus_grade_design.md §7.1 (the bp3 hist-key pattern one
step further: the b"A" segment = street byte + raw engine action-id bytes of the WHOLE hand, hashed
into EVERY street's key — preflop b"w" AND postflop b"B" branches).

Walks GAME-FLOW states ONLY (real hands dealt through the pure-Python engine; standing rule: never
hand-build nodes). For every decision state ON EVERY STREET the acting seat's blueprint key is
computed three ways and all three must be byte-identical:

  * py_key  — ``jax_weva_fast.fingerprint_fast`` (the serve twin under test);
  * nat_key — ``battery_native.fingerprint_fast`` (the Rust trainer/serve keying; reads
              POKERBOT_FULLHIST_KEY itself via OnceLock, so this process's env drives it);
  * ref_key — an INDEPENDENT in-script reimplementation of the exact message layout
              blake2b16([b"w"+LE32(pos*169+class) | b"B"+LE32(bucket)] [+ b"A"+street+hist]
              + b"S" + betting6), with hist/betting6 recomputed from the raw state.

Run with --fullhist 1 (the v4 Stage-0 key space) AND --fullhist 0 (must be byte-identical to the
deployed layout, i.e. ref_key WITHOUT the A segment). One env state per process — the flags are
read at import (Python) / first use (Rust OnceLock) — so the pytest wrapper
(tests/test_fullhist_key_parity.py) runs this twice via subprocess.

Coverage floors guarantee the walk exercises the aliasing-critical shapes: every street (incl.
>=512 POSTFLOP states — the whole point of v4 is postflop keying), limped/raised/3bet+ preflop
pots, postflop raise chains, and multiple history lengths.

Usage:
  python scripts/probe_fullhist_key_parity.py --fullhist 1 --min-states 512 \
      --out artifacts/fullhist_parity_on.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import struct
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# ---------------------------------------------------------------------------
# Env MUST be resolved before any pokerbot import (flags are read at import; the Rust side reads
# POKERBOT_FULLHIST_KEY once via OnceLock at first fingerprint call in this same process).
# The bounded made-hand config (the fingerprint.rs header config) is the minimal parity regime;
# the A segment is card-abstraction-independent (same bytes under EMD keying).
# ---------------------------------------------------------------------------
_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--fullhist", type=int, choices=(0, 1), required=True,
                 help="1 = POKERBOT_FULLHIST_KEY=1 (v4 Stage-0 key space); 0 = deployed layout")
_ap.add_argument("--presum", type=int, choices=(0, 1), default=0,
                 help="1 = POKERBOT_PRESUM_KEY=1 (arm-4 postflop preflop-summary key; requires "
                      "--fullhist 1). The POSTFLOP A-segment becomes street + 4 summary bytes + "
                      "exact postflop bytes; preflop is unchanged fullhist. 0 = plain fullhist.")
_ap.add_argument("--translate", type=int, choices=(0, 1), default=0,
                 help="1 = POKERBOT_ACTION_TRANSLATE=1 (serve-coverage off-menu pseudo-harmonic "
                      "translation; requires --fullhist 1). The independent ref layout cannot "
                      "reproduce the replay-translation, so this mode gates py==nat byte-parity "
                      "plus a floor of actually-TRANSLATED states. 0 = no translation (3-way).")
_ap.add_argument("--min-states", type=int, default=512)
_ap.add_argument("--min-postflop", type=int, default=512)
_ap.add_argument("--max-hands", type=int, default=12000)
_ap.add_argument("--seed", type=int, default=7)
_ap.add_argument("--num-buckets", type=int, default=200)
_ap.add_argument("--out", default="")
ARGS = _ap.parse_args()

os.environ.setdefault("JAX_PLATFORMS", "cpu")
for _k, _v in {
    "POKERBOT_BETTING_ABSTRACTION": "1",
    "POKERBOT_POSTFLOP_BUCKETS": "1",
    "POKERBOT_POSTFLOP_NUM_BUCKETS": str(ARGS.num_buckets),
    "POKERBOT_POSTFLOP_MODE": "fast",
    "POKERBOT_PREFLOP_LOSSLESS": "1",
    "POKERBOT_ALLOW_MADE_HAND_BUCKETS": "1",  # parity probe, not a production train
}.items():
    os.environ.setdefault(_k, _v)
if ARGS.presum == 1 and ARGS.fullhist != 1:
    _ap.error("--presum 1 requires --fullhist 1 (presum refines the fullhist A-segment)")
if ARGS.translate == 1 and ARGS.fullhist != 1:
    _ap.error("--translate 1 requires --fullhist 1 (translation exists for history-baked keys)")
if ARGS.translate == 1:
    os.environ["POKERBOT_ACTION_TRANSLATE"] = "1"
    # Exercise the M-RR postflop thinning too (the deployed fullhist regimes train with it), so the
    # walk's free-menu bets are off the trained menu on every street.
    os.environ.setdefault("POKERBOT_MRR", "1")
else:
    os.environ.pop("POKERBOT_ACTION_TRANSLATE", None)
if ARGS.fullhist == 1:
    os.environ["POKERBOT_FULLHIST_KEY"] = "1"
else:
    os.environ.pop("POKERBOT_FULLHIST_KEY", None)
if ARGS.presum == 1:
    os.environ["POKERBOT_PRESUM_KEY"] = "1"
else:
    os.environ.pop("POKERBOT_PRESUM_KEY", None)
os.environ.pop("POKERBOT_PREFLOP_HIST_KEY", None)  # v4 arms do NOT stack the bp3 3-byte key

import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# Independent reference layout (mirrors fingerprint.rs, NOT the twin under test).
# ---------------------------------------------------------------------------
_POS_BY_OFFSET_REF = (3, 4, 5, 0, 1, 2)
_POT_EDGES = (3.0, 6.0, 12.0, 25.0, 50.0, 100.0, 200.0)
_TOCALL_EDGES = (0.01, 0.34, 0.67, 1.01, 2.01)
_SPR_EDGES = (1.0, 2.5, 5.0, 10.0, 20.0)


def _bucket(x: float, edges) -> int:
    b = 0
    for e in edges:
        if x >= e:
            b += 1
        else:
            break
    return b


def _canon_class(c0: int, c1: int) -> int:
    r0, s0 = c0 // 4, c0 % 4
    r1, s1 = c1 // 4, c1 % 4
    if r0 == r1:
        return r0
    hr, lr = (r0, r1) if r0 > r1 else (r1, r0)
    tri = hr * (hr - 1) // 2 + lr
    return (13 + tri) if s0 == s1 else (13 + 78 + tri)


def _ref_betting6(env, seat: int) -> bytes:
    committed = np.asarray(env.committed_street)
    seat_status = np.asarray(env.seat_status)
    stacks = np.asarray(env.stacks)
    pot = int(np.asarray(env.pot_dead)) + int(committed.sum())
    to_call = max(0, int(np.asarray(env.current_bet)) - int(committed[seat]))
    live = [int(stacks[s]) for s in range(6) if int(seat_status[s]) not in (0, 1)]
    eff = (min(live) if live else 0) / 100.0
    pot_bb = pot / 100.0
    last_agg = int(np.asarray(env.last_aggressor))
    return bytes((
        int(np.asarray(env.street)) & 7,
        _bucket(pot_bb, _POT_EDGES),
        _bucket(to_call / max(pot, 1), _TOCALL_EDGES),
        _bucket(eff / max(pot_bb, 1e-6), _SPR_EDGES),
        1 if seat == int(np.asarray(env.button)) else 0,
        1 if (last_agg >= 0 and last_agg != seat) else 0,
    ))


def _ref_fullhist(env) -> bytes:
    """street byte + raw action-id bytes (the b"A" segment payload)."""
    hl = max(0, min(int(np.asarray(env.hist_len)), 64))
    act = np.asarray(env.hist_action)
    street = int(np.asarray(env.street)) & 0xFF
    return bytes([street] + [int(act[i]) & 0xFF for i in range(hl)])


def _ref_presum(env, n_preflop: int) -> bytes:
    """INDEPENDENT presum A-segment payload (the bytes after b"A"): street + 4 summary bytes +
    exact postflop action bytes. `n_preflop` is supplied by the walk as GROUND TRUTH (the count of
    leading actions whose ORIGIN street was preflop — tracked forward during the game walk, NOT
    recomputed from the twin's engine-replay), so this reference is independent of the twin."""
    hl = max(0, min(int(np.asarray(env.hist_len)), 64))
    act = np.asarray(env.hist_action)
    seat = np.asarray(env.hist_seat)
    street = int(np.asarray(env.street)) & 0xFF
    n_calls = 0
    n_raises = 0
    opener_seat = -1
    for i in range(min(n_preflop, hl)):
        a = int(act[i])
        if a == 2:  # CALL
            if n_raises == 0 and n_calls < 3:
                n_calls += 1
        elif 3 <= a <= 10:  # BET_BUCKET_START..ALL_IN
            if n_raises == 0:
                opener_seat = int(seat[i])
            if n_raises < 3:
                n_raises += 1
    if opener_seat >= 0:
        opener_pos = _POS_BY_OFFSET_REF[(opener_seat - int(np.asarray(env.button))) % 6]
    else:
        opener_pos = -1
    seat_status = np.asarray(env.seat_status)
    n_active = int(np.sum((seat_status == 2) | (seat_status == 3)))  # SEAT_LIVE | SEAT_ALLIN
    out = [street, n_calls & 0xFF, n_raises & 0xFF, (opener_pos + 1) & 0xFF, n_active & 0xFF]
    out += [int(act[i]) & 0xFF for i in range(n_preflop, hl)]
    return bytes(out)


def _ref_made_hand_bucket(env, seat: int, num_buckets: int) -> int:
    """Independent made-hand bucket via phevaluator (matches fast_made_hand_bucket)."""
    import phevaluator

    holes = np.asarray(env.holes)
    board = [int(c) for c in np.asarray(env.board) if int(c) >= 0]
    h0, h1 = int(holes[seat, 0]), int(holes[seat, 1])
    if len(board) < 3 or h0 < 0 or h1 < 0:
        return -1
    rank = phevaluator.evaluate_cards(*[_card_str(c) for c in [h0, h1] + board])
    return min((int(rank) - 1) * num_buckets // 7462, num_buckets - 1)


_RANKS = "23456789TJQKA"
_SUITS = "cdhs"


def _card_str(idx: int) -> str:
    return _RANKS[idx // 4] + _SUITS[idx % 4]


def _ref_key(env, seat: int, fullhist_on: bool, num_buckets: int,
             presum_on: bool = False, n_preflop: int = 0) -> bytes:
    holes = np.asarray(env.holes)
    bucket = _ref_made_hand_bucket(env, seat, num_buckets)
    if bucket >= 0:
        msg = b"B" + struct.pack("<i", bucket)
    else:
        h0, h1 = int(holes[seat, 0]), int(holes[seat, 1])
        lo, hi = (h0, h1) if h0 <= h1 else (h1, h0)
        pos_enum = _POS_BY_OFFSET_REF[(seat - int(np.asarray(env.button))) % 6]
        weva_id = pos_enum * 169 + _canon_class(lo, hi)
        msg = b"w" + struct.pack("<i", weva_id)
    if fullhist_on:
        # arm-4 presum: POSTFLOP (bucket >= 0) uses the preflop-summary A-segment; preflop is plain
        # fullhist. Presum off => plain fullhist on every street (the Stage-0 layout).
        if presum_on and bucket >= 0:
            msg += b"A" + _ref_presum(env, n_preflop)
        else:
            msg += b"A" + _ref_fullhist(env)
    msg += b"S" + _ref_betting6(env, seat)
    return hashlib.blake2b(msg, digest_size=16).digest()


def _pick_action(rng: random.Random, legal_set: set[int]) -> int:
    """Game-flow policy tuned for HISTORY variety across all streets: checks/calls (long histories),
    sized bets/raises (raise chains), a little jam, some folds."""
    u = rng.random()
    sized = [a for a in legal_set if 3 <= a <= 9]
    if u < 0.16 and 0 in legal_set and len(legal_set) > 1:
        return 0
    if u < 0.62:
        for a in (1, 2):
            if a in legal_set:
                return a
    if u < 0.96 and sized:
        return rng.choice(sized)
    if 10 in legal_set and rng.random() < 0.25:
        return 10
    return rng.choice(sorted(legal_set))


def main() -> int:
    import jax
    from pokerbot.engine.jax_engine import init_hand
    from pokerbot.engine.np_engine import (
        apply_action_np, is_chance_node_np, legal_action_mask_np, sample_chance_np, to_numpy_state,
    )
    from pokerbot.engine.jax_weva_fast import fingerprint_fast as py_fingerprint
    from pokerbot.search import native_solver as _ns

    fullhist_on = ARGS.fullhist == 1
    if not _ns.available():
        print("FATAL: battery_native extension unavailable (rebuild + install the .pyd)", flush=True)
        return 2
    ext = _ns._EXT
    caps = set(ext.native_caps()) if hasattr(ext, "native_caps") else set()
    if fullhist_on and "full_hist_key" not in caps:
        print(f"FATAL: battery_native binary predates the full-hist-key port (caps={sorted(caps)}); "
              "rebuild --features python and reinstall the .pyd", flush=True)
        return 2
    presum_on = ARGS.presum == 1
    if presum_on and "presum_key" not in caps:
        print(f"FATAL: battery_native binary predates the presum-key port (caps={sorted(caps)}); "
              "rebuild --features python and reinstall the .pyd", flush=True)
        return 2
    translate_on = ARGS.translate == 1
    if translate_on and "action_translate" not in caps:
        print(f"FATAL: battery_native binary predates the action-translate port (caps={sorted(caps)}); "
              "rebuild --features python and reinstall the .pyd", flush=True)
        return 2
    print(f"fullhist={ARGS.fullhist} presum={ARGS.presum} num_buckets={ARGS.num_buckets} "
          f"native_caps={sorted(caps)}", flush=True)

    if translate_on:
        from pokerbot.engine.action_translate import translate_state_for_key as _py_translate

    rng = random.Random(ARGS.seed)
    total = 0
    translated = 0
    fails = 0
    mismatches = []
    cov = {"preflop": 0, "flop": 0, "turn": 0, "river": 0,
           "limp1": 0, "raise2plus": 0, "postflop_raise": 0, "postflop_reraise": 0,
           "hist_len_10plus": 0, "hist_len_16plus": 0}
    hands = 0
    t0 = time.time()

    def covered() -> bool:
        postflop = cov["flop"] + cov["turn"] + cov["river"]
        return (total >= ARGS.min_states and postflop >= ARGS.min_postflop
                and cov["preflop"] >= 200 and cov["flop"] >= 200 and cov["turn"] >= 100
                and cov["river"] >= 60 and cov["limp1"] >= 20 and cov["raise2plus"] >= 12
                and cov["postflop_raise"] >= 40 and cov["postflop_reraise"] >= 8
                and cov["hist_len_10plus"] >= 40 and cov["hist_len_16plus"] >= 8)

    while hands < ARGS.max_hands and not covered():
        key = jax.random.PRNGKey(rng.randrange(0, 2**31 - 1))
        env = to_numpy_state(init_hand(key, button=hands % 6))
        hands += 1
        depth = 0
        street_raises = 0
        last_street = 0
        # GROUND-TRUTH preflop boundary for the presum reference: count of leading actions applied
        # from a street==0 state (tracked forward, independent of the twin's engine-replay). Once the
        # hand leaves preflop this stops incrementing, so it is the true n_preflop for every later
        # (postflop) decision state of this hand.
        true_n_preflop = 0
        while not bool(env.done) and depth < 64:
            if is_chance_node_np(env):
                env = sample_chance_np(env)
                continue
            seat = int(env.action_seat)
            if seat < 0:
                break
            mask = legal_action_mask_np(env, seat)
            legal = set(int(a) for a in np.flatnonzero(mask))
            if not legal:
                break
            # --- the three-way parity check on this GAME-FLOW decision state ---
            py_key = bytes(py_fingerprint(env, seat))
            nat_key = bytes(ext.fingerprint_fast(_ns._to_flat(env).tobytes(), seat,
                                                 ARGS.num_buckets))
            ref_key = _ref_key(env, seat, fullhist_on, ARGS.num_buckets,
                               presum_on=presum_on, n_preflop=true_n_preflop)
            total += 1
            street = int(np.asarray(env.street))
            street_name = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}.get(street)
            if street_name:
                cov[street_name] += 1
            hl = int(np.asarray(env.hist_len))
            act = np.asarray(env.hist_action)[:hl]
            n_pre_raises = 0
            n_pre_calls = 0
            for a in act:
                a = int(a)
                if 3 <= a <= 10:
                    n_pre_raises += 1
                elif a == 2 and n_pre_raises == 0:
                    n_pre_calls += 1
            if street == 0 and n_pre_calls >= 1:
                cov["limp1"] += 1
            if street == 0 and n_pre_raises >= 2:
                cov["raise2plus"] += 1
            if street != last_street:
                street_raises = 0
                last_street = street
            if street > 0 and street_raises >= 1:
                cov["postflop_raise"] += 1
            if street > 0 and street_raises >= 2:
                cov["postflop_reraise"] += 1
            if hl >= 10:
                cov["hist_len_10plus"] += 1
            if hl >= 16:
                cov["hist_len_16plus"] += 1
            if translate_on:
                # The independent ref cannot replay-translate; the gate here is Rust==Python
                # byte-parity of the TRANSLATED key + a floor of actually-translated states.
                t_env = _py_translate(env)
                if t_env is not env:
                    translated += 1
                key_ok = py_key == nat_key
            else:
                key_ok = py_key == nat_key == ref_key
            if not key_ok:
                fails += 1
                if len(mismatches) < 20:
                    mismatches.append({
                        "seat": seat, "street": street, "hist_len": hl,
                        "py": py_key.hex(), "nat": nat_key.hex(), "ref": ref_key.hex(),
                        "hist_seq": [int(x) for x in act],
                    })
            a_pick = _pick_action(rng, legal)
            if 3 <= a_pick <= 10:
                street_raises += 1
            was_preflop = street == 0
            env = apply_action_np(env, a_pick)
            if was_preflop and (int(np.asarray(env.street)) >= 1 or bool(env.pending_deal)):
                # GROUND TRUTH = hist_len snapshot at the FIRST street boundary: every recorded entry
                # so far (INCLUDING a POKERBOT_CLOSE_CALLED_STREET closure-injected closing CHECK,
                # which closure.rs/np_engine record in hist like any action) originated preflop.
                # This matches fingerprint.rs::n_preflop_boundary, which replays the recorded bytes
                # through the raw core step — the semantics the trainer keyed the checkpoint with.
                # (The old per-applied-action counter missed the injected CHECK: off-by-one under
                # closure-ON on every called preflop raise.)
                true_n_preflop = min(int(np.asarray(env.hist_len)), 64)
            depth += 1
        if hands % 500 == 0:
            print(f"[progress] hands={hands} states={total} cov={cov} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    ok = fails == 0 and covered()
    if translate_on and translated < 100:
        ok = False  # the walk must actually exercise translation, not just pass vacuously
    print(f"\n=== FULL-HIST KEY PARITY (fullhist={ARGS.fullhist} presum={ARGS.presum} "
          f"translate={ARGS.translate}) ===")
    print(f"hands={hands} states={total} mismatches={fails} translated={translated}")
    print(f"coverage: {cov}")
    print(f"PARITY GATE: {'PASS' if ok else 'FAIL'}", flush=True)
    result = {"fullhist": ARGS.fullhist, "presum": ARGS.presum, "translate": ARGS.translate,
              "translated_states": translated, "hands": hands, "states": total,
              "mismatches": fails, "coverage": cov, "covered": covered(), "pass": bool(ok),
              "mismatch_samples": mismatches, "seed": ARGS.seed,
              "num_buckets": ARGS.num_buckets}
    if ARGS.out:
        os.makedirs(os.path.dirname(ARGS.out) or ".", exist_ok=True)
        with open(ARGS.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {ARGS.out}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
