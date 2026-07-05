#!/usr/bin/env python3
"""bp3 forced-response probe — the H-ii vs H-iii discriminator (bp3_gate_verdict.md §7).

QUESTION: bp3_prefhist_1 limps AA on the BTN (cum-regret CALL 15,374 vs raises <=931). Is that
because (H-ii) the in-model postflop game genuinely values the limp higher, or (H-iii) the field's
learned responses punish the raise only because the raise was never played (stale/starved rows)?

MODE rollout — Monte-Carlo EV of hero (BTN, seat 3, forced hole cards) at the folded-to-BTN node
under 5 arms, everything downstream (villain responses, later streets) played by the FROZEN
blueprint average strategy (TabularBlueprintAgent, sampling):
    own     : the model's own mixed policy at the node (limp-dominant for AA)
    limp    : forced CALL (action index 2)
    open325 : forced raise-to 3.25bb (index 8 = B1.5p; smallest sized open in the serve tree)
    open400 : forced raise-to 4.0bb (index 9 = B2.0p)
    jam     : forced 100bb all-in (index 10)
Deals are REAL game-flow walks: full 6-max hands with a rigged deck (hero's 2 deck slots hold the
forced hand class; remaining 50 cards random), discarded unless UTG/HJ/CO all fold (node reached).
CRN pairing: identical deck + identical per-seat agent seeds + identical all-in-settle RNG across
arms, so per-deal deltas pair exactly. All-in pots settle by side-pot-correct EQUITY
(equity_allin_settle, mean-preserving) to kill runout variance on the jam-shaped tree.

MODE rows — direct-key extraction (preflop_root_diag machinery, --hist-key space) of the response
rows a forced BTN open/jam hits: BB/SB vs BTN 2.5-4bb open (one aliased row), BB vs BTN jam, plus
the limp-economy comparison rows (BB/SB vs BTN limp). Reports per-row visit mass (strat_sum row
sums), the LATE-WINDOW (40k->45.5k) mass = current visitation (H-iii's starvation signature), and
avg + regret-matched-current response distributions.

Serve env (EMD-K160 + river-OCHS + preflop hist-key) is baked at import, so rollout mode sets env
from argv BEFORE any pokerbot import — run one process per shard:

  python scripts/blueprint2/bp3_forced_response_probe.py rollout \
      --blueprint runs/native_cfr/bp3_prefhist_1.nlcfr --hand AA \
      --start 0 --deals 4000 --out artifacts/_bp3_probe_AA.shard0.json

  python scripts/blueprint2/bp3_forced_response_probe.py rows \
      --blueprint runs/native_cfr/bp3_prefhist_1.nlcfr \
      --window-base artifacts/bp3_ladder/iter_040000.nlcfr \
      --out artifacts/_bp3_probe_rows.json

  python scripts/blueprint2/bp3_forced_response_probe.py merge \
      --out artifacts/bp3_forced_response_probe.json artifacts/_bp3_probe_AA.shard*.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_HERE, ".."))  # scripts/ (f32_regret_strategy_delta)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)  # aof_equity_grids lives at repo root (tournament.sim imports it bare)

# Serve-tree note: the engine's RFI menu at the folded-to-BTN node is CALL(limp) + RAISE(+2.25 =
# raise-to 3.25bb, action idx 8 = B1.5p) + RAISE(+3.0 = raise-to 4.0bb, idx 9 = B2.0p) + ALL_IN.
# The trainer's 2.5bb (B1.0p, idx 7) open does NOT exist in the serve tree (idx 7 nearest-maps to
# the 3.25bb raise) -- so the forced sized opens are 3.25bb and 4.0bb, exactly the sizes every
# serve-path eval (the 8k gate included) realizes. Verified: the BB's serve fingerprint after a
# forced 3.25bb open == trainer row (0,1,2,5,0,1)+hist(0,1,4); after 4.0bb == (0,1,2,4,0,1); after
# jam == (0,6,3,0,0,1) -- all present in the checkpoint.
ARMS = ("own", "limp", "open325", "open400", "jam")
ARM_IDX = {"own": None, "limp": 2, "open325": 8, "open400": 9, "jam": 10}
PROGRESS_LOG = os.path.join(_ROOT, "artifacts", "bp3_probe_progress.log")


def _plog(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------------------------
# rollout mode
# ---------------------------------------------------------------------------------------------
def _configure_env(args) -> None:
    """Bake the serve env for the checkpoint's key space BEFORE any pokerbot import.

    Defaults reproduce the bp3 regime exactly (EMD-K160 + river-OCHS + preflop hist-key).
    blueprint-v4 Stage 0 (G5, design doc §7.1.4) passes --key-mode fullhist and, for the
    made-hand arms, --postflop-mode fast --num-buckets 200|1000."""
    os.environ["POKERBOT_BETTING_ABSTRACTION"] = "1"
    os.environ["POKERBOT_POSTFLOP_BUCKETS"] = "1"
    os.environ["POKERBOT_PREFLOP_LOSSLESS"] = "1"
    os.environ["POKERBOT_POSTFLOP_NUM_BUCKETS"] = str(args.num_buckets)
    # Measure per-street fingerprint COVERAGE on the rollout serve path (lbr.make_blueprint_policy reads
    # this env AT IMPORT, so it must be set before any pokerbot import). CRITICAL for the G5 verdict:
    # ``adapter_fallback_rate`` does NOT detect a fingerprint MISS — a miss returns a *valid* uniform-legal
    # distribution, so it is silently absorbed and never increments fallback_count. Under the vast fullhist
    # key space the postflop TAIL misses far more than the summary control, biasing EV(open)-EV(limp) in an
    # arm-dependent way. Enabling coverage records hits/misses per street so the driver can gate on the real
    # skew signal. Recording only (a bool check per lookup) — does NOT change EV/CRN.
    os.environ["POKERBOT_BP_COVERAGE"] = "1"
    # blueprint-v4 M-RR serve twin (R2 probe-fidelity port, lbr._MRR_ON — read AT IMPORT like the
    # coverage env): a fullhist+M-RR-trained arm MUST be probed under the SAME thinned postflop menus
    # it trained with, else a single miss's uniform fallback walks the playout off the trained action
    # tree and every later fullhist key misses too (cascade). Summary arms trained the full menu —
    # leave the flag off for them (byte-identical serve).
    if getattr(args, "mrr", False):
        os.environ["POKERBOT_MRR"] = "1"
    else:
        os.environ.pop("POKERBOT_MRR", None)
    if args.postflop_mode == "emd":
        os.environ["POKERBOT_POSTFLOP_MODE"] = "emd"
        os.environ["POKERBOT_POSTFLOP_EMD_FROZEN"] = os.path.abspath(args.emd_frozen)
        os.environ["POKERBOT_EMD_FROZEN_MIRROR_FALLBACK"] = "1"
        os.environ["POKERBOT_RIVER_OCHS"] = os.path.abspath(args.river_ochs)
    else:  # made-hand ("fast") — the Stage-0 K200/K1000 attribution arms
        os.environ["POKERBOT_POSTFLOP_MODE"] = "fast"
        os.environ["POKERBOT_ALLOW_MADE_HAND_BUCKETS"] = "1"
        os.environ.pop("POKERBOT_RIVER_OCHS", None)
        os.environ.pop("POKERBOT_POSTFLOP_EMD_FROZEN", None)
    if args.key_mode == "prefhist":
        os.environ["POKERBOT_PREFLOP_HIST_KEY"] = "1"
        os.environ.pop("POKERBOT_FULLHIST_KEY", None)
    elif args.key_mode == "fullhist":
        os.environ["POKERBOT_FULLHIST_KEY"] = "1"
        os.environ.pop("POKERBOT_PREFLOP_HIST_KEY", None)
    else:  # summary — no history key extension
        os.environ.pop("POKERBOT_PREFLOP_HIST_KEY", None)
        os.environ.pop("POKERBOT_FULLHIST_KEY", None)


class AbortHand(Exception):
    """UTG/HJ/CO did not fold -> the folded-to-BTN node was not reached; deal discarded."""


class _IndexedSnap:
    """mmap-backed ``{fingerprint(bytes) -> avg-strategy row}`` over an NLCFR001 blob + the sorted
    (A,B,O) offset index built by ``bpv4_nlcfr_index.py``.

    WHY: the blueprint-v4 Stage-0 CLOUD fullhist arm (bpv4_h200 @20k) has 347.7M strat rows / 55.3 GB
    — the historical full in-RAM snapshot needs >100 GB and cannot run on the 31 GB local box. This
    class serves the SAME mapping contract the in-RAM dict serves (``lbr.make_blueprint_policy`` only
    calls ``snapshot.get(key)``; the probe log only calls ``len()``), with each lookup a binary search
    over two mmapped uint64 arrays (numeric order == byte-lexicographic order because the build stores
    the big-endian interpretation) + one 88-byte row read from the mmapped blob. Values reproduce
    ``avg_strategy`` EXACTLY (row/sum; uniform 1/11 on an all-zero row) so hit/miss + coverage
    semantics are byte-identical to the in-RAM path. Page cache is shared across shard processes.
    A bounded per-process dict caches hot rows (preflop rows repeat every deal).
    """

    _MISS = object()
    _CACHE_CAP = 500_000

    def __init__(self, index_dir: str):
        import mmap as _mmap

        import numpy as np

        meta = json.load(open(os.path.join(index_dir, "meta.json"), encoding="utf-8"))
        self._n = int(meta["n_strat"])
        self._np = np
        self._A = np.memmap(os.path.join(index_dir, "A.bin"), dtype="<u8", mode="r")
        self._B = np.memmap(os.path.join(index_dir, "B.bin"), dtype="<u8", mode="r")
        self._O = np.memmap(os.path.join(index_dir, "O.bin"), dtype="<u8", mode="r")
        assert len(self._A) == self._n == len(self._B) == len(self._O), "index length mismatch"
        f = open(meta["blob"], "rb")
        self._mm = _mmap.mmap(f.fileno(), 0, access=_mmap.ACCESS_READ)
        self._cache: dict[bytes, object] = {}
        self.iters = int(meta["iters"])

    def __len__(self) -> int:
        return self._n

    def get(self, key, default=None):
        v = self._cache.get(key, self._MISS)
        if v is not self._MISS:
            return default if v is None else v
        np = self._np
        # np.uint64 casts are LOAD-BEARING: a plain Python int scalar sends numpy's NEP50 promotion
        # down an object-comparison path and searchsorted degrades to a FULL 348M-element scan
        # (~0.9 s/call, measured via cProfile); the uint64 scalar keeps it a true binary search (~30 µs).
        a = np.uint64(int.from_bytes(key[:8], "big"))
        b = np.uint64(int.from_bytes(key[8:16], "big"))
        lo = int(np.searchsorted(self._A, a, "left"))
        hi = int(np.searchsorted(self._A, a, "right"))
        out = None
        if hi > lo:
            j = lo + int(np.searchsorted(self._B[lo:hi], b, "left"))
            if j < hi and int(self._B[j]) == b:
                off = int(self._O[j])
                row = struct.unpack_from("<11d", self._mm, off + 16)
                tot = sum(row)
                # EXACT avg_strategy semantics (f32_regret_strategy_delta.avg_strategy)
                if tot > 0.0:
                    out = np.asarray([x / tot for x in row], dtype=np.float64)
                else:
                    out = np.full(11, 1.0 / 11.0, dtype=np.float64)
        if len(self._cache) < self._CACHE_CAP:
            self._cache[key] = out
        return default if out is None else out


def _rollout_main(args) -> int:
    _configure_env(args)
    import random

    import numpy as np  # noqa: F401

    from f32_regret_strategy_delta import avg_strategy, load_checkpoint
    from pokerbot.engine.cards import Card, Rank, Suit
    from pokerbot.engine.game import ActionType, GameState, play_hand
    from pokerbot.eval.blueprint_ranking import TabularBlueprintAgent
    from pokerbot.tournament.rollout_leaf import equity_allin_settle
    from pokerbot.training.self_play import _index_to_action

    import numpy as np

    tag = f"{args.hand}/s{args.start}"
    if args.snapshot_index:
        # blueprint-v4 Stage-0 CLOUD arm: 347.7M rows / 55.3 GB blob — mmap-backed lookup (the full
        # in-RAM snapshot needs >100 GB). Values/coverage semantics identical to the in-RAM path.
        _plog(f"{tag} mmap-index {args.snapshot_index} over {args.blueprint}")
        snap = _IndexedSnap(args.snapshot_index)
    else:
        _plog(f"{tag} loading {args.blueprint}")
        ck = load_checkpoint(args.blueprint)
        snap = {bytes(k): np.asarray(avg_strategy(row), dtype=np.float64)
                for k, row in ck["strat"].items()}
        del ck
    _plog(f"{tag} snapshot {len(snap):,} infosets; env {args.postflop_mode}/{args.num_buckets} "
          f"key-mode={args.key_mode}")

    agents = {s: TabularBlueprintAgent(snap, random.Random(1000 + s)) for s in range(6)}

    # --- hero hole-card rigging: seat 3 (BTN under button_seat=3) draws deck slots 6:8 ------------
    hand = args.hand.strip()
    def rig_deck(state, rng) -> None:
        cards = state._deck._cards
        if len(hand) == 2 and hand[0] == hand[1]:  # pair, e.g. AA
            rank = Rank(2 + "23456789TJQKA".index(hand[0]))
            suits = rng.sample(list(Suit), 2)
            want = [Card(rank, suits[0]), Card(rank, suits[1])]
        else:  # suited/offsuit two-ranker, e.g. K9s
            r1 = Rank(2 + "23456789TJQKA".index(hand[0]))
            r2 = Rank(2 + "23456789TJQKA".index(hand[1]))
            if hand[2] == "s":
                s1 = s2 = rng.choice(list(Suit))
            else:
                s1, s2 = rng.sample(list(Suit), 2)
            want = [Card(r1, s1), Card(r2, s2)]
        for slot, c in zip((6, 7), want):
            j = cards.index(c)
            cards[slot], cards[j] = cards[j], cards[slot]

    def play_arm(h: int, arm: str):
        """One arm of one deal. Returns record dict or None if node unreached."""
        rng = random.Random(args.seed * 77 + h)
        state = GameState.new_hand(rng=rng)  # button_seat=3 -> seats: 0 UTG .. 3 BTN, 4 SB, 5 BB
        rig_deck(state, rng)
        for s, ag in agents.items():
            ag.reseed(args.seed * 1_000_003 + h * 31 + s)  # arm-independent -> CRN
        settle_seed = 987_001 + h
        forced_idx = ARM_IDX[arm]
        st = {"hero_acted": False, "hero_first": None}

        def hero_policy(state):
            if not st["hero_acted"]:
                st["hero_acted"] = True
                hist = state._history
                assert len(hist) == 3 and all(
                    a.action_type == ActionType.FOLD for _s, a in hist), "node guard"
                a = agents[3](state) if forced_idx is None else _index_to_action(state, 3, forced_idx)
                st["hero_first"] = (a.action_type.name, round(getattr(a, "amount_bb", 0.0) or 0.0, 2))
                return a
            return agents[3](state)

        def early_policy(seat):
            ag = agents[seat]
            def pol(state):
                a = ag(state)
                if not st["hero_acted"] and a.action_type != ActionType.FOLD:
                    raise AbortHand()
                return a
            return pol

        policies = {0: early_policy(0), 1: early_policy(1), 2: early_policy(2),
                    3: hero_policy, 4: agents[4], 5: agents[5]}
        # blueprint-v4 v3 G5 INSTRUMENT REBUILD (--postflop-field checkcall; amendment doc section 5,
        # Stage-0 verdict section 7-B): replace EVERY seat's POSTFLOP play (hero included -- his
        # forced node action is preflop, protected by the node guard) with a SYMMETRIC FIXED
        # check-preferring/call-otherwise continuation field. Rationale: at H200@20k the blueprint
        # postflop lookup missed ~92% (probe) / ~72% (on-policy) -- the "frozen field" the EV
        # rollout measured was ~uniform-random postflop noise, so G5's EV comparison was
        # coverage-BLOCKED. Under a fixed symmetric field the preflop response game (100% key
        # coverage on BOTH arms) stays trained while postflop collapses to pure equity
        # realization -- the SAME field for both arms and for every forced arm, so the
        # EV(open)-EV(limp) contrast is well-defined at ANY postflop coverage. What it gives up:
        # postflop SKILL differences (positional betting leverage) are erased, so G5-checkcall is
        # a LOWER-BOUND instrument for the open's value (opens realize extra EV postflop via
        # initiative; limps do not). Default "blueprint" = byte-identical legacy behavior.
        if args.postflop_field == "checkcall":
            def _postflop_fixed(seat, pol):
                def wrapped(state):
                    if len(state.board) > 0:
                        legal = state.legal_actions(seat)
                        for want in (ActionType.CHECK, ActionType.CALL):
                            for a in legal:
                                if a.action_type == want:
                                    return a
                        return legal[0]  # defensive; CHECK/CALL is legal at every postflop node
                    return pol(state)
                return wrapped
            policies = {s: _postflop_fixed(s, pol) for s, pol in policies.items()}
        settle = lambda s, nb: equity_allin_settle(  # noqa: E731
            s, nb, n_runouts=args.runouts, rng=random.Random(settle_seed))
        try:
            final, deltas = play_hand(policies, state=state, allin_settle=settle)
        except AbortHand:
            return None
        if not st["hero_acted"]:  # e.g. everyone folded to BTN is guaranteed reached; safety only
            return None
        # villain first responses AFTER hero's node action (preflop only)
        resp = {}
        hist = list(final.action_history_with_street())
        seen_hero = False
        for seat, a, street in hist:
            if street.name != "PREFLOP":
                break
            if seat == 3 and not seen_hero:
                seen_hero = True
                continue
            if seen_hero and seat in (4, 5) and seat not in resp:
                nm = a.action_type.name
                if nm in ("RAISE", "BET") and getattr(a, "amount_bb", 0.0) >= 90.0:
                    nm = "JAM"
                resp[seat] = nm
        return {
            "delta": float(deltas.get(3, 0.0)),
            "hero_first": st["hero_first"],
            "sb": resp.get(4), "bb": resp.get(5),
            "end_street": len(final.board),
        }

    # --- deal loop: h indexes candidate deals; keep the first --deals REACHED ones ---------------
    out = {arm: [] for arm in ARMS}
    meta = {"hero_first_own": {}, "resp": {arm: {"sb": {}, "bb": {}} for arm in ARMS},
            "end_street": {arm: {} for arm in ARMS}}
    reached = 0
    tried = 0
    h = args.start
    t0 = time.time()
    last_hb = t0
    while reached < args.deals:
        tried += 1
        rec0 = play_arm(h, "own")
        if rec0 is None:
            h += 1
            continue
        recs = {"own": rec0}
        ok = True
        for arm in ARMS[1:]:
            r = play_arm(h, arm)
            if r is None:  # cannot happen (prefix identical) but stay safe
                ok = False
                break
            recs[arm] = r
        h += 1
        if not ok:
            continue
        reached += 1
        for arm in ARMS:
            r = recs[arm]
            out[arm].append(r["delta"])
            for side in ("sb", "bb"):
                if r[side]:
                    d = meta["resp"][arm][side]
                    d[r[side]] = d.get(r[side], 0) + 1
            es = meta["end_street"][arm]
            es[str(r["end_street"])] = es.get(str(r["end_street"]), 0) + 1
        hf = str(rec0["hero_first"])
        meta["hero_first_own"][hf] = meta["hero_first_own"].get(hf, 0) + 1
        now = time.time()
        if now - last_hb > 30:
            last_hb = now
            pct = 100.0 * reached / args.deals
            eta = (now - t0) / max(1, reached) * (args.deals - reached)
            _plog(f"{tag} {reached}/{args.deals} ({pct:.1f}%)  reach-rate "
                  f"{reached/max(1,tried):.2f}  eta {eta/60:.1f}m")

    fb = sum(a.fallback_count for a in agents.values())
    dc = sum(a.decision_count for a in agents.values())
    # Per-street fingerprint coverage on the serve path (POKERBOT_BP_COVERAGE, set in _configure_env).
    # This is the ONLY signal that catches the fullhist postflop-tail MISS that adapter_fallback_rate
    # cannot (a miss returns a valid uniform-legal distribution -> no fallback increment). The driver
    # gates the G5 verdict on the postflop (flop/turn/river) miss-rate.
    from pokerbot.eval import lbr as _lbr
    bp_cov = _lbr.bp_coverage_report()
    res = {
        "mode": "rollout", "hand": hand, "blueprint": args.blueprint, "seed": args.seed,
        "start": args.start, "deals": reached, "tried": tried,
        "reach_rate": reached / max(1, tried), "runouts": args.runouts,
        "postflop_field": args.postflop_field,
        "adapter_fallback_rate": fb / max(1, dc),
        "bp_coverage": bp_cov,
        "deltas_bb": out, "meta": meta,
        "wall_s": round(time.time() - t0, 1),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f)
    _pf_h = sum(bp_cov.get(s, {}).get("hits", 0) for s in ("flop", "turn", "river"))
    _pf_m = sum(bp_cov.get(s, {}).get("misses", 0) for s in ("flop", "turn", "river"))
    _pf_miss = _pf_m / max(1, _pf_h + _pf_m)
    _plog(f"{tag} DONE {reached} deals in {res['wall_s']:.0f}s -> {args.out} "
          f"(fallback {res['adapter_fallback_rate']:.4f}; postflop miss-rate {_pf_miss:.4f} "
          f"[{_pf_m}/{_pf_h + _pf_m}]; per-street {bp_cov})")
    return 0


# ---------------------------------------------------------------------------------------------
# rows mode — response-row health straight off the checkpoint (no pokerbot imports)
# ---------------------------------------------------------------------------------------------
def _rows_main(args) -> int:
    import preflop_root_diag as prd
    from f32_regret_strategy_delta import load_checkpoint

    prd.HIST_KEY = True
    POS = prd.POS
    FOLD, CHECK, CALL, ALL_IN = 0, 1, 2, 10
    RAISES = (7, 8, 9)
    LEGAL_VS_BET = (FOLD, CALL) + RAISES + (ALL_IN,)
    LEGAL_VS_CHECK = (CHECK,) + RAISES + (ALL_IN,)  # facing a limp, BB may check
    # node: label -> (pos_enum, betting6, legal, hist3)
    nodes = {
        # what a FORCED BTN 3.25bb open hits (== the trainer's 2.5/3.25bb-open response row):
        "SB_vs_BTN_open": (POS["SB"], bytes((0, 1, 2, 5, 0, 1)), LEGAL_VS_BET, bytes((0, 1, POS["BTN"] + 1))),
        "BB_vs_BTN_open": (POS["BB"], bytes((0, 1, 2, 5, 0, 1)), LEGAL_VS_BET, bytes((0, 1, POS["BTN"] + 1))),
        # what a FORCED BTN 4.0bb open hits (spr bucket drops 5 -> 4; serve-key verified):
        "SB_vs_BTN_open4": (POS["SB"], bytes((0, 1, 2, 4, 0, 1)), LEGAL_VS_BET, bytes((0, 1, POS["BTN"] + 1))),
        "BB_vs_BTN_open4": (POS["BB"], bytes((0, 1, 2, 4, 0, 1)), LEGAL_VS_BET, bytes((0, 1, POS["BTN"] + 1))),
        # what a FORCED BTN jam hits: pot 101.5 -> pot_b 6; tc/pot .975 -> b3 (SB .985 -> b3);
        # spr eff/pot ~ .97 -> b0:
        "SB_vs_BTN_jam": (POS["SB"], bytes((0, 6, 3, 0, 0, 1)), (FOLD, CALL, ALL_IN), bytes((0, 1, POS["BTN"] + 1))),
        "BB_vs_BTN_jam": (POS["BB"], bytes((0, 6, 3, 0, 0, 1)), (FOLD, CALL, ALL_IN), bytes((0, 1, POS["BTN"] + 1))),
        # the LIMP-economy comparison rows (what the model's own line hits):
        "SB_vs_BTN_limp": (POS["SB"], bytes((0, 0, 1, 5, 0, 0)), (FOLD, CALL) + RAISES + (ALL_IN,), bytes((1, 0, 0))),
        "BB_vs_BTN_limp": (POS["BB"], bytes((0, 0, 0, 5, 0, 0)), LEGAL_VS_CHECK, bytes((1, 0, 0))),
        # hero RFI row itself (context):
        "BTN_rfi": (POS["BTN"], bytes((0, 0, 2, 5, 1, 0)), (FOLD, CALL) + RAISES + (ALL_IN,), bytes((0, 0, 0))),
        # BB vs a BTN open after SB flat? (secondary; skip)
    }
    detail = [prd.CLASS_BY_NAME[h] for h in
              ("AA", "KK", "QQ", "JJ", "TT", "99", "AKs", "AKo", "AQs", "KQs", "T9s",
               "K9s", "A5s", "76s", "J9o", "QTo", "72o", "32o")]

    ck = load_checkpoint(args.blueprint)
    base = load_checkpoint(args.window_base) if args.window_base else None
    _plog(f"rows: final iters={ck['iters_done']}  window_base="
          f"{base['iters_done'] if base else None}")

    def row_report(label, pe, betting, legal, hist):
        strat, cum = ck["strat"], ck["cum"]
        per_class = {}
        avg_by, cur_by, win_by = {}, {}, {}
        tot_mass = tot_win_mass = tot_reg = 0.0
        n_present = 0
        for cls in range(169):
            k = prd.key_for(pe, cls, betting, hist)
            srow = strat.get(k)
            crow = cum.get(k)
            if srow is None and crow is None:
                continue
            n_present += 1
            mass = sum(srow) if srow else 0.0
            tot_mass += mass
            if srow:
                avg_by[cls] = prd.renorm(list(srow), legal)
            if crow:
                cur_by[cls] = prd.regret_match(list(crow), legal)
                tot_reg += sum(abs(x) for x in crow)
            wmass = 0.0
            if base is not None and srow:
                b = base["strat"].get(k)
                w = [max(0.0, srow[a] - (b[a] if b else 0.0)) for a in range(11)]
                wmass = sum(w)
                tot_win_mass += wmass
                win_by[cls] = prd.renorm(w, legal)
            if cls in detail:
                d = {"strat_sum_total": round(mass, 1),
                     "window_mass_40k_45k": round(wmass, 1)}
                if srow:
                    d["avg"] = {prd.ACTION_NAMES[a]: round(avg_by[cls][a], 4) for a in legal}
                if crow:
                    d["cum_regret"] = [round(crow[a], 1) for a in legal]
                    d["current"] = {prd.ACTION_NAMES[a]: round(cur_by[cls][a], 4) for a in legal}
                if cls in win_by:
                    d["window_current"] = {prd.ACTION_NAMES[a]: round(win_by[cls][a], 4)
                                           for a in legal}
                per_class[prd.class_name(cls)] = d
        return {
            "classes_present": n_present,
            "strat_sum_total_mass": round(tot_mass, 1),
            "window_mass_40k_45k_total": round(tot_win_mass, 1),
            "cum_regret_total_abs": round(tot_reg, 1),
            "avg_agg": prd.agg(avg_by, legal),
            "current_agg": prd.agg(cur_by, legal),
            "window_current_agg": prd.agg(win_by, legal) if win_by else None,
            "detail": per_class,
        }

    out = {"mode": "rows", "blueprint": args.blueprint, "iters": ck["iters_done"],
           "window_base": args.window_base,
           "nodes": {}}
    for label, (pe, betting, legal, hist) in nodes.items():
        out["nodes"][label] = row_report(label, pe, betting, legal, hist)
        a = out["nodes"][label]
        _plog(f"rows: {label}: {a['classes_present']} classes, mass {a['strat_sum_total_mass']:.0f}, "
              f"window {a['window_mass_40k_45k_total']:.0f}")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    _plog(f"rows -> {args.out}")
    return 0


# ---------------------------------------------------------------------------------------------
# merge mode — combine rollout shards, paired stats
# ---------------------------------------------------------------------------------------------
def _merge_main(args) -> int:
    import numpy as np

    shards = [json.load(open(p, encoding="utf-8")) for p in args.inputs]
    by_hand = {}
    for s in shards:
        by_hand.setdefault(s["hand"], []).append(s)
    out = {"mode": "merged", "hands": {}}
    for hand, ss in by_hand.items():
        deltas = {arm: np.concatenate([np.asarray(x["deltas_bb"][arm]) for x in ss])
                  for arm in ARMS}
        n = len(deltas["own"])
        stats = {}
        for arm in ARMS:
            d = deltas[arm]
            stats[arm] = {"n": int(n), "ev_bb100": float(d.mean() * 100),
                          "se_bb100": float(d.std(ddof=1) / math.sqrt(n) * 100)}
        paired = {}
        for arm in ("own", "open325", "open400", "jam"):
            diff = deltas[arm] - deltas["limp"]
            paired[f"{arm}_minus_limp"] = {
                "delta_bb100": float(diff.mean() * 100),
                "se_bb100": float(diff.std(ddof=1) / math.sqrt(n) * 100),
                "t": float(diff.mean() / (diff.std(ddof=1) / math.sqrt(n)) if diff.std(ddof=1) > 0 else 0.0),
            }
        meta = {"hero_first_own": {}, "resp": {a: {"sb": {}, "bb": {}} for a in ARMS},
                "end_street": {a: {} for a in ARMS}}
        tried = sum(x["tried"] for x in ss)
        for x in ss:
            m = x["meta"]
            for k, v in m["hero_first_own"].items():
                meta["hero_first_own"][k] = meta["hero_first_own"].get(k, 0) + v
            for a in ARMS:
                for side in ("sb", "bb"):
                    for k, v in m["resp"][a][side].items():
                        meta["resp"][a][side][k] = meta["resp"][a][side].get(k, 0) + v
                for k, v in m["end_street"][a].items():
                    meta["end_street"][a][k] = meta["end_street"][a].get(k, 0) + v
        # Aggregate per-street fingerprint coverage across this hand's shards so the A1/A2 verdict step
        # can compare postflop miss-rate between the fullhist arm and the summary control (a large
        # DIFFERENTIAL is the biased-verdict signal). ``adapter_fallback_rate`` cannot see this.
        cov_agg: dict[str, list[int]] = {}
        for x in ss:
            for st, c in (x.get("bp_coverage") or {}).items():
                a = cov_agg.setdefault(st, [0, 0])
                a[0] += int(c.get("hits", 0))
                a[1] += int(c.get("misses", 0))
        bp_coverage = {st: {"hits": h, "misses": m,
                            "hit_rate": (h / (h + m) if (h + m) else 0.0)}
                       for st, (h, m) in cov_agg.items()}
        pf_h = sum(cov_agg.get(st, [0, 0])[0] for st in ("flop", "turn", "river"))
        pf_m = sum(cov_agg.get(st, [0, 0])[1] for st in ("flop", "turn", "river"))
        out["hands"][hand] = {"n_deals": int(n), "tried": tried,
                              "reach_rate": n / max(1, tried),
                              "arms": stats, "paired_vs_limp": paired, "meta": meta,
                              "fallback_rates": [x["adapter_fallback_rate"] for x in ss],
                              "bp_coverage": bp_coverage,
                              "postflop_miss_rate": pf_m / max(1, pf_h + pf_m)}
        _plog(f"merge {hand}: n={n}  " + "  ".join(
            f"{a}={stats[a]['ev_bb100']:+.1f}±{stats[a]['se_bb100']:.1f}" for a in ARMS))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    _plog(f"merge -> {args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    r = sub.add_parser("rollout")
    r.add_argument("--blueprint", required=True)
    r.add_argument("--hand", required=True, help="AA | K9s | ...")
    r.add_argument("--deals", type=int, default=2000, help="REACHED deals to keep")
    r.add_argument("--start", type=int, default=0, help="first candidate-deal index (shard offset)")
    r.add_argument("--seed", type=int, default=13)
    r.add_argument("--runouts", type=int, default=128)
    r.add_argument("--emd-frozen", default="data/abstraction/emd_frozen_flop_turn_160_r400.emdfrzn")
    r.add_argument("--river-ochs", default="data/abstraction/river_ochs_160.rochs")
    # blueprint-v4 Stage-0 env adaptation (defaults reproduce the bp3 regime byte-for-byte):
    r.add_argument("--key-mode", choices=("prefhist", "fullhist", "summary"), default="prefhist")
    r.add_argument("--postflop-mode", choices=("emd", "fast"), default="emd")
    r.add_argument("--num-buckets", type=int, default=160)
    r.add_argument("--mrr", action="store_true",
                   help="serve under the M-RR thinned postflop menus (POKERBOT_MRR=1, the trainer's "
                        "abstracted_legal_mrr keep-lists mirrored in lbr.make_blueprint_policy) — "
                        "REQUIRED for fullhist+M-RR-trained arms (train/probe menu skew otherwise). "
                        "Default off = byte-identical bp3/summary behavior.")
    r.add_argument("--postflop-field", choices=("blueprint", "checkcall"), default="blueprint",
                   help="postflop continuation field for the EV rollout. 'blueprint' (default) = the "
                        "legacy frozen-blueprint field (byte-identical behavior; coverage-gated by "
                        "--max-postflop-miss). 'checkcall' = the v3 G5 instrument: EVERY seat plays "
                        "a SYMMETRIC fixed check/call policy postflop, so the EV(open)-EV(limp) "
                        "contrast measures the TRAINED preflop response economics under pure equity "
                        "realization and stays well-defined at any postflop coverage (the fix for "
                        "the Stage-0 G5 coverage-BLOCK; amendment doc section 5).")
    r.add_argument("--snapshot-index", default=None,
                   help="dir built by bpv4_nlcfr_index.py — mmap-backed snapshot lookup instead of the "
                        "full in-RAM load (REQUIRED for the 55 GB cloud fullhist blobs). Default None "
                        "= historical in-RAM path, byte-identical.")
    r.add_argument("--out", required=True)

    w = sub.add_parser("rows")
    w.add_argument("--blueprint", required=True)
    w.add_argument("--window-base", default=None)
    w.add_argument("--out", required=True)

    m = sub.add_parser("merge")
    m.add_argument("inputs", nargs="+")
    m.add_argument("--out", required=True)

    args = ap.parse_args()
    if args.mode == "rollout":
        return _rollout_main(args)
    if args.mode == "rows":
        return _rows_main(args)
    return _merge_main(args)


if __name__ == "__main__":
    sys.exit(main())
