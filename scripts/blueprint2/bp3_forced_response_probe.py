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
# blueprint-v4 v3 root-cause coda: the BB-defend-vs-fold node target. Instead of forcing the BTN's
# first action (folded-to-BTN, seat 3), we RIG THE BIG BLIND (seat 5), let UTG/HJ/CO fold + the BTN
# OPEN (forced to a fixed modal open, default idx 8 = 3.25bb raise-to, the smallest sized open present
# in the serve tree — its BB response row (0,1,2,5,0,1)+hist(0,1,BTN+1) is the trainer row the `rows`
# mode already verified present) + the SB FOLD, then force the BB to CALL (defend) vs FOLD and measure
# the BB's paired EV(defend)-EV(fold). ARMS_BB[0] ("fold") is the CRN baseline (EV of the fold = the
# BB's already-posted -1bb, deterministic; kept as an arm so the paired structure/merge is reused
# verbatim). Under --postflop-field checkcall this is a PURE-EQUITY-REALIZATION defend test.
ARMS_BB = ("fold", "defend")
ARM_BB_IDX = {"fold": 0, "defend": 2}  # engine action ids: FOLD=0, CALL=2
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
    # blueprint-v4 arm-3 serve twins (read AT IMPORT by lbr, like POKERBOT_MRR). Explicit set/pop keeps
    # the probe env HERMETIC: a baseline shard must not inherit a stale arm-3 env from the shell.
    for flag, env_name in (("pre_keep", "POKERBOT_PRE_KEEP"),
                           ("pre_rr_keep", "POKERBOT_PRE_RR_KEEP"),
                           ("post_keep", "POKERBOT_POST_KEEP"),
                           ("flop_rr_keep", "POKERBOT_FLOP_RR_KEEP")):
        val = getattr(args, flag, None)
        if val:
            os.environ[env_name] = str(val)
        else:
            os.environ.pop(env_name, None)
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
    # Presum postflop key (POKERBOT_PRESUM_KEY, composes with fullhist — jax_weva_encode.py:62, read
    # AT IMPORT). Explicit set/pop like the arm-3 twins above: a presum-trained snapshot (e.g.
    # bpv4_presum_1) served WITHOUT this flag misses ~100% of postflop keys (fullhist-exact keys the
    # presum table never wrote) and the whole postflop field silently degrades to uniform — the exact
    # skew that made the first Stage-0' rerun unreproducible (the original §37 shards inherited the
    # flag from their launcher shell; nothing in the CLI recorded it).
    if getattr(args, "presum", False):
        os.environ["POKERBOT_PRESUM_KEY"] = "1"
    else:
        os.environ.pop("POKERBOT_PRESUM_KEY", None)


class AbortHand(Exception):
    """UTG/HJ/CO did not fold -> the folded-to-BTN node was not reached; deal discarded."""


class SettleNow(Exception):
    """v5.1 Stage-0' idealized-realization re-serve (blueprint_v5_pluribus_grade_plan.md §11.4): raised
    by the ``--postflop-field idealized`` wrapper at the FIRST turn/river decision whose blueprint
    fingerprint MISSES (uniform-legal fallback). The hand is then settled at that state by
    ``equity_allin_settle`` (side-pot-correct check-down equity, mean-preserving) instead of continuing
    under uniform noise — replacing the node's VALUE realization while holding upstream reach (and the
    CRN pairing) byte-identical. Carries the state at the decision point (the sampled action is
    discarded UNAPPLIED, so ``state`` is exactly the pre-action node)."""

    def __init__(self, state):
        super().__init__()
        self.state = state


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


class _MmapSortedSnap:
    """Serve current-iterate probs from a SORTED NLSNAP01 file by DIRECT mmap binary search — no
    separate A/B/O index. The snapshot is globally key-sorted (gather_sorted_all), so binary search
    over the mmapped fixed-width rows suffices. A 3.9B-row snapshot's index would be ~94 GB + ~94 GB
    temp; this needs ZERO extra disk. Same .get(key) contract as _IndexedSnap / _IndexedSnapFile
    (np.float64[11] or None). Row = 16-byte key + 11 f32 probs (LE)."""

    _MISS = object()
    _CACHE_CAP = 2_000_000
    ROW = 60
    HDR = 24

    def __init__(self, path: str):
        import mmap as _mmap

        import numpy as np
        f = open(path, "rb")
        hdr = f.read(self.HDR)
        assert hdr[:8] == b"NLSNAP01", f"bad magic {hdr[:8]!r}"
        self.iters = struct.unpack_from("<Q", hdr, 8)[0]
        self._n = struct.unpack_from("<Q", hdr, 16)[0]
        self._mm = _mmap.mmap(f.fileno(), 0, access=_mmap.ACCESS_READ)
        self._np = np
        self._cache: dict[bytes, object] = {}

    def __len__(self) -> int:
        return self._n

    def get(self, key, default=None):
        c = self._cache.get(key, self._MISS)
        if c is not self._MISS:
            return default if c is None else c
        mm, ROW, HDR = self._mm, self.ROW, self.HDR
        lo, hi, out = 0, self._n, None
        while lo < hi:
            mid = (lo + hi) >> 1
            off = HDR + mid * ROW
            k = mm[off:off + 16]
            if k == key:                       # lexicographic == the serializer's BE-u128 order
                row = struct.unpack_from("<11f", mm, off + 16)
                tot = sum(row)
                out = (self._np.asarray([x / tot for x in row], dtype=self._np.float64)
                       if tot > 0.0 else self._np.full(11, 1.0 / 11.0, dtype=self._np.float64))
                break
            if k < key:
                lo = mid + 1
            else:
                hi = mid
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
    if getattr(args, "snapshot_nlsnap_file", None):
        # SORTED NLSNAP01 served by direct mmap binary search — no index (a 3.9B-row snapshot's index
        # is ~94 GB and would not fit). Trained current-iterate incl POSTFLOP.
        _plog(f"{tag} NLSNAP mmap-sorted (no index) {args.snapshot_nlsnap_file}")
        snap = _MmapSortedSnap(args.snapshot_nlsnap_file)
    elif getattr(args, "snapshot_nlsnap_index", None):
        # TRAINED-POSTFLOP field source: an NLSNAP01 current-iterate snapshot (16B key + 11 f32 probs).
        # Under snapshot_avg the .nlcfr STRAT section is PREFLOP-ONLY, so the trained POSTFLOP strategy
        # lives only in the .nlsnap current-iterate. Same .get() contract as _IndexedSnap (reused verbatim
        # from bpv4_onpolicy_coverage_snap). Used by the root-cause probe (trained vs check-call fields).
        from bpv4_onpolicy_coverage_snap import _IndexedSnapFile
        _plog(f"{tag} NLSNAP mmap-index {args.snapshot_nlsnap_index}")
        snap = _IndexedSnapFile(args.snapshot_nlsnap_index)
    elif args.snapshot_index:
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

    # node target: legacy folded-to-BTN (hero=seat 3) vs the BB-defend coda (hero=seat 5).
    node_target = getattr(args, "node_target", "btn_open")
    arm_set = ARMS_BB if node_target == "bb_defend" else ARMS
    # Which deck slots the rigged hand occupies. new_hand deals hole cards in seat order 0..5, each
    # taking 2 slots (deal_hole_cards loops p in self.players by seat): seat s -> slots (2s, 2s+1).
    # seat 3 (BTN) -> (6,7); seat 5 (BB) -> (10,11). (The legacy btn_open path hard-coded (6,7).)
    hero_seat = 5 if node_target == "bb_defend" else 3
    rig_slots = (2 * hero_seat, 2 * hero_seat + 1)
    # Forced BTN open size for the bb_defend node (idx into ACTION_VOCAB). Default 8 = 3.25bb raise-to
    # (BET_SIZE_BUCKETS[8-3]=1.5x pot; the smallest sized open present in the serve tree — see the
    # module note lines 59-65 and the `rows`-mode BB_vs_BTN_open row). Forcing a FIXED size makes every
    # reached deal key the SAME BB-facing node (clean pot odds + clean coverage). The BTN action is
    # FORCED (not served) so no menu-skew on the BTN; the SB/BB response rows it hits ARE in the table.
    btn_open_idx = int(getattr(args, "btn_open_idx", 8) or 8)

    # --- hero hole-card rigging: draw rig_slots to hold the forced hand class -----------------------
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
        for slot, c in zip(rig_slots, want):
            j = cards.index(c)
            cards[slot], cards[j] = cards[j], cards[slot]

    def play_arm_bb(h: int, arm: str):
        """One arm of one deal for the BB-defend node target. Returns record dict or None if unreached.

        Node reached (all guarded): UTG(0)/HJ(1)/CO(2) FOLD -> BTN(3) forced OPEN (fixed idx
        ``btn_open_idx``, default 3.25bb) -> SB(4) folds PER BLUEPRINT (else abort: we condition on the
        clean SB-out BB-closing spot with defined pot odds) -> BB(5, rigged hand) forced to ``arm``
        (fold | defend=CALL). CRN: identical deck + per-seat agent seeds + settle rng across arms, so
        the BTN open and the SB fold are IDENTICAL between the fold and defend arms -> per-deal delta
        pairs exactly. Only the BB's own action differs. ``delta`` is seat 5's chip delta in bb."""
        rng = random.Random(args.seed * 77 + h)
        state = GameState.new_hand(rng=rng)  # button_seat=3 -> seats: 0 UTG .. 3 BTN, 4 SB, 5 BB
        rig_deck(state, rng)
        for s, ag in agents.items():
            ag.reseed(args.seed * 1_000_003 + h * 31 + s)  # arm-independent -> CRN
        settle_seed = 987_001 + h
        forced_idx = ARM_BB_IDX[arm]  # 0=FOLD, 2=CALL
        st = {"btn_opened": False, "sb_folded": False, "bb_acted": False,
              "btn_open": None, "pot_before_bb": None, "to_call_bb": None}

        def early_fold_policy(seat):
            """UTG/HJ/CO: must fold, else this deal never reaches the folded-to-BTN open."""
            ag = agents[seat]
            def pol(state):
                a = ag(state)
                if a.action_type != ActionType.FOLD:
                    raise AbortHand()
                return a
            return pol

        def btn_policy(state):
            # Guard: BTN acts folded-to (UTG/HJ/CO all folded). Force the fixed modal open.
            hist = state._history
            if not (len(hist) == 3 and all(a.action_type == ActionType.FOLD for _s, a in hist)):
                raise AbortHand()
            a = _index_to_action(state, 3, btn_open_idx)
            # Require it to actually be a RAISE (an open). At 100bb the 3.25bb raise is always legal, so
            # this holds; the guard protects against a short-stack/edge nearest-map to CALL/CHECK.
            if a.action_type not in (ActionType.RAISE, ActionType.BET):
                raise AbortHand()
            st["btn_opened"] = True
            st["btn_open"] = round(getattr(a, "amount_bb", 0.0) or 0.0, 2)
            return a

        def sb_policy(state):
            # SB plays the BLUEPRINT facing the BTN open; we condition on it FOLDING (clean HU BB spot).
            a = agents[4](state)
            if a.action_type != ActionType.FOLD:
                raise AbortHand()
            st["sb_folded"] = True
            return a

        def bb_policy(state):
            if not st["bb_acted"]:
                st["bb_acted"] = True
                # node guard: BTN opened + SB folded + BB is closing action. history == 3 folds + BTN
                # raise + SB fold == 5 entries; BB (seat 5) faces the open with SB out.
                hist = state._history
                assert st["btn_opened"] and st["sb_folded"] and len(hist) == 5, "bb node guard"
                player = state.players[5]
                st["pot_before_bb"] = round(state.pot_bb, 3)
                st["to_call_bb"] = round(max(0.0, state.current_bet_bb - player.committed_street_bb), 3)
                return _index_to_action(state, 5, forced_idx)
            return agents[5](state)

        policies = {0: early_fold_policy(0), 1: early_fold_policy(1), 2: early_fold_policy(2),
                    3: btn_policy, 4: sb_policy, 5: bb_policy}
        if args.postflop_field == "checkcall":
            def _postflop_fixed(seat, pol):
                def wrapped(state):
                    if len(state.board) > 0:
                        legal = state.legal_actions(seat)
                        for want in (ActionType.CHECK, ActionType.CALL):
                            for a in legal:
                                if a.action_type == want:
                                    return a
                        return legal[0]
                    return pol(state)
                return wrapped
            policies = {s: _postflop_fixed(s, pol) for s, pol in policies.items()}
        settle = lambda s, nb: equity_allin_settle(  # noqa: E731
            s, nb, n_runouts=args.runouts, rng=random.Random(settle_seed))
        try:
            final, deltas = play_hand(policies, state=state, allin_settle=settle)
        except AbortHand:
            return None
        if not st["bb_acted"]:
            return None
        return {
            "delta": float(deltas.get(5, 0.0)),
            "btn_open": st["btn_open"], "pot_before_bb": st["pot_before_bb"],
            "to_call_bb": st["to_call_bb"],
            "end_street": len(final.board),
        }

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
        # v5.1 Stage-0' idealized-realization re-serve (plan §11.4; spec: bpv4_pluribus_grade_design.md
        # VERDICT "$0 discriminator"). EVERY seat plays the trained blueprint postflop, but the FIRST
        # turn/river decision whose fingerprint MISSES (the ~83%/~91% uniform-fallback locus,
        # lbr.py make_blueprint_policy) settles the hand THERE by check-down equity — an upper bound on
        # what ANY realization fix (river abstraction / coverage / leaf values) could recover, with
        # reach frequency held fixed (upstream play byte-identical to --postflop-field blueprint; the
        # miss action is discarded unapplied). Flop misses keep the legacy uniform fallback (§37: flop
        # contribution is stable; the localized deficit is turn+river). Miss detection = the
        # _BP_COVERAGE per-street miss counters (the ONLY signal — a miss returns a valid uniform-legal
        # dist; adapter fallback_count never increments; env.street ints: 2=turn, 3=river).
        if args.postflop_field == "idealized":
            from pokerbot.eval import lbr as _lbr

            def _tr_misses() -> int:
                cov = _lbr._BP_COVERAGE
                return cov.get(2, (0, 0))[1] + cov.get(3, (0, 0))[1]

            def _postflop_idealized(pol):
                def wrapped(state):
                    if len(state.board) >= 4:  # turn (4) / river (5) decisions only
                        before = _tr_misses()
                        a = pol(state)
                        if _tr_misses() > before:
                            raise SettleNow(state)
                        return a
                    return pol(state)
                return wrapped
            policies = {s: _postflop_idealized(pol) for s, pol in policies.items()}
        settle = lambda s, nb: equity_allin_settle(  # noqa: E731
            s, nb, n_runouts=args.runouts, rng=random.Random(settle_seed))
        try:
            final, deltas = play_hand(policies, state=state, allin_settle=settle)
        except AbortHand:
            return None
        except SettleNow as e:
            # Same settle RNG construction as the all-in hook (CRN: fixed per candidate deal h, arm-
            # independent). deltas has the resolve_showdown shape {seat: net_bb}; the rest of the
            # record path (delta, resp extraction, end_street tag) works on the mid-hand state
            # unchanged — end_street tags 4/5 so ev_by_street attributes the settled EV to turn/river.
            final = e.state
            deltas = settle(e.state, None)
            idealized_settles[str(len(final.board))] = (
                idealized_settles.get(str(len(final.board)), 0) + 1)
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
    if args.postflop_field == "idealized" and node_target == "bb_defend":
        raise SystemExit("--postflop-field idealized supports --node-target btn_open only")
    idealized_settles: dict[str, int] = {}  # board-len ('4'=turn,'5'=river) -> settles fired
    _play = play_arm_bb if node_target == "bb_defend" else play_arm
    out = {arm: [] for arm in arm_set}
    # Per-deal end-street tag (0=preflop, 3=flop, 4=turn, 5=river) paired 1:1 with out[arm], so the
    # merge step can attribute each arm's EV to the street its OWN line ended on (open-vs-limp localizer).
    street_out = {arm: [] for arm in arm_set}
    if node_target == "bb_defend":
        meta = {"btn_open": {}, "pot_before_bb": {}, "to_call_bb": {},
                "end_street": {arm: {} for arm in arm_set}}
    else:
        meta = {"hero_first_own": {}, "resp": {arm: {"sb": {}, "bb": {}} for arm in arm_set},
                "end_street": {arm: {} for arm in arm_set}}
    reached = 0
    tried = 0
    h = args.start
    t0 = time.time()
    last_hb = t0
    while reached < args.deals:
        tried += 1
        rec0 = _play(h, arm_set[0])
        if rec0 is None:
            h += 1
            continue
        recs = {arm_set[0]: rec0}
        ok = True
        for arm in arm_set[1:]:
            r = _play(h, arm)
            if r is None:  # cannot happen (prefix identical) but stay safe
                ok = False
                break
            recs[arm] = r
        h += 1
        if not ok:
            continue
        reached += 1
        for arm in arm_set:
            r = recs[arm]
            out[arm].append(r["delta"])
            street_out[arm].append(int(r["end_street"]))
            if node_target != "bb_defend":
                for side in ("sb", "bb"):
                    if r[side]:
                        d = meta["resp"][arm][side]
                        d[r[side]] = d.get(r[side], 0) + 1
            es = meta["end_street"][arm]
            es[str(r["end_street"])] = es.get(str(r["end_street"]), 0) + 1
        if node_target == "bb_defend":
            for fld in ("btn_open", "pot_before_bb", "to_call_bb"):
                k = str(rec0[fld])
                meta[fld][k] = meta[fld].get(k, 0) + 1
        else:
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
        "node_target": node_target,
        "start": args.start, "deals": reached, "tried": tried,
        "reach_rate": reached / max(1, tried), "runouts": args.runouts,
        "postflop_field": args.postflop_field,
        "adapter_fallback_rate": fb / max(1, dc),
        "bp_coverage": bp_cov,
        "idealized_settles": idealized_settles,
        "deltas_bb": out, "street_tags_bb": street_out, "meta": meta,
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
    # node target is uniform across a merge set; pick the arm set + pairing accordingly.
    nt = shards[0].get("node_target", "btn_open")
    arm_set = ARMS_BB if nt == "bb_defend" else ARMS
    out = {"mode": "merged", "node_target": nt, "hands": {}}
    for hand, ss in by_hand.items():
        deltas = {arm: np.concatenate([np.asarray(x["deltas_bb"][arm]) for x in ss])
                  for arm in arm_set}
        n = len(deltas[arm_set[0]])
        stats = {}
        for arm in arm_set:
            d = deltas[arm]
            stats[arm] = {"n": int(n), "ev_bb100": float(d.mean() * 100),
                          "se_bb100": float(d.std(ddof=1) / math.sqrt(n) * 100)}
        paired = {}
        if nt == "bb_defend":
            # THE discriminator: BB's paired EV(defend=CALL) - EV(fold), per deal (CRN: identical BTN
            # open + SB fold across the two arms). Positive => defending is +EV under this postflop
            # field for the average reached hand.
            diff = deltas["defend"] - deltas["fold"]
            paired["defend_minus_fold"] = {
                "delta_bb100": float(diff.mean() * 100),
                "se_bb100": float(diff.std(ddof=1) / math.sqrt(n) * 100),
                "t": float(diff.mean() / (diff.std(ddof=1) / math.sqrt(n)) if diff.std(ddof=1) > 0 else 0.0),
                "frac_defend_positive": float((diff > 1e-9).mean()),
            }
        else:
            for arm in ("own", "open325", "open400", "jam"):
                diff = deltas[arm] - deltas["limp"]
                paired[f"{arm}_minus_limp"] = {
                    "delta_bb100": float(diff.mean() * 100),
                    "se_bb100": float(diff.std(ddof=1) / math.sqrt(n) * 100),
                    "t": float(diff.mean() / (diff.std(ddof=1) / math.sqrt(n)) if diff.std(ddof=1) > 0 else 0.0),
                }
        # --- per-street EV attribution (the open-vs-limp deficit localizer) --------------------------
        # contrib_s(arm) = mean over ALL n deals of delta * 1[arm's line ended on street s], so
        # sum_s contrib_s(arm) == EV(arm) exactly. The gap EV(open)-EV(limp) then decomposes additively
        # as sum_s [contrib_s(open) - contrib_s(limp)]. This is an UNCONDITIONAL partition by the street
        # each arm's OWN line ended on -- deliberately NOT a same-deal conditional diff (which would be
        # selection-biased: opens fold villains out preflop more often, so conditioning on both arms
        # reaching river silently drops the very deals where the open's fold-equity lives). Comparing the
        # checkcall vs blueprint fields per street then tells us whether the deficit is postflop-play
        # (coverage/realization -> presum lever) or preflop (equilibrium/jam -> preflop-menu lever).
        ev_by_street = None
        gap_by_street = None
        if nt != "bb_defend" and all("street_tags_bb" in x for x in ss):
            STREETS = [(0, "preflop"), (3, "flop"), (4, "turn"), (5, "river")]
            stags = {arm: np.concatenate([np.asarray(x["street_tags_bb"][arm]) for x in ss])
                     for arm in arm_set}
            ev_by_street = {}
            for arm in arm_set:
                d = deltas[arm]
                tg = stags[arm]
                row = {}
                for sn, lab in STREETS:
                    mask = tg == sn
                    ns = int(mask.sum())
                    if ns:
                        ds = d[mask]
                        row[lab] = {
                            "n": ns, "frac": ns / n,
                            "cond_ev_bb100": float(ds.mean() * 100),
                            "cond_se_bb100": float(ds.std(ddof=1) / math.sqrt(ns) * 100) if ns > 1 else 0.0,
                            "contrib_bb100": float(ds.sum() / n * 100),
                        }
                    else:
                        row[lab] = {"n": 0, "frac": 0.0, "cond_ev_bb100": 0.0,
                                    "cond_se_bb100": 0.0, "contrib_bb100": 0.0}
                ev_by_street[arm] = row
            gap_by_street = {}
            for arm in ("own", "open325", "open400", "jam"):
                gap_by_street[f"{arm}_minus_limp"] = {
                    lab: float(ev_by_street[arm][lab]["contrib_bb100"]
                               - ev_by_street["limp"][lab]["contrib_bb100"])
                    for _sn, lab in STREETS
                }
        if nt == "bb_defend":
            meta = {"btn_open": {}, "pot_before_bb": {}, "to_call_bb": {},
                    "end_street": {a: {} for a in arm_set}}
        else:
            meta = {"hero_first_own": {}, "resp": {a: {"sb": {}, "bb": {}} for a in arm_set},
                    "end_street": {a: {} for a in arm_set}}
        tried = sum(x["tried"] for x in ss)
        for x in ss:
            m = x["meta"]
            if nt == "bb_defend":
                for fld in ("btn_open", "pot_before_bb", "to_call_bb"):
                    for k, v in m.get(fld, {}).items():
                        meta[fld][k] = meta[fld].get(k, 0) + v
            else:
                for k, v in m["hero_first_own"].items():
                    meta["hero_first_own"][k] = meta["hero_first_own"].get(k, 0) + v
                for a in arm_set:
                    for side in ("sb", "bb"):
                        for k, v in m["resp"][a][side].items():
                            meta["resp"][a][side][k] = meta["resp"][a][side].get(k, 0) + v
            for a in arm_set:
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
        pair_key = "paired_defend_vs_fold" if nt == "bb_defend" else "paired_vs_limp"
        out["hands"][hand] = {"n_deals": int(n), "tried": tried,
                              "reach_rate": n / max(1, tried),
                              "arms": stats, pair_key: paired, "meta": meta,
                              "fallback_rates": [x["adapter_fallback_rate"] for x in ss],
                              "bp_coverage": bp_coverage,
                              "postflop_miss_rate": pf_m / max(1, pf_h + pf_m)}
        if ev_by_street is not None:
            out["hands"][hand]["ev_by_street"] = ev_by_street
            out["hands"][hand]["open_minus_limp_by_street"] = gap_by_street
        _plog(f"merge {hand}: n={n}  " + "  ".join(
            f"{a}={stats[a]['ev_bb100']:+.1f}±{stats[a]['se_bb100']:.1f}" for a in arm_set))
        if ev_by_street is not None:
            for arm in ("open325", "open400", "jam"):
                g = gap_by_street[f"{arm}_minus_limp"]
                _plog(f"merge {hand}: ({arm}-limp) by street  "
                      + "  ".join(f"{lab}={g[lab]:+.1f}" for lab in ("preflop", "flop", "turn", "river"))
                      + f"  [sum={sum(g.values()):+.1f} bb/100]")
        if nt == "bb_defend":
            p = paired["defend_minus_fold"]
            _plog(f"merge {hand}: defend-fold = {p['delta_bb100']:+.1f}±{p['se_bb100']:.1f} bb/100 "
                  f"(t={p['t']:.1f}, frac_defend_+EV={p['frac_defend_positive']:.3f})")
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
    r.add_argument("--presum", action="store_true",
                   help="serve under the presum postflop key (POKERBOT_PRESUM_KEY=1, composes with "
                        "--key-mode fullhist) — REQUIRED for presum-trained snapshots (bpv4_presum_1): "
                        "without it every postflop lookup misses (fullhist-exact keys) and the field "
                        "silently degrades to uniform. Default off = byte-identical legacy.")
    r.add_argument("--mrr", action="store_true",
                   help="serve under the M-RR thinned postflop menus (POKERBOT_MRR=1, the trainer's "
                        "abstracted_legal_mrr keep-lists mirrored in lbr.make_blueprint_policy) — "
                        "REQUIRED for fullhist+M-RR-trained arms (train/probe menu skew otherwise). "
                        "Default off = byte-identical bp3/summary behavior.")
    r.add_argument("--pre-keep", default=None,
                   help="arm-3 preflop FIRST-raise keep-list (POKERBOT_PRE_KEEP), comma-separated "
                        "bet-bucket ids e.g. '2,3,4,5,6' — applies while no raise exists yet this "
                        "hand (current_bet <= BB). Default None = trainer default (4,5,6), "
                        "byte-identical. MUST match the probed checkpoint's trained menu.")
    r.add_argument("--pre-rr-keep", default=None,
                   help="arm-3 preflop RE-raise (3-bet+) keep-list (POKERBOT_PRE_RR_KEEP), "
                        "comma-separated bet-bucket ids e.g. '4,5,6' — applies at preflop nodes with "
                        "a raise already in (current_bet > BB). Default None = the --pre-keep menu "
                        "applies everywhere preflop (legacy, byte-identical).")
    r.add_argument("--post-keep", default=None,
                   help="arm-3 postflop keep-list (POKERBOT_POST_KEEP), comma-separated bet-bucket "
                        "ids. Default None = trainer default (0,1,3,4,5), byte-identical.")
    r.add_argument("--flop-rr-keep", default=None,
                   help="arm-3 FLOP re-raise keep-list under --mrr (POKERBOT_FLOP_RR_KEEP), "
                        "comma-separated bet-bucket ids e.g. '1,4'. Turn/river re-raises always stay "
                        "the legacy {4} thin. Default None = legacy M-RR flop thin {4}, "
                        "byte-identical. Only meaningful with --mrr.")
    r.add_argument("--postflop-field", choices=("blueprint", "checkcall", "idealized"),
                   default="blueprint",
                   help="postflop continuation field for the EV rollout. 'blueprint' (default) = the "
                        "legacy frozen-blueprint field (byte-identical behavior; coverage-gated by "
                        "--max-postflop-miss). 'checkcall' = the v3 G5 instrument: EVERY seat plays "
                        "a SYMMETRIC fixed check/call policy postflop, so the EV(open)-EV(limp) "
                        "contrast measures the TRAINED preflop response economics under pure equity "
                        "realization and stays well-defined at any postflop coverage (the fix for "
                        "the Stage-0 G5 coverage-BLOCK; amendment doc section 5). 'idealized' = the "
                        "v5.1 Stage-0' re-serve discriminator: blueprint postflop everywhere, but the "
                        "FIRST turn/river fingerprint MISS settles the hand there by check-down "
                        "equity (reach held fixed) — the upper bound on what any realization fix "
                        "could recover (plan §11.4; btn_open only).")
    r.add_argument("--snapshot-index", default=None,
                   help="dir built by bpv4_nlcfr_index.py — mmap-backed snapshot lookup instead of the "
                        "full in-RAM load (REQUIRED for the 55 GB cloud fullhist blobs). Default None "
                        "= historical in-RAM path, byte-identical.")
    r.add_argument("--snapshot-nlsnap-file", default=None,
                   help="path to a SORTED NLSNAP01 snapshot — served by DIRECT mmap binary search (no "
                        "index; a multi-billion-row snapshot's A/B/O index would be ~90 GB). Trained "
                        "current-iterate INCL POSTFLOP. Highest precedence among snapshot sources.")
    r.add_argument("--snapshot-nlsnap-index", default=None,
                   help="dir built by bpv4_nlsnap_index.py over an NLSNAP01 current-iterate snapshot — "
                        "serves the TRAINED current-strategy INCL POSTFLOP (the .nlcfr strat section is "
                        "preflop-only under snapshot_avg). For the root-cause probe: EV under the TRAINED "
                        "postflop field vs --postflop-field checkcall. Takes precedence over --snapshot-index.")
    r.add_argument("--node-target", choices=("btn_open", "bb_defend"), default="btn_open",
                   help="which forced node to measure. 'btn_open' (default, byte-identical legacy) = the "
                        "folded-to-BTN first-action arms (own/limp/open325/open400/jam). 'bb_defend' = the "
                        "BB-defend-vs-fold coda: rig the BIG BLIND (seat 5), let UTG/HJ/CO fold + the BTN "
                        "force-OPEN (--btn-open-idx) + the SB fold PER BLUEPRINT, then force the BB to "
                        "CALL(defend) vs FOLD and measure the BB's paired EV(defend)-EV(fold). Arms = "
                        "(fold, defend). Use with --postflop-field checkcall for a pure-equity-realization "
                        "defend test.")
    r.add_argument("--btn-open-idx", type=int, default=8,
                   help="bb_defend only: ACTION_VOCAB index the BTN open is forced to. Default 8 = 3.25bb "
                        "raise-to (BET_SIZE_BUCKETS[5]=1.5x pot), the smallest sized open in the serve "
                        "tree (its BB response row is verified present). 9 = 4.0bb.")
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
