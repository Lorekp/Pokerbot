#!/usr/bin/env python3
"""Blueprint SERVE-COVERAGE miss probe (fullhist/presum key family, blueprint-v4).

Measures the blueprint lookup MISS RATE — the fraction of serve-time policy lookups whose key is
absent from the served table (each miss degrades to uniform-over-menu play) — on REAL trajectories:

  * ``--mode selfplay``: all 6 seats served by the RAW blueprint policy (search off). This is the
    NEW-vs-NEW surface where the 2026-07-10 strength cert measured a 10.3% miss.
  * ``--mode pool``: seat-rotated hero (RAW blueprint) vs the CoinPoker-mirror pool archetypes the
    cert used (responsive_station / responsive_reg / pro_balanced / hs_nit) — the OFF-MENU opponent
    surface (21-25% missing in the cert).

Every miss is CLASSIFIED against a full-checkpoint key index (scripts/nlcfr_key_index.py):
  * ``in_full``  — the key IS in the full trained table; the serve table dropped the row
                   (mass-filter truncation). NOT a key-construction bug.
  * ``off_space`` — the key is not in the full table: an infoset the trainer never stored
                   (unvisited, or an off-menu opponent action baked into the fullhist key).

The blueprint can be served from a pkl (dict / arrays formats, e.g. the top-25M mass-filtered cert
extract) or from a full-table sorted index dir built with ``--with-rows`` (``SortedSnapshot``:
mmap + binary search, ~0 RSS — the practical full-table serve object).

Run with POKERBOT_ACTION_TRANSLATE=1 in the environment to measure the pseudo-harmonic action
translation layer (set it BEFORE python; this module never overrides pinned env).

Usage (env defaults = the bpv4_pluribus_finemenu_1 serve regime, worktree-local paths):
  python scripts/serve_miss_probe.py --blueprint <pkl-or-indexdir> \
      --full-index artifacts/_serve_cov/idx_finemenu50k \
      --mode selfplay --hands 4000 --seed 11 --out artifacts/_serve_cov/selfplay_before.json
  python scripts/serve_miss_probe.py --blueprint ... --mode pool --decks 40 \
      --archetypes responsive_station,responsive_reg,pro_balanced --out ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--blueprint", required=True,
                 help="serve table: .pkl (strat_sum dict or arrays) OR a key-index dir with rows")
_ap.add_argument("--full-index", default="",
                 help="full-checkpoint key index dir (keys_sorted.bin) for miss classification")
_ap.add_argument("--mode", choices=("selfplay", "pool"), default="selfplay")
_ap.add_argument("--hands", type=int, default=4000, help="selfplay hands")
_ap.add_argument("--decks", type=int, default=40, help="pool decks per archetype (x6 rotations)")
_ap.add_argument("--archetypes", default="responsive_station,responsive_reg,pro_balanced")
_ap.add_argument("--seed", type=int, default=11)
_ap.add_argument("--miss-log", default="", help="JSONL of missed keys + history context (sampled)")
_ap.add_argument("--miss-log-cap", type=int, default=50000)
_ap.add_argument("--out", required=True)
ARGS = _ap.parse_args()

# ---------------------------------------------------------------------------
# Env BEFORE any pokerbot import (import-latched gates). setdefault: pinned env always wins.
# Defaults = the bpv4_pluribus_finemenu_1 training/serve regime with worktree-local data paths.
# ---------------------------------------------------------------------------
os.environ.setdefault("JAX_PLATFORMS", "cpu")
for _k, _v in {
    "POKERBOT_BETTING_ABSTRACTION": "1",
    "POKERBOT_POSTFLOP_BUCKETS": "1",
    "POKERBOT_PREFLOP_LOSSLESS": "1",
    "POKERBOT_POSTFLOP_MODE": "emd",
    "POKERBOT_POSTFLOP_NUM_BUCKETS": "160",
    "POKERBOT_POSTFLOP_EMD_FROZEN": os.path.join(
        _ROOT, "data", "abstraction", "emd_frozen_flop_turn_160_r400.emdfrzn"),
    "POKERBOT_EMD_FROZEN_MIRROR_FALLBACK": "1",
    "POKERBOT_RIVER_OCHS": os.path.join(_ROOT, "data", "abstraction", "river_ochs_160.rochs"),
    "POKERBOT_FULLHIST_KEY": "1",
    "POKERBOT_PRESUM_KEY": "1",
    "POKERBOT_MRR": "1",
    "POKERBOT_CLOSE_CALLED_STREET": "1",
    "POKERBOT_PRE_KEEP": "2,3,4,5,6",
    "POKERBOT_PRE_RR_KEEP": "4,5,6",
}.items():
    os.environ.setdefault(_k, _v)

import numpy as np  # noqa: E402

_KEY_ENVS = ("POKERBOT_FULLHIST_KEY", "POKERBOT_PRESUM_KEY", "POKERBOT_MRR",
             "POKERBOT_CLOSE_CALLED_STREET", "POKERBOT_PRE_KEEP", "POKERBOT_PRE_RR_KEEP",
             "POKERBOT_RIVER_OCHS", "POKERBOT_POSTFLOP_MODE", "POKERBOT_POSTFLOP_NUM_BUCKETS",
             "POKERBOT_POSTFLOP_EMD_FROZEN", "POKERBOT_EMD_FROZEN_MIRROR_FALLBACK",
             "POKERBOT_ACTION_TRANSLATE")


# ---------------------------------------------------------------------------
# Serve tables.
# ---------------------------------------------------------------------------
class SortedSnapshot:
    """Full-table blueprint serve object: lexicographically sorted S16 keys + aligned f32 rows,
    both memory-mapped; ``get`` = binary search. Duck-types the only method
    ``lbr.make_blueprint_policy`` uses (``snapshot.get(key)``)."""

    def __init__(self, dirpath: str):
        self.keys = np.memmap(os.path.join(dirpath, "keys_sorted.bin"), dtype="S16", mode="r")
        rows_path = os.path.join(dirpath, "rows_sorted.f32.bin")
        self.rows = None
        if os.path.exists(rows_path):
            self.rows = np.memmap(rows_path, dtype=np.float32, mode="r").reshape(-1, 11)
            assert self.rows.shape[0] == self.keys.shape[0]

    def __len__(self) -> int:
        return int(self.keys.shape[0])

    # NOTE numpy 'S16' comparison semantics: trailing NULs are ignored, and comparing an np.bytes_
    # element against a raw python bytes object is EXACT — so a digest ending in b"\\x00" (~1/256 of
    # keys) would searchsorted to the right slot but fail a bytes-vs-np equality check (a false
    # MISS). Keep every comparison numpy-vs-numpy (both sides NUL-stripped) for consistency; blake2b
    # digests distinct-modulo-trailing-NULs are astronomically unlikely to collide.
    def contains(self, key: bytes) -> bool:
        kk = np.asarray(key, dtype="S16")
        i = int(np.searchsorted(self.keys, kk))
        return i < self.keys.shape[0] and self.keys[i] == kk

    def contains_many(self, key_arr: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.keys, key_arr)
        idx = np.minimum(idx, self.keys.shape[0] - 1)
        return self.keys[idx] == key_arr

    def get(self, key: bytes, default=None):
        kk = np.asarray(key, dtype="S16")
        i = int(np.searchsorted(self.keys, kk))
        if i < self.keys.shape[0] and self.keys[i] == kk:
            if self.rows is None:
                raise RuntimeError("index built without --with-rows cannot serve values")
            return self.rows[i]
        return default


def _load_serve_table(path: str):
    if os.path.isdir(path):
        return SortedSnapshot(path)
    import pickle
    with open(path, "rb") as f:
        raw = pickle.load(f)
    inner = raw.get("strat_sum", raw) if isinstance(raw, dict) else raw
    if isinstance(inner, dict) and "keys" in inner and "vals" in inner:
        # arrays format: sort once -> binary-search table (no 25M-entry dict RSS).
        keys = np.frombuffer(inner["keys"], dtype="S16")
        vals = np.asarray(inner["vals"], dtype=np.float32)
        order = np.argsort(keys, kind="stable")
        snap = SortedSnapshot.__new__(SortedSnapshot)
        snap.keys = keys[order]
        snap.rows = vals[order]
        return snap
    return inner  # plain {bytes: row} dict


class CountingTable:
    """Wraps a serve table; counts hits/misses per street and logs missed keys + history bytes.
    The policy wrapper sets ``ctx`` (street, hist bytes) before each lookup (single-threaded)."""

    def __init__(self, inner, log_path: str, log_cap: int):
        self.inner = inner
        self.ctx_street = -1
        self.ctx_hist = b""
        self.hits = [0, 0, 0, 0]
        self.misses = [0, 0, 0, 0]
        self.zero_rows = [0, 0, 0, 0]
        self.miss_keys: list[tuple[bytes, int]] = []
        self._log = open(log_path, "w", encoding="utf-8") if log_path else None
        self._log_cap = log_cap

    def get(self, k, default=None):
        v = self.inner.get(k, default)
        st = min(max(self.ctx_street, 0), 3)
        if v is None:
            self.misses[st] += 1
            if len(self.miss_keys) < self._log_cap:
                self.miss_keys.append((bytes(k), st))
                if self._log is not None:
                    self._log.write(json.dumps({
                        "key": bytes(k).hex(), "street": st,
                        "hist": self.ctx_hist.hex()}) + "\n")
        else:
            self.hits[st] += 1
            if float(np.asarray(v, dtype=np.float64).sum()) <= 1e-12:
                self.zero_rows[st] += 1
        return v

    def close(self):
        if self._log is not None:
            self._log.close()


def _counted_policy(pf, ctable: CountingTable):
    def policy(env, seat: int):
        ctable.ctx_street = int(env.street)
        hl = int(np.asarray(env.hist_len))
        ctable.ctx_hist = bytes(int(a) & 0xFF for a in np.asarray(env.hist_action)[:hl])
        return pf(env, seat)
    return policy


def _report(ctable: CountingTable, full_index, extra: dict) -> dict:
    streets = ["preflop", "flop", "turn", "river"]
    per = {}
    for i, s in enumerate(streets):
        tot = ctable.hits[i] + ctable.misses[i]
        per[s] = {"lookups": tot, "hits": ctable.hits[i], "misses": ctable.misses[i],
                  "zero_rows": ctable.zero_rows[i],
                  "miss_rate": (ctable.misses[i] / tot) if tot else 0.0}
    tot_h, tot_m = sum(ctable.hits), sum(ctable.misses)
    cls = None
    if full_index is not None and ctable.miss_keys:
        karr = np.asarray([k for k, _ in ctable.miss_keys], dtype="S16")
        member = full_index.contains_many(karr)
        cls = {"classified": int(karr.size),
               "in_full_table": int(member.sum()),
               "off_space": int((~member).sum())}
        per_st = {}
        sts = np.asarray([st for _, st in ctable.miss_keys])
        for i, s in enumerate(streets):
            m = sts == i
            per_st[s] = {"n": int(m.sum()), "in_full": int(member[m].sum()),
                         "off_space": int((~member[m]).sum())}
        cls["per_street"] = per_st
    return {
        "per_street": per,
        "total": {"lookups": tot_h + tot_m, "hits": tot_h, "misses": tot_m,
                  "miss_rate": (tot_m / (tot_h + tot_m)) if (tot_h + tot_m) else 0.0},
        "miss_classification": cls,
        **extra,
    }


def main() -> int:
    print("[probe] keying env: " + " ".join(f"{k}={os.environ.get(k, '')}" for k in _KEY_ENVS),
          flush=True)
    t0 = time.time()
    snap = _load_serve_table(ARGS.blueprint)
    n_rows = len(snap)
    print(f"[probe] serve table {ARGS.blueprint}: {n_rows:,} rows "
          f"({type(snap).__name__})", flush=True)
    full_index = SortedSnapshot(ARGS.full_index) if ARGS.full_index else None
    if full_index is not None:
        print(f"[probe] full index: {len(full_index):,} keys", flush=True)

    ctable = CountingTable(snap, ARGS.miss_log, ARGS.miss_log_cap)

    from pokerbot.eval.lbr import make_blueprint_policy
    pf = _counted_policy(make_blueprint_policy(ctable), ctable)

    extra = {"blueprint": os.path.abspath(ARGS.blueprint), "mode": ARGS.mode,
             "seed": ARGS.seed, "rows": n_rows,
             "keying_env": {k: os.environ.get(k, "") for k in _KEY_ENVS}}

    if ARGS.mode == "selfplay":
        import jax
        from pokerbot.engine.jax_engine import init_hand
        from pokerbot.engine.np_engine import (
            apply_action_np, current_player_np, is_chance_node_np, is_terminal_np,
            legal_action_mask_np, sample_chance_np, to_numpy_state,
        )
        occ = jax.numpy.array([True] * 6)
        n_hands = ARGS.hands
        for h in range(n_hands):
            env = to_numpy_state(init_hand(
                jax.random.PRNGKey(ARGS.seed * 100003 + h), occupancy=occ))
            g = 0
            while not is_terminal_np(env) and g < 400:
                g += 1
                if is_chance_node_np(env):
                    env = sample_chance_np(env)
                    continue
                seat = current_player_np(env)
                if seat < 0:
                    break
                p = np.asarray(pf(env, int(seat)), dtype=np.float64)
                pc = np.clip(p, 0, None)
                s = pc.sum()
                if s <= 0:
                    m = legal_action_mask_np(env, seat).astype(np.float64)
                    pc = m / m.sum()
                else:
                    pc = pc / s
                a = int(np.random.default_rng(
                    ARGS.seed * 7 + h * 131 + g).choice(len(pc), p=pc))
                env = apply_action_np(env, a)
            if (h + 1) % 500 == 0:
                tm = sum(ctable.misses)
                tl = tm + sum(ctable.hits)
                print(f"  [selfplay] hand {h + 1}/{n_hands} lookups={tl} "
                      f"miss={tm} ({100.0 * tm / max(1, tl):.2f}%) "
                      f"t={time.time() - t0:.0f}s", flush=True)
        extra["hands"] = n_hands
        result = _report(ctable, full_index, extra)
    else:  # pool
        import random
        from track2_s3c_pool import SearchHeroAgent, _run_arm
        per_arch = {}
        for name in [a.strip() for a in ARGS.archetypes.split(",") if a.strip()]:
            before_h, before_m = list(ctable.hits), list(ctable.misses)
            mark = len(ctable.miss_keys)

            def factory(*, exploit: bool):
                return SearchHeroAgent(pf, random.Random(0))

            _run_arm(factory, name, decks=ARGS.decks, seed=ARGS.seed, workers=1,
                     tag=f"miss_{name}", progress_every=10)
            d_h = sum(ctable.hits) - sum(before_h)
            d_m = sum(ctable.misses) - sum(before_m)
            arch_cls = None
            if full_index is not None and len(ctable.miss_keys) > mark:
                karr = np.asarray([k for k, _ in ctable.miss_keys[mark:]], dtype="S16")
                member = full_index.contains_many(karr)
                arch_cls = {"in_full_table": int(member.sum()),
                            "off_space": int((~member).sum())}
            per_arch[name] = {"lookups": d_h + d_m, "misses": d_m,
                              "miss_rate": d_m / max(1, d_h + d_m),
                              "miss_classification": arch_cls}
            print(f"  [pool] {name}: lookups={d_h + d_m} misses={d_m} "
                  f"({100.0 * d_m / max(1, d_h + d_m):.2f}%) cls={arch_cls}", flush=True)
        extra["decks"] = ARGS.decks
        result = _report(ctable, full_index, extra)
        result["per_archetype"] = per_arch

    ctable.close()
    result["wall_s"] = time.time() - t0
    os.makedirs(os.path.dirname(os.path.abspath(ARGS.out)), exist_ok=True)
    with open(ARGS.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(f"[probe] total miss_rate={result['total']['miss_rate']:.4f} "
          f"({result['total']['misses']}/{result['total']['lookups']}) "
          f"cls={result['miss_classification']} -> {ARGS.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
