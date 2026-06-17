#!/usr/bin/env python3
"""Diagnose Leduc encode injectivity and py_info_key vs encode / strat_sum alignment."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from pokerbot.engine.leduc_bridge import (
    apply_action_py,
    py_fingerprint,
    py_info_key,
    py_legal,
    py_state_from_deal,
)
from pokerbot.engine.leduc_jax import NUM_ACTIONS
from pokerbot.training.leduc_collision_check import check_leduc_encode_collisions
from pokerbot.training.leduc_tabular_cfr import all_leduc_deals
from pokerbot.training.sd_cfr import encode_leduc_py


def encode_key(state, seat: int) -> tuple:
    return tuple(np.asarray(encode_leduc_py(state, seat)).tolist())


def walk_all_nodes():
    deals = list(all_leduc_deals())
    ik_to_enc: dict[tuple, set[tuple]] = defaultdict(set)
    enc_to_ik: dict[tuple, set[tuple]] = defaultdict(set)
    ik_to_pot: dict[tuple, set[int]] = defaultdict(set)
    n_nodes = 0

    for cards, pub in deals:
        stack = [py_state_from_deal(cards, pub)]
        seen: set[tuple] = set()
        while stack:
            state = stack.pop()
            fp = py_fingerprint(state)
            if fp in seen:
                continue
            seen.add(fp)
            if state.done:
                continue
            seat = state.acting
            ik = py_info_key(state, seat)
            ek = encode_key(state, seat)
            ik_to_enc[ik].add(ek)
            enc_to_ik[ek].add(ik)
            ik_to_pot[ik].add(state.pot)
            n_nodes += 1
            mask = py_legal(state)
            for a in range(NUM_ACTIONS):
                if mask[a]:
                    stack.append(apply_action_py(state, a))

    return deals, ik_to_enc, enc_to_ik, ik_to_pot, n_nodes


def main() -> None:
    print("=== (1) encode_leduc injectivity (distinct infosets -> distinct encodings) ===")
    coll = check_leduc_encode_collisions()
    print(f"  n_infosets (py_info_key nodes): {coll['n_infosets']}")
    print(f"  n_unique_encodes: {coll['n_unique_encodes']}")
    print(f"  encode collision groups (multiple infosets -> same encode): {coll['collision_groups']}")
    print(f"  inconsistent encode per single infoset key: {coll['inconsistent_keys']}")
    if coll["sample_collision"]:
        print(f"  sample collision infosets: {coll['sample_collision'][:3]}...")

    _, ik_to_enc, enc_to_ik, ik_to_pot, n_nodes = walk_all_nodes()
    print(f"  acting nodes walked: {n_nodes}")

    ik_multi_enc = {ik: encs for ik, encs in ik_to_enc.items() if len(encs) > 1}
    enc_multi_ik = {ek: iks for ek, iks in enc_to_ik.items() if len(iks) > 1}
    ik_multi_pot = {ik: pots for ik, pots in ik_to_pot.items() if len(pots) > 1}

    print(f"\n  py_info_key -> multiple encodings (same tabular key, different network input): {len(ik_multi_enc)}")
    if ik_multi_enc:
        ik = next(iter(ik_multi_enc))
        print(f"    example ik={ik} encodings={len(ik_multi_enc[ik])}")

    print(f"  encode -> multiple py_info_keys (encode collision / merged infosets): {len(enc_multi_ik)}")
    if enc_multi_ik:
        ek = next(iter(enc_multi_ik))
        print(f"    example {len(enc_multi_ik[ek])} infosets share one encode")

    print(f"  py_info_key -> multiple pot values (pot in encode but NOT in py_info_key): {len(ik_multi_pot)}")
    if ik_multi_pot:
        ik = next(iter(ik_multi_pot))
        print(f"    example ik={ik} pots={ik_multi_pot[ik]}")

    print("\n=== (2) strat_sum accumulation key vs BR / policy evaluation key ===")
    print("  tabular CFR strat_sum / avg dict key: py_info_key(state, seat)")
    print("  SD-CFR _strat_sum / _cum_regret key:    encode_leduc_py tuple (_encode_key)")
    print("  BR policy_fn lookup:                    encode_leduc (jax) via average_strategy_policy")
    print("  BR infoset grouping (_InfosetBestResponse): py_info_key(state, br_seat)")

    aligned = len(ik_multi_enc) == 0 and len(enc_multi_ik) == 0
    print(f"\n  keys aligned (bijection py_info_key <-> encode): {aligned}")
    if not aligned:
        print("  BUG: strat_sum fills encode buckets; tabular/L1/BR grouping use py_info_key — policies diverge.")

    print("\n=== (3) avg-strategy head (only if 1–2 clean) ===")
    if aligned and coll["passed"]:
        print("  Encoding and keys clean — avg-strategy head / recovery weighting is next suspect.")
    else:
        print("  Skipped — fix injectivity / key alignment first.")


if __name__ == "__main__":
    main()
