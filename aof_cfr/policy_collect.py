"""Collect tabular policy rows without full chance-tree enumeration."""

from __future__ import annotations

import numpy as np
import pyspiel

from aof_cfr.buckets import COMBOS_169, LABELS_169
from aof_cfr.combo_deal import card_id
from aof_cfr.game_pushfold import _encode_pair
from aof_cfr.infoset import encode_infoset_key
from aof_ranges import active_seats


def _pick_disjoint_combos(bucket_idxs: list[int]) -> list[tuple[str, str]] | None:
    chosen: list[tuple[str, str]] = []
    used: set[str] = set()

    def dfs(i: int) -> bool:
        if i == len(bucket_idxs):
            return True
        for c1, c2 in COMBOS_169[bucket_idxs[i]]:
            if c1 in used or c2 in used:
                continue
            chosen.append((c1, c2))
            used.add(c1)
            used.add(c2)
            if dfs(i + 1):
                return True
            used.remove(c1)
            used.remove(c2)
            chosen.pop()
        return False

    return chosen if dfs(0) else None


def _deal_to_state(game: pyspiel.Game, combos: list[tuple[str, str]]) -> pyspiel.State:
    st = game.new_initial_state()
    for c0, c1 in combos:
        st = st.child(_encode_pair(card_id(c0), card_id(c1)))
    return st


def _replay_to_history(game: pyspiel.Game, combos: list[tuple[str, str]], history: str) -> pyspiel.State | None:
    st = _deal_to_state(game, combos)
    tokens = [] if history in ("", "-") else list(history)
    for tok in tokens:
        if st.is_terminal() or st.is_chance_node():
            return None
        action = 0 if tok == "F" else 1
        if action not in st.legal_actions(st.current_player()):
            return None
        st = st.child(action)
    if st.is_terminal() or st.is_chance_node():
        return None
    return st


def _histories_for_table(n: int) -> dict[str, set[str]]:
    seats = active_seats(n)
    histories: dict[str, set[str]] = {s: set() for s in seats}

    def rec(step: int, history: str, n_players: int) -> None:
        if step >= n_players:
            return
        seat = seats[step]
        h = history if history else "-"
        histories[seat].add(h)
        rec(step + 1, history + "F", n_players)
        rec(step + 1, history + "A", n_players)

    rec(0, "", n)
    return histories


def collect_pushfold_policy_rows(game: pyspiel.Game, policy) -> dict[str, np.ndarray]:
    n = game.num_players()
    seats = active_seats(n)
    histories = _histories_for_table(n)
    rows: dict[str, np.ndarray] = {}

    for si, seat in enumerate(seats):
        for bi, label in enumerate(LABELS_169):
            for hist in histories[seat]:
                bucket_idxs = []
                ok = True
                for sj in range(n):
                    if sj == si:
                        bucket_idxs.append(bi)
                    else:
                        bucket_idxs.append(0)
                combos = _pick_disjoint_combos(bucket_idxs)
                if combos is None:
                    ok = False
                    for fallback in range(1, len(LABELS_169)):
                        bucket_idxs = [bi if sj == si else fallback for sj in range(n)]
                        combos = _pick_disjoint_combos(bucket_idxs)
                        if combos is not None:
                            ok = True
                            break
                if not ok or combos is None:
                    continue
                st = _replay_to_history(game, combos, hist)
                if st is None:
                    continue
                p = st.current_player()
                key = st.information_state_string(p)
                if key != encode_infoset_key(seat, label, hist):
                    continue
                if key in rows:
                    continue
                probs = policy.action_probabilities(st)
                vec = np.zeros(2, dtype=np.float64)
                for action, prob in probs.items():
                    vec[int(action)] = float(prob)
                ssum = vec.sum()
                rows[key] = vec / ssum if ssum > 0 else np.array([0.5, 0.5])
    return rows
