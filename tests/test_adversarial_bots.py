"""Phase 5 (G5): adversarial strong-player opponents — tilt detection direction and battery hookup."""

from __future__ import annotations

import os
import random

os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")

from pokerbot.engine.actions import Action, ActionType
from pokerbot.eval.bots import CounterExploiterBot, build_bot


def _feed(bot: CounterExploiterBot, kinds: str) -> None:
    m = {"a": ActionType.BET, "f": ActionType.FOLD, "c": ActionType.CALL, "k": ActionType.CHECK}
    for ch in kinds:
        bot.observe_hero(Action(m[ch]))


def test_tilt_detection_directions():
    b = CounterExploiterBot(random.Random(0))
    assert b._tilt() == "balanced"                       # no observations -> baseline
    _feed(b, "a" * 30 + "k" * 20)                        # 60% aggressive
    assert b._tilt() == "aggro"
    b2 = CounterExploiterBot(random.Random(0))
    _feed(b2, "f" * 25 + "k" * 25)                       # 50% folds
    assert b2._tilt() == "fold"
    b3 = CounterExploiterBot(random.Random(0))
    _feed(b3, "c" * 30 + "k" * 20)                       # 60% calls
    assert b3._tilt() == "call"
    b4 = CounterExploiterBot(random.Random(0))
    _feed(b4, "a" * 6 + "f" * 6 + "c" * 6 + "k" * 12)    # mixed, nothing exceeds
    assert b4._tilt() == "balanced"


def test_counter_exploiter_battery_runs():
    """The battery's adaptive hook feeds observe_hero and the run completes (ON vs OFF arms)."""
    from pokerbot.eval.exploitation_battery import run_archetype

    r = run_archetype("counter_exploiter", hands=25, seed=0, progress_every=0)
    assert r["hands"] == 25
    assert "uplift_bb_per_hand" in r


def test_blueprint_bot_plays_legal_actions():
    """BlueprintBot samples the snapshot's mixed strategy (or its fallbacks) and always returns a
    LEGAL action across many states. Uses an empty snapshot via env override (adapter fallback
    path) so the test has no checkpoint dependency."""
    import pickle
    import tempfile
    from pathlib import Path

    from pokerbot.engine.game import GameState

    with tempfile.TemporaryDirectory() as td:
        snap = Path(td) / "empty.pkl"
        snap.write_bytes(pickle.dumps({}))
        os.environ["POKERBOT_BLUEPRINT_SNAPSHOT"] = str(snap)
        try:
            bot = build_bot("blueprint", random.Random(3))
            for seed in range(8):
                st = GameState.new_hand(button_seat=seed % 6, rng=random.Random(seed))
                st.deal_hole_cards()
                for _ in range(12):
                    if sum(1 for p in st.players if not p.folded) < 2:
                        break
                    a = bot(st)
                    assert a in st.legal_actions() or a.action_type in {
                        x.action_type for x in st.legal_actions()}
                    st.apply_action(a)
                    if not st.advance_to_next_actor():
                        break
        finally:
            del os.environ["POKERBOT_BLUEPRINT_SNAPSHOT"]
