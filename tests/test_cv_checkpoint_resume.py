"""Checkpoint/resume + off_only + parallel-pool gate for run_field_absolute_rotated (the CV rotated run).

These pin the three crash-recovery features added for the overnight CV "absolute rotated-duplicate
win-rate" job, each of which is OPT-IN and must reduce to the byte-identical legacy path when unset:

A. ``off_only`` — skip the discarded ON (exploit) seat rotation + the bot_stats learn block for the
   ``arm=="off"`` CV path. The OFF value is a pure fn of (seed, deck) computed BEFORE the ON arm, so
   ``off_deck_seat_means`` / ``off_base`` must be BYTE-IDENTICAL to a full run (==, not a tolerance).
B. ``POKERBOT_ROTATED_WORKERS`` process pool — already gated by test_rotated_parallel_equivalence;
   here we co-exercise it WITH the checkpoint stream.
C. per-deck checkpoint + resume — each completed OFF deck is appended (one JSONL line) so a hard kill
   resumes instead of restarts. The aggregate is order-independent (np.mean of a pure-fn series), so a
   truncated-then-resumed run is byte-identical to an uninterrupted one.

Uses the lightweight ExploitHero (no search, milliseconds) like the sibling rotated/CV wiring tests.
"""

from __future__ import annotations

import json
import os
import random

import pytest

from pokerbot.eval.exploitation_battery import (
    ExploitHero,
    field_spec,
    run_field_absolute_rotated,
)

FIELD = field_spec("homog:calling_station")


def _hero_factory(*, exploit: bool):
    return ExploitHero(random.Random(0), exploit=exploit)


def _run(*, decks, seed, workers=0, **kw):
    """Run the rotated eval with POKERBOT_ROTATED_WORKERS scoped to this call (restored after)."""
    prev = os.environ.get("POKERBOT_ROTATED_WORKERS")
    os.environ["POKERBOT_ROTATED_WORKERS"] = str(workers)
    try:
        return run_field_absolute_rotated(
            FIELD, decks=decks, seed=seed, hero_factory=_hero_factory,
            allin_ev=True, progress_every=0, **kw)
    finally:
        if prev is None:
            os.environ.pop("POKERBOT_ROTATED_WORKERS", None)
        else:
            os.environ["POKERBOT_ROTATED_WORKERS"] = prev


def _read_ckpt(path):
    """Parse a checkpoint JSONL -> (meta_dict_or_None, {d: val})."""
    meta = None
    done = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "_meta" in rec:
                meta = rec["_meta"]
            else:
                done[int(rec["d"])] = float(rec["val"])
    return meta, done


# --- A: off_only -------------------------------------------------------------------------------

def test_off_only_off_arm_byte_identical_to_full():
    full = _run(decks=8, seed=2)
    off = _run(decks=8, seed=2, off_only=True)
    # OFF series + aggregate are byte-identical (off_v is computed before the ON arm + bot_stats).
    assert off["off_deck_seat_means"] == full["off_deck_seat_means"]
    assert off["off_base"]["mean_bb"] == full["off_base"]["mean_bb"]
    assert off["off_base"]["std"] == full["off_base"]["std"]
    assert off["off_base"]["se"] == full["off_base"]["se"]
    # off_only carries NO ON arm.
    assert off["on_deck_seat_means"] == []
    assert off["on_exploit"]["decks"] == 0


def test_off_only_rejects_chance_cv_and_rake():
    with pytest.raises(ValueError):
        _run(decks=4, seed=0, off_only=True, chance_cv=True)
    # a real stake key from pokerbot.economics.rake.STAKES
    from pokerbot.economics.rake import STAKES
    stake = next(iter(STAKES))
    with pytest.raises(ValueError):
        _run(decks=4, seed=0, off_only=True, net_of_rake_stake=stake)


# --- C: checkpoint write ------------------------------------------------------------------------

def test_checkpoint_writes_one_line_per_deck(tmp_path):
    ckpt = tmp_path / "search@0.jsonl"
    r = _run(decks=8, seed=0, off_only=True, checkpoint_path=str(ckpt))
    assert ckpt.exists()
    meta, done = _read_ckpt(ckpt)
    assert meta is not None
    assert meta["seed"] == 0 and meta["decks"] == 8 and meta["n_seats"] == 6
    assert meta["pass_key"] == "search@0"
    assert len(done) == 8 and set(done) == set(range(8))
    # the checkpointed {d: val} reproduces the reported per-deck series exactly
    for d, v in enumerate(r["off_deck_seat_means"]):
        assert done[d] == v


# --- C: resume aggregate == uninterrupted (both pool seams) ------------------------------------

@pytest.mark.parametrize("workers", [0, 2])
def test_resume_aggregate_equals_uninterrupted(tmp_path, workers):
    # Reference: one clean uninterrupted run.
    ref = _run(decks=8, seed=0, off_only=True)

    ckpt = tmp_path / f"search@0_w{workers}.jsonl"
    # First run to completion to populate the checkpoint.
    _run(decks=8, seed=0, off_only=True, workers=workers, checkpoint_path=str(ckpt))
    meta, done = _read_ckpt(ckpt)
    assert len(done) == 8

    # Simulate a crash at deck 4: keep the header + first 4 deck lines only.
    lines = ckpt.read_text(encoding="utf-8").splitlines()
    header = lines[0]
    assert "_meta" in header
    deck_lines = lines[1:]
    truncated = [header] + deck_lines[:4]
    ckpt.write_text("\n".join(truncated) + "\n", encoding="utf-8")
    _, partial = _read_ckpt(ckpt)
    assert len(partial) == 4

    # Resume: only the missing decks recompute; the aggregate must equal the uninterrupted run.
    res = _run(decks=8, seed=0, off_only=True, workers=workers,
               checkpoint_path=str(ckpt), resume=True)
    assert res["off_deck_seat_means"] == ref["off_deck_seat_means"]
    assert res["off_base"]["mean_bb"] == ref["off_base"]["mean_bb"]
    assert res["off_base"]["std"] == ref["off_base"]["std"]
    # No deck is double-counted: the file has exactly 8 unique deck records after resume.
    _, after = _read_ckpt(ckpt)
    assert set(after) == set(range(8))


def test_resume_header_mismatch_raises(tmp_path):
    ckpt = tmp_path / "search@0.jsonl"
    # Write a checkpoint whose header is for a DIFFERENT run (seed/decks).
    with open(ckpt, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"_meta": {"seed": 99, "decks": 999, "n_seats": 6,
                                       "off_only": True, "pass_key": "search@0"}}) + "\n")
        fh.write(json.dumps({"d": 0, "val": 1.23}) + "\n")
    with pytest.raises(ValueError):
        _run(decks=8, seed=0, off_only=True, checkpoint_path=str(ckpt), resume=True)


# --- opt-in proof: defaults are byte-identical -------------------------------------------------

def test_checkpoint_disabled_is_byte_identical():
    a = _run(decks=8, seed=4)                      # no checkpoint kwargs at all
    b = _run(decks=8, seed=4, checkpoint_path=None, resume=False)  # explicit-None opt-out
    assert a["off_deck_seat_means"] == b["off_deck_seat_means"]
    assert a["on_deck_seat_means"] == b["on_deck_seat_means"]
    assert a["off_base"] == b["off_base"]
    assert a["on_exploit"] == b["on_exploit"]
    assert a["uplift_bb_per_hand"] == b["uplift_bb_per_hand"]
    assert a["profile_contexts"] == b["profile_contexts"]
