"""A2 depth-aware preflop key: the effective-starting-depth bucket must be hashed IDENTICALLY into the
train-encode and serve-fingerprint preflop keys (else train/serve skew -> 0% coverage), be CONSTANT within a
hand, route distinct depths to distinct keys, and be byte-identical to the non-depth key when the flag is OFF.

The depth/betting flags are import-time module constants; we monkeypatch them (the encode/fingerprint paths
re-read them per call, and _digest reads the module global) so the test needs no subprocess."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pokerbot.engine import jax_weva_encode as _enc
from pokerbot.engine.jax_engine import CHIP_UNIT, NUM_PLAYERS, init_hand
from pokerbot.engine.jax_weva_fast import (
    _DEPTH_EDGES,
    depth_bucket,
    effective_depth_bb,
    encode_nlhe_bundle_fast,
    fingerprint_fast,
)
from pokerbot.engine.np_engine import apply_action_np, legal_action_mask_np, to_numpy_state

_OCC = jnp.asarray(np.array([True] * 6, dtype=bool))


def _preflop_env(stack_bb: float, *, seed: int = 0):
    stacks = jnp.full(NUM_PLAYERS, int(stack_bb * CHIP_UNIT), dtype=jnp.int32)
    return to_numpy_state(init_hand(jax.random.PRNGKey(seed), occupancy=_OCC, stacks=stacks))


@pytest.fixture
def depth_on(monkeypatch):
    monkeypatch.setattr(_enc, "_BETTING_ABSTRACTION", True)
    monkeypatch.setattr(_enc, "_DEPTH_BUCKETS", True)
    return True


def test_effective_depth_and_bucket():
    assert abs(effective_depth_bb(_preflop_env(100.0)) - 100.0) < 1e-6
    assert abs(effective_depth_bb(_preflop_env(40.0)) - 40.0) < 1e-6
    # edges (50,90,140) -> 40bb=0 (short), 70bb=1 (mid), 110bb=2 (full/100bb cluster), 170bb=3 (deep run-up)
    assert depth_bucket(_preflop_env(40.0)) == 0
    assert depth_bucket(_preflop_env(70.0)) == 1
    assert depth_bucket(_preflop_env(110.0)) == 2
    assert depth_bucket(_preflop_env(170.0)) == 3


def test_v1_train_serve_key_identity(depth_on):
    # The serve fingerprint must equal the train-encode fingerprint at every bucket + boundary.
    for bb in (39.0, 40.0, 55.0, 70.0, 90.0, 100.0, 140.0, 200.0):
        env = _preflop_env(bb)
        seat = int(env.action_seat)
        assert fingerprint_fast(env, seat) == encode_nlhe_bundle_fast(env, seat).fingerprint(), bb


def test_distinct_depths_distinct_keys_same_bucket_same_key(depth_on):
    # Same deck/cards/seat, only stack differs -> different DEPTH BUCKET => different key.
    e50, e100 = _preflop_env(50.0), _preflop_env(100.0)
    seat = int(e50.action_seat)
    assert fingerprint_fast(e50, seat) != fingerprint_fast(e100, seat)
    # Within the SAME bucket [90,140), 100 and 120bb must collapse to the SAME key.
    assert fingerprint_fast(_preflop_env(100.0), seat) == fingerprint_fast(_preflop_env(120.0), seat)


def test_v2_depth_bucket_constant_within_hand(depth_on):
    # The effective STARTING depth (and thus the bucket) must not drift as chips commit preflop.
    env = _preflop_env(200.0)
    b0 = depth_bucket(env)
    for _ in range(3):
        seat = int(env.action_seat)
        if seat < 0 or bool(env.done):
            break
        mask = legal_action_mask_np(env, seat).astype(bool)
        a = int(np.argmax(mask))  # first legal action (fold/check) — just advance the betting
        env = apply_action_np(env, a)
        if int(env.street) == 0 and not bool(env.done):
            assert depth_bucket(env) == b0


def test_flag_off_is_byte_identical(monkeypatch):
    # With DEPTH_BUCKETS OFF, the depth byte is never hashed -> different depths share the preflop key.
    monkeypatch.setattr(_enc, "_BETTING_ABSTRACTION", True)
    monkeypatch.setattr(_enc, "_DEPTH_BUCKETS", False)
    seat = int(_preflop_env(50.0).action_seat)
    assert fingerprint_fast(_preflop_env(50.0), seat) == fingerprint_fast(_preflop_env(200.0), seat)


def test_depth_sampler_env_gated(monkeypatch):
    # A3 sampler: env-gated, deterministic per deal key, ONE log-U depth per deal broadcast to all seats
    # (so the effective/min-keyed depth sweeps the full range), fail-safe to None.
    from pokerbot.training.sd_cfr_nlhe_hogwild import _sample_depth_stacks
    k = jax.random.PRNGKey(7)
    monkeypatch.delenv("POKERBOT_TRAIN_DEPTH_BB", raising=False)
    assert _sample_depth_stacks(k) is None                      # unset -> fixed 100bb (None)
    monkeypatch.setenv("POKERBOT_TRAIN_DEPTH_BB", "40,200")
    s = np.asarray(_sample_depth_stacks(k))
    assert s.shape == (NUM_PLAYERS,) and s.dtype == np.int32
    assert (s >= 40 * CHIP_UNIT).all() and (s <= 200 * CHIP_UNIT).all()
    assert np.array_equal(s, np.asarray(_sample_depth_stacks(k)))  # deterministic for the same key
    assert len(set(s.tolist())) == 1                            # SHARED depth: all seats equal per deal
    monkeypatch.setenv("POKERBOT_TRAIN_DEPTH_BB", "garbage")
    assert _sample_depth_stacks(k) is None                      # malformed -> fail-safe None
    # the pool/distributed trainer's copy MUST produce identical stacks for the same deal key (both
    # production trainers must randomize depth the same way, or merged tables would be incoherent).
    monkeypatch.setenv("POKERBOT_TRAIN_DEPTH_BB", "40,200")
    from pokerbot.training.sd_cfr_nlhe_distributed import _sample_depth_stacks as _dist_sampler
    assert np.array_equal(np.asarray(_sample_depth_stacks(k)), np.asarray(_dist_sampler(k)))


def test_depth_sampler_fills_all_buckets(monkeypatch):
    # The keyed variable is the effective (min-over-seats) depth. With a SHARED per-deal depth that equals
    # the drawn depth, every depth bucket must receive a healthy share of deals over many keys — the old
    # independent per-seat draw collapsed the min to the short end and starved the deep buckets (~0%),
    # which is the entire failure this sampler shape fixes.
    from pokerbot.training.sd_cfr_nlhe_hogwild import _sample_depth_stacks
    monkeypatch.setenv("POKERBOT_TRAIN_DEPTH_BB", "30,200")
    counts = [0, 0, 0, 0]
    N = 600
    for i in range(N):
        env = to_numpy_state(init_hand(jax.random.PRNGKey(1000 + i), occupancy=_OCC,
                                       stacks=_sample_depth_stacks(jax.random.PRNGKey(1000 + i))))
        counts[depth_bucket(env)] += 1
    assert len(_DEPTH_EDGES) + 1 == 4
    # every bucket including the deep tail [140,200) must be non-trivially sampled (log-U over 30-200)
    for b, c in enumerate(counts):
        assert c >= 0.08 * N, f"bucket {b} starved: {c}/{N} ({counts})"


def test_sampled_stacks_feed_init_hand(monkeypatch):
    # The sampled stacks must produce a hand whose effective depth equals the min sampled stack.
    from pokerbot.training.sd_cfr_nlhe_hogwild import _sample_depth_stacks
    monkeypatch.setenv("POKERBOT_TRAIN_DEPTH_BB", "40,200")
    k = jax.random.PRNGKey(3)
    stacks = _sample_depth_stacks(k)
    env = to_numpy_state(init_hand(k, occupancy=_OCC, stacks=stacks))
    expect = float(np.asarray(stacks).min()) / CHIP_UNIT
    assert abs(effective_depth_bb(env) - expect) < 1e-6


def test_occupancy_sampler_env_gated_and_parity(monkeypatch):
    # C1 sampler: env-gated, deterministic per deal key, a size-n subset of the 6 seats, fail-safe to None,
    # and BYTE-identical across the hogwild + distributed trainer copies (same deal key -> same occupancy).
    from pokerbot.training.sd_cfr_nlhe_hogwild import _sample_occupancy
    k = jax.random.PRNGKey(11)
    monkeypatch.delenv("POKERBOT_TRAIN_OCCUPANCY_MIX", raising=False)
    assert _sample_occupancy(k) is None                       # unset -> fixed --occupancy (None)
    monkeypatch.setenv("POKERBOT_TRAIN_OCCUPANCY_MIX", "2,6")
    occ = np.asarray(_sample_occupancy(k))
    assert occ.shape == (NUM_PLAYERS,) and occ.dtype == bool
    assert 2 <= int(occ.sum()) <= 6
    assert np.array_equal(occ, np.asarray(_sample_occupancy(k)))   # deterministic for the same key
    for bad in ("garbage", "1,6", "2,7", "6,2"):
        monkeypatch.setenv("POKERBOT_TRAIN_OCCUPANCY_MIX", bad)
        assert _sample_occupancy(k) is None                    # malformed / out-of-range -> fail-safe None
    monkeypatch.setenv("POKERBOT_TRAIN_OCCUPANCY_MIX", "2,6")
    from pokerbot.training.sd_cfr_nlhe_distributed import _sample_occupancy as _dist_occ
    assert np.array_equal(np.asarray(_sample_occupancy(k)), np.asarray(_dist_occ(k)))


def test_occupancy_sampler_spans_all_counts(monkeypatch):
    # Over many keys the live-seat count must span the whole {2,3,4,5,6} range (so every short-handed
    # config gets trained), weighted toward fuller tables (the live mode).
    from pokerbot.training.sd_cfr_nlhe_hogwild import _sample_occupancy
    monkeypatch.setenv("POKERBOT_TRAIN_OCCUPANCY_MIX", "2,6")
    counts = {}
    for i in range(400):
        occ = np.asarray(_sample_occupancy(jax.random.PRNGKey(5000 + i)))
        counts[int(occ.sum())] = counts.get(int(occ.sum()), 0) + 1
    assert set(counts) == {2, 3, 4, 5, 6}, counts          # every seat-count sampled
    assert counts[6] > counts[2]                            # weighted toward fuller tables


def test_occupancy_feeds_init_hand(monkeypatch):
    # The sampled occupancy must produce a hand with exactly that many LIVE seats.
    from pokerbot.engine.jax_engine import SEAT_LIVE
    from pokerbot.training.sd_cfr_nlhe_hogwild import _sample_occupancy
    monkeypatch.setenv("POKERBOT_TRAIN_OCCUPANCY_MIX", "2,6")
    k = jax.random.PRNGKey(123)
    occ = _sample_occupancy(k)
    env = to_numpy_state(init_hand(k, occupancy=jnp.asarray(occ)))
    assert int((np.asarray(env.seat_status) == SEAT_LIVE).sum()) == int(np.asarray(occ).sum())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
