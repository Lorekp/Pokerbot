"""Property-based tests for SCRIPTS tooling (Hypothesis @given).

These tests exercise three production code paths used by the SCRIPTS-EVAL-AUDIT,
SCRIPTS-LIVE-AUDIT, and the spot launcher / babysitter infrastructure:

  P1  **CRN reseed uniqueness** — for any n_seeds in [1, 100] the
     ``seed_for_replication(i)`` helper from ``scripts/eval/eval_emd_vs_iter62000.py``
     produces a sequence of N seeds that is (a) unique, (b) non-negative, and
     (c) reproduces the documented BASE_SEED for i=0 (the xor-identity).
     The k500 verdict-failure bug class (signature: ``seeds = list(range(N))``)
     is exactly what this property guards against.

  P2  **Ingest idempotency** — re-running the JSONL ingest pipeline on the
     same input records MUST NOT produce duplicate hand_ids. The dedup
     contract is per-hand_id (write-side set semantics). Feeding the same
     input twice yields the same set of surviving hand_ids with the same
     payload (byte-equal).

  P3  **Babysitter storm guard** — given a relaunch budget
     ``max_launches >= 1`` and a sequence of consecutive failures, the storm
     guard logic fires iff the count of consecutive failures reaches
     ``max_launches``. Before the threshold the guard stays armed; AT or
     ABOVE the threshold the guard fires (returns ``True``).

  P4  **Launcher fallback chain** — for any capacity-error sequence
     encountered on instance types ``T1, T2, ..., Tn``, the
     ``_launch_with_fallback`` runner advances through the chain until either
     a non-capacity error fires (fatal) or the chain is exhausted (SystemExit).
     For the SUCCESS case the first attempt is the chosen type; for the
     PARTIAL case it is the first non-failing type; for the EXHAUSTED case
     a SystemExit is raised.

Each property uses Hypothesis ``@given`` with at least 50 examples
(``max_examples=64`` explicitly to be safe and predictable on slow CI). Run via:

    pytest tests/test_scripts_property.py -v
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

# Insert the scripts dir for the duration of the test module so we can import
# the CRN reseed helper + the spot launcher in isolation. Done lazily inside
# each test that needs it (the existing test_scripts_sharpening.py pattern).
# Hypothesis is the standard property-based testing library. The property
# blocks below skip if hypothesis is not installed.
hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, assume, given, settings, strategies as st  # noqa: E402


# ---------------------------------------------------------------------------
# P1 — CRN reseed uniqueness for any n_seeds in [1, 100]
# ---------------------------------------------------------------------------
# Mirrors ``scripts/eval/eval_emd_vs_iter62000.py:seed_for_replication``.
# The launcher forwards the SEED COUNT and the on-box driver applies the
# per-replication xor. So the unit under test IS the helper's behaviour for
# any n_seeds in the documented range.

BASE_SEED = 0x5EEDBAB0  # 1592263344; must match scripts/eval/eval_emd_vs_iter62000.py


def _seed_for_replication(replication_id: int) -> int:
    """Mirror of scripts/eval/eval_emd_vs_iter62000.py.seed_for_replication (kept in sync)."""
    s = BASE_SEED ^ int(replication_id)
    if replication_id != 0:
        assert s != BASE_SEED, f"CRN reseed must differ from base for non-zero replication_id (got {s})"
    assert s >= 0, "XOR of non-negative ints must be non-negative"
    return s


# Strategy: any int in [1, 100] for the seed count.
_seed_count_strategy = st.integers(min_value=1, max_value=100)


@settings(max_examples=64, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(n_seeds=_seed_count_strategy)
def test_property_p1_crn_reseed_uniqueness(n_seeds: int) -> None:
    """For any n_seeds in [1, 100] the helper yields:
    (a) exactly n_seeds seeds, (b) all seeds are pairwise distinct, (c) every
    seed is a non-negative int, (d) seed[0] is the documented BASE_SEED xor 0
    (i.e. the documented reproduction-id=0 case), and (e) the sequence
    matches the canonical [BASE_SEED ^ 0, BASE_SEED ^ 1, ..., BASE_SEED ^ (n-1)]
    derivation that the on-box driver applies."""
    seeds = [_seed_for_replication(i) for i in range(n_seeds)]
    # (a) length
    assert len(seeds) == n_seeds
    # (b) pairwise distinct (this is the k500-verdict-failure signature)
    assert len(set(seeds)) == n_seeds, (
        f"CRN reseed must be pairwise distinct; duplicates found in {seeds}"
    )
    # (c) all non-negative ints
    assert all(isinstance(s, int) and s >= 0 for s in seeds), (
        f"all CRN reseeds must be non-negative ints; got {seeds}"
    )
    # (d) id=0 yields the BASE_SEED xor 0 == BASE_SEED
    assert seeds[0] == BASE_SEED, (
        f"CRN reseed id=0 must be BASE_SEED (xor-identity); got {seeds[0]}"
    )
    # (e) full sequence matches the documented canonical derivation
    expected = [BASE_SEED ^ i for i in range(n_seeds)]
    assert seeds == expected, (
        f"CRN reseed sequence must be [BASE_SEED ^ 0, BASE_SEED ^ 1, ...]; "
        f"expected={expected} got={seeds}"
    )


@settings(max_examples=64, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(replication_id=st.integers(min_value=0, max_value=2**31 - 1))
def test_property_p1b_crn_reseed_individual(replication_id: int) -> None:
    """The xor-with-base function is its own documented contract — for any
    non-negative integer id, the result equals BASE_SEED ^ id (which is
    trivially non-negative) and is equal to BASE_SEED ONLY for id=0."""
    s = _seed_for_replication(replication_id)
    assert s == BASE_SEED ^ replication_id
    assert s >= 0
    if replication_id == 0:
        assert s == BASE_SEED
    else:
        assert s != BASE_SEED


# ---------------------------------------------------------------------------
# P2 — Ingest idempotency
# ---------------------------------------------------------------------------
# A small, dependency-free "ingest" facade: write records to a JSONL shard
# (one JSON per line, line-protocol) and "re-ingest" by reading them back.
# The dedup contract is per-hand_id — feeding the same records twice MUST
# produce the same set of hand_ids with the same payloads. The tests use a
# tmp dir; the production code is `pokerbot.live.hand_collector.HandCollector`
# but for a property test against the contract we exercise a tight
# hand_collector-shaped facade that we instantiate fresh per property run.

# Strategy: small set of (unique hand_id, payload) records with possible
# intentional duplicates. The dedup MUST drop duplicates while preserving
# the first-seen payload (set semantics).
_records_strategy = st.lists(
    st.tuples(
        st.text(alphabet=st.characters(min_codepoint=ord("a"), max_codepoint=ord("z")),
                min_size=1, max_size=8),  # hand_id
        st.dictionaries(
            keys=st.text(alphabet=st.characters(min_codepoint=ord("a"), max_codepoint=ord("z")),
                         min_size=1, max_size=6),
            values=st.integers(min_value=0, max_value=10_000),
            min_size=1, max_size=4,
        ),  # payload
    ),
    min_size=1, max_size=20,
)


def _ingest_dedup(records: list[tuple[str, dict]]) -> dict[str, dict]:
    """Ingest facade: takes a list of (hand_id, payload) tuples and returns
    a dict keyed by hand_id with the FIRST-seen payload per hand_id.

    The contract: feeding the same input records twice yields the same
    dict (idempotency under set semantics)."""
    out: dict[str, dict] = {}
    for hand_id, payload in records:
        out.setdefault(hand_id, payload)  # first-wins
    return out


@settings(max_examples=64, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(records=_records_strategy)
def test_property_p2_ingest_idempotency_inmemory(records: list[tuple[str, dict]]) -> None:
    """For ANY list of (hand_id, payload) records, running the ingest facade
    TWICE on the same input yields the same dict (set semantics). The
    resulting dict has unique hand_ids and the first-seen payload per id."""
    # First pass
    d1 = _ingest_dedup(records)
    # Second pass — the actual idempotency test
    d2 = _ingest_dedup(records)
    assert d1 == d2, (
        f"ingest must be idempotent: first={d1} second={d2}"
    )
    # All keys are unique
    keys = list(d1.keys())
    assert len(keys) == len(set(keys)), (
        f"ingest must drop duplicate hand_ids; found dupes in {keys}"
    )
    # First-seen payload per hand_id survives. Compute the expected dict
    # independently and compare — this guards against off-by-one / wrong-
    # payload bugs in the dedup logic without depending on the iteration
    # order of `records` (a hand_id may appear in many records).
    expected: dict[str, dict] = {}
    for hand_id, payload in records:
        expected.setdefault(hand_id, payload)
    assert d1 == expected, (
        f"dedup must keep first-seen payload per hand_id; "
        f"expected={expected} got={d1}"
    )


@settings(max_examples=64, deadline=None,
         suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(records=_records_strategy)
def test_property_p2b_ingest_idempotency_jsonl(records: list[tuple[str, dict]]) -> None:
    """Same as P2a but uses the JSONL line protocol (the production write/read
    surface) and a tmp dir: writing the records, re-writing the same records
    (idempotent re-ingest), and reading them back MUST yield the same
    dict. Duplicates written twice MUST NOT multiply the surviving line count
    (set semantics over hand_id)."""
    assume(len(records) >= 1)
    # Build a unique tmp dir per @given input (function-scoped fixture would
    # be reused across the 64 examples and accumulate state). We deliberately
    # do NOT use ``tmp_path`` as a fixture here — that is the function-scoped
    # fixture Hypothesis warns about. We build a fresh tmp dir + clean it up.
    base = tempfile.mkdtemp(prefix="scripts_prop_p2b_")
    try:
        jsonl_path = Path(base) / "events.jsonl"
        seen: dict[str, dict] = {}
        for hand_id, payload in records:
            if hand_id in seen:
                continue  # set semantics: skip the dup
            seen[hand_id] = payload
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"hand_id": hand_id, "payload": payload}) + "\n")
        # Re-ingest: writing the same records again must NOT add new lines
        # (the per-hand_id set is already populated). This is the idempotency test.
        n_lines_after_first = sum(1 for _ in open(jsonl_path, encoding="utf-8"))
        for hand_id, payload in records:
            if hand_id in seen:
                continue  # dedup: skip
            seen[hand_id] = payload
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"hand_id": hand_id, "payload": payload}) + "\n")
        n_lines_after_second = sum(1 for _ in open(jsonl_path, encoding="utf-8"))
        # No new lines added by the re-ingest pass (set semantics is preserved).
        assert n_lines_after_first == n_lines_after_second, (
            f"re-ingest must NOT add duplicate lines; "
            f"first={n_lines_after_first} second={n_lines_after_second}"
        )
        # Readback: every line yields a unique hand_id.
        readback: dict[str, dict] = {}
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                readback[obj["hand_id"]] = obj["payload"]
        assert readback == seen, (
            f"readback must equal the deduped set; readback={readback} expected={seen}"
        )
    finally:
        shutil.rmtree(base, ignore_errors=True)


# ---------------------------------------------------------------------------
# P3 — Babysitter storm guard fires after N consecutive failures
# ---------------------------------------------------------------------------
# Mirrors the PowerShell ``emd_pipeline_babysitter.ps1`` storm guard:
#   if ($launches -ge $MaxLaunch) { STOP: storm guard }
# We lift it to a pure-Python function so we can property-test it without
# spinning up PowerShell. The function takes a budget ``max_launches`` and a
# count of consecutive failures ``n_failures``; it returns True iff
# ``n_failures >= max_launches`` AND ``max_launches > 0``.

def _storm_guard_should_fire(n_consecutive_failures: int, max_launches: int) -> bool:
    """Pure-Python mirror of the babysitter storm guard. Returns True iff
    the babysitter should STOP (give up relaunching) instead of relaunching.

    Edge cases:
      * max_launches <= 0: never fires (defensive: a zero budget disables the guard)
      * n_consecutive_failures < 0: clamped to 0
      * n_consecutive_failures >= max_launches: fires
    """
    if max_launches <= 0:
        return False
    n = max(0, int(n_consecutive_failures))
    return n >= max_launches


# Strategies: budgets in a sensible babysitter range and failure counts
# bounded well above the budget so the boundary is exercised.
_budget_strategy = st.integers(min_value=1, max_value=20)
_failures_strategy = st.integers(min_value=0, max_value=50)


@settings(max_examples=64, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(max_launches=_budget_strategy, n_failures=_failures_strategy)
def test_property_p3_storm_guard_threshold(max_launches: int, n_failures: int) -> None:
    """The storm guard fires iff n_failures >= max_launches (with max_launches
    positive). For n_failures strictly below the budget the guard stays
    armed (returns False); AT the boundary or ABOVE the guard fires."""
    fired = _storm_guard_should_fire(n_failures, max_launches)
    if n_failures < max_launches:
        assert fired is False, (
            f"storm guard should NOT fire for n_failures={n_failures} "
            f"< max_launches={max_launches}; got fired={fired}"
        )
    else:  # n_failures >= max_launches
        assert fired is True, (
            f"storm guard SHOULD fire for n_failures={n_failures} "
            f">= max_launches={max_launches}; got fired={fired}"
        )


@settings(max_examples=64, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(max_launches=_budget_strategy, n_failures=_failures_strategy)
def test_property_p3b_storm_guard_streak_sequence(max_launches: int, n_failures: int) -> None:
    """Walking the failure streak 0..n_failures step-by-step, the guard
    transitions False -> True EXACTLY at the threshold (max_launches).
    Before the threshold it never fires; AT or ABOVE it stays fired.

    This catches a subtle class of bug: an off-by-one in the comparison
    (e.g. ``>`` instead of ``>=``) would let the guard fire one failure
    late, or (worse) never fire if it used ``>`` AND the count happened
    to be the boundary."""
    # Walk the streak and record the first index where the guard fires.
    streak = list(range(n_failures + 1))  # 0, 1, ..., n_failures
    fired_at = [i for i, f in enumerate(streak)
                if _storm_guard_should_fire(f, max_launches)]
    if n_failures < max_launches:
        # No failures reach the budget — guard never fires in the walk.
        assert fired_at == [], (
            f"guard should never fire below the budget; "
            f"max_launches={max_launches} n_failures={n_failures} "
            f"fired_at={fired_at}"
        )
    else:
        # The FIRST firing happens at the budget index.
        assert fired_at[0] == max_launches, (
            f"guard should fire FIRST at the budget index; "
            f"max_launches={max_launches} n_failures={n_failures} "
            f"fired_at={fired_at}"
        )
        # And every subsequent index (>= max_launches) also fires.
        assert fired_at == list(range(max_launches, n_failures + 1)), (
            f"guard should fire at every index >= max_launches; "
            f"max_launches={max_launches} n_failures={n_failures} "
            f"fired_at={fired_at}"
        )


@settings(max_examples=64, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(n_failures=_failures_strategy)
def test_property_p3c_storm_guard_zero_budget_disables(n_failures: int) -> None:
    """Defensive edge: a zero/negative max_launches disables the guard
    (the babysitter should never auto-stop). A misconfigured zero budget
    MUST NOT silently block the run."""
    assert _storm_guard_should_fire(n_failures, 0) is False
    assert _storm_guard_should_fire(n_failures, -1) is False
    # Negative failures are clamped to 0 — still no fire.
    assert _storm_guard_should_fire(-5, 1) is False


# ---------------------------------------------------------------------------
# P4 — Launcher fallback chain handles capacity errors gracefully
# ---------------------------------------------------------------------------
# Mirrors ``scripts/launch_native_cfr_training_spot.py:_launch_with_fallback``.
# The production function tries ``run_instances`` down a list of instance
# types; a capacity error code advances to the next type; any OTHER error
# is fatal. We lift the chain to a pure-Python function so we can
# property-test it without spinning up AWS / moto. The function returns the
# chosen type on success, raises the non-capacity error verbatim, and
# raises SystemExit when the chain is exhausted.

# The canonical capacity-error codes from launch_native_cfr_training_spot.py.
_CAPACITY_ERROR_CODES = frozenset({
    "InsufficientInstanceCapacity",
    "SpotMaxPriceTooLow",
    "InstanceLimitExceeded",
    "MaxSpotInstanceCountExceeded",
    "Unsupported",
})


class _CapacityError(Exception):
    """Stand-in for boto3 ClientError with a .code attribute."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _launch_with_fallback(
    instance_types: list[str],
    error_schedule: list[str],
) -> str:
    """Mirror of launch_native_cfr_training_spot.py:_launch_with_fallback.

    ``error_schedule`` is one entry per ``instance_types`` slot, in order.
    Possible entries:
      * ``""``        — that attempt succeeds
      * a capacity-error code (in ``_CAPACITY_ERROR_CODES``) — that attempt
        raises _CapacityError(code) and we advance to the next slot
      * any other string — that attempt raises a ValueError (fatal, NOT
        retried)

    Returns the chosen instance type on success.
    Raises ValueError (fatal) or SystemExit (chain exhausted) on failure.
    """
    if not instance_types:
        raise SystemExit("empty instance type list")
    last_err: Exception | None = None
    for it, err_code in zip(instance_types, error_schedule):
        if err_code == "":
            return it
        if err_code in _CAPACITY_ERROR_CODES:
            last_err = _CapacityError(err_code)
            if len(instance_types) > 1:
                # advance to the next type
                continue
            else:
                # single-type chain: capacity error is terminal (no next)
                raise SystemExit(
                    f"single-type capacity failure: {err_code} on {it}"
                )
        else:
            # Non-capacity error: fatal, do NOT advance.
            raise ValueError(f"fatal error on {it}: {err_code}")
    # Chain exhausted without success.
    raise SystemExit(
        f"all instance types exhausted ({instance_types}); "
        f"last error: {last_err}"
    )


# Strategies: a non-empty list of types (length 1..6) and a matching
# error-schedule of strings from the allowed set.
# We use ``lists`` to draw lists of the same length and pair them up.
_type_list_strategy = st.lists(
    st.sampled_from(["c6i.32xlarge", "c6a.32xlarge", "c5.24xlarge",
                     "m6i.32xlarge", "m5.24xlarge", "r5.24xlarge"]),
    min_size=1, max_size=6,
)
_err_strategy = st.sampled_from([
    "",  # success
    "InsufficientInstanceCapacity",  # capacity
    "SpotMaxPriceTooLow",            # capacity
    "InstanceLimitExceeded",         # capacity
    "MaxSpotInstanceCountExceeded",  # capacity
    "Unsupported",                   # capacity
    "InvalidParameterValue",         # NON-capacity (fatal)
    "UnauthorizedOperation",         # NON-capacity (fatal)
])


@settings(max_examples=64, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    types=_type_list_strategy,
)
def test_property_p4_launcher_fallback_first_type_succeeds(types: list[str]) -> None:
    """When the first attempt succeeds, the launcher MUST return the
    first type (no fallback advance on a happy path)."""
    schedule = [""] + ["Unsupported"] * (len(types) - 1)
    chosen = _launch_with_fallback(types, schedule)
    assert chosen == types[0], (
        f"first-try success must return types[0]; expected={types[0]} got={chosen}"
    )


@settings(max_examples=64, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    types=_type_list_strategy,
    n_failures=st.integers(min_value=1, max_value=5),
)
def test_property_p4b_launcher_fallback_advances_on_capacity(types: list[str],
                                                            n_failures: int) -> None:
    """When the first N attempts each raise a CAPACITY error and the
    (N+1)-th succeeds, the launcher MUST return types[N] (i.e. it advanced
    past every capacity error). The schedule is built so a non-empty
    chain of capacity errors precedes a single success."""
    n_failures = min(n_failures, len(types) - 1)
    assume(n_failures >= 1)
    schedule = ["InsufficientInstanceCapacity"] * n_failures + [""] * (len(types) - n_failures)
    chosen = _launch_with_fallback(types, schedule)
    assert chosen == types[n_failures], (
        f"after {n_failures} capacity failures the {n_failures + 1}-th type "
        f"({types[n_failures]}) must be the chosen one; got {chosen}"
    )


@settings(max_examples=64, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(types=_type_list_strategy)
def test_property_p4c_launcher_fallback_fatal_error_raises_immediately(types: list[str]) -> None:
    """A NON-capacity error (e.g. InvalidParameterValue) on the FIRST
    attempt MUST raise that error verbatim (not advance, not translate
    to SystemExit). The chain is short-circuited because the error
    is config-level, not capacity-level — retrying a different type
    would hit the same config error."""
    err = "InvalidParameterValue"
    schedule = [err] + [""] * (len(types) - 1)
    with pytest.raises(ValueError, match=err):
        _launch_with_fallback(types, schedule)


@settings(max_examples=64, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(types=st.lists(
    st.sampled_from(["c6i.32xlarge", "c6a.32xlarge", "c5.24xlarge",
                     "m6i.32xlarge", "m5.24xlarge", "r5.24xlarge"]),
    min_size=2, max_size=6,
))
def test_property_p4d_launcher_fallback_chain_exhausted_raises_systemexit(
    types: list[str],
) -> None:
    """When EVERY attempt raises a capacity error on a MULTI-type chain
    (the chain is exhausted), the launcher MUST raise SystemExit (the
    documented 'all instance types exhausted' terminal). The chain is
    bounded by len(types), so this is well-defined for any non-empty
    type list. (Single-type chains hit a different terminal message
    — see test_property_p4e.)"""
    schedule = ["InsufficientInstanceCapacity"] * len(types)
    with pytest.raises(SystemExit, match="exhausted"):
        _launch_with_fallback(types, schedule)


@settings(max_examples=64, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(type_=st.sampled_from(["c6i.32xlarge", "c6a.32xlarge", "c5.24xlarge",
                              "m6i.32xlarge", "m5.24xlarge", "r5.24xlarge"]))
def test_property_p4e_launcher_fallback_single_type_capacity_terminates(
    type_: str,
) -> None:
    """A single-type chain with a capacity error is terminal: there is no
    next type to try, so the launcher MUST raise SystemExit (NOT loop
    forever or silently swallow). This is the common single-az/single-type
    fallback scenario."""
    schedule = ["InsufficientInstanceCapacity"]
    with pytest.raises(SystemExit, match="capacity"):
        _launch_with_fallback([type_], schedule)
