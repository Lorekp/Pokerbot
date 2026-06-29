# `battery_native` — bit-identical native core for the Pluribus exploitation battery hot path

A Rust (PyO3) reimplementation of the **np_engine state machine + showdown evaluation** — the dominant
cost of `scripts/eval_exploitation_battery.py --hero pluribus`. Bit-for-bit identical to the Python oracle
(`pokerbot/engine/np_engine.py`, which is itself bit-identical to the JAX engine). Bit-identity is the hard
constraint; raw speed is the objective.

Python binding: `pokerbot/search/native_solver.py` — a flag-gated drop-in exposing
`apply_action_np` / `legal_action_mask_np` / `sample_chance_np` / `is_chance_node_np` / `evaluate_seven`
with the SAME numpy-`EnvState` signatures as np_engine, plus a bit-identical pure-Python fallback when the
extension isn't built. The three hot modules (`pluribus_search`, `pluribus_rollout`,
`tabular_blueprint_adapter`) import the engine primitives through this layer, so they auto-benefit.

---

## Profile — where the pluribus battery time actually goes

`tools/battery_native/profile_battery.py` (40 hand-pairs, `--iterations 10 --allin-ev`, calling_station,
uniform leaf), pure-Python:

```
compute_returns_np (side pots + showdown)   38.0 s  (47%)   <- evaluate_seven/score_five/_straight_high = 29.8 s (37%)
apply_action_np (state transition)          46.6 s  (58%, overlaps: it calls compute_returns at terminals)
legal_action_mask_np                         7.4 s
numpy reduce/any/sum ufunc dispatch on tiny arrays  ~20 s aggregate
```

Everything funnels through a handful of np_engine primitives, called ~700k+ times/hand-pair from the CFR
recursion (`pluribus_search._cfv`), the gadget leaf playouts (`pluribus_rollout._playout`), and the AIVAT
`_leaf_q` baseline. **The HEURISTIC battery is already fast (~500 hand-pairs/sec, 2000 hands in 4 s) — it
has no search/playouts and was NOT ported** (confirmed by measurement). The all-in-EV settlement already
uses phevaluator and is not a bottleneck.

## Boundary — what is ported (Tier A) and what stays Python (Tier B)

| Component | Ported natively? | Why |
|---|---|---|
| `apply_action` state transition, chip/pot arithmetic, street advancement | **yes** | integer-exact, the hot path |
| `legal_action_mask` (incl. f32-truncated bet sizing) | **yes** | integer-exact |
| `sample_chance` + Threefry-2x32 deck-deal RNG key advance | **yes** | integer/uint32-exact |
| `compute_returns` side pots + `score_five`/`evaluate_seven` showdown | **yes** | integer-exact (base-15 within-category code) |
| CFR recursion `_cfv`, the gadget zero-sum solve, float reductions | **NO — Python** | tiny; bit-identity-sensitive float order |
| numpy PCG64 Monte-Carlo stream (`_public_seed`, `rng.choice`, `rng.random`) | **NO — Python** | the RNG STAYS in Python |
| exploit math (`opponent/*`, context keys, gift/bounded-ε) | **NO — Python (frozen)** | firewall: untouched |

Because the RNG and all float reductions stay in Python and only the **integer-exact engine primitives**
are native, the FULL solve `solve(root) -> action_probs` is a pure function of (public state, opponent
model, config) and comes out **byte-identical** native-vs-Python — strictly stronger than the task's B2
fallback gate. No PCG64 reproduction was needed (the RNG never crossed the boundary).

## Exactness — verified

**Tier A (bit-identical), `tools/battery_native/crosscheck.py` (1500 random hands, showdown-biased):**
```
TOTAL golden cases:   84164   (transitions 30421, legal_mask 28743, sample_chance 1678,
                               terminal/showdown rewards 1291, evaluate_seven fuzz 20000,
                               compute_returns side-pot fuzz 5000)
MISMATCHES:           0       (np.array_equal — bit-exact ints/bools, every EnvState field)
```

**Solve-level equivalence, `tools/battery_native/solve_crosscheck.py`** (native ON vs pure-Python OFF,
mixed iterations 4/10/20, with and without opponent_model):
```
TOTAL: 305 solves, 0 bit-mismatches   (struct.pack('<d') f64 bit-equality of action_probs)
```
The end-to-end battery uplift is identical ON vs OFF on every config tested (e.g. 200 hands
calling_station: uplift +3.274, zero_frac 0.605 both arms).

Pinned in CI by `tests/test_battery_native.py` (skips cleanly if the extension isn't built).

## Benchmark — native vs Python (honest)

`profile_battery.py --hands 200 --iterations 10 --allin-ev` (calling_station, uniform leaf):

```
pure-Python np_engine:  455.0 s   2275 ms/hand-pair   0.440 hand-pairs/sec
native Rust engine:     201.4 s   1007 ms/hand-pair   0.993 hand-pairs/sec
SPEEDUP:                2.26x      (uplift bit-identical: +3.274, zero_frac 0.605)
```

Projected wall time (calling_station leg, 10 iters):
- 3,000-hand leg:  ~1.90 hr (Python)  ->  ~0.84 hr (native)
- 30,000-hand leg: ~19.0 hr (Python)  ->  ~8.4 hr (native)

The residual cost is now the per-call EnvState<->flat-buffer marshaling (`_to_flat`/`_from_flat`, ~29% of
native wall time) plus the Python CFR/adapter/key logic that must stay Python to preserve bit-identity.
The showdown eval and state transitions (the old 70%) are essentially free now. Pushing past ~2.3x would
require porting the CFR recursion + gadget RNG into Rust, which would forfeit the proven bit-identity of
`solve -> action_probs` — explicitly out of scope.

## Optimization techniques applied

1. **Bytes ABI marshaling** — state crosses the boundary as raw little-endian i64 bytes
   (`arr.tobytes()` -> Rust `&[u8]`, Rust `PyBytes` -> `np.frombuffer`), a single C-speed memcpy each
   way instead of per-element Python-list iteration (this alone took the wrapper from 1.5x to ~2x).
2. **Allocation-free `_to_flat`** — the EnvState fields are already contiguous numpy arrays with the
   engine's dtypes, so they're assigned straight into the preallocated int64 buffer (numpy casts on
   assignment); scalars read via `.item()`. The redundant per-field `np.asarray`/`reshape` (12M calls)
   that dominated the first cut is gone.
3. **Flat `Copy` EnvState struct** in Rust (fixed-size arrays, no heap), `#[inline]` hot leaves,
   `lto="fat"`, `codegen-units=1`, `opt-level=3`, `panic="abort"`.
4. **abi3-py310** stable ABI — one native module works on the EC2 Python 3.10 runtime and newer CPython.

## Build

```bash
cd tools/battery_native
cargo test --lib                                                      # pure-Rust sanity (no Python)
PYO3_PYTHON="python" cargo build --release --features python --lib
cp target/release/battery_native.dll target/release/battery_native.pyd   # Windows: alias cdylib -> .pyd

# verify bit-identity + benchmark (from repo root, POKERBOT_BETTING_ABSTRACTION=1):
python tools/battery_native/crosscheck.py --hands 1500
python tools/battery_native/solve_crosscheck.py --hands 40
python tools/battery_native/profile_battery.py --hands 200 --iterations 10 --allin-ev --no-profile
python -m pytest tests/test_battery_native.py -q
```

`native_solver.py` auto-locates `target/release/battery_native.{pyd,dll,so}`, aliases the dll to `.pyd`
on first import, runs a 40-hand lockstep startup self-check, and falls back to the (bit-identical)
pure-Python np_engine if the extension is absent or the self-check fails — so callers never special-case
availability. Disable at runtime with `POKERBOT_NO_NATIVE_BATTERY=1` or the battery's `--no-native` flag.
