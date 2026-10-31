# `aivat_native` — bit-identical native core for the AIVAT control variates

A Rust (PyO3) reimplementation of the deterministic arithmetic of the AIVAT control variates, **bit-for-bit
identical** to the Python oracle (`pokerbot/eval/aivat.py` + `pokerbot/eval/chance_term.py`). Bit-identity to
the oracle is the hard constraint; raw speed is the objective.

Python binding: `pokerbot/eval/aivat_native.py` (same API as the oracle, with a bit-identical pure-Python
fallback when the extension isn't built).

---

## Profile — where the AIVAT time actually goes

Profiled the chance term over 8 real showdown hands on the matured blueprint
(`bp6max_dcfrvr_iter200.pkl`), `gadget` leaf, `rollout_samples=2`, `gadget_iters=64`
(`tools/aivat_native/capture_golden.py`):

```
chance_corrections_for_hand            54.5 s   (100%)   8 hands -> 6.8 s/hand
└─ _value / continuation_leaf_value    52.6 s   (96.5%)  THE GADGET VALUE FUNCTION
   └─ _payoff_matrix                    51.9 s   (95%)    RNG-driven playouts (numpy PCG64)
      └─ _playout (19,200 calls)        51.1 s
         ├─ adapter.continuation_set    24.1 s   blueprint lookup + bias arithmetic
         ├─ apply_action_np             17.1 s   np_engine state transition
         ├─ legal_action_mask_np        13.5 s
         └─ compute_returns_np / eval    8.6 s   (evaluate_seven/score_five = 6.5 s, deterministic)
The AIVAT correction arithmetic itself (np.mean over <=47 vals, realized-expected, the accumulator
mean/std) is NEGLIGIBLE in the profile — it never appears above noise.
```

**Conclusion:** 96% of the chance-term cost is the RNG-driven gadget value function `v`
(`continuation_leaf_value`): numpy PCG64 sampling (`default_rng`, `rng.random()`, `rng.shuffle`) + the
`np_engine` playout + a small matrix-game CFR. The correction/accumulator math the task names is already
fast.

## Boundary — what is ported natively (and what is NOT)

| Component | RNG? | Ported natively? | Why |
|---|---|---|---|
| `decision_correction` | no | **yes** | deterministic dot/normalize |
| `chance_correction` | no | **yes** | scalar subtract |
| `chance_reduce` (`realized − mean(vals)`, the per-reveal enumeration reduction) | no | **yes** | deterministic; needs NumPy-exact pairwise mean |
| `AIVATAccumulator` mean/std/variance_reduction | no | **yes** | needs NumPy-exact pairwise + CPython `sum()` Neumaier |
| showdown eval (`evaluate_seven`/`score_five`) | no | not standalone | deterministic & native-able, but it lives *inside* the RNG playout (next row) — porting it alone yields no end-to-end win |
| **gadget `v` = `continuation_leaf_value`** (`_payoff_matrix`→`_playout`) | **YES (PCG64)** | **NO — stays in Python** | reproducing numpy's PCG64 stream + the engine's exact float-reduction order bit-identically is infeasible; this is the documented hard boundary |

The native core therefore consumes **precomputed** gadget `v` values: Python runs the RNG gadget, hands the
`(realized, [candidate_v…])` arrays to `chance_reduce` / `chance_corrections_from_values`, and the native
code does the bit-identical reduction. This is the largest bit-identical subset — but because `v` (the 96%)
stays in Python, the realistic end-to-end speedup on the chance term is small (see Benchmark).

## Optimization techniques applied

1. **NumPy-exact pairwise summation** (`core::pairwise_sum`) — the load-bearing one. NumPy's `sum/mean/std`
   are NOT naive left-to-right above ~8 elements; they use an 8-way unrolled base case (`n<=128`) + a
   recursive half-split rounded to a multiple of 8 (`n>128`). Replicated exactly so every reduction is
   bit-identical (verified across n=0..8192). *This is correctness, not speed — but it is the whole point.*
2. **CPython `sum()` Neumaier compensation** (`core::py_sum`) — `AIVATAccumulator.add_hand` uses builtin
   `sum()`, which (CPython ≥3.12) does Neumaier *compensated* summation, not a naive fold. A naive fold is
   off by 1 ULP on ~8% of 3-element correction lists. Replicated exactly.
3. **Zero-allocation / `#[inline]` hot leaves**, `lto="fat"`, `codegen-units=1`, `opt-level=3`,
   `panic="abort"` — the reduction kernels are branch-light and the recursion is on borrowed-slice indices
   (sequential, cache-friendly).
4. **abi3-py314** stable ABI — one `.pyd` works across CPython 3.14+ without re-linking.

## Benchmark — native vs Python (honest)

`tools/aivat_native/bench_and_crosscheck.py`:

```
chance_reduce x2000 (47-elt enumeration): python=7.72 ms  native=4.65 ms  speedup=1.66x   <-- the hot path
accumulator_summary (50k hands):          python=0.12 ms  native=3.07 ms  speedup=0.04x   <-- numpy wins
pairwise sum (100k):                      np.sum=0.03 ms   native=0.92 ms                  <-- numpy wins
```

- **`chance_reduce` is 1.66x faster** natively — this is the reduction actually invoked per board reveal in
  the chance term, where per-call numpy dispatch (`np.asarray` + `np.mean` on a tiny 47-elt array) dominates
  and a tight native loop wins.
- **Bulk contiguous reductions (`accumulator_summary`, `np_sum`) are SLOWER natively**: numpy's `sum/std`
  are already C+SIMD over contiguous memory, and the native path pays Python-list→`Vec<f64>` marshaling.
  Honest finding — for the once-per-battery summary, keep numpy; the native core helps only the
  per-reveal scalar reductions.
- **End-to-end chance term:** since the gadget `v` (96% of time, RNG-bound) stays in Python, the realistic
  end-to-end speedup is ≈1.0x (the 1.66x applies to the <4% reduction slice). This is the expected,
  acceptable outcome the task anticipated: bit-identity forces keeping the RNG value function in Python,
  capping the end-to-end gain.

## Bit-identity proof

```
$ POKERBOT_BETTING_ABSTRACTION=1 python -m pokerbot.eval.aivat_native
[aivat_native] backend = native
  decision_correction            11 cases  OK (bit-identical)
  chance_correction              12 cases  OK (bit-identical)
  chance_reduce                  17 cases  OK (bit-identical)   # incl. pairwise-trap n=127/128/129/200/1000/4096 + real reveals
  accumulator                     4 cases  OK (bit-identical)

$ POKERBOT_BETTING_ABSTRACTION=1 python tools/aivat_native/bench_and_crosscheck.py
decision_correction: 20000 random cases, mismatches=0
chance_reduce: 19955 random cases, mismatches=0
accumulator: 200 random series, mismatches=0
TOTAL MISMATCHES vs live oracle: 0
```

Equality is exact f64 bit equality (`struct.pack('<d', …)` comparison), not approximate. Both the compiled
Rust backend and the pure-Python fallback pass.

## Build

Requires the Rust toolchain (cargo) and a CPython 3.14 install (for linking via PyO3; `python314.lib` /
`python3.lib`). No `maturin`, `cmake`, or external C compiler needed — cargo locates the MSVC linker.

```bash
# 1) Pure-Rust core tests (no Python) — proves the algorithm bit-identical without an interpreter:
cd tools/aivat_native
cargo test --lib

# 2) Build the CPython extension (cdylib):
PYO3_PYTHON="$(python -c 'import sys; print(sys.executable)')" cargo build --release --features python --lib

# 3) On Windows, alias the cdylib to a .pyd so CPython can import it:
cp target/release/aivat_native.dll target/release/aivat_native.pyd
#   (Linux/macOS: the cdylib is already aivat_native.so — copy it as-is.)

# 4) Verify bit-identity + benchmark:
POKERBOT_BETTING_ABSTRACTION=1 python -m pokerbot.eval.aivat_native
POKERBOT_BETTING_ABSTRACTION=1 python tools/aivat_native/bench_and_crosscheck.py
POKERBOT_BETTING_ABSTRACTION=1 python -m pytest tests/test_aivat_native.py -q
```

`pokerbot/eval/aivat_native.py` auto-locates `target/release/aivat_native.{pyd,dll,so}`, aliases the dll to
`.pyd` on first import, and falls back to the (bit-identical) pure-Python path if the extension is absent —
so callers never special-case availability.

## Golden vectors

`tools/aivat_native/golden/golden.json` — inputs + EXACT little-endian-hex f64 outputs from the Python
oracle, regenerable with `POKERBOT_BETTING_ABSTRACTION=1 python tools/aivat_native/capture_golden.py`
(also re-profiles the chance term). Covers synthetic edge cases, the NumPy pairwise-summation trap sizes,
and real precomputed gadget reveal values from the matured blueprint.
