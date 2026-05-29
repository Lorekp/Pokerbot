#!/usr/bin/env python3
"""Export equity/payoff optimization handoff to a single markdown file."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(r"data/external/aof-equity-optimization-handoff/EQUITY_OPTIMIZATION_HANDOFF.md")

FILES = [
    "aof_equity_showdown.py",
    "aof_equity_grids.py",
    "aof_rl/payoff.py",
    "aof_rl/reward_noise.py",
    "aof_rl/env.py",
    "aof_rl/rollout.py",
    "aof_rl/config.py",
    "aof_rl/league_trainer.py",
    "scripts/census_hu_showdown_keys.py",
    "scripts/build_aof_hu_showdown_equity.py",
    "scripts/measure_payoff_equity_error.py",
    "scripts/benchmark_eval_backend.py",
    "scripts/eval_multiway_payoff_gate.py",
    "scripts/build_aof_hu_equity_grid.py",
    "tools/ompeval_cli/ompeval_backend.py",
    "tools/ompeval_cli/enumerate_hu_equity.cpp",
    "tools/ompeval_cli/CMakeLists.txt",
    "tools/ompeval_cli/build.ps1",
    "tools/ompeval_cli/README.md",
    "tests/test_aof_hu_showdown_equity.py",
    "tests/test_aof_batched_rollout.py",
    "tests/test_aof_payoff_hu_table.py",
    "configs/aof_v3_ppo_league.yaml",
    "docs/aof-training-bottlenecks.md",
    "artifacts/audit/hu_showdown_key_census.json",
    "artifacts/audit/payoff_equity_error.json",
]

EQUITY_EXCERPT = """## excerpt: `aof_equity.py` — `equity_hu_exact` (legacy HU payoff MC)

```python
def equity_hu_exact(cards_a, cards_b, samples, seed):
    # P(player A wins) at showdown vs known B hole (numba MC).
    a0, a1 = _card_id_str(cards_a[0]), _card_id_str(cards_a[1])
    b0, b1 = _card_id_str(cards_b[0]), _card_id_str(cards_b[1])
    rng = np.random.default_rng(seed)
    boards = _sample_boards(a0, a1, b0, b1, max(samples, 1), rng)
    return float(_mc_hu_from_boards(a0, a1, b0, b1, boards))
```

## excerpt: `aof_equity.py` — multiway MC (still hot path after HU table)

```python
def _sample_boards_3way(h0, h1, v0, v1, w0, w1, n, rng):
    dead = np.zeros(52, dtype=np.bool_)
    for c in (h0, h1, v0, v1, w0, w1):
        dead[c] = True
    deck = np.nonzero(~dead)[0].astype(np.uint8)
    noise = rng.random((n, deck.size))
    pick = np.argsort(noise, axis=1)[:, :5]
    return deck[pick]

@njit(parallel=True, cache=True)
def _mc_3way_shares_all(h0, h1, v0, v1, w0, w1, boards):
    # Normalized pot shares for players 0,1,2 — used by aof_rl.payoff compute_payoffs
    ...

# 4-way: _sample_boards_4way, _mc_4way_shares_all (same file)
```

## excerpt: `aof_rl/env.py` — payoff seed + profiler split

```python
def payoff_step_seed(hand_id: int, guard: int) -> int:
    return hand_id * 1009 + guard

def _finalize(state, *, payoff_samples, payoff_seed):
    alive = [s for s in state.seats if s not in state.folded]
    bucket = "rollout.step_payoff_hu" if len(alive) == 2 else (
        "rollout.step_payoff_multiway" if len(alive) > 2 else "rollout.step_payoff"
    )
    state.payoffs = compute_payoffs(state, samples=payoff_samples, seed=seed)
```

---

"""

README = """# AoF HU Payoff / Equity Optimization Handoff

Self-contained context for optimizing **combo-exact HU showdown payoffs** (replacing MC in `rollout.step_payoff`).

## Problem (post-LUT profile)

After teacher LUT + batched rollout, **`rollout.step_payoff` ~21% wall** (Example C). HU path used `equity_hu_exact` which is **board MC** (~1200 samples), not enumeration. PPO rewards need **combo-exact** equity (specific hole cards), not 169-class means.

## Target architecture

1. **Offline build**: enumerate C(48,5) runouts per canonical 4-card pattern (~97k keys)
2. **Runtime**: O(1) NPZ lookup via `equity_hu_showdown_exact`
3. **Multiway**: still MC (`payoff_samples`); E3 stratified MC deferred

## Key metrics (measured on this machine)

| Metric | Value |
|--------|-------|
| Canonical keys | 97,513 |
| Artifact size est | ~1.1 MB |
| Build time (numba) | ~0.25 s/key → ~7 h total |
| reward_noise_floor_bb | ~0.055 bb |
| MC p95 ΔEV vs exact | ~0.44 bb (fails floor → combo-exact justified) |

## Build / verify commands

```bash
pip install -e ".[aof,dev]"
python scripts/census_hu_showdown_keys.py
python scripts/benchmark_eval_backend.py
python scripts/build_aof_hu_showdown_equity.py
python scripts/measure_payoff_equity_error.py -n 4000
pytest tests/test_aof_hu_showdown_equity.py tests/test_aof_batched_rollout.py -q
python scripts/profile_aof_league_gpu.py --total-updates 60  # Example E after table lands
python scripts/eval_multiway_payoff_gate.py --profile logs/aof_rl_profile_latest.json
```

## Training config

```yaml
payoff_hu_mode: table   # table | exact_enum | mc_legacy
hu_showdown_table_path: data/aof/hu_showdown_equity.npz
payoff_samples: 1200    # multiway MC only
```

## Gates

| Gate | Criterion |
|------|-----------|
| A | AA vs AKo combo-exact (AcAd vs KhKd) ≈ 0.81946 |
| B | 10k matchups: lookup vs `_equity_two_combos_exact`, max err < 1e-7 |
| Phase 0 | p95 abstraction ΔEV vs `reward_noise_floor_bb` |
| N=1 #1a | Pre-payoff trajectory bitwise |
| N=1 #1b | Reward match when same `payoff_hu_mode` |

## Deferred work

- **E3**: stratified multiway MC if multiway wall ≥10% AND bucket error > floor
- **E2**: CUDA kernel
- **OMPEval pybind**: optional; CLI speeds offline build only

## Original repo

`.`

---

"""


def _lang(rel: str) -> str:
    if rel.endswith(".json"):
        return "json"
    if rel.endswith(".yaml"):
        return "yaml"
    if rel.endswith(".cpp"):
        return "cpp"
    if rel.endswith(".ps1"):
        return "powershell"
    if "CMake" in rel:
        return "cmake"
    if rel.endswith(".md"):
        return "markdown"
    return "python"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    parts = [README, EQUITY_EXCERPT]
    for rel in FILES:
        p = ROOT / rel
        if not p.is_file():
            parts.append(f"## MISSING: `{rel}`\n\n(file not found at export time)\n\n---\n")
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        parts.append(f"## `{rel}`\n\n```{_lang(rel)}\n{text.rstrip()}\n```\n\n---\n")
    OUT.write_text("\n".join(parts), encoding="utf-8")
    kb = OUT.stat().st_size // 1024
    print(f"Wrote {OUT} ({kb} KB)")


if __name__ == "__main__":
    main()
