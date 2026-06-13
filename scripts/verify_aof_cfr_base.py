#!/usr/bin/env python3
"""Verify AoF CFR base artifacts (Phase A gates)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aof_cfr.buckets import LABELS_169
from aof_cfr.hu_validate import (
    compare_freqs,
    dcfr_oracle_freqs,
    load_policy_freqs,
)
from aof_cfr.pushfold_payoffs import audit_terminal_ev
from aof_cfr.query import base_strategy, load_cfr_base
from aof_cfr.trio_payoff_tensor import load_trio_tensor

HU_FREQ_TOLERANCE = 0.02
TARGET_MBBG_HU = 10.0
TRIVIAL_RATIO = 0.25
FP_SOLVER_3 = "fictitious_play_3max"
FP_SOLVER_4 = "fictitious_play_4max"


def validate_hu_oracle(base_dir: Path, *, eq_path: Path) -> dict[str, object]:
    path = base_dir / "cfr_base_2p.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    sb_sol, bb_sol = load_policy_freqs(path)
    sb_oracle, bb_oracle = dcfr_oracle_freqs(eq_path)
    return compare_freqs(
        sb_sol,
        bb_sol,
        sb_oracle,
        bb_oracle,
        tolerance=HU_FREQ_TOLERANCE,
    )


def chart_sanity_check(base_dir: Path) -> dict[str, object]:
    """Premium hands push often; trash folds on BTN open spot."""
    load_cfr_base(base_dir)
    checks = [
        ("BTN", "-", "AA", 0.85),
        ("BTN", "-", "72o", 0.15),
        ("BTN", "-", "AKs", 0.70),
    ]
    results = []
    ok = True
    for pos, hist, hand, floor in checks:
        p = base_strategy(3, pos, hist, hand)
        passed = p >= floor if floor > 0.5 else p <= floor
        results.append({"spot": f"{pos}|{hist}|{hand}", "p_push": p, "floor": floor, "passed": passed})
        ok = ok and passed
    return {"checks": results, "passed": ok}


def validate_manifest(base_dir: Path, *, trio_path: Path) -> dict[str, object]:
    manifest_path = base_dir / "cfr_base.manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    trivial = None
    if trio_path.is_file():
        from aof_cfr.fp_br import load_3max_payoffs, trivial_baselines_3max

        shares, pair_eq, deal_w = load_trio_tensor(trio_path)
        st, deal_w = load_3max_payoffs(shares, pair_eq, deal_w)
        trivial = trivial_baselines_3max(st, deal_w)

    table_results = []
    all_ok = True
    seen_n = set()
    for table in manifest.get("tables", []):
        mbb = float(table["exploitability_mbb_g"])
        n = int(table["num_players"])
        solver = str(table.get("solver", ""))
        ok = True
        if n == 2:
            ok = mbb <= TARGET_MBBG_HU
        elif n == 3:
            # "exact" / "exact_legal_marginalized" / "legal_fp" are all exact-BR methods; the
            # "legal*" ones are the non-clairvoyant marginalization (the canonical one post-fix).
            ok = str(table.get("br_method", "")).startswith(("exact", "legal"))
            ok = ok and solver == FP_SOLVER_3
            if trivial:
                push_t = trivial["always_push"]
                ok = ok and mbb < push_t * TRIVIAL_RATIO
        elif n == 4:
            ok = mbb < 3000.0 * TRIVIAL_RATIO
            ok = ok and solver in (FP_SOLVER_4, FP_SOLVER_4 + "_vec")
        seen_n.add(n)
        table_results.append(
            {
                "num_players": n,
                "solver": solver,
                "exploitability_mbb_g": mbb,
                "br_method": table.get("br_method"),
                "passed": ok,
            }
        )
        all_ok = all_ok and ok

    four_p = base_dir / "cfr_base_4p.npz"
    if not four_p.is_file():
        all_ok = False
        table_results.append(
            {
                "num_players": 4,
                "solver": "missing",
                "exploitability_mbb_g": float("nan"),
                "br_method": None,
                "passed": False,
            }
        )
    return {"tables": table_results, "trivial_baselines_mbb_g": trivial, "passed": all_ok}


def main() -> None:
    p = argparse.ArgumentParser(description="Verify AoF CFR base artifacts")
    p.add_argument("--base-dir", default="data/aof/cfr_base")
    p.add_argument("--eq-path", default="data/aof/cfr_base/hu_eq_matrix.npz")
    p.add_argument("--trio-path", default="data/aof/cfr_base/trio_eq_tensor.npz")
    args = p.parse_args()

    base_dir = Path(args.base_dir)
    audit_terminal_ev()
    print("audit_terminal_ev: OK")

    hu = validate_hu_oracle(base_dir, eq_path=Path(args.eq_path))
    print(
        f"HU oracle agreement (NPZ eq DCFR): max_diff={hu['max_diff']:.6f} "
        f"passed={hu['passed']}"
    )
    if not hu["passed"]:
        print(
            "HU diff exceeds tolerance — model bug (payoff/deal), not tolerance tuning.",
            file=sys.stderr,
        )

    man = validate_manifest(base_dir, trio_path=Path(args.trio_path))
    if man.get("trivial_baselines_mbb_g"):
        print(f"Trivial baselines (3-max): {man['trivial_baselines_mbb_g']}")
    print(f"Manifest gates passed={man['passed']}")
    for t in man["tables"]:
        print(
            f"  {t['num_players']}-max: {t['exploitability_mbb_g']:.3f} mbb/g "
            f"solver={t['solver']} method={t['br_method']} passed={t['passed']}"
        )

    load_cfr_base(base_dir)
    print("Runtime load_cfr_base: OK")

    chart = chart_sanity_check(base_dir)
    print(f"Chart sanity passed={chart['passed']}")
    for c in chart["checks"]:
        print(f"  {c['spot']}: p_push={c['p_push']:.3f} floor={c['floor']} passed={c['passed']}")

    if not hu["passed"] or not man["passed"] or not chart["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
