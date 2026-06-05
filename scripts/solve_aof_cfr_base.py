#!/usr/bin/env python3
"""Build AoF CFR base policies (OpenSpiel) for 2/3/4-max."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aof_cfr.export import export_manifest, export_policy_npz
from aof_cfr.hu_payoff_matrix import (
    build_hu_bucket_equity_matrix,
    build_hu_bucket_equity_matrix_solver,
    save_equity_matrix,
)
from aof_cfr.solve import (
    solve_3max_cfr_plus,
    solve_3max_fp,
    solve_4max_fp,
    solve_hu_cfr_plus,
    solve_pushfold_mccfr,
)
from aof_cfr.trio_payoff_tensor import (
    DEFAULT_TENSOR,
    build_trio_equity_tensor,
    save_trio_tensor,
)
from aof_equity import warmup


def main() -> None:
    p = argparse.ArgumentParser(description="Solve AoF CFR base (Phase A)")
    p.add_argument("--out-dir", default="data/aof/cfr_base")
    p.add_argument("--hu-iterations", type=int, default=2500)
    p.add_argument("--trio-iterations", type=int, default=3000)
    p.add_argument("--fp-rounds", type=int, default=200)
    p.add_argument("--fp-damping", type=float, default=1.0)
    p.add_argument("--fp-4-br-samples", type=int, default=4000)
    p.add_argument("--mccfr-iterations", type=int, default=20_000)
    p.add_argument("--mccfr-max", type=int, default=80_000)
    p.add_argument("--target-mbb-g", type=float, default=50.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eq-source", choices=("npz", "solver"), default="npz")
    p.add_argument("--eq-stride", type=int, default=6)
    p.add_argument("--eq-samples", type=int, default=500)
    p.add_argument("--3way-eq-path", dest="trio_eq_path", default=str(DEFAULT_TENSOR))
    p.add_argument("--tensor-workers", type=int, default=None)
    p.add_argument("--mc-boards", type=int, default=768)
    p.add_argument("--rebuild-tensor", action="store_true")
    p.add_argument(
        "--solver",
        choices=("fp", "cfr", "mccfr"),
        default="fp",
        help="3-max/4-max backend (default: fp)",
    )
    p.add_argument("--only", choices=("hu", "3", "4", "all"), default="all")
    p.add_argument(
        "--diagnostic-cfr",
        action="store_true",
        help="Run CFR+ on 3-max instead of FP (root-cause diagnostic)",
    )
    args = p.parse_args()

    warmup()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eq_path = out_dir / "hu_eq_matrix.npz"
    trio_path = Path(args.trio_eq_path)
    need_hu = args.only in ("hu", "all")
    need_3 = args.only in ("3", "all")
    if need_hu or not eq_path.is_file():
        if args.eq_source == "npz":
            if not eq_path.is_file():
                print("Building HU bucket equity matrix from hu_showdown_equity.npz ...")
                eq = build_hu_bucket_equity_matrix()
                save_equity_matrix(eq_path, eq)
                print(f"Wrote {eq_path}")
        else:
            print("Building solver-family HU eq matrix (MC, oracle-compatible) ...")
            eq = build_hu_bucket_equity_matrix_solver(
                equity_samples=args.eq_samples,
                seed=args.seed,
                bucket_stride=args.eq_stride,
            )
            save_equity_matrix(eq_path, eq)
            print(f"Wrote {eq_path}")
    elif not need_hu:
        print(f"Using existing {eq_path}")

    if need_3 and (args.rebuild_tensor or not trio_path.is_file()):
        print(f"Building 3-way equity tensor -> {trio_path} ...")
        shares, pair_eq, deal_weights, combo_counts = build_trio_equity_tensor(
            workers=args.tensor_workers,
            mc_boards=args.mc_boards,
            progress=True,
        )
        save_trio_tensor(
            trio_path, shares, pair_eq, deal_weights, combo_counts, mc_boards=args.mc_boards
        )
        print(f"Wrote {trio_path}")

    results = []
    if args.only in ("hu", "all"):
        print(f"Solving HU CFR+ ({args.hu_iterations} iterations) ...")
        hu = solve_hu_cfr_plus(
            iterations=args.hu_iterations,
            eq_matrix_path=str(eq_path).replace("\\", "/"),
            use_efg=False,
        )
        export_policy_npz(out_dir / "cfr_base_2p.npz", hu)
        results.append(hu)
        print(
            f"HU exploitability: {hu.exploitability_mbb_g:.3f} mbb/g "
            f"({hu.exploitability_bb_per_hand:.6f} bb/hand)"
        )

    if args.only in ("3", "all"):
        use_cfr = args.diagnostic_cfr or args.solver == "cfr"
        if use_cfr:
            print(f"Solving 3-max CFR+ diagnostic (seed={args.seed}) ...")
            r3 = solve_3max_cfr_plus(
                iterations=args.trio_iterations,
                seed=args.seed,
                tensor_path=str(trio_path).replace("\\", "/"),
            )
        else:
            print(f"Solving 3-max FP (seed={args.seed}) ...")
            r3 = solve_3max_fp(
                max_rounds=args.fp_rounds,
                damping=args.fp_damping,
                seed=args.seed,
                tensor_path=str(trio_path).replace("\\", "/"),
            )
        export_policy_npz(out_dir / "cfr_base_3p.npz", r3)
        results.append(r3)
        print(
            f"3-max Nash gap: {r3.exploitability_mbb_g:.3f} mbb/g "
            f"(method={r3.br_method}, solver={r3.solver_name})"
        )

    if args.only in ("4", "all"):
        if args.solver == "mccfr":
            print(f"Solving 4-max MCCFR regression (seed={args.seed}) ...")
            r4 = solve_pushfold_mccfr(
                4,
                iterations=args.mccfr_iterations,
                seed=args.seed,
                target_mbb_g=args.target_mbb_g,
                max_iterations=args.mccfr_max,
            )
        else:
            print(f"Solving 4-max FP (seed={args.seed}) ...")
            r4 = solve_4max_fp(
                max_rounds=min(args.fp_rounds, 30),
                br_samples=args.fp_4_br_samples,
                seed=args.seed,
                damping=args.fp_damping,
            )
        export_policy_npz(out_dir / "cfr_base_4p.npz", r4)
        results.append(r4)
        print(
            f"4-max Nash gap: {r4.exploitability_mbb_g:.3f} mbb/g "
            f"(method={r4.br_method}, solver={r4.solver_name})"
        )

    export_manifest(out_dir / "cfr_base.manifest.json", results, merge_existing=True)
    print(f"Wrote manifest {out_dir / 'cfr_base.manifest.json'}")


if __name__ == "__main__":
    main()
