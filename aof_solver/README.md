# aof_solver — experimental CFR skeleton

**`EXPERIMENTAL_SOLVER_SKELETON_NOT_VALIDATED`**

This package is an isolated HU push/fold toy solver for plumbing tests. It does **not** produce production Nash ranges for 3/4-max AoF.

- Do not wire exports into `aof_ranges.lookup_*` without validation.
- Production HU 8bb ranges remain `data/aof/hu_pushfold_8bb.json` (`solver_hu_dcfr`).
- 3/4-max advisor ranges remain hand-tuned static profile.

See also: [docs/aof-4max-cfr-solver-design.md](../docs/aof-4max-cfr-solver-design.md).
