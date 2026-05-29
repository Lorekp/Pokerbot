# texas_solver_cli

Drop-in **JSON stdin/stdout** wrapper around [TexasSolver](https://github.com/bupticybee/TexasSolver) `console_solver`, matching the contract of `external/postflop_solver_cli` (Rust / postflop-solver).

## Prerequisites

1. Download a [TexasSolver release](https://github.com/bupticybee/TexasSolver/releases) (console build with `console_solver` + `resources/`).
2. Set environment:
   ```bash
   export TEXAS_SOLVER_ROOT=/path/to/TexasSolver/install
   # or
   export TEXAS_SOLVER_CONSOLE=/path/to/console_solver
   ```
3. Optional: `TEXAS_SOLVER_THREADS=N` (default: hardware concurrency).

## Build

```bash
cmake -S tools/texas_solver_cli -B tools/texas_solver_cli/build
cmake --build tools/texas_solver_cli/build --config Release
```

Binary: `tools/texas_solver_cli/build/texas_solver_cli[.exe]`

## Usage

Same as Rust CLI:

```bash
cat tools/texas_solver_cli/fixtures/golden_request.json | tools/texas_solver_cli/build/texas_solver_cli
```

From Python label pipeline:

```bash
export POSTFLOP_SOLVER_CLI=tools/texas_solver_cli/build/texas_solver_cli
python scripts/build_solver_labels.py --config config/solver_labels_smoke.yaml --max-spots 1 --cli-path "$POSTFLOP_SOLVER_CLI"
```

## Parity gate

```bash
python scripts/compare_solver_clis.py --request tools/texas_solver_cli/fixtures/golden_request.json
```

**Do not** switch production default until golden semantic diff passes. See [`docs/native_rewrite/postflop_parity_gaps.md`](../../docs/native_rewrite/postflop_parity_gaps.md).

## License

TexasSolver is **AGPL-3.0**. Internal offline batch use is typically fine; network-facing solver services may require a commercial license from the author.
