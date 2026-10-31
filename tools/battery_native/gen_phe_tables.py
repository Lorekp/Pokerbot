#!/usr/bin/env python3
"""Generate `src/phe_tables.rs` from the installed `phevaluator` package.

The native postflop fingerprint bucket (`fast_made_hand_bucket`) needs phevaluator's exact 1..7462
hand-rank, which is table-driven (perfect-hash). Rather than transcribe ~400 KB of tables by hand
(transcription risk), we emit them directly from the live Python package, guaranteeing byte-identity
by construction. Cargo runs this generator using the pinned phevaluator==0.5.3.1 dependency.
Generated tables are local build inputs and are ignored by Git.
"""
from __future__ import annotations

from pathlib import Path
from importlib.metadata import version

if version("phevaluator") != "0.5.3.1":
    raise SystemExit("Install phevaluator==0.5.3.1 before generating lookup tables")

from phevaluator.tables import (  # type: ignore
    BINARIES_BY_ID,
    DP,
    FLUSH,
    NO_FLUSH_5,
    NO_FLUSH_6,
    NO_FLUSH_7,
    SUITBIT_BY_ID,
    SUITS,
)

out: list[str] = []


def emit_u16(name: str, arr) -> None:
    out.append(f"pub static {name}: [u16; {len(arr)}] = [")
    line: list[str] = []
    for v in arr:
        assert 0 <= v <= 65535, (name, v)
        line.append(str(v))
        if len(line) == 32:
            out.append("    " + ",".join(line) + ",")
            line = []
    if line:
        out.append("    " + ",".join(line) + ",")
    out.append("];")


def emit_i32(name: str, arr) -> None:
    out.append(f"pub static {name}: [i32; {len(arr)}] = [")
    line: list[str] = []
    for v in arr:
        line.append(str(v))
        if len(line) == 32:
            out.append("    " + ",".join(line) + ",")
            line = []
    if line:
        out.append("    " + ",".join(line) + ",")
    out.append("];")


out.append("// AUTO-GENERATED from the installed `phevaluator` package tables. DO NOT EDIT BY HAND.")
out.append("// Regenerate via tools/battery_native/gen_phe_tables.py. Byte-identical to phevaluator.")
emit_u16("FLUSH", FLUSH)
emit_u16("NO_FLUSH_5", NO_FLUSH_5)
emit_u16("NO_FLUSH_6", NO_FLUSH_6)
emit_u16("NO_FLUSH_7", NO_FLUSH_7)
emit_i32("SUITBIT_BY_ID", SUITBIT_BY_ID)
emit_i32("SUITS", SUITS)
emit_i32("BINARIES_BY_ID", BINARIES_BY_ID)
out.append(f"// DP dims: {len(DP)} x {len(DP[0])} x {len(DP[0][0])}")
out.append(f"pub static DP: [[[u32; {len(DP[0][0])}]; {len(DP[0])}]; {len(DP)}] = [")
for a in DP:
    out.append("  [")
    for b in a:
        out.append("    [" + ",".join(str(x) for x in b) + "],")
    out.append("  ],")
out.append("];")

text = "\n".join(out) + "\n"
dest = Path(__file__).resolve().parent / "src" / "phe_tables.rs"
with dest.open("w", encoding="ascii") as f:
    f.write(text)
print("wrote local phe_tables.rs:", len(text), "bytes; DP dims", len(DP), len(DP[0]), len(DP[0][0]))
