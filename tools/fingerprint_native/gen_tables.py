"""Regenerate src/tables.rs from the installed phevaluator package (perfect-hash lookup tables).

The tables are pure integer perfect-hash tables (~130KB), so vendoring them makes the Rust postflop
made-hand bucket bit-exact by construction. Card id == rank*4+suit, identical to the encoder convention.
Run from the repo root:  python tools/fingerprint_native/gen_tables.py
"""
from __future__ import annotations

from pathlib import Path

from phevaluator.tables import (
    BINARIES_BY_ID,
    DP,
    FLUSH,
    NO_FLUSH_5,
    NO_FLUSH_6,
    NO_FLUSH_7,
    SUITBIT_BY_ID,
    SUITS,
)


def emit_u16(name: str, arr) -> str:
    out = [f"pub static {name}: [u16; {len(arr)}] = ["]
    line: list[str] = []
    for v in arr:
        line.append(str(int(v)))
        if len(line) == 32:
            out.append("    " + ",".join(line) + ",")
            line = []
    if line:
        out.append("    " + ",".join(line) + ",")
    out.append("];")
    return "\n".join(out)


def main() -> None:
    parts = [
        "//! AUTO-GENERATED from the installed phevaluator package tables. DO NOT EDIT BY HAND.",
        "//! Regenerate via tools/fingerprint_native/gen_tables.py. Card id == rank*4+suit (encoder == phe).",
        "",
    ]
    for name, arr in (
        ("BINARIES_BY_ID", BINARIES_BY_ID),
        ("SUITBIT_BY_ID", SUITBIT_BY_ID),
        ("SUITS", SUITS),
        ("FLUSH", FLUSH),
        ("NO_FLUSH_5", NO_FLUSH_5),
        ("NO_FLUSH_6", NO_FLUSH_6),
        ("NO_FLUSH_7", NO_FLUSH_7),
    ):
        parts.append(emit_u16(name, arr))
        parts.append("")
    out = ["pub static DP: [[[u32; 10]; 14]; 5] = ["]
    for a in DP:
        out.append("  [")
        for b in a:
            out.append("    [" + ",".join(str(int(x)) for x in b) + "],")
        out.append("  ],")
    out.append("];")
    parts.append("\n".join(out))
    parts.append("")
    dest = Path(__file__).resolve().parent / "src" / "tables.rs"
    dest.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {dest} ({dest.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
