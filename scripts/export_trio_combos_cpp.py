#!/usr/bin/env python3
"""Generate C++ combo tables for native trio tensor builder (169 buckets)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aof_cfr.buckets import COMBOS_169
from aof_equity_grids import _card_str_to_id

OUT = _ROOT / "tools" / "ompeval_cli" / "trio_combos_data.hpp"


def main() -> None:
    lines = [
        "#pragma once",
        "",
        "#include <array>",
        "#include <cstdint>",
        "",
        "struct BucketComboPair {",
        "    int c0;",
        "    int c1;",
        "};",
        "",
        "static constexpr int TRIO_N_BUCKETS = 169;",
        "static constexpr int TRIO_MAX_COMBOS_PER_BUCKET = 6;",
        "",
    ]
    ncombos = []
    for i, combos in enumerate(COMBOS_169):
        ncombos.append(len(combos))
        lines.append(f"static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_{i} = {{ {{")
        entries = []
        for c0, c1 in combos:
            id0 = _card_str_to_id(c0)
            id1 = _card_str_to_id(c1)
            entries.append(f"    {{ {id0}, {id1} }}")
        while len(entries) < 6:
            entries.append("    { -1, -1 }")
        lines.append(",\n".join(entries))
        lines.append("} };")
        lines.append("")

    lines.append("static constexpr std::array<int, TRIO_N_BUCKETS> TRIO_BUCKET_NCOMBOS = {")
    lines.append("    " + ", ".join(str(n) for n in ncombos))
    lines.append("};")
    lines.append("")
    lines.append(
        "static constexpr std::array<const BucketComboPair*, TRIO_N_BUCKETS> TRIO_BUCKET_COMBOS = {"
    )
    lines.append("    " + ", ".join(f"TRIO_BUCKET_{i}.data()" for i in range(169)))
    lines.append("};")
    lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
