"""169 canonical hand buckets (shared with HU DCFR oracle)."""

from __future__ import annotations

from pokerbot.solver.hu_preflop_dcfr import _build_169_buckets

LABELS_169, COMBOS_169 = _build_169_buckets()

LABEL_TO_INDEX: dict[str, int] = {lab: i for i, lab in enumerate(LABELS_169)}


def bucket_index(label: str) -> int:
    try:
        return LABEL_TO_INDEX[label]
    except KeyError as exc:
        raise KeyError(f"Unknown hand bucket {label!r}") from exc


def combo_to_bucket(c0: str, c1: str) -> str:
    """Map two card strings to 169 label."""
    r0, s0 = c0[0], c0[1]
    r1, s1 = c1[0], c1[1]
    if r0 == r1:
        return f"{r0}{r1}"
    hi, lo = (r0, r1) if "23456789TJQKA".index(r0) > "23456789TJQKA".index(r1) else (r1, r0)
    suited = s0 == s1
    return f"{hi}{lo}{'s' if suited else 'o'}"
