"""
Archived tight Nash-style opponent profile (pre-chart-default).

Used only for A/B comparison scripts (``aof_perfect_player_snapshot.py``,
``aof_range_sensitivity_snapshot.py``). Production advisor uses ``aof_ranges.py``
(chart pool default).
"""

from __future__ import annotations

HAND_TUNED_CALL_RANGE_8: dict[tuple[str, str], str] = {
    ("BB", "SB"): "66+,A9s+,ATo+,KJs+,KQo",
    ("BB", "BTN"): "99+,ATs+,AJo+,KQs+",
    ("BB", "CO"): "TT+,AJs+,AQo+",
    ("BB", "UTG"): "JJ+,AQs+,AKo",
    ("SB", "BTN"): "88+,A9s+,ATo+,KQs",
    ("SB", "CO"): "99+,ATs+,AJo+,KQs+",
    ("SB", "UTG"): "TT+,AJs+,AQo+",
    ("CO", "UTG"): "JJ+,AQs+,AKo+",
    ("BTN", "UTG"): "TT+,AJs+,AQo+",
    ("BTN", "CO"): "TT+,AJs+,AQo+",
}

CALL_RANGE_STR: dict[int, dict[tuple[str, str], str]] = {
    8: dict(HAND_TUNED_CALL_RANGE_8),
    10: dict(HAND_TUNED_CALL_RANGE_8),
    6: dict(HAND_TUNED_CALL_RANGE_8),
}

CALL_PCT: dict[int, dict[tuple[str, str], float]] = {
    8: {
        ("BB", "SB"): 0.40,
        ("BB", "BTN"): 0.06,
        ("BB", "CO"): 0.05,
        ("BB", "UTG"): 0.04,
        ("SB", "BTN"): 0.14,
        ("SB", "CO"): 0.09,
        ("SB", "UTG"): 0.05,
        ("CO", "UTG"): 0.07,
        ("BTN", "UTG"): 0.05,
        ("BTN", "CO"): 0.07,
    },
    10: {
        ("BB", "SB"): 0.36,
        ("BB", "BTN"): 0.07,
        ("BB", "CO"): 0.06,
        ("BB", "UTG"): 0.05,
        ("SB", "BTN"): 0.12,
        ("SB", "CO"): 0.08,
        ("SB", "UTG"): 0.05,
        ("CO", "UTG"): 0.05,
        ("BTN", "UTG"): 0.04,
        ("BTN", "CO"): 0.06,
    },
    6: {
        ("BB", "SB"): 0.45,
        ("BB", "BTN"): 0.09,
        ("BB", "CO"): 0.06,
        ("BB", "UTG"): 0.05,
        ("SB", "BTN"): 0.16,
        ("SB", "CO"): 0.10,
        ("SB", "UTG"): 0.06,
        ("CO", "UTG"): 0.05,
        ("BTN", "UTG"): 0.04,
        ("BTN", "CO"): 0.06,
    },
}

HAND_TUNED_SHOVE_8: dict[str, str] = {
    "UTG": "88+,ATs+,AJo+,KQs+",
    "CO": "66+,A9s+,ATo+,KJs+,KQo",
    "BTN": "55+,A7s+,A9o+,K9s+,KJo+,Q9s+,J9s+",
    "SB": "44+,A5s+,A8o+,K9s+,KJo+,Q9s+,J9s+,T9s",
}

SHOVE_RANGE_STR: dict[int, dict[str, str]] = {
    8: dict(HAND_TUNED_SHOVE_8),
    10: {
        "UTG": "77+,ATs+,AJo+,KQs+",
        "CO": "55+,A8s+,A9o+,K9s+,KJo+,Q9s+,J9s+",
        "BTN": "44+,A5s+,A8o+,K9s+,KJo+,Q9s+,J9s+,T9s",
        "SB": "33+,A4s+,A7o+,K8s+,KTo+,Q9s+,QJo,J9s+,T9s",
    },
    6: {
        "UTG": "99+,ATs+,AJo+,KQs+",
        "CO": "77+,A9s+,ATo+,KJs+,KQo+",
        "BTN": "66+,A8s+,A9o+,K9s+,KJo+,Q9s+,J9s+",
        "SB": "55+,A6s+,A8o+,K9s+,KJo+,Q9s+,J9s+,T9s",
    },
}
