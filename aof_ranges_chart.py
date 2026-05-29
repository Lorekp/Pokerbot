"""
Loose chart-pool opponent profile (ICMIZER / coaching-style).

Default production tables for ``aof_ranges.py`` (open/call/shove + multi-jam).

Multi-jam chart rules (hand-tuned, not solver-derived):
- SECOND_JAM: reshove over first open — wider than continue-vs-open (CALL_*).
- THIRD_JAM: squeeze over two jams — wider than SB continue vs UTG alone.
"""

from __future__ import annotations

CALL_RANGE_STR: dict[int, dict[tuple[str, str], str]] = {
    8: {
        ("BB", "SB"): "22+,A2s+,A2o+,K2s+,K5o+,Q4s+,Q8o+,J6s+,J9o+,T7s+,T9o,98s,87s,76s,65s",
        ("BB", "BTN"): "77+,A9s+,ATo+,KQs",
        ("BB", "CO"): "88+,A9s+,ATo+,KQs,KQo",
        ("BB", "UTG"): "TT+,AJs+,AQo+",
        ("SB", "BTN"): "22+,A2s+,A3o+,K5s+,K9o+,Q7s+,QTo+,J8s+,JTo,T8s+,98s,87s",
        ("SB", "CO"): "22+,A5s+,A8o+,K9s+,KJo+,Q9s+,QJo,J9s+,T9s",
        ("SB", "UTG"): "88+,A9s+,ATo+,KQs",
        ("CO", "UTG"): "99+,ATs+,AJo+,KQs",
        ("BTN", "UTG"): "TT+,AJs+,AQo+",
        ("BTN", "CO"): "99+,ATs+,AJo+,KQs",
    },
    10: {
        ("BB", "SB"): "22+,A2s+,A4o+,K5s+,K9o+,Q7s+,QTo+,J8s+,JTo,T8s+,98s,87s",
        ("BB", "BTN"): "88+,A9s+,ATo+,KJs+,KQo",
        ("BB", "CO"): "99+,ATs+,AJo+,KQs",
        ("BB", "UTG"): "TT+,AJs+,AQo+",
        ("SB", "BTN"): "22+,A3s+,A7o+,K8s+,KTo+,Q9s+,QJo,J9s+,T9s",
        ("SB", "CO"): "22+,A6s+,A9o+,K9s+,KJo+,Q9s+,J9s+,T9s",
        ("SB", "UTG"): "99+,ATs+,AJo+,KQs",
        ("CO", "UTG"): "TT+,AJs+,AQo+",
        ("BTN", "UTG"): "JJ+,AQs+,AKo",
        ("BTN", "CO"): "TT+,ATs+,AJo+,KQs",
    },
    6: {
        ("BB", "SB"): "22+,A2s+,A2o+,K2s+,K4o+,Q3s+,Q7o+,J5s+,J8o+,T6s+,T8o,97s+,86s+,75s+,64s+",
        ("BB", "BTN"): "66+,A8s+,A9o+,KTs+,KJo+,QJs",
        ("BB", "CO"): "77+,A9s+,ATo+,KQs",
        ("BB", "UTG"): "88+,ATs+,AJo+,KQs+",
        ("SB", "BTN"): "22+,A2s+,A5o+,K7s+,KTo+,Q8s+,QJo,J8s+,T9s,98s",
        ("SB", "CO"): "22+,A7s+,A9o+,K9s+,KJo+,Q9s+,J9s+",
        ("SB", "UTG"): "99+,ATs+,AJo+,KQs",
        ("CO", "UTG"): "TT+,AJs+,AQo+",
        ("BTN", "UTG"): "TT+,AJs+,AQo+",
        ("BTN", "CO"): "99+,ATs+,AJo+,KQs",
    },
}

CALL_PCT: dict[int, dict[tuple[str, str], float]] = {
    8: {
        ("BB", "SB"): 0.62,
        ("BB", "BTN"): 0.13,
        ("BB", "CO"): 0.10,
        ("BB", "UTG"): 0.08,
        ("SB", "BTN"): 0.35,
        ("SB", "CO"): 0.22,
        ("SB", "UTG"): 0.12,
        ("CO", "UTG"): 0.15,
        ("BTN", "UTG"): 0.10,
        ("BTN", "CO"): 0.14,
    },
    10: {
        ("BB", "SB"): 0.55,
        ("BB", "BTN"): 0.15,
        ("BB", "CO"): 0.11,
        ("BB", "UTG"): 0.09,
        ("SB", "BTN"): 0.30,
        ("SB", "CO"): 0.20,
        ("SB", "UTG"): 0.11,
        ("CO", "UTG"): 0.10,
        ("BTN", "UTG"): 0.08,
        ("BTN", "CO"): 0.12,
    },
    6: {
        ("BB", "SB"): 0.70,
        ("BB", "BTN"): 0.18,
        ("BB", "CO"): 0.12,
        ("BB", "UTG"): 0.09,
        ("SB", "BTN"): 0.40,
        ("SB", "CO"): 0.25,
        ("SB", "UTG"): 0.14,
        ("CO", "UTG"): 0.12,
        ("BTN", "UTG"): 0.10,
        ("BTN", "CO"): 0.15,
    },
}

SHOVE_RANGE_STR: dict[int, dict[str, str]] = {
    8: {
        "UTG": "55+,A9s+,ATo+,KTs+,KQo,QTs+,JTs",
        "CO": "22+,A2s+,A3o+,K7s+,KTo+,Q8s+,QJo,J8s+,JTo,T8s+,98s,87s",
        "BTN": "22+,A2s+,A2o+,K4s+,K9o+,Q8s+,QTo+,J7s+,JTo,T7s+,97s+,86s+,76s",
        "SB": "22+,A2s+,A2o+,K2s+,K5o+,Q4s+,Q8o+,J6s+,J9o+,T7s+,T9o,98s,87s,76s,65s",
    },
    10: {
        "UTG": "66+,ATs+,AJo+,KQs",
        "CO": "22+,A3s+,A7o+,K8s+,KTo+,Q9s+,QJo,J9s+,T9s",
        "BTN": "22+,A2s+,A4o+,K5s+,K9o+,Q7s+,QTo+,J8s+,JTo,T8s+,98s,87s",
        "SB": "22+,A2s+,A4o+,K5s+,K9o+,Q7s+,QTo+,J8s+,JTo,T8s+,98s,87s",
    },
    6: {
        "UTG": "88+,ATs+,AJo+,KQs+",
        "CO": "22+,A5s+,A8o+,K9s+,KJo+,Q9s+,J9s+,T9s",
        "BTN": "22+,A2s+,A5o+,K7s+,KTo+,Q8s+,QJo,J8s+,T9s,98s",
        "SB": "22+,A2s+,A2o+,K2s+,K4o+,Q3s+,Q7o+,J5s+,J8o+,T6s+,T8o,97s+,86s+,75s+,64s+",
    },
}

# Second jammer reshove vs first open (4-max legal pairs). Wider than CALL_* continue.
SECOND_JAM_RANGE_STR: dict[int, dict[tuple[str, str], str]] = {
    8: {
        ("BTN", "UTG"): "99+,ATs+,AJo+,KQs,KQo,QTs+,JTs,T9s",
        ("SB", "UTG"): "77+,A8s+,A9o+,ATo+,KJs+,KQo,QJs+,JTs",
        ("SB", "BTN"): "22+,A2s+,A2o+,K4s+,K8o+,Q6s+,QTo+,J7s+,JTo,T7s+,T9o,98s,87s,76s,65s",
    },
    10: {
        ("BTN", "UTG"): "99+,ATs+,AJo+,KQs,KQo,QTs+,JTs",
        ("SB", "UTG"): "88+,A9s+,ATo+,KJs+,KQo,QJs+",
        ("SB", "BTN"): "22+,A2s+,A3o+,K5s+,K9o+,Q7s+,QTo+,J8s+,JTo,T8s+,98s,87s,76s",
    },
    6: {
        ("BTN", "UTG"): "99+,ATs+,AJo+,KQs,KQo,QTs+,JTs",
        ("SB", "UTG"): "88+,A9s+,ATo+,KJs+,KQo,QJs+",
        ("SB", "BTN"): "22+,A2s+,A4o+,K6s+,K9o+,Q7s+,QTo+,J8s+,JTo,T8s+,98s,87s,76s",
    },
}

# Third jammer vs two prior jams (4-max: SB over UTG+BTN).
THIRD_JAM_RANGE_STR: dict[int, dict[tuple[str, str, str], str]] = {
    8: {
        ("SB", "UTG", "BTN"): (
            "22+,A2s+,A2o+,K3s+,K7o+,Q5s+,Q9o+,J6s+,J9o+,T6s+,T8o+,"
            "97s+,86s+,76s,65s,54s"
        ),
    },
    10: {
        ("SB", "UTG", "BTN"): (
            "22+,A2s+,A3o+,K5s+,K9o+,Q6s+,QTo+,J7s+,J9o+,T7s+,T9o,98s,87s,76s,65s"
        ),
    },
    6: {
        ("SB", "UTG", "BTN"): (
            "22+,A2s+,A4o+,K5s+,K9o+,Q6s+,QTo+,J7s+,J9o+,T7s+,T9o,98s,87s,76s,65s"
        ),
    },
}
