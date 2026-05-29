"""Parse AoF advisor shorthand for open-shove and call-vs-shove spots."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Union

from aof_ranges import ACTION_ORDER_4, active_seats, position_aliases

RANKS = "23456789TJQKA"
SUITS = "cdhs"
RANK_RE = f"[{RANKS}]"
PAIR_RE = re.compile(rf"^({RANK_RE})\1$", re.I)
TYPE_RE = re.compile(
    rf"^({RANK_RE})({RANK_RE})([so])?$",
    re.I,
)
CARD_RE = re.compile(
    rf"^({RANK_RE})([{SUITS}])({RANK_RE})([{SUITS}])$",
    re.I,
)
POS_RE = re.compile(r"^[A-Za-z+0-9]+$")
BB_SUFFIX = re.compile(r"^(\d+(?:\.\d+)?)\s*bb$", re.I)
STACK_BB_RE = re.compile(r"^\d+(?:\.\d+)?bb$", re.I)
BEHIND_RE = re.compile(r"^behind$", re.I)


def _is_stack_bb_token(tok: str) -> bool:
    if BB_SUFFIX.match(tok):
        return True
    return bool(STACK_BB_RE.match(tok))


@dataclass(frozen=True)
class OpenSpot:
    hand_label: str
    hero_cards: tuple[str, str]
    position: str
    stack_bb: float
    table_size: int


@dataclass(frozen=True)
class CallSpot:
    hand_label: str
    hero_cards: tuple[str, str]
    position: str
    shover: str
    stack_bb: float
    table_size: int
    players_behind: int
    behind_seats: tuple[str, ...]
    second_shover: str | None = None
    third_shover: str | None = None

    @property
    def shovers(self) -> tuple[str, ...]:
        out: list[str] = [self.shover]
        if self.second_shover:
            out.append(self.second_shover)
        if self.third_shover:
            out.append(self.third_shover)
        return tuple(out)

    @property
    def n_prior_shovers(self) -> int:
        return len(self.shovers)


Spot = Union[OpenSpot, CallSpot]


def _norm_pos(token: str, table_size: int = 4) -> str:
    key = token.strip().upper()
    aliases = position_aliases(table_size)
    if key in aliases:
        key = aliases[key]
    order = _action_order(table_size)
    if key not in order:
        raise ValueError(
            f"Unknown position {token!r}. "
            f"Use one of: {', '.join(order)} (aliases: EP, CO→BTN in 4-max, …)."
        )
    return key


def _parse_hand(token: str) -> tuple[str, str, str]:
    t = token.strip().replace(" ", "")
    if not t:
        raise ValueError("Missing hand.")

    m = CARD_RE.match(t)
    if m:
        r1, s1, r2, s2 = m.groups()
        c1, c2 = f"{r1.upper()}{s1.lower()}", f"{r2.upper()}{s2.lower()}"
        if c1 == c2:
            raise ValueError(f"Duplicate card: {token}")
        return f"{c1}{c2}", (c1, c2)

    m = PAIR_RE.match(t)
    if m:
        r = m.group(1).upper()
        c1, c2 = f"{r}c", f"{r}d"
        return f"{r}{r}", (c1, c2)

    m = TYPE_RE.match(t)
    if not m:
        raise ValueError(
            f"Cannot parse hand {token!r}. Use 99, AKs, AKo, or AhKd."
        )
    r1, r2, suited = m.group(1).upper(), m.group(2).upper(), m.group(3)
    if r1 == r2 and suited:
        raise ValueError(f"Pair {token} cannot be suited.")
    label = f"{r1}{r2}{suited.lower()}" if suited else f"{r1}{r2}"
    if r1 == r2:
        c1, c2 = f"{r1}c", f"{r2}d"
    elif suited and suited.lower() == "s":
        c1, c2 = f"{r1}s", f"{r2}s"
    elif suited and suited.lower() == "o":
        c1, c2 = f"{r1}c", f"{r2}d"
    else:
        c1, c2 = f"{r1}c", f"{r2}d"
        label = f"{r1}{r2}o"
    return label, (c1, c2)


def _action_order(table_size: int) -> tuple[str, ...]:
    return active_seats(table_size)


def _validate_shovers_before_hero(
    hero: str,
    shover: str,
    table_size: int,
    *,
    second_shover: str | None = None,
    third_shover: str | None = None,
) -> tuple[str, str | None, str | None]:
    shovers: list[str] = [shover]
    if second_shover:
        shovers.append(second_shover)
    if third_shover:
        shovers.append(third_shover)
    _validate_shover_sequence(hero, tuple(shovers), table_size)
    return shover, second_shover, third_shover


def _validate_shover_sequence(
    hero: str,
    shovers: tuple[str, ...],
    table_size: int,
) -> None:
    order = _action_order(table_size)
    if hero not in order:
        raise ValueError(f"Position {hero} not in {table_size}-max order.")
    if len(shovers) > 3:
        raise ValueError("At most 3 prior shovers supported.")
    if len(shovers) >= 3 and table_size < 4:
        raise ValueError(
            f"Facing 3 requires 4-max table (got {table_size}-max)."
        )
    hi = order.index(hero)
    seen: set[str] = set()
    prev_i = -1
    for s in shovers:
        if s in seen:
            raise ValueError(f"Duplicate shover {s}.")
        seen.add(s)
        if s not in order:
            raise ValueError(f"Unknown shover position {s!r}.")
        si = order.index(s)
        if si >= hi:
            raise ValueError(
                f"Shover {s} must act before hero {hero} in preflop order."
            )
        if si <= prev_i:
            raise ValueError(
                f"Shovers must be in preflop order "
                f"({', '.join(order[:hi])}); got {', '.join(shovers)}."
            )
        prev_i = si


def _infer_players_behind(
    hero: str,
    shover: str,
    table_size: int,
) -> tuple[int, tuple[str, ...]]:
    order = _action_order(table_size)
    if hero not in order:
        raise ValueError(f"Position {hero} not in {table_size}-max order.")
    if shover not in order:
        raise ValueError(f"Shover position {shover} not in {table_size}-max order.")
    si, hi = order.index(shover), order.index(hero)
    if si >= hi:
        raise ValueError(
            f"Shover {shover} must act before hero {hero} in preflop order."
        )
    behind = tuple(order[hi + 1 :])
    return len(behind), behind


def _parse_nums(
    nums: list[float],
    *,
    stack_bb: float,
    table_size: int,
) -> tuple[float, int]:
    if len(nums) == 1:
        n = nums[0]
        if n <= 10 and float(int(n)) == n and int(n) in (2, 3, 4, 5, 6, 7, 8, 9):
            return stack_bb, int(n)
        return n, table_size
    if len(nums) >= 2:
        a, b = nums[0], nums[1]
        if a <= 10 and float(int(a)) == a and int(a) in (2, 3, 4, 5, 6, 7, 8, 9):
            return b, int(a)
        return a, int(b)
    return stack_bb, table_size


def _is_call_trigger(parts: list[str], force_call: bool) -> bool:
    if force_call:
        return True
    low = [p.lower() for p in parts]
    return "vs" in low or "call" in low


def parse_spot_line(
    text: str,
    *,
    default_stack: float = 8.0,
    default_table: int = 4,
    force_call: bool = False,
) -> Spot:
    parts = text.strip().split()
    if len(parts) < 2:
        raise ValueError(
            "Need at least hand and position, e.g. 'AJo BTN' or 'AA BB vs BTN'."
        )

    if not _is_call_trigger(parts, force_call):
        return _parse_open(parts, default_stack, default_table)
    return _parse_call(parts, default_stack, default_table)


def _parse_open(
    parts: list[str],
    default_stack: float,
    default_table: int,
) -> OpenSpot:
    hand_tok = parts[0]
    pos_tok = parts[1]
    if not POS_RE.match(pos_tok) or _is_stack_bb_token(pos_tok):
        raise ValueError(f"Expected position, got {pos_tok!r}.")

    label, cards = _parse_hand(hand_tok)
    stack_bb = default_stack
    table_size = default_table
    nums: list[float] = []

    for tok in parts[2:]:
        stack_bb, table_size, nums = _consume_token(tok, stack_bb, table_size, nums)

    stack_bb, table_size = _parse_nums(nums, stack_bb=stack_bb, table_size=table_size)
    _validate_table_stack(table_size, stack_bb)
    position = _norm_pos(pos_tok, table_size)
    return OpenSpot(label, cards, position, stack_bb, table_size)


def _parse_call(
    parts: list[str],
    default_stack: float,
    default_table: int,
) -> CallSpot:
    hand_tok = parts[0]
    hero_tok = parts[1]
    if not POS_RE.match(hero_tok) or _is_stack_bb_token(hero_tok):
        raise ValueError(f"Expected hero position, got {hero_tok!r}.")

    label, cards = _parse_hand(hand_tok)

    shover_tok: str | None = None
    second_shover_tok: str | None = None
    third_shover_tok: str | None = None
    shover: str | None = None
    players_behind: int | None = None
    stack_bb = default_stack
    table_size = default_table
    nums: list[float] = []

    i = 2
    while i < len(parts):
        tok = parts[i]
        low = tok.lower()

        if low == "vs":
            if i + 1 >= len(parts):
                raise ValueError("'vs' needs shover position, e.g. 'AA BB vs BTN'.")
            tok = parts[i + 1]
            if "," in tok:
                bits = [b.strip() for b in tok.split(",") if b.strip()]
                if len(bits) < 1 or len(bits) > 3:
                    raise ValueError(
                        "One to three shovers required, e.g. "
                        "'AA BB vs BTN' or 'Ks4d BB vs UTG,BTN,SB'."
                    )
                shover_tok = bits[0]
                if len(bits) >= 2:
                    second_shover_tok = bits[1]
                if len(bits) == 3:
                    third_shover_tok = bits[2]
                i += 2
            else:
                shover_tok = tok
                if i + 2 < len(parts):
                    nxt = parts[i + 2]
                    if (
                        not BEHIND_RE.match(nxt)
                        and not _is_stack_bb_token(nxt)
                        and nxt.lower() not in ("hu", "call")
                        and _norm_pos_optional(nxt, default_table) is not None
                    ):
                        second_shover_tok = nxt
                        i += 3
                        continue
                i += 2
            continue

        if low == "call":
            if shover_tok is None:
                if i + 1 >= len(parts):
                    raise ValueError(
                        "'call' needs shover position, e.g. '99 SB call BTN'."
                    )
                shover_tok = parts[i + 1]
                i += 2
                continue
            i += 1
            continue

        if BEHIND_RE.match(tok):
            if i + 1 >= len(parts):
                raise ValueError("'behind' needs a count, e.g. 'behind 1'.")
            players_behind = int(parts[i + 1])
            i += 2
            continue

        stack_bb, table_size, nums = _consume_token(tok, stack_bb, table_size, nums)
        i += 1

    if shover_tok is None:
        raise ValueError(
            "Call-vs-shove spot needs shover: use 'vs BTN' or 'call BTN', "
            "e.g. 'AA BB vs BTN 8bb'."
        )

    stack_bb, table_size = _parse_nums(nums, stack_bb=stack_bb, table_size=table_size)
    _validate_table_stack(table_size, stack_bb)
    hero = _norm_pos(hero_tok, table_size)
    shover = _norm_pos(shover_tok, table_size)
    second_shover = (
        _norm_pos(second_shover_tok, table_size) if second_shover_tok else None
    )
    third_shover = (
        _norm_pos(third_shover_tok, table_size) if third_shover_tok else None
    )
    _validate_shovers_before_hero(
        hero,
        shover,
        table_size,
        second_shover=second_shover,
        third_shover=third_shover,
    )

    if second_shover is not None or third_shover is not None:
        players_behind = 0

    if players_behind is not None:
        max_behind = max(0, table_size - 2)
        if players_behind < 0 or players_behind > max_behind:
            raise ValueError(
                f"players behind must be 0–{max_behind} for {table_size}-handed, "
                f"got {players_behind}."
            )
        n_behind = players_behind
    else:
        n_behind, _ = _infer_players_behind(hero, shover, table_size)

    _, behind_seats = _infer_players_behind(hero, shover, table_size)
    if n_behind != len(behind_seats):
        if n_behind < len(behind_seats):
            behind_seats = behind_seats[:n_behind]
        elif n_behind > len(behind_seats) and players_behind is not None:
            raise ValueError(
                f"behind {n_behind} invalid for {hero} vs {shover} "
                f"(at most {len(behind_seats)} in {table_size}-max order)."
            )

    return CallSpot(
        label,
        cards,
        hero,
        shover,
        stack_bb,
        table_size,
        n_behind,
        behind_seats,
        second_shover=second_shover,
        third_shover=third_shover,
    )


def _consume_token(
    tok: str,
    stack_bb: float,
    table_size: int,
    nums: list[float],
) -> tuple[float, int, list[float]]:
    bm = BB_SUFFIX.match(tok)
    if bm:
        return float(bm.group(1)), table_size, nums
    if _is_stack_bb_token(tok):
        return float(tok.lower().removesuffix("bb")), table_size, nums
    if tok.lower() == "hu":
        return stack_bb, 2, nums
    if tok.isdigit() or re.match(r"^\d+(\.\d+)?$", tok):
        nums.append(float(tok))
        return stack_bb, table_size, nums
    if _norm_pos_optional(tok) is not None:
        raise ValueError(
            f"Unexpected position {tok!r} in call spot "
            "(use 'vs SHOVER' or 'call SHOVER')."
        )
    raise ValueError(f"Unexpected token {tok!r}.")


def _norm_pos_optional(token: str, table_size: int = 4) -> str | None:
    key = token.strip().upper()
    aliases = position_aliases(table_size)
    if key in aliases:
        return aliases[key]
    if key in _action_order(table_size):
        return key
    return None


def _validate_table_stack(table_size: int, stack_bb: float) -> None:
    if table_size < 2 or table_size > 9:
        raise ValueError(f"table_size must be 2–9, got {table_size}.")
    if stack_bb <= 0:
        raise ValueError("stack must be positive.")


# Backward-compatible alias
def parse_spot(text: str, *, default_stack: float = 8.0, default_table: int = 4) -> OpenSpot:
    spot = parse_spot_line(text, default_stack=default_stack, default_table=default_table)
    if not isinstance(spot, OpenSpot):
        raise ValueError(
            "This looks like a call-vs-shove spot (vs/call). "
            "Use --call or add 'vs SHOVER'."
        )
    return spot
