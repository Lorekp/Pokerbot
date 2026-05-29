"""Python adapter: JSON solve request -> TexasSolver console_solver -> contract response."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from pokerkit import parse_range

_RANKS = "23456789TJQKA"


def _combo_ranks_suits(combo: frozenset) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for card in combo:
        text = str(card)
        m = re.search(r"\(([2-9TJQKA])([cdhs])\)", text, re.I)
        if m:
            out.append((m.group(1).upper(), m.group(2).lower()))
    out.sort(key=lambda x: _RANKS.index(x[0]), reverse=True)
    return out


def _hand_class(combo: frozenset) -> str:
    cards = _combo_ranks_suits(combo)
    if len(cards) != 2:
        return ""
    (r1, s1), (r2, s2) = cards
    if r1 == r2:
        return r1 + r2
    hi, lo = (r1, s1), (r2, s2)
    if _RANKS.index(r1) < _RANKS.index(r2):
        hi, lo = (r2, s2), (r1, s1)
    suited = hi[1] == lo[1]
    return hi[0] + lo[0] + ("s" if suited else "o")


def postflop_range_to_texas(range_str: str) -> str:
    """Expand postflop-solver range DSL to TexasSolver Pio-style hand classes."""
    try:
        classes: set[str] = set()
        for combo in parse_range(range_str):
            hc = _hand_class(combo)
            if hc:
                classes.add(hc)
    except Exception:
        return range_str
    if not classes:
        return range_str

    def sort_key(h: str) -> tuple:
        if len(h) == 2:
            return (0, -_RANKS.index(h[0]))
        return (1, -_RANKS.index(h[0]), -_RANKS.index(h[1]), 0 if h.endswith("s") else 1)

    return ",".join(sorted(classes, key=sort_key))


def _parse_raise_pct(raise_sizes: str) -> int:
    if raise_sizes.endswith("x"):
        return int(float(raise_sizes[:-1]) * 100)
    return 250


def build_texassolver_config(request: dict[str, Any], *, thread_num: int | None = None) -> str:
    pot_chips = int(float(request["starting_pot_bb"]) * float(request.get("chip_unit_bb", 100)))
    stack_chips = int(float(request["effective_stack_bb"]) * float(request.get("chip_unit_bb", 100)))
    board = request["board"]
    raise_pct = _parse_raise_pct(str(request.get("raise_sizes", "2.5x")))
    threads = thread_num if thread_num is not None else (os.cpu_count() or 8)

    lines = [
        f"set_pot {pot_chips}",
        f"set_effective_stack {stack_chips}",
        f"set_board {board[0]},{board[1]},{board[2]}",
        f"set_range_oop {postflop_range_to_texas(str(request['oop_range']))}",
        f"set_range_ip {postflop_range_to_texas(str(request['ip_range']))}",
    ]
    for street in ("flop", "turn", "river"):
        for role in ("oop", "ip"):
            lines.append(f"set_bet_sizes {role},{street},bet,50")
            lines.append(f"set_bet_sizes {role},{street},raise,{raise_pct}")
            lines.append(f"set_bet_sizes {role},{street},allin")
    lines.extend(
        [
            "set_allin_threshold 0.15",
            "build_tree",
            f"set_thread_num {threads}",
            f"set_accuracy {request.get('target_exploitability_pct', 0.5)}",
            f"set_max_iteration {int(request.get('max_iterations', 10000))}",
            "set_print_interval 50",
            "set_use_isomorphism 1",
            "start_solve",
            "set_dump_rounds 2",
            "dump_result output_result.json",
        ]
    )
    return "\n".join(lines) + "\n"


def _normalize_action_key(raw: str) -> str:
    k = raw.lower()
    if k == "call":
        return "check"
    if "all" in k:
        return "all_in"
    if "bet" in k or "raise" in k:
        return "bet_0.50"
    return raw


def _map_texas_action(action: str, pot_chips: int) -> str:
    a = action.upper().strip()
    if a == "CHECK" or a == "CALL":
        return "check"
    if a == "FOLD":
        return "fold"
    if a.startswith("BET ") or a.startswith("RAISE "):
        try:
            amount = float(a.split()[1])
        except (IndexError, ValueError):
            return "bet_0.50"
        if pot_chips > 0 and amount >= pot_chips * 5:
            return "all_in"
        if pot_chips > 0:
            frac = round(amount / pot_chips, 2)
            return f"bet_{frac:.2f}"
        return "bet_0.50"
    return action.lower()


def _root_strategy_distribution(output: dict[str, Any], pot_chips: int) -> tuple[dict[str, float], list[str]]:
    """Extract average root action probabilities from TexasSolver dump."""
    dist: dict[str, float] = {}
    avail: list[str] = []

    strategy_block = output.get("strategy")
    if not isinstance(strategy_block, dict):
        return dist, avail

    actions = strategy_block.get("actions") or output.get("actions") or []
    hands = strategy_block.get("strategy")
    if not isinstance(actions, list) or not isinstance(hands, dict):
        return dist, avail

    n_actions = len(actions)
    if n_actions == 0:
        return dist, avail

    sums = [0.0] * n_actions
    count = 0
    for vec in hands.values():
        if not isinstance(vec, list) or len(vec) != n_actions:
            continue
        for i, p in enumerate(vec):
            if isinstance(p, (int, float)):
                sums[i] += float(p)
        count += 1

    if count == 0:
        return dist, avail

    for i, action in enumerate(actions):
        key = _map_texas_action(str(action), pot_chips)
        if key not in avail:
            avail.append(key)
        dist[key] = dist.get(key, 0.0) + sums[i] / count

    total = sum(dist.values())
    if total > 0:
        dist = {k: v / total for k, v in dist.items()}
    return dist, avail


def _parse_exploitability_from_stderr(stderr: str) -> tuple[float, int]:
    """Return (exploitability_pct_of_pot, iterations_used) from TexasSolver log."""
    exploit_pct = 100.0
    iterations = 0
    for line in stderr.splitlines():
        if "Total exploitability" in line:
            parts = line.split()
            for token in parts:
                try:
                    exploit_pct = float(token)
                    break
                except ValueError:
                    continue
        if line.strip().startswith("Iter:"):
            try:
                iterations = int(line.split(":")[1].strip())
            except (IndexError, ValueError):
                pass
    return exploit_pct, iterations


def _walk_actions(node: dict[str, Any], dist: dict[str, float], avail: list[str]) -> None:
    """Legacy tree walk — prefer _root_strategy_distribution for TexasSolver dumps."""
    actions = node.get("actions")
    if isinstance(actions, dict):
        for name, val in actions.items():
            key = _normalize_action_key(str(name))
            if key not in avail:
                avail.append(key)
            mass = 0.0
            count = 0
            if isinstance(val, (int, float)):
                mass = float(val)
                count = 1
            elif isinstance(val, dict):
                for v in val.values():
                    if isinstance(v, (int, float)):
                        mass += float(v)
                        count += 1
            if count:
                dist[key] = dist.get(key, 0.0) + mass / count
    for child_key in ("childrens", "children"):
        children = node.get(child_key)
        if isinstance(children, dict):
            for child in children.values():
                if isinstance(child, dict):
                    _walk_actions(child, dist, avail)


def adapt_texassolver_output(
    output: dict[str, Any],
    *,
    target_exploitability_pct: float,
    max_iterations: int,
    pot_chips: int,
    stderr: str = "",
    stdout: str = "",
) -> dict[str, Any]:
    exploit_pct, iterations_used = _parse_exploitability_from_stderr(stderr + "\n" + stdout)
    if exploit_pct >= 100.0 and isinstance(output.get("meta"), dict):
        meta = output["meta"]
        if "exploitability" in meta:
            exploit_pct = float(meta["exploitability"])
        if "iteration" in meta:
            iterations_used = int(meta["iteration"])

    target_exploitability = pot_chips * (target_exploitability_pct / 100.0)
    exploitability = pot_chips * (exploit_pct / 100.0)
    converged = exploit_pct <= target_exploitability_pct + 1e-6

    dist, avail = _root_strategy_distribution(output, pot_chips)
    if not dist:
        _walk_actions(output, dist, avail)
        total = sum(dist.values())
        if total > 0:
            dist = {k: v / total for k, v in dist.items()}

    return {
        "exploitability": exploitability,
        "target_exploitability": target_exploitability,
        "exploitability_pct_of_pot": exploit_pct,
        "converged": converged,
        "iterations": max_iterations,
        "iterations_used": iterations_used,
        "available_actions": avail,
        "action_distribution": dist,
        "solver_version": "texas-solver@console",
        "error": None,
    }


def resolve_texas_root() -> Path | None:
    root = os.environ.get("TEXAS_SOLVER_ROOT")
    if root:
        return Path(root)
    console = os.environ.get("TEXAS_SOLVER_CONSOLE")
    if console:
        return Path(console).parent
    return None


def resolve_console_binary(root: Path) -> Path:
    env = os.environ.get("TEXAS_SOLVER_CONSOLE")
    if env:
        return Path(env)
    name = "console_solver.exe" if os.name == "nt" else "console_solver"
    return root / name


def run_texas_solver(request: dict[str, Any], *, timeout_sec: float = 7200.0) -> dict[str, Any]:
    root = resolve_texas_root()
    if root is None or not root.is_dir():
        return {
            "exploitability": 0.0,
            "target_exploitability": 0.0,
            "exploitability_pct_of_pot": 0.0,
            "converged": False,
            "iterations": int(request.get("max_iterations", 0)),
            "iterations_used": 0,
            "available_actions": [],
            "action_distribution": {},
            "solver_version": "texas-solver@console",
            "error": "TEXAS_SOLVER_ROOT or TEXAS_SOLVER_CONSOLE not set",
        }

    console = resolve_console_binary(root)
    if not console.is_file():
        return {
            "exploitability": 0.0,
            "target_exploitability": 0.0,
            "exploitability_pct_of_pot": 0.0,
            "converged": False,
            "iterations": int(request.get("max_iterations", 0)),
            "iterations_used": 0,
            "available_actions": [],
            "action_distribution": {},
            "solver_version": "texas-solver@console",
            "error": f"console_solver not found at {console}",
        }

    pot_chips = int(float(request["starting_pot_bb"]) * float(request.get("chip_unit_bb", 100)))

    with tempfile.TemporaryDirectory(prefix="texas_solver_") as tmp:
        work = Path(tmp)
        config_path = work / "input.txt"
        output_path = work / "output_result.json"
        config_path.write_text(build_texassolver_config(request), encoding="utf-8")

        proc = subprocess.run(
            [str(console), "-i", str(config_path)],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )

        # TexasSolver writes output_result.json relative to cwd (install root)
        root_output = root / "output_result.json"
        if root_output.is_file():
            raw_text = root_output.read_text(encoding="utf-8")
            try:
                root_output.unlink()
            except OSError:
                pass
        elif output_path.is_file():
            raw_text = output_path.read_text(encoding="utf-8")
        else:
            return {
                "exploitability": 0.0,
                "target_exploitability": 0.0,
                "exploitability_pct_of_pot": 0.0,
                "converged": False,
                "iterations": int(request.get("max_iterations", 0)),
                "iterations_used": 0,
                "available_actions": [],
                "action_distribution": {},
                "solver_version": "texas-solver@console",
                "error": f"TexasSolver failed exit={proc.returncode} stderr={proc.stderr[:400]!r}",
            }

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            return {
                "exploitability": 0.0,
                "target_exploitability": 0.0,
                "exploitability_pct_of_pot": 0.0,
                "converged": False,
                "iterations": int(request.get("max_iterations", 0)),
                "iterations_used": 0,
                "available_actions": [],
                "action_distribution": {},
                "solver_version": "texas-solver@console",
                "error": f"invalid TexasSolver JSON: {exc}",
            }

        return adapt_texassolver_output(
            parsed,
            target_exploitability_pct=float(request.get("target_exploitability_pct", 0.5)),
            max_iterations=int(request.get("max_iterations", 10000)),
            pot_chips=pot_chips,
            stderr=proc.stderr or "",
            stdout=proc.stdout or "",
        )
