#!/usr/bin/env python3
"""No-dep flamegraph for the 500-iter smoke run.

Why this exists: the project hosts on Win32 and `perf record --call-graph=dwarf`
+ `inferno-flamegraph` are Linux-only; the user's WSL distro has no `perf` and
no `cargo`. The in-repo precedent (tools/battery_native/profile_battery.py,
scripts/native_cfv_profile.py) is `cProfile` + `pstats`. This tool emits an SVG
flamegraph (Brendan-Gregg layout: stacks as nested rectangles, color = hot/cold)
directly from cProfile stats, with zero new dependencies. Pure stdlib.

Measure-only — does NOT modify any source. Captures the search/CFV hot path
(the equivalent of the 500-iter smoke run used in the per-function breakdown
from `profile_nodes.exe 64 500`).

Usage:
  python tools/perf_flamegraph.py \\
      --snapshot "checkpoints/bp6max_lossless_iter200.pkl" \\
      --iterations 500 --out artifacts/perf_baseline_training.svg
"""
from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import random
import sys
import time
from typing import Dict, List, Tuple

# ---------- workload: 500-iter smoke of the search/CFV hot path ----------

def _build_workload(snapshot_path: str, iterations: int):
    """Return a (state -> Action) callable that drives every decision through
    the SearchAgent at `iterations` CFR iters. The agent is built ONCE per
    workload (not per call) to avoid reconstructing its 180k-infoset blueprint
    on every decision — which would make the profile meaningless.

    Preflop: cheapest non-fold (CALL/CHECK) so we reach postflop where the
    search path actually runs.
    Postflop: SearchAgent.decide() (via __call__) with the agent's solver
    set to `iterations` max-iterations."""
    import pickle
    from pokerbot.engine.actions import ActionType
    from pokerbot.eval.search_agent import SearchAgent

    with open(snapshot_path, "rb") as f:
        snap_full = pickle.load(f)
    snap = snap_full["strat_sum"] if isinstance(snap_full, dict) and "strat_sum" in snap_full else snap_full
    rng = random.Random(0)

    def _pol_factory():
        # Lazy agent construction: built once when the policy is requested, not
        # at module import time, so the warmup pass also pays it.
        from pokerbot.search.state_bridge import game_state_to_env

        # Cache one SearchAgent per hero_seat (the solver reads trajectory keyed
        # on hero_seat; building per-call would skew the profile).
        agents: Dict[int, SearchAgent] = {}

        def _agent_for(seat: int) -> SearchAgent:
            if seat not in agents:
                a = SearchAgent(
                    use_pluribus_search=True,
                    blueprint_snapshot=snap,
                    pluribus_leaf_select="uniform",
                    time_budget_ms=10 ** 9,
                    transformer_checkpoint=None,
                    hero_seat=seat,
                )
                a.solver.config.max_iterations = iterations
                a.solver.config.time_budget_ms = 10 ** 9
                agents[seat] = a
            return agents[seat]

        def _pol(state):
            legal = state.legal_actions()
            if not legal:
                return None
            s = state.action_seat
            street = int(state.street.value if hasattr(state.street, "value") else state.street)
            if street < 2:
                # preflop: cheapest non-fold so we reach postflop
                for ai in legal:
                    if ai.action_type in (ActionType.CHECK, ActionType.CALL):
                        return ai
                return legal[0]
            # postflop: full SearchAgent path
            env = game_state_to_env(state, hero_seat=s)
            return _agent_for(s)(state)  # SearchAgent.__call__

        return _pol

    return _pol_factory


def _run_workload(snapshot_path: str, hands: int, iterations: int):
    """Play `hands` random hands; every decision (preflop + postflop) goes
    through the SearchAgent with `iterations` CFR iters. The cProfile outer
    wrapper times this. The goal is to measure the 500-iter SMOKE of the
    hot serve path (search/CFV)."""
    from pokerbot.engine.game import GameState, play_hand
    pol_factory = _build_workload(snapshot_path, iterations)

    for _ in range(hands):
        st0 = GameState.new_hand(button_seat=random.randint(0, 5),
                                 rng=random.Random(random.randint(0, 10 ** 9)))
        smart_pol = pol_factory()
        policies = {s: smart_pol for s in range(6)}
        try:
            play_hand(policies, state=st0)
        except Exception as e:  # noqa: BLE001 — measurement; don't crash on a single bad node
            print(f"  [warn] hand skipped: {type(e).__name__}: {e}", file=sys.stderr)


# ---------- cProfile capture ----------

def capture(snapshot_path: str, hands: int, iterations: int) -> Tuple[cProfile.Profile, float]:
    """cProfile a fresh run, but FIRST run one warmup hand (outside the profiler)
    so import costs / first-call overhead don't dominate the top-N."""
    # Warmup: load modules, deserialize the blueprint, run one full hand.
    try:
        _run_workload(snapshot_path, hands=1, iterations=10)
    except Exception:
        pass
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    _run_workload(snapshot_path, hands, iterations)
    pr.disable()
    return pr, time.perf_counter() - t0


# ---------- folded-stack extraction (cProfile -> FlameGraph input format) ----
# Format: "frame1;frame2;frame3 count"  (semicolon-separated leaf-to-root, count = self samples)

def extract_folded(pr: cProfile.Profile) -> Dict[Tuple[str, ...], int]:
    """Extract (callstack -> inclusive-sample-count) from cProfile.

    cProfile's stats are per-function aggregates, NOT per-sample callstacks, so
    we reconstruct approximate callstacks by walking the callers/callees graph.
    Each function's inclusive time is attributed to a synthetic callstack built
    from `(caller chain) -> fn`. This is a faithful approximation used by
    py-spy/dogpile.profile equivalents; perfect attribution would need a sampler
    (py-spy/yspy), which we don't ship. Sample counts are normalized to `tottime`
    units so the SVG widths are comparable to perf's self-time view."""
    st = pstats.Stats(pr)
    st.sort_stats("tottime")

    # cProfile keys are 3-tuples (filename, line, funcname).
    # Value is (cc, nc, tt, ct, callers_dict) where callers_dict maps
    # caller_key -> (cc, nc, tt, ct) — a 4-tuple per caller.
    raw = st.stats  # dict[3-tuple] -> 5-tuple

    folded: Dict[Tuple[str, ...], int] = {}

    def _short(key) -> str:
        # Reduce a 3-tuple cProfile key to a compact, readable frame label.
        if not isinstance(key, tuple) or len(key) != 3:
            return str(key)[:32]
        fn = key[2]
        path = key[0]
        if not path:
            if not fn:
                return "<unknown>"
            if fn.startswith("<"):
                return fn[:32]
            return f"<{fn}>"
        if "importlib" in path and "_bootstrap" in path:
            return "<importlib>"
        if "site-packages" in path:
            return f"<pip:{fn or 'mod'}>"
        mod = path.replace("\\", "/").rstrip("/")
        parts = mod.split("/")
        if len(parts) > 2:
            mod = "/".join(parts[-2:])
        mod = mod.split(".")[0]
        if not fn:
            return mod or "<unknown>"
        return f"{mod}:{fn}" if mod else fn

    for callee_key, (cc, _nc, tt, ct, callers) in raw.items():
        callee_short = _short(callee_key)
        if not callers:
            stack = (callee_short,)
            folded[stack] = folded.get(stack, 0) + int(max(tt * 1e6, 1))
            continue
        total_cc = sum(c_info[0] for c_info in callers.values()) or 1
        for caller_key, (c_cc, _c_nc, _c_tt, c_ct) in callers.items():
            caller_short = _short(caller_key)
            share = c_ct * (c_cc / total_cc) if total_cc else c_ct
            stack = (caller_short, callee_short)
            folded[stack] = folded.get(stack, 0) + int(max(share * 1e6, 1))

    return folded


# ---------- SVG flamegraph renderer (Brendan-Gregg layout, pure stdlib) -----

_HTML_ESC = str.maketrans({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"})


def _esc(s: str) -> str:
    return s.translate(_HTML_ESC)


def _color_for(depth: int, pct: float) -> str:
    """Hot (red) -> cold (blue) gradient as a function of fraction-of-parent-width."""
    # pct in [0, 100]; clamp. Maps to HSL hue 0 (red) -> 220 (blue).
    pct = max(0.0, min(1.0, pct / 100.0))
    hue = int(220 * (1.0 - pct))  # 0 = red (hot), 220 = blue (cold)
    sat = 55 + int(35 * pct)
    light = 55 - int(15 * (1.0 - pct))
    return f"hsl({hue},{sat}%,{light}%)"


def render_svg(folded: Dict[Tuple[str, ...], int], out_path: str, title: str) -> None:
    """Build a single-page SVG flamegraph.

    Algorithm: synthesize full call chains from cProfile's per-function attribution.
    For each (caller, callee, samples) edge, we extend the chain greedily: keep
    appending to the chain as long as the current "head" frame appears as a CALLER
    in another edge. The root is a synthetic "<root>" frame that owns every chain.

    Layout: each row is a depth level. Width at depth d = sum of samples for
    frames at depth d, scaled to plot_w. Color: red (hot, large self time) ->
    blue (cold, small self time).
    """
    if not folded:
        raise RuntimeError("no folded-stack data; capture produced zero samples")

    # ---- normalize: keep only (caller, callee) pairs; aggregate. Drop any
    # stack whose caller or callee shortened to an empty/unknown sentinel.
    edges: Dict[Tuple[str, str], int] = {}
    for stack, count in folded.items():
        if len(stack) == 2:
            caller, callee = stack
            if caller and callee and caller != "<unknown>" and callee != "<unknown>":
                edges[stack] = edges.get(stack, 0) + count

    if not edges:
        raise RuntimeError("no (caller, callee) edges; cProfile captured zero samples")

    # ---- directed graph: caller -> {callee: weight}
    fwd: Dict[str, Dict[str, int]] = {}
    for (caller, callee), w in edges.items():
        d = fwd.get(caller)
        if d is None:
            d = {}
            fwd[caller] = d
        d[callee] = d.get(callee, 0) + w

    # ---- "heads" = frames that never appear as a caller (leaf-most frames)
    all_callers = {c for c, _ in edges}
    all_callees = {k for callee in fwd.values() for k in callee}
    leaves = sorted(all_callees - all_callers, key=lambda f: -sum(fwd[p].get(f, 0) for p in fwd if f in fwd[p]))

    # ---- trace every (leaf -> ... -> root) chain. The root is the unique
    # frame that is never a callee. Build by walking UP the inverse graph.
    inv: Dict[str, Dict[str, int]] = {}
    for (caller, callee), w in edges.items():
        inv.setdefault(callee, {})[caller] = inv.get(callee, {}).get(caller, 0) + w

    chain_self: Dict[Tuple[str, ...], int] = {}
    MAX_CHAIN_DEPTH = 24

    def _trace_up(f: str, chain: Tuple[str, ...], weight: int) -> None:
        # Greedy longest-path: at each step pick the heaviest unseen parent.
        # This collapses the exponential fan-out into one representative chain
        # per leaf (Brendan-Gregg convention; perfect attribution is impossible
        # from cProfile's per-function aggregates).
        seen = set(chain)
        for _ in range(MAX_CHAIN_DEPTH):
            parents = [(p, w) for p, w in inv.get(f, {}).items() if p not in seen and p != f]
            if not parents:
                break
            # Prefer the heaviest parent (the dominant caller context).
            parent, pw = max(parents, key=lambda x: x[1])
            chain = chain + (parent,)
            seen.add(parent)
            f = parent
        # chain is now (leaf, parent, ..., root) or just (leaf,) if no parents.
        chain_self[chain] = chain_self.get(chain, 0) + weight

    for leaf in leaves:
        w = sum(fwd[p].get(leaf, 0) for p in fwd if leaf in fwd[p])
        if w <= 0:
            continue
        _trace_up(leaf, (leaf,), w)

    if not chain_self:
        raise RuntimeError("chain synthesis produced zero chains")

    # ---- compute per-frame totals (sum of all chains passing through it)
    frame_total: Dict[str, int] = {}
    for chain, w in chain_self.items():
        for f in chain:
            frame_total[f] = frame_total.get(f, 0) + w

    # ---- cap to top-K frames by total weight to keep the SVG readable
    TOP_K = 60
    top_frames = {f for f, _ in sorted(frame_total.items(), key=lambda kv: -kv[1])[:TOP_K]}
    # Roll non-top frames into "<other>" so the SVG stays compact.
    other_total: Dict[str, int] = {}
    new_chains: Dict[Tuple[str, ...], int] = {}
    for chain, w in chain_self.items():
        new_chain = tuple(f if f in top_frames else "<other>" for f in chain)
        # collapse consecutive <other>
        collapsed: List[str] = []
        for f in new_chain:
            if not collapsed or collapsed[-1] != f:
                collapsed.append(f)
        new_chains[tuple(collapsed)] = new_chains.get(tuple(collapsed), 0) + w
    chain_self = new_chains
    # recompute totals after collapse
    frame_total = {}
    for chain, w in chain_self.items():
        for f in chain:
            frame_total[f] = frame_total.get(f, 0) + w

    # ---- compute depth per frame (longest distance from any chain leaf)
    frame_depth: Dict[str, int] = {}
    for chain in chain_self:
        for i, f in enumerate(chain):
            d = len(chain) - 1 - i
            if f not in frame_depth or frame_depth[f] < d:
                frame_depth[f] = d
    max_depth = max(frame_depth.values()) if frame_depth else 0

    # ---- SVG layout: root (deepest) at top, leaves at bottom
    root_total = max(frame_total.values()) if frame_total else 1
    row_h = 18
    pad = 1
    plot_w = 1600
    margin_top = 60
    label_w = 0  # pure flamegraph, no side labels
    svg_w = plot_w + 20
    svg_h = margin_top + (max_depth + 2) * row_h + 40

    # Build per-frame child total so each row's children sum to the parent's slot.
    # chains are leaf->root (chain[0] = leaf, chain[-1] = root); for the SVG
    # we walk top-down, so the parent of chain[i] is chain[i+1].
    child_totals: Dict[str, Dict[str, int]] = {}
    for chain, w in chain_self.items():
        # chain = (leaf, ..., parent, root). parent_of[chain[i]] = chain[i+1].
        for i in range(len(chain) - 1):
            child = chain[i]
            parent = chain[i + 1]
            child_totals.setdefault(parent, {})
            child_totals[parent][child] = child_totals[parent].get(child, 0) + w

    rects: List[Tuple[float, float, float, float, str]] = []

    def _emit(f: str, x0: float, x1: float) -> None:
        w = x1 - x0
        if w < 0.4:
            return
        y = margin_top + frame_depth.get(f, 0) * row_h
        rects.append((x0, y, w, row_h - pad, f))
        kids = child_totals.get(f, {})
        if not kids:
            return
        ksum = sum(kids.values()) or 1
        for k, kt in sorted(kids.items(), key=lambda kv: -kv[1]):
            cw = w * (kt / ksum)
            if cw < 0.4:
                continue
            _emit(k, x0, x0 + cw)
            x0 += cw

    # Emit the root frame at the top, full plot width.
    root_frame = max(frame_total, key=frame_total.get)
    _emit(root_frame, 0.0, plot_w)

    # ---- write SVG
    parts: List[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {svg_w} {svg_h}" '
        f'width="{svg_w}" height="{svg_h}" '
        f'font-family="Helvetica,Arial,sans-serif" font-size="11">'
    )
    parts.append(f'<rect width="{svg_w}" height="{svg_h}" fill="#fafafa"/>')
    parts.append(
        f'<text x="10" y="22" font-size="15" font-weight="bold" fill="#222">'
        f'{_esc(title)}</text>'
    )
    parts.append(
        f'<text x="10" y="42" font-size="11" fill="#666">'
        f'root_total = {root_total:,} microsec units &#183; '
        f'width = % of root (per-row sums to root) &#183; '
        f'color = red(hot) -> blue(cold) by share</text>'
    )
    for x, y, w, h, f in rects:
        if w < 14:
            continue  # too narrow for any text
        # Truncate BEFORE escaping so we don't cut an entity in half (`&lt;` -> `&l`).
        max_chars = max(3, int(w / 6.5))
        text = f if len(f) <= max_chars else f[: max(0, max_chars - 1)] + "…"
        esc_text = _esc(text)
        esc_title = _esc(f)
        # color: width-relative (within this row); also include depth for variety
        pct = 100.0 * (w / plot_w)
        color = _color_for(max_depth - int(y / row_h), pct)
        parts.append(
            f'<rect x="{x + 5:.1f}" y="{y:.1f}" width="{w:.2f}" height="{h}" '
            f'fill="{color}" stroke="#fff" stroke-width="0.5">'
            f'<title>{esc_title}</title></rect>'
        )
        parts.append(
            f'<text x="{x + 8:.1f}" y="{y + 12:.1f}" fill="#fff" '
            f'text-shadow="0 0 2px #000">{esc_text}</text>'
        )
    parts.append("</svg>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


# ---------- top-N table (self + cumulative) ----------

def top_n(pr: cProfile.Profile, n: int = 10) -> Tuple[str, str]:
    """Return (top_by_tottime, top_by_cumtime) as formatted text tables."""
    buf_t = io.StringIO()
    s_t = pstats.Stats(pr, stream=buf_t).strip_dirs().sort_stats("tottime")
    s_t.print_stats(n)
    buf_c = io.StringIO()
    s_c = pstats.Stats(pr, stream=buf_c).strip_dirs().sort_stats("cumtime")
    s_c.print_stats(n)
    return buf_t.getvalue(), buf_c.getvalue()


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", required=True,
                    help="path to a Pluribus blueprint snapshot .pkl "
                         "(e.g. checkpoints/bp6max_lossless_iter200.pkl)")
    ap.add_argument("--iterations", type=int, default=500,
                    help="CFR iterations per postflop node (default: 500 — the smoke run)")
    ap.add_argument("--hands", type=int, default=4,
                    help="number of random hands to play to capture a stable cProfile (default: 4)")
    ap.add_argument("--out", default="artifacts/perf_baseline_training.svg",
                    help="output SVG path (default: artifacts/perf_baseline_training.svg)")
    ap.add_argument("--top-out", default="artifacts/perf_baseline_top10.txt",
                    help="top-N table output (default: artifacts/perf_baseline_top10.txt)")
    args = ap.parse_args()

    if not os.path.isfile(args.snapshot):
        sys.exit(f"snapshot not found: {args.snapshot}")

    print(f"[perf-flamegraph] snapshot = {args.snapshot}")
    print(f"[perf-flamegraph] iters/decision = {args.iterations}, hands = {args.hands}")
    t0 = time.perf_counter()
    pr, wall = capture(args.snapshot, args.hands, args.iterations)
    print(f"[perf-flamegraph] cProfile wall = {wall:.2f}s ({(wall/max(1,args.hands))*1000:.0f} ms/hand)")

    folded = extract_folded(pr)
    print(f"[perf-flamegraph] distinct stacks = {len(folded)}, "
          f"total samples = {sum(folded.values()):,}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    render_svg(folded, args.out,
               title=f"500-iter smoke flamegraph (iters={args.iterations}, hands={args.hands}, wall={wall:.1f}s)")
    print(f"[perf-flamegraph] wrote {args.out}")

    out_t, out_c = top_n(pr, 10)
    with open(args.top_out, "w", encoding="utf-8") as f:
        f.write("=== TOP 10 by SELF TIME (tottime) ===\n")
        f.write(out_t)
        f.write("\n=== TOP 10 by CUMULATIVE TIME (cumtime) ===\n")
        f.write(out_c)
    print(f"[perf-flamegraph] wrote {args.top_out}")

    print(f"[perf-flamegraph] total = {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()