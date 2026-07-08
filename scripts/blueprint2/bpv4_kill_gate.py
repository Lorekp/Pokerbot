#!/usr/bin/env python3
"""Mid-run KILL GATE for a blueprint-v4 Stage-0 cloud FULLHIST arm (design v2 sec 6.3).

Schedule: run at ~20k AND ~30k iters (the bp3 lesson: the 30k check would have saved the run;
cost at risk ~$5 / ~$12). Downloads the latest bpv4 S3 ladder checkpoint (plus the immediately
preceding ladder point for the trend check), STREAM-extracts the canonical preflop nodes with the
FULLHIST key space (preflop_root_diag.py --full-hist node/key machinery, run memory-safe via
bpv4_stage0_gates.scan_checkpoint -- NEVER load_checkpoint, a fullhist blob is 30-55 GB), and
evaluates the pre-registered gates calibrated at the mid-run point:

  G1  monotonicity  : UTG <= HJ <= CO <= BTN open%, each step >= -0.5pt tolerance, BTN > UTG.
  G2  BTN open      : >= 20% (HARD at K1). At >= --trend-hard-iters (default 30k) the
                      TREND form is HARD too: decline prior->latest >= -3pt.
  G3  AA-BTN-limp   : <= 30% (HARD at K1). At >= --trend-hard-iters: non-rising prior->latest (HARD).
  G4  BB-vs-BTN-open fold : <= 70% (fullhist: weighted over the arm's own BTN open-size mix,
                      mirroring bpv4_stage0_gates.py).
  G6  jam sanity    : BTN RFI all-in share of open <= 10%.

TWO KILL REGIMES (amendment v3 sec 24.3, pre-registered):
  * K1a @<--k1-hard-iters (default 20000) -- the "EARLY KILL". HARD on ONE clause ONLY:
    AA-BTN-limp (w=t weighted avg) >= --k1a-aa-cap (default 0.50) => KILL. (arm-2b signature:
    61.8% @20k; an arm that will fail G3 shows it by 10k, so this halves the spend on a dead arm.)
    EVERY OTHER read at this regime -- G1, G2-level, G4, G6, and the 0.30 G3-level cap -- is
    ADVISORY (printed WARN-only, NEVER folded into the verdict). This is deliberate: the arm-3
    widened menu is 2-3x slower / sampling-limited, so it is mid-convergence at 10k and a hard
    level cap would SPURIOUSLY KILL a converging run.
  * K1 @>=--k1-hard-iters (default 20000) -- the BINDING all-clause gate. AA-limp <= 0.30 AND
    G1/G2-level/G4/G6 all HARD (arm-2b registration verbatim + menu extension). This is unchanged
    from the historical instrument.
The --trend-hard-iters knob (default 30k) is ORTHOGONAL: it relaxes ONLY the G2/G3 TREND clauses
below it (the E5-class positive controls slide early, so an early hard trend-kill would kill a
validated-good arm -- same discipline as check_prefhist_kill_gates.py), and never touches a level cap.

NEGATIVE CONTROL (bp3 kill-gate pattern): the OPPOSITE (summary) key space must read 0 rows from a
fullhist checkpoint -- a nonzero count means the extraction key space is wrong / the run is not
training fullhist, so the gate output cannot be trusted: ABORT (rc 2), do not emit a verdict.

PRE-MENU CONSISTENCY GUARD (fail-closed, same rc 2 as the negative control): the negative control
only checks the KEY SPACE, never the raise MENU. A DEFAULT-menu scan of an arm-3 (widened-menu) blob
silently drops the id-5/6 open mass from both the renorm denominator and the opens numerator --
inflating AA-limp (a true 0.25 reads 0.357 -> false KILL) and understating opens (G1/G2 false-fail),
with neg_control still 0 and nothing aborting. So the guard reads the ACTUAL menu straight off the
loaded blob (id-5/6 open-bucket mass at rfi_BTN, and whether the vsopen_BB_200 / vsopen_BB_212 arm-3
nodes carry trained rows) and asserts it agrees with --pre-menu: an arm-3 blob under --pre-menu
default, or --pre-menu arm3 over a blob with no arm-3 rows, ABORTS (rc 2) with a loud error and no
verdict.

Prints PASS/KILL with every number. KILL => terminate the box + stop the babysitter:
  aws ec2 terminate-instances --instance-ids <iid> --region eu-central-1
Field strength is still decided by the FINAL Stage-0 gates (bpv4_stage0_gates.py @30k/40k) + the G5
probe; this gate only prevents paying a fullhist arm to the backstop once it already reproduces the
limp-trap pathology.

Usage (default = latest bpv4_h200 ladder point on S3):
  python scripts/blueprint2/bpv4_kill_gate.py
  python scripts/blueprint2/bpv4_kill_gate.py --iter 20000            # pin a ladder point
  python scripts/blueprint2/bpv4_kill_gate.py --run-subdir bpv4_h1000
  python scripts/blueprint2/bpv4_kill_gate.py --checkpoint latest.nlcfr --prior-checkpoint prev.nlcfr
  python scripts/blueprint2/bpv4_kill_gate.py --iter 20000 --no-prior # skip the trend-baseline download

ROBUST BIG-PULL ROUTE (recommended on this box -- a fullhist checkpoint is 30-55 GB, and python's
urllib3 cannot verify the SSL-inspecting proxy while the AWS CLI can): download with the CLI (it has
a live progress bar + a working cert bundle), then point the gate at the local file:
  aws s3 cp s3://video-processing-6789/pokerbot/prod6000/native_cfr/bpv4_h200/checkpoints/iter_020000.nlcfr \
      artifacts/bpv4_ladder/bpv4_h200_iter20000.nlcfr --region eu-central-1
  python scripts/blueprint2/bpv4_kill_gate.py --checkpoint artifacts/bpv4_ladder/bpv4_h200_iter20000.nlcfr

PRE-MENU (arm-3): for an arm trained with the WIDENED preflop raise menu (engine ids 5/6/7/8/9 =
0.66/0.75/1.0/1.5/2.0 pot, trainer env POKERBOT_PRE_KEEP="2,3,4,5,6"), pass --pre-menu arm3 (or
env POKERBOT_PRE_MENU=arm3). Ids 5/6 then count as opens in G1/G2/G6, the two BB-response nodes
vs the NEW BTN open sizes (vsopen_BB_200 / vsopen_BB_212) are added, and the G4 weighting spans
all 5 sizes (weighted by the arm's own BTN open-size mix, identical construction). DEFAULT IS
BYTE-IDENTICAL to the pre-arm-3 instrument; off-default runs stamp "pre_menu" +
"open_raise_ids" into params so evidence files are self-describing (absence == default 3-size).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import threading
import time
import warnings

# This box reaches S3 through an SSL-inspecting proxy: python's urllib3 cannot verify the re-signed
# chain, so boto3 is used with verify=False (the repo convention -- see poll_dcfr_curve.py /
# poll_hogthru.py) and the resulting per-request InsecureRequestWarning is silenced (a 51 GB download
# otherwise emits dozens of them). The AWS CLI ships its own working CA bundle, so `aws s3 cp` stays
# fully TLS-verified -- prefer it for the big pull (see the ROBUST BIG-PULL ROUTE in the docstring).
try:
    import urllib3  # noqa: E402

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:  # pragma: no cover - urllib3 is always present via botocore
    warnings.filterwarnings("ignore", message="Unverified HTTPS request")

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))

# The FULLHIST node/key machinery (preflop_root_diag --full-hist) + the memory-safe streaming
# extractor, reused verbatim from the final Stage-0 gate evaluator (single source of truth).
import bpv4_stage0_gates as g0  # noqa: E402
import bpv4_kcov as kcov  # noqa: E402
import preflop_root_diag as prd  # noqa: E402

DEFAULT_BUCKET = "video-processing-6789"
DEFAULT_PREFIX = "pokerbot/prod6000/native_cfr"
DEFAULT_RUN = "bpv4_h200"
DEFAULT_REGION = "eu-central-1"

DETAIL_HANDS = ("AA", "KK", "QQ", "AKs", "K9s", "72o")


def _list_ladder(s3, bucket: str, run_prefix: str) -> list[tuple[int, str]]:
    """Sorted [(iters, key)] for the versioned iter_XXXXXX.nlcfr ladder checkpoints."""
    out: list[tuple[int, str]] = []
    token = None
    while True:
        kw = {"Bucket": bucket, "Prefix": f"{run_prefix}/checkpoints/"}
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        for obj in resp.get("Contents", []):
            m = re.search(r"iter_(\d+)\.nlcfr$", obj["Key"])
            if m:
                out.append((int(m.group(1)), obj["Key"]))
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return sorted(out)


class _Progress:
    """Thread-safe, throttled S3 download progress line (s3transfer calls it from N worker threads)."""

    def __init__(self, total: int, every_s: float = 2.0):
        self._total = max(1, int(total))
        self._done = 0
        self._every = every_s
        self._t0 = time.time()
        self._last = 0.0
        self._lock = threading.Lock()

    def __call__(self, n: int) -> None:
        with self._lock:
            self._done += n
            now = time.time()
            if now - self._last < self._every and self._done < self._total:
                return
            self._last = now
            el = max(1e-6, now - self._t0)
            rate = self._done / el
            pct = 100.0 * self._done / self._total
            eta = (self._total - self._done) / rate if rate > 0 else 0.0
            sys.stdout.write(
                f"\r[kill-gate]   {self._done / 1e9:6.2f}/{self._total / 1e9:.2f} GB "
                f"({pct:5.1f}%)  {rate / 1e6:6.1f} MB/s  ETA {eta / 60:5.1f} min   ")
            sys.stdout.flush()


def _download(s3, bucket: str, key: str, dst: str) -> str:
    """Download one checkpoint with a live progress line; size-verified cache reuse.

    Writes to a .part temp then atomically renames, so an interrupted download never masquerades as a
    complete cache hit (the old check trusted any existing file, so a Ctrl-C mid-pull left a corrupt
    'cache' that the next run would have scanned)."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    total = int(s3.head_object(Bucket=bucket, Key=key)["ContentLength"])
    if os.path.exists(dst) and os.path.getsize(dst) == total:
        print(f"[kill-gate] reusing cached {dst} ({total / 1e9:.1f} GB)", flush=True)
        return dst
    free = shutil.disk_usage(os.path.dirname(dst) or ".").free
    if free < total * 1.1:
        print(f"[kill-gate] WARNING: only {free / 1e9:.1f} GB free on the cache drive for a "
              f"{total / 1e9:.1f} GB checkpoint -- the download may fail; free space or pass "
              f"--cache-dir on a bigger drive.", flush=True)
    tmp = dst + ".part"
    print(f"[kill-gate] downloading s3://{bucket}/{key} -> {dst} ({total / 1e9:.1f} GB)", flush=True)
    s3.download_file(bucket, key, tmp, Callback=_Progress(total))
    os.replace(tmp, dst)
    print(f"\n[kill-gate] downloaded {dst}", flush=True)
    return dst


# Engine action ids of the arm-3-only open sizes (0.66/0.75 pot -> RFI 2.0/2.12bb). Used ONLY by the
# menu-consistency guard to read the ACTUAL trained menu off the blob, independent of --pre-menu.
ARM3_OPEN_IDS = (5, 6)


def _arm3_probe_keys() -> dict:
    """Build the arm-3 menu-detection probe keys, INDEPENDENT of the active --pre-menu setting.

    Returns {'rfi_BTN': <key>, 'vsopen_BB_200': <key>, 'vsopen_BB_212': <key>}. The rfi_BTN fullhist
    key is menu-invariant (it depends on pos/class/betting6/action-history, not the legal action set),
    so its raw row exposes whether the trainer accumulated id-5/6 open mass. The two vsopen_BB_2xx
    keys exist ONLY in the arm-3 node list (BTN opens 2.0/2.12bb) -- their presence in the blob is a
    direct arm-3 fingerprint. Restores prd globals to fullhist on the way out."""
    prev = prd.PRE_MENU
    prd.set_pre_menu("arm3")
    nl = g0.build_nodes("fullhist")  # arm-3 fullhist node list (includes vsopen_BB_200/212)
    ks = g0.node_keys(nl, "fullhist")
    want = ("rfi_BTN", "vsopen_BB_200", "vsopen_BB_212")
    # class 0 ("22") key per node -- any class row suffices to detect a trained node; we scan all 169
    # classes at rfi_BTN so a nonzero id-5/6 mass on ANY class trips the guard.
    probe = {"rfi_BTN": [ks[("rfi_BTN", c)] for c in range(169)],
             "vsopen_BB_200": [ks[("vsopen_BB_200", c)] for c in range(169)],
             "vsopen_BB_212": [ks[("vsopen_BB_212", c)] for c in range(169)]}
    prd.set_pre_menu(prev)
    g0.build_nodes("fullhist")  # restore prd globals to the (real) fullhist mode after the probe build
    return probe


def _extract(path: str, node_list, keys, neg_keys, detail_cls, arm3_probe=None):
    """Memory-safe streaming extraction of the fullhist nodes off one checkpoint.

    Returns (iters, nodes_dict, m1_dict, neg_present, menu). NEVER loads the full table into a dict --
    scan_checkpoint mmaps the blob and keeps only the ~few-thousand wanted node keys. `menu` is the
    arm-3-menu-consistency read from arm3_probe (or None if no probe requested)."""
    probe_keys = set()
    if arm3_probe:
        for lst in arm3_probe.values():
            probe_keys |= set(lst)
    wanted = set(keys.values()) | set(neg_keys.values()) | probe_keys
    it, n_cum, n_strat, cum_hits, strat_hits = g0.scan_checkpoint(path, wanted)
    nodes = g0.extract_nodes(node_list, keys, cum_hits, strat_hits, detail_cls)
    neg_present = sum(1 for k in set(neg_keys.values()) if k in strat_hits or k in cum_hits)
    menu = _menu_read(arm3_probe, cum_hits, strat_hits) if arm3_probe else None
    m1 = {"iters": it, "cum_rows": n_cum, "strat_rows": n_strat,
          "blob_bytes": os.path.getsize(path)}
    return it, nodes, m1, neg_present, menu


def _menu_read(arm3_probe, cum_hits, strat_hits) -> dict:
    """Read the ACTUAL trained raise menu off the blob via the arm-3 probe keys.

    Returns {'btn_id56_mass': float, 'bb200_rows': int, 'bb212_rows': int, 'arm3_present': bool}.
    btn_id56_mass = the largest id-5/6 strat_sum mass seen on any class at rfi_BTN (0.0 == the trainer
    never put mass on the arm-3 open sizes)."""
    id56_mass = 0.0
    for k in arm3_probe["rfi_BTN"]:
        row = strat_hits.get(k)
        if row is not None:
            id56_mass = max(id56_mass, sum(row[a] for a in ARM3_OPEN_IDS))
    bb200 = sum(1 for k in arm3_probe["vsopen_BB_200"] if k in strat_hits or k in cum_hits)
    bb212 = sum(1 for k in arm3_probe["vsopen_BB_212"] if k in strat_hits or k in cum_hits)
    return {"btn_id56_mass": id56_mass, "bb200_rows": bb200, "bb212_rows": bb212,
            "arm3_present": id56_mass > 0.0 or bb200 > 0 or bb212 > 0}


def _btn_open(nodes) -> float | None:
    return g0.open_pct(nodes["rfi_BTN"])  # already *100


def _aa_limp(nodes) -> float | None:
    aa = nodes["rfi_BTN"]["detail"].get("AA", {}).get("avg", {})
    return aa.get("CALL")


def _bb_fold_weighted(nodes) -> tuple[float | None, dict]:
    """BB fold vs BTN open, weighted by the arm's own BTN open-size mix (mirrors bpv4_stage0_gates;
    under --pre-menu arm3 the g0.bb_size_weights() table spans all 5 open sizes)."""
    btn = nodes["rfi_BTN"]["avg"]
    mix = btn["by_action"] if btn else {}
    w = {node: mix.get(lbl, 0.0) for node, lbl in g0.bb_size_weights()}
    per_size = {}
    acc = used = 0.0
    for n, wt in w.items():
        a = nodes.get(n, {}).get("avg")
        if a is not None:
            per_size[n] = a["fold"]
            acc += wt * a["fold"]
            used += wt
    return (acc / used if used > 0 else None), per_size


def _jam_share(nodes) -> float | None:
    a = nodes["rfi_BTN"]["avg"]
    if a and a["open_or_raise"] > 0:
        return a["allin"] / a["open_or_raise"]
    return None


def evaluate(latest, prior, *, mono_tol_pt: float, btn_floor_pct: float, aa_cap: float,
             bb_fold_cap: float, jam_cap: float, decline_floor_pt: float,
             trend_hard: bool, k1a_regime: bool, k1a_aa_cap: float
             ) -> tuple[bool, list[str], dict]:
    """Evaluate the mid-run gates.

    Two regimes (amendment v3 sec 24.3):
      * k1a_regime=True  (latest_it < --k1-hard-iters): the ONLY hard kill clause is
        AA-BTN-limp level >= k1a_aa_cap (default 0.50). G1/G2-level/G3-at-aa_cap/G4/G6 are all
        computed and PRINTED as ADVISORY (WARN-only) but NEVER folded into `ok`. This mirrors how
        `trend_hard` conditionally softens the G2/G3 trend clauses -- here the SOFTENED set is the
        level caps, because the widened arm-3 menu is sampling-limited mid-run and a hard level cap
        would spuriously kill a converging run.
      * k1a_regime=False (latest_it >= --k1-hard-iters): the binding K1 gate. AA-limp <= aa_cap
        (0.30) AND G1/G2-level/G4/G6 all HARD -- unchanged historical behavior.
    """
    lines: list[str] = []
    detail: dict = {}
    ok = True
    detail["k1a_regime"] = k1a_regime
    # In the K1a regime a level clause is advisory (WARN-only, never folded into `ok`); in K1 it is
    # hard. `_fold(cond)` folds `cond` into `ok` only when hard; `_tag()` labels the printed line.
    lvl_hard = not k1a_regime

    def _fold(passed: bool) -> None:
        nonlocal ok
        if lvl_hard:
            ok &= passed

    def _tag(passed: bool) -> str:
        if lvl_hard:
            return "PASS" if passed else "FAIL"
        return "PASS" if passed else "WARN (advisory @K1a -- not a kill clause)"

    rfi = ["rfi_UTG", "rfi_HJ", "rfi_CO", "rfi_BTN"]

    opens = {n: g0.open_pct(latest[n]) for n in rfi}
    detail["opens_pct"] = opens
    lines.append("opens (avg open_or_raise): " + "  ".join(
        f"{n.split('_')[1]}={'NA' if opens[n] is None else f'{opens[n]:.1f}%'}" for n in rfi))

    # G1 monotonicity (level clause: hard @K1, advisory @K1a).
    if any(opens[n] is None for n in rfi):
        g1 = False
        lines.append("G1 monotonicity: " + _tag(g1) + " (an RFI node read 0 rows -- checkpoint too "
                     "early or wrong key space; verify the run exports POKERBOT_FULLHIST_KEY=1)")
    else:
        seq = [opens[n] for n in rfi]
        steps_ok = all(seq[i + 1] - seq[i] >= -mono_tol_pt for i in range(3))
        g1 = steps_ok and seq[3] > seq[0]
        lines.append(f"G1 monotonicity (each step >= -{mono_tol_pt:.1f}pt, BTN>UTG): {_tag(g1)}")
    _fold(g1)
    detail["G1"] = g1

    # G2 BTN open floor (level clause: hard @K1, advisory @K1a) + trend (relaxed by trend_hard).
    btn = opens["rfi_BTN"]
    btn_prev = g0.open_pct(prior["rfi_BTN"]) if prior else None
    decline = None if (btn is None or btn_prev is None) else btn - btn_prev
    g2_level = btn is not None and btn >= btn_floor_pct
    g2_trend_ok = decline is None or decline >= decline_floor_pt
    # The trend clause never fires @K1a (level advisory there); at K1 it is hard iff trend_hard.
    g2 = g2_level and (g2_trend_ok or not trend_hard)
    _fold(g2)
    tnote = ("no prior ladder point" if btn_prev is None else
             f"prior {btn_prev:.1f}% -> latest {btn:.1f}% (decline {decline:+.1f}pt)")
    trend_status = ("HARD" if trend_hard else "advisory") + (
        "" if g2_trend_ok else " -- TREND BREACH"
        + ("" if trend_hard else " (WARN, advisory below trend-hard-iters)"))
    lines.append(f"G2 BTN open {'NA' if btn is None else f'{btn:.1f}%'} >= {btn_floor_pct:.0f}% "
                 f"AND decline >= {decline_floor_pt:+.0f}pt [{trend_status}]: "
                 f"{_tag(g2)}  [{tnote}]")
    detail["G2"] = {"btn_open_pct": btn, "btn_open_prev_pct": btn_prev, "decline_pt": decline,
                    "level_ok": g2_level, "trend_ok": g2_trend_ok, "trend_hard": trend_hard,
                    "level_hard": lvl_hard, "pass": g2}

    # G3 AA-BTN-limp. THE kill clause. @K1a: HARD at k1a_aa_cap (0.50), the ONLY hard clause. @K1:
    # HARD at aa_cap (0.30) + non-rising trend (relaxed by trend_hard).
    g3_cap = k1a_aa_cap if k1a_regime else aa_cap
    aa = _aa_limp(latest)
    aa_prev = _aa_limp(prior) if prior else None
    if aa is None:
        g3 = False
        lines.append(f"G3 AA-BTN-limp: FAIL (AA row missing at rfi_BTN -- checkpoint too early / "
                     f"wrong key space) [{'K1a' if k1a_regime else 'K1'} HARD cap {g3_cap*100:.0f}%]")
    else:
        g3_level = aa <= g3_cap
        g3_trend_ok = aa_prev is None or aa <= aa_prev + 1e-9  # non-rising
        # @K1a the trend clause is NOT applied (only the level cap is the kill); @K1 trend is hard
        # iff trend_hard.
        g3 = g3_level and (k1a_regime or g3_trend_ok or not trend_hard)
        tnote3 = ("no prior ladder point" if aa_prev is None else
                  f"prior {aa_prev*100:.1f}% -> latest {aa*100:.1f}% "
                  f"({'non-rising' if g3_trend_ok else 'RISING'})")
        if k1a_regime:
            ts3 = "K1a EARLY-KILL (HARD, sole kill clause)"
        else:
            ts3 = ("HARD" if trend_hard else "advisory") + (
                "" if g3_trend_ok else " -- RISING"
                + ("" if trend_hard else " (WARN, advisory)"))
        lines.append(f"G3 AA-BTN-limp {aa*100:.1f}% <= {g3_cap*100:.0f}%"
                     + ("" if k1a_regime else " AND non-rising")
                     + f" [{ts3}]: {'PASS' if g3 else 'KILL' if k1a_regime else 'FAIL'}  [{tnote3}]")
        detail["G3"] = {"aa_limp": aa, "aa_limp_prev": aa_prev, "cap": g3_cap,
                        "level_ok": g3_level, "trend_ok": g3_trend_ok, "trend_hard": trend_hard,
                        "k1a_regime": k1a_regime, "pass": g3}
    # G3 is ALWAYS hard (@K1a it is the sole kill clause; @K1 it is a binding cap).
    ok &= g3

    # G4 BB fold vs BTN open (weighted; level clause: hard @K1, advisory @K1a).
    bb_fold, per_size = _bb_fold_weighted(latest)
    g4 = bb_fold is not None and bb_fold <= bb_fold_cap
    _fold(g4)
    lines.append(f"G4 BB-vs-BTN-open fold {'NA' if bb_fold is None else f'{bb_fold*100:.1f}%'} "
                 f"<= {bb_fold_cap*100:.0f}% (open-size-weighted): {_tag(g4)}  "
                 f"[per-size folds: {{ {', '.join(f'{k}={v*100:.1f}%' for k, v in per_size.items())} }}]")
    detail["G4"] = {"bb_fold_weighted": bb_fold, "per_size": per_size, "level_hard": lvl_hard,
                    "pass": g4}

    # G6 jam sanity (level clause: hard @K1, advisory @K1a).
    js = _jam_share(latest)
    g6 = js is not None and js <= jam_cap
    _fold(g6)
    lines.append(f"G6 BTN jam share of open {'NA' if js is None else f'{js*100:.1f}%'} "
                 f"<= {jam_cap*100:.0f}%: {_tag(g6)}")
    detail["G6"] = {"jam_share": js, "level_hard": lvl_hard, "pass": g6}

    return ok, lines, detail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    ap.add_argument("--run-subdir", default=DEFAULT_RUN,
                    help="S3 run subdir (bpv4_h200 | bpv4_h1000 | bpv4_hd200 | bpv4_he200)")
    ap.add_argument("--region", default=DEFAULT_REGION)
    ap.add_argument("--iter", type=int, default=None,
                    help="pin a ladder point (default: the latest iter_XXXXXX.nlcfr)")
    ap.add_argument("--checkpoint", default=None, help="local .nlcfr override (skips S3)")
    ap.add_argument("--prior-checkpoint", default=None,
                    help="local prior-ladder-point .nlcfr for the G2/G3 trend check")
    ap.add_argument("--no-prior", action="store_true",
                    help="skip downloading the preceding ladder point (the G2/G3 trend baseline). "
                         "At the 20k check the trend is advisory anyway, so this saves a ~28-55 GB "
                         "download; the hard caps G1/G2-level/G3-level/G4/G6 still fully evaluate.")
    ap.add_argument("--mono-tol-pt", type=float, default=0.5)
    ap.add_argument("--btn-floor-pct", type=float, default=20.0)
    ap.add_argument("--aa-limp-cap", type=float, default=0.30,
                    help="K1 (>= --k1-hard-iters) hard AA-BTN-limp cap (design 24.3: 0.30 @20k)")
    ap.add_argument("--k1a-aa-cap", type=float, default=None,
                    help="K1a (< --k1-hard-iters) EARLY-KILL AA-BTN-limp cap -- the SOLE hard clause "
                         "below --k1-hard-iters. Default resolves to 0.50 for a fullhist arm (design "
                         "24.3: >=50%% @10k => KILL; arm-2b signature was 61.8%%) or 0.40 under "
                         "--presum (design 36: presum should FIX the limp-trap, so AA-limp <= 40%% "
                         "@10k is the pre-registered target). All other reads below --k1-hard-iters "
                         "are advisory.")
    ap.add_argument("--k1-hard-iters", type=int, default=20000,
                    help="the K1a->K1 boundary (design 24.3). BELOW it (K1a regime) the ONLY hard "
                         "kill clause is AA-limp >= --k1a-aa-cap; G1/G2-level/G4/G6/G3-at-0.30 are "
                         "advisory. AT/ABOVE it (K1) the 0.30 AA cap + G1/G2/G4/G6 are all hard.")
    ap.add_argument("--bb-fold-cap", type=float, default=0.70)
    ap.add_argument("--jam-cap", type=float, default=0.10)
    ap.add_argument("--decline-floor-pt", type=float, default=-3.0,
                    help="G2 trend: BTN open decline prior->latest must be >= this (pt)")
    ap.add_argument("--trend-hard-iters", type=int, default=30000,
                    help="G2/G3 trend clauses are ADVISORY below this and HARD from it (design 6.3: "
                         "trend forms added AT the 30k check; early positive-control arms slide)")
    ap.add_argument("--pre-menu", choices=("default", "arm3"),
                    default=os.environ.get("POKERBOT_PRE_MENU", "default"),
                    help="preflop raise menu of the arm's trainer: 'default' = 3-size ids 7/8/9 "
                         "(byte-identical historical instrument) | 'arm3' = widened 5-size ids "
                         "5/6/7/8/9 (POKERBOT_PRE_KEEP=2..6). Env POKERBOT_PRE_MENU sets the "
                         "default.")
    ap.add_argument("--presum", action="store_true",
                    help="blueprint-v4 arm-6 presum run (design 36): (a) resolves the K1a AA-limp cap "
                         "to 0.40 (presum should FIX the limp-trap), and (b) ACTIVATES the K-COV "
                         "gate (opener-faced raised-pot on-policy coverage, from the train log's "
                         "COVERAGE line) as a verdict clause. Presum's PREFLOP nodes are fullhist-keyed "
                         "(presum only rewrites POSTFLOP keys), so G1-G6 + the node/negative-control "
                         "machinery are unchanged. WITHOUT --presum the K-COV read is advisory-only "
                         "(printed, never folded), so the historical fullhist verdict is unchanged.")
    ap.add_argument("--train-log", default=None,
                    help="local train log for the K-COV COVERAGE lines (default: fetch "
                         "s3://.../{run_subdir}/logs/train.log when going through S3).")
    ap.add_argument("--kcov-cap", type=float, default=kcov.CAP_PCT,
                    help="K1 K-COV: raised-pot postflop MISS%% must be <= this (coverage >= 100-cap).")
    ap.add_argument("--kcov-early-kill", type=float, default=kcov.EARLY_KILL_PCT,
                    help="K1a K-COV EARLY-KILL: raised-pot postflop MISS%% > this => KILL.")
    ap.add_argument("--kcov-min-sample", type=int, default=kcov.MIN_SAMPLE,
                    help="raised_decisions below this => K-COV INSUFFICIENT_DATA (advisory, never folds).")
    ap.add_argument("--k1a-drift-pt", type=float, default=kcov.DRIFT_PT,
                    help="advisory leading indicator: AA-limp and raised-pot miss should each fall >= "
                         "this many pt vs the prior checkpoint (design 36: >=5pt 7.5k->10k).")
    ap.add_argument("--cache-dir", default=os.path.join(REPO, "artifacts", "bpv4_ladder"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    # Resolve the K1a AA cap default per --presum (kept None above so the fullhist path stays 0.50).
    if args.k1a_aa_cap is None:
        args.k1a_aa_cap = 0.40 if args.presum else 0.50
    out = args.out or os.path.join(REPO, "artifacts", f"bpv4_kill_gate_{args.run_subdir}.json")
    prd.set_pre_menu(args.pre_menu)

    # --- build the fullhist node list + keys ONCE (engine-replayed, exact) -------------------------
    # The node list / keys / weighting table all follow the SELECTED --pre-menu (set above). The
    # arm-3 menu-detection probe keys are built INDEPENDENTLY (always in arm3 mode) so the guard can
    # read the blob's real menu regardless of the flag.
    node_list = g0.build_nodes("fullhist")
    keys = g0.node_keys(node_list, "fullhist")
    neg_name, neg_nodes = g0.wrong_space_nodes("fullhist")
    neg_keys = g0.node_keys(neg_nodes, neg_name)
    g0.build_nodes("fullhist")  # restore prd globals to the real mode after neg-space build
    arm3_probe = _arm3_probe_keys()  # also restores prd globals to fullhist + the selected menu
    prd.set_pre_menu(args.pre_menu)  # re-assert the selected menu (probe build touched RAISE_SET)
    g0.build_nodes("fullhist")
    detail_cls = {prd.CLASS_BY_NAME[h] for h in DETAIL_HANDS}

    latest_path, prior_path = args.checkpoint, args.prior_checkpoint
    train_log_path = args.train_log
    latest_iters = prior_iters = None
    if latest_path is None:
        import boto3  # noqa: PLC0415
        s3 = boto3.client("s3", region_name=args.region, verify=False)
        run_prefix = f"{args.prefix}/{args.run_subdir}"
        ladder = _list_ladder(s3, args.bucket, run_prefix)
        if not ladder:
            print(f"ERROR: no iter_XXXXXX.nlcfr ladder under "
                  f"s3://{args.bucket}/{run_prefix}/checkpoints/ (run not started / too early?)")
            return 2
        print(f"[kill-gate] ladder: {[i for i, _ in ladder]}")
        if args.iter is not None:
            match = [(i, k) for i, k in ladder if i == args.iter]
            if not match:
                print(f"ERROR: iter {args.iter} not in ladder {[i for i, _ in ladder]}")
                return 2
            idx = ladder.index(match[0])
        else:
            idx = len(ladder) - 1
        latest_iters, latest_key = ladder[idx]
        latest_path = _download(s3, args.bucket, latest_key,
                                os.path.join(args.cache_dir, f"{args.run_subdir}_iter{latest_iters}.nlcfr"))
        if idx > 0 and prior_path is None and not args.no_prior:
            # Trend baseline = the immediately preceding ladder point (checkpoint cadence is 10k, so
            # adjacent points already carry a trend signal, unlike the bp3 500-iter cadence).
            prior_iters, prior_key = ladder[idx - 1]
            prior_path = _download(s3, args.bucket, prior_key,
                                   os.path.join(args.cache_dir, f"{args.run_subdir}_iter{prior_iters}.nlcfr"))
        # K-COV source: the tee'd train log (the driver uploads it at every checkpoint, key
        # {run_prefix}/logs/train.log). Tiny; a miss (run too early / no COVERAGE probing) is fine --
        # K-COV then reads INSUFFICIENT_DATA (advisory). Never aborts the gate.
        if train_log_path is None:
            tl_dst = os.path.join(args.cache_dir, f"{args.run_subdir}_train.log")
            try:
                s3.download_file(args.bucket, f"{run_prefix}/logs/train.log", tl_dst)
                train_log_path = tl_dst
                print(f"[kill-gate] fetched train log -> {tl_dst}", flush=True)
            except Exception as e:  # noqa: BLE001 - absence is expected early; K-COV degrades gracefully
                print(f"[kill-gate] no train log yet ({e!r}) -- K-COV will read INSUFFICIENT_DATA", flush=True)

    os.makedirs(args.cache_dir, exist_ok=True)

    # --- extract (streaming, memory-safe) ----------------------------------------------------------
    latest_it, latest_nodes, latest_m1, latest_neg, latest_menu = _extract(
        latest_path, node_list, keys, neg_keys, detail_cls, arm3_probe=arm3_probe)
    print(f"[kill-gate] latest iter {latest_it}: strat_rows={latest_m1['strat_rows']:,} "
          f"cum_rows={latest_m1['cum_rows']:,} blob={latest_m1['blob_bytes']/1e9:.1f}GB "
          f"neg_ctrl_rows={latest_neg}  arm3_menu_present={latest_menu['arm3_present']} "
          f"(btn_id56_mass={latest_menu['btn_id56_mass']:.3g} bb200_rows={latest_menu['bb200_rows']} "
          f"bb212_rows={latest_menu['bb212_rows']})", flush=True)
    prior_nodes = None
    prior_m1 = None
    if prior_path:
        prior_it, prior_nodes, prior_m1, prior_neg, _prior_menu = _extract(
            prior_path, node_list, keys, neg_keys, detail_cls)
        print(f"[kill-gate] prior  iter {prior_it}: strat_rows={prior_m1['strat_rows']:,} "
              f"neg_ctrl_rows={prior_neg}", flush=True)
        if prior_neg != 0:
            print(f"ABORT: prior checkpoint negative control read {prior_neg} rows under the "
                  f"{neg_name} key space -- extraction key space is wrong; not emitting a verdict.")
            return 2

    if latest_neg != 0:
        print(f"ABORT: negative control read {latest_neg} rows under the {neg_name} key space -- "
              f"the checkpoint is NOT in the fullhist key space (run not training fullhist, or a "
              f"key/extractor bug). The gate numbers cannot be trusted; not emitting a verdict.")
        return 2

    # --- PRE-MENU CONSISTENCY GUARD (fail-closed, rc 2 -- mirrors the negative-control abort) -------
    # The blob's REAL menu (read via the arm-3 probe, independent of --pre-menu) MUST agree with the
    # selected --pre-menu. A default-menu scan of an arm-3 blob drops the id-5/6 open mass and biases
    # every gate toward a false KILL; the negative control (key-space only) never catches this.
    blob_is_arm3 = latest_menu["arm3_present"]
    flag_is_arm3 = args.pre_menu == "arm3"
    if blob_is_arm3 != flag_is_arm3:
        if blob_is_arm3 and not flag_is_arm3:
            print(
                "ABORT (pre-menu mismatch): the checkpoint carries ARM-3 widened-menu rows "
                f"(btn_id56_mass={latest_menu['btn_id56_mass']:.4g}, "
                f"vsopen_BB_200 rows={latest_menu['bb200_rows']}, "
                f"vsopen_BB_212 rows={latest_menu['bb212_rows']}) but --pre-menu is 'default'. A "
                "default-menu scan DROPS the id-5/6 open mass -> AA-limp inflates and opens deflate "
                "-> the gate would FALSELY KILL a healthy arm-3 run. Re-run with --pre-menu arm3 "
                "(or env POKERBOT_PRE_MENU=arm3). Not emitting a verdict.")
        else:
            print(
                "ABORT (pre-menu mismatch): --pre-menu is 'arm3' but the checkpoint shows NO arm-3 "
                f"rows (btn_id56_mass={latest_menu['btn_id56_mass']:.4g}, "
                f"vsopen_BB_200 rows={latest_menu['bb200_rows']}, "
                f"vsopen_BB_212 rows={latest_menu['bb212_rows']} -- all zero). This blob was NOT "
                "trained with the widened menu (POKERBOT_PRE_KEEP=2,3,4,5,6); the arm-3 node set / "
                "G4 weighting do not apply. Re-run with --pre-menu default. Not emitting a verdict.")
        return 2

    trend_hard = int(latest_it) >= args.trend_hard_iters
    k1a_regime = int(latest_it) < args.k1_hard_iters
    regime = "K1a (EARLY-KILL: sole hard clause AA-limp)" if k1a_regime else "K1 (BINDING all-clause)"
    aa_cap_applied = args.k1a_aa_cap if k1a_regime else args.aa_limp_cap
    print(f"\n=== bpv4 {args.run_subdir} MID-RUN KILL GATE @ iters={latest_it}  [{regime}, "
          f"AA cap {aa_cap_applied*100:.0f}%]"
          + (f" (trend vs {prior_m1['iters']}, {'HARD' if trend_hard else 'advisory'})"
             if prior_m1 else "") + " ===")
    ok, lines, gdetail = evaluate(
        latest_nodes, prior_nodes, mono_tol_pt=args.mono_tol_pt, btn_floor_pct=args.btn_floor_pct,
        aa_cap=args.aa_limp_cap, bb_fold_cap=args.bb_fold_cap, jam_cap=args.jam_cap,
        decline_floor_pt=args.decline_floor_pt, trend_hard=trend_hard,
        k1a_regime=k1a_regime, k1a_aa_cap=args.k1a_aa_cap)
    for ln in lines:
        print("  " + ln)
    print(f"  M1: strat_rows={latest_m1['strat_rows']:,} cum_rows={latest_m1['cum_rows']:,} "
          f"blob={latest_m1['blob_bytes']/1e9:.2f}GB  neg_control_rows={latest_neg} (must be 0)")

    # --- K-COV (opener-faced raised-pot on-policy coverage; design 36) -------------------------------
    # Sourced from the train log's COVERAGE line (trainer coverage_probe raised-pot subclass), NOT the
    # checkpoint. Under --presum it is a verdict clause (hard early-kill @K1a / hard cap @K1); without
    # --presum it is advisory-only (printed, never folded) so the historical fullhist verdict is
    # byte-identical.
    cov_list = kcov.parse_coverage(train_log_path) if (train_log_path and os.path.exists(train_log_path)) else []
    cov_latest = kcov.cov_at(cov_list, int(latest_it))
    cov_prior = kcov.cov_at(cov_list, int(prior_m1["iters"])) if prior_m1 else None
    kcov_res = kcov.evaluate_kcov(
        cov_latest, cov_prior, k1a_regime=k1a_regime, cap_pct=args.kcov_cap,
        early_kill_pct=args.kcov_early_kill, min_sample=args.kcov_min_sample, drift_pt=args.k1a_drift_pt)
    print("  " + kcov_res["line"] + ("" if args.presum else "  [advisory: --presum not set, not folded]"))
    kcov_killed = False
    if args.presum and kcov_res["hard"]:
        kcov_killed = not bool(kcov_res["passed"])
        ok &= bool(kcov_res["passed"])
    # FAIL-LOUD (design 36 calibration fix): under --presum, an INSUFFICIENT_DATA K-COV read means the
    # spend-gate coverage clause could NOT be evaluated here -- most likely raised_decisions <
    # --kcov-min-sample because --v3-cov-hands is too small for the measured jam-heavy raised-pot
    # density (~0.07-0.10/hand; need ~40000 hands to clear 2000). A dark clause is NOT a pass; make it
    # impossible to mistake for one (printed loud + stamped kcov_dark). It does NOT auto-KILL (an early
    # checkpoint legitimately lacks data), but it must never read as a silent clean K-COV.
    kcov_dark = bool(args.presum and kcov_res["status"] == "INSUFFICIENT_DATA")
    if kcov_dark:
        ndec = kcov_res["detail"].get("raised_decisions")
        print(f"  !! K-COV DARK: the presum coverage spend-gate clause was NOT evaluated at iters="
              f"{latest_it} (INSUFFICIENT_DATA"
              + (f", raised_decisions={ndec} < min_sample={args.kcov_min_sample}" if ndec is not None
                 else ", no COVERAGE data")
              + "). This is NOT a K-COV pass -- the raised-pot coverage pathology is UNCHECKED here. "
              "Raise --v3-cov-hands (~40000 at the measured density; see README_LAUNCH.md) and re-gate.")

    # AA-limp drift advisory (design 36 K1a leading indicator: AA-limp should fall >=5pt 7.5k->10k).
    aa_now, aa_pre = _aa_limp(latest_nodes), (_aa_limp(prior_nodes) if prior_nodes else None)
    aa_drift = None if (aa_now is None or aa_pre is None) else (aa_pre - aa_now)
    if aa_drift is not None:
        print(f"  AA-limp drift (advisory): fell {aa_drift*100:+.1f}pt "
              f"({'>=' if aa_drift*100 >= args.k1a_drift_pt else '<'} {args.k1a_drift_pt:.0f}pt target) "
              f"[{aa_pre*100:.1f}% -> {aa_now*100:.1f}%]")

    verdict = "PASS" if ok else "KILL"
    print(f"\nVERDICT: {verdict}  [regime {regime}; AA cap {aa_cap_applied*100:.0f}%"
          + ("; +K-COV" if args.presum else "") + "]")
    if not ok:
        if k1a_regime:
            # List ONLY the clauses that actually breached (G3 pass=False, and/or K-COV kill); under
            # --presum a run can be killed by K-COV alone with AA-limp passing, so do not mislabel.
            hard_clauses = []
            if not gdetail.get("G3", {}).get("pass", False):
                hard_clauses.append(f"AA-BTN-limp > {aa_cap_applied*100:.0f}% cap")
            if kcov_killed:
                hard_clauses.append(f"K-COV raised-pot miss > {args.kcov_early_kill:.0f}%")
            print(f"KILL @K1a EARLY-KILL: breached {' and/or '.join(hard_clauses) or 'a hard clause'} "
                  "(the hard clause(s) below --k1-hard-iters). Any G1/G2/G4/G6 WARN above is "
                  "advisory and did NOT contribute to this verdict.")
        elif kcov_killed:
            print(f"KILL: K-COV raised-pot coverage below the {100-args.kcov_cap:.0f}% floor "
                  "(presum failed to fix the postflop-coverage pathology).")
        print("KILL the run now: terminate the box (aws ec2 terminate-instances --instance-ids "
              "<iid> --region " + args.region + ") + stop the babysitter. The limp-trap pathology "
              + ("persisted under presum" if args.presum else "reproduced under fullhist")
              + " -- do not fund it to the backstop.")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "verdict": verdict, "run_subdir": args.run_subdir, "iters": latest_it,
            "prior_iters": prior_m1["iters"] if prior_m1 else None,
            "regime": "K1a" if k1a_regime else "K1", "k1a_regime": k1a_regime,
            "aa_cap_applied": aa_cap_applied,
            "trend_hard": trend_hard, "gates": lines, "gate_detail": gdetail,
            "m1_latest": latest_m1, "m1_prior": prior_m1,
            "neg_control_rows": latest_neg, "menu_read": latest_menu,
            # K-COV (design 36): folded into the verdict only under --presum; advisory otherwise.
            # kcov_dark = INSUFFICIENT_DATA under --presum: the clause could not be evaluated (sample
            # too small / no data) -- NOT a pass; raise --v3-cov-hands. Fails loud, never silent.
            "presum": args.presum, "kcov_status": kcov_res["status"], "kcov_folded": args.presum,
            "kcov_dark": kcov_dark, "kcov": kcov_res["detail"],
            "aa_limp_drift_pt": (round(aa_drift * 100, 2) if aa_drift is not None else None),
            "params": {"mono_tol_pt": args.mono_tol_pt, "btn_floor_pct": args.btn_floor_pct,
                       "aa_limp_cap": args.aa_limp_cap, "k1a_aa_cap": args.k1a_aa_cap,
                       "k1_hard_iters": args.k1_hard_iters, "bb_fold_cap": args.bb_fold_cap,
                       "jam_cap": args.jam_cap, "decline_floor_pt": args.decline_floor_pt,
                       "trend_hard_iters": args.trend_hard_iters,
                       # stamped ONLY off-default so the default JSON stays byte-identical to
                       # the pre-arm-3 instrument (absence == default 3-size menu).
                       **({"pre_menu": args.pre_menu,
                           "open_raise_ids": list(prd.RAISE_SET)}
                          if args.pre_menu != "default" else {}),
                       # presum K-COV thresholds stamped only under --presum (self-describing evidence).
                       **({"presum": True, "kcov_cap": args.kcov_cap,
                           "kcov_early_kill": args.kcov_early_kill,
                           "kcov_min_sample": args.kcov_min_sample,
                           "k1a_drift_pt": args.k1a_drift_pt} if args.presum else {})},
        }, f, indent=2)
    print(f"wrote {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
