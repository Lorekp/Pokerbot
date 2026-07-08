#!/usr/bin/env python3
"""blueprint-v4 blocker D: K-COV (opener-faced raised-pot on-policy coverage) -- parser + gate.

SINGLE SOURCE OF TRUTH for the presum K-COV threshold. K-COV is measured by the trainer's on-policy
coverage probe (tools/battery_native/src/trainer.rs::coverage_probe, raised-pot subclass) and emitted
at checkpoint cadence as the

    COVERAGE iters_done=<it> ... raised_postflop=<miss%> ... raised_decisions=<n> ...

line in the run's train log (BOTH ab_train.exe -- `iter=` -- and the cloud driver -- `iters_done=` --
emit it; this parser accepts either spelling). `raised_postflop` is the on-policy MISS rate among
RAISED-pot ("opener-faced", preflop n_raises>=1) postflop decisions; coverage = 100 - miss. It is the
exact pot-type region presum collapses, so it directly measures whether presum is fixing the
postflop-coverage pathology the fullhist limp-trap is driven by.

GROUNDING (artifacts/_kcov_grounding_ab.json -- local A/B at an EQUAL ~40M-row budget, i.e. equal RAM,
the binding Stage-3 constraint):
  * fullhist: raised-pot postflop miss 95.3%  (coverage ~5%)
  * presum  : raised-pot postflop miss 24.8%  (coverage ~75%)
A ~15x coverage win in exactly the starved region, at equal RAM. The offline instrument
(examples/kcov_probe.rs) reproduces the in-trainer number bit-identically at a matched seed. The gate
below is calibrated to that contrast with margin for scale/seed drift on the funded run.

PRE-REGISTERED PRESUM K-COV GATE (design amendment sec 36):
  * INSUFFICIENT_DATA (advisory, never folds a verdict): no COVERAGE data at the target iter, or
    raised_decisions < MIN_SAMPLE (default 2000) -- a coverage number over too few raised-pot visits
    is noise. SIZING: the MEASURED raised-pot-postflop decision density is only ~0.07-0.10 per probe
    hand on jam-heavy runs (grounding: presum 286/4000=0.072, fullhist 361/4000=0.090), and a BROKEN
    presum run stays jam-heavy (high preflop-all-in rate) => LOWEST density on exactly the run that
    must be killed. So clearing 2000 raised decisions needs ~28k hands at that density; the funded run
    uses POKERBOT_COV_HANDS / -V3CovHands ~40000 for margin (~2800). If a gate checkpoint STILL reads
    INSUFFICIENT_DATA under --presum, bpv4_kill_gate.py flags it LOUD + stamps kcov_dark -- a dark
    spend-gate clause is never treated as a pass (fail-loud, not fail-open-silently).
  * K1a EARLY-KILL (iters < k1_hard_iters): raised_postflop miss > EARLY_KILL_PCT (default 80%) => KILL.
    A presum run stuck near fullhist's ~95% miss even early means the key-sharing is not working
    (a bug / wrong config); kill fast. Presum hit 25% miss by ~800 local iters, so >80% at a real
    multi-thousand-iter checkpoint is decisively broken. The drift clause is advisory at K1a.
  * K1 BINDING (iters >= k1_hard_iters): raised_postflop miss <= CAP_PCT (default 50%) HARD
    (coverage >= 50%); presum's local A/B already cleared 75% coverage at a tiny budget, and the
    funded run trains far longer, so 50% is a generous floor that fullhist (95% miss) fails outright.
    Falling-vs-prior is advisory (a coverage plateau above the cap is what the CAP already catches).
"""
from __future__ import annotations

import os
import re

# Pre-registered thresholds (design amendment sec 36). Overridable via evaluate() kwargs / gate flags.
CAP_PCT = 50.0          # K1 binding: raised-pot miss must be <= this (coverage >= 50%).
EARLY_KILL_PCT = 80.0   # K1a early-kill: raised-pot miss > this => KILL.
MIN_SAMPLE = 2000       # raised-pot decisions below this => INSUFFICIENT_DATA (advisory).
DRIFT_PT = 5.0          # advisory: raised-pot miss should fall >= this many pt vs the prior checkpoint.

# Accepts both `iters_done=` (cloud driver) and `iter=` (ab_train.exe). Fields are order-independent.
_ITERS_RE = re.compile(r"\biters?(?:_done)?=(\d+)")
_RPF_RE = re.compile(r"\braised_postflop=([\d.]+)")
_RDEC_RE = re.compile(r"\braised_decisions=(\d+)")
_POST_RE = re.compile(r"\bpostflop=([\d.]+)")
_ALLIN_RE = re.compile(r"\bpre_allin_runout=([\d.]+)")


def parse_coverage(src: str) -> list[dict]:
    """Parse every COVERAGE line from a train-log string OR a path to one. Returns
    [{iters, raised_postflop_miss, raised_decisions, postflop_miss, pre_allin_runout}] sorted by iters
    (last-wins on duplicate iters, so a resumed run's newest probe supersedes an older one)."""
    text = src
    if "\n" not in src and os.path.exists(src):
        with open(src, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    by_iter: dict[int, dict] = {}
    for ln in text.splitlines():
        if "COVERAGE" not in ln or "raised_postflop=" not in ln:
            continue
        mi, mr, md = _ITERS_RE.search(ln), _RPF_RE.search(ln), _RDEC_RE.search(ln)
        if not (mi and mr and md):
            continue
        it = int(mi.group(1))
        mp, ma = _POST_RE.search(ln), _ALLIN_RE.search(ln)
        by_iter[it] = {
            "iters": it,
            "raised_postflop_miss": float(mr.group(1)),
            "raised_decisions": int(md.group(1)),
            "postflop_miss": float(mp.group(1)) if mp else None,
            "pre_allin_runout": float(ma.group(1)) if ma else None,
        }
    return [by_iter[k] for k in sorted(by_iter)]


def cov_at(cov: list[dict], iters: int | None) -> dict | None:
    """The COVERAGE record at `iters` (exact), else the nearest record with iters <= target, else the
    latest available if `iters` is None. Returns None if there are no records."""
    if not cov:
        return None
    if iters is None:
        return cov[-1]
    exact = [c for c in cov if c["iters"] == iters]
    if exact:
        return exact[-1]
    leq = [c for c in cov if c["iters"] <= iters]
    return leq[-1] if leq else None


def evaluate_kcov(latest: dict | None, prior: dict | None, *, k1a_regime: bool,
                  cap_pct: float = CAP_PCT, early_kill_pct: float = EARLY_KILL_PCT,
                  min_sample: int = MIN_SAMPLE, drift_pt: float = DRIFT_PT) -> dict:
    """Evaluate the presum K-COV gate at one checkpoint.

    Returns a dict:
      status   : "PASS" | "KILL" | "WARN" | "INSUFFICIENT_DATA"
      hard      : bool  -- whether `passed` should be folded into the run verdict
      passed    : bool  -- the hard pass/fail (meaningful only when hard=True)
      line      : str   -- one printable summary line
      detail    : dict  -- the raw numbers + thresholds for the evidence JSON
    Mirrors the kill-gate's K1a/K1 discipline: at K1a only a decisive early-kill (miss > early_kill)
    is hard; at K1 the cap is hard. Insufficient/absent data is ALWAYS advisory (never folds)."""
    d: dict = {"cap_pct": cap_pct, "early_kill_pct": early_kill_pct, "min_sample": min_sample,
               "drift_pt": drift_pt, "k1a_regime": k1a_regime}
    if latest is None:
        d["reason"] = "no COVERAGE line at/<= target iter (run too early, or POKERBOT_COV_HANDS unset)"
        return {"status": "INSUFFICIENT_DATA", "hard": False, "passed": None,
                "line": "K-COV: INSUFFICIENT_DATA (no COVERAGE data) -- advisory, not folded", "detail": d}

    miss = latest["raised_postflop_miss"]
    ndec = latest["raised_decisions"]
    cov = 100.0 - miss
    d.update({"iters": latest["iters"], "raised_postflop_miss_pct": miss,
              "raised_postflop_coverage_pct": round(cov, 2), "raised_decisions": ndec})
    drift = None
    if prior is not None:
        drift = prior["raised_postflop_miss"] - miss  # positive == miss fell (coverage rose)
        d["prior_iters"] = prior["iters"]
        d["drift_pt"] = round(drift, 2)
        d["drift_ok"] = drift >= drift_pt
    tnote = "no prior COVERAGE point" if drift is None else (
        f"prior {prior['raised_postflop_miss']:.1f}% -> {miss:.1f}% miss "
        f"(fell {drift:+.1f}pt, {'>=' if drift >= drift_pt else '<'} {drift_pt:.0f}pt target)")

    if ndec < min_sample:
        d["reason"] = f"raised_decisions {ndec} < min_sample {min_sample}"
        return {"status": "INSUFFICIENT_DATA", "hard": False, "passed": None,
                "line": (f"K-COV: INSUFFICIENT_DATA (raised_decisions={ndec} < {min_sample}) -- "
                         f"raised-pot coverage {cov:.1f}% is noise at this sample; advisory, not folded"),
                "detail": d}

    if k1a_regime:
        # Early-kill only on a decisive breach; otherwise advisory (converging).
        killed = miss > early_kill_pct
        d["passed"] = not killed
        status = "KILL" if killed else ("PASS" if cov >= cap_pct else "WARN")
        line = (f"K-COV @K1a raised-pot coverage {cov:.1f}% (miss {miss:.1f}% vs early-kill "
                f">{early_kill_pct:.0f}%): {'KILL' if killed else ('PASS' if cov >= cap_pct else 'WARN (advisory @K1a -- converging)')}"
                f"  [{tnote}; n={ndec}]")
        return {"status": status, "hard": killed, "passed": not killed, "line": line, "detail": d}

    # K1 binding: the cap is hard.
    passed = miss <= cap_pct
    d["passed"] = passed
    line = (f"K-COV @K1 raised-pot coverage {cov:.1f}% (miss {miss:.1f}% <= cap {cap_pct:.0f}%): "
            f"{'PASS' if passed else 'KILL'}  [{tnote}; n={ndec}]")
    return {"status": "PASS" if passed else "KILL", "hard": True, "passed": passed,
            "line": line, "detail": d}
