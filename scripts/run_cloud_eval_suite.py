#!/usr/bin/env python3
"""ON-BOX driver for the cloud exploitation-eval suite (Exploitation Program v2 batteries).

Runs the ENTIRE remaining measurement program on one many-core spot box, restart-proof:

  Phase A (one shared pool):
    - P0 root-mixer fair battery: arms A (facing-only) / B (+unopened) x 4 hs bots x N seeds
      (pluribus search hero, the production config candidates)
    - G4 adaptation sweep: 27 configs x 6 seeds (heuristic hero, env-overridden constants)
    - G3 prior A/B: shipped vs IRC-fitted prior x N seeds (heuristic hero)
  GATE (automatic, the plan's pre-registered rule): an arm passes iff >=1 weak archetype pooled
  CI>0 AND the hs_tag control CI lower bound > -0.05; prefer the passing arm with the larger
  summed weak-archetype uplift.
  Phase B (winning arm config; skipped if no arm passes):
    - P2 bluff-model battery (4 hs bots, --bluff-model)
    - G3 weak_mixture fitted-gauntlet battery
    - P5 do-no-harm: counter_exploiter + blueprint near-Nash villain

Every job uploads its result JSON to s3://{bucket}/{prefix}/results/ IMMEDIATELY and is SKIPPED
if that key already exists — a spot reclaim loses only in-flight jobs; the babysitter relaunches
and the suite resumes. Final: {prefix}/results/SUMMARY.json + {prefix}/eval_suite_DONE.marker.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
ARCHES = ("hs_station", "hs_nit", "hs_lag", "hs_tag")
WEAK = ("hs_station", "hs_nit", "hs_lag")
SWEEP_GRID = {
    "POKERBOT_PRIOR_STRENGTH": ["4.0", "8.0", "16.0"],
    "POKERBOT_K_RAMP": ["2.0", "4.0", "8.0"],
    "POKERBOT_GAMMA_FAST": ["0.85", "0.90", "0.95"],
}


def battery_cmd(out: Path, *, hero: str, arch: str, seed: int, hands: int, bp: str,
                extra: tuple[str, ...] = (), ckpt: str | None = None,
                is_field: bool = False, iters: int = 10, leaf: str = "uniform",
                per_seat: bool = False) -> list[str]:
    cmd = [sys.executable, "scripts/eval_exploitation_battery.py", "--hero", hero,
           "--hands", str(hands), "--seed", str(seed), "--allin-ev", "--out", str(out)]
    # ``is_field`` ⇒ heterogeneous realistic-table base-strength cell (run_field_absolute);
    # else the homogeneous archetype on-off A/B (run_archetype).
    cmd += (["--field", arch] if is_field else ["--archetypes", arch])
    if is_field and per_seat:
        # PER-VILLAIN targeting on a MIXED table (matches live per-seat play): one profile per seat,
        # the hero reads the aggressor it faces. Fixes the pooled-blend +exploit regression on the
        # blueprint base. No-op for homogeneous cells (all seats are the same archetype).
        cmd += ["--per-seat-profiles"]
    if hero == "pluribus":
        # iters/leaf are the basestrength STRONG-CONFIG sweep knobs (Stage 1a). Floor = 10/uniform.
        # AIVAT leaf-Q stays off (anti-correlated for the gadget hero); the OFF arm is the headline.
        cmd += ["--iterations", str(iters), "--leaf-select", leaf, "--no-aivat-leafq",
                "--blueprint-snapshot", bp]
    if hero == "production" and ckpt:
        cmd += ["--checkpoint", ckpt]
    if "--gear-before" not in extra:
        cmd += ["--no-gear-changer"]
    cmd += list(extra)
    return cmd


class Suite:
    def __init__(self, args) -> None:
        self.a = args
        self.s3 = boto3.client("s3", region_name=args.region)
        self.rk = f"{args.prefix}/results"
        RUNS.mkdir(exist_ok=True)
        self._existing = self._list_existing()

    def _list_existing(self) -> set[str]:
        keys, token = set(), None
        while True:
            kw = {"Bucket": self.a.bucket, "Prefix": self.rk + "/"}
            if token:
                kw["ContinuationToken"] = token
            r = self.s3.list_objects_v2(**kw)
            for o in r.get("Contents", []):
                keys.add(o["Key"].rsplit("/", 1)[-1])
            if not r.get("IsTruncated"):
                return keys
            token = r.get("NextContinuationToken")

    def run_job(self, name: str, cmd: list[str], env_extra: dict[str, str] | None = None) -> None:
        fn = f"{name}.json"
        out = RUNS / fn
        if fn in self._existing:
            if not out.exists():  # resume on a fresh box: pull for local pooling
                self.s3.download_file(self.a.bucket, f"{self.rk}/{fn}", str(out))
            return
        env = {**os.environ, "POKERBOT_BETTING_ABSTRACTION": "1", "PYTHONIOENCODING": "utf-8",
               # Blueprint serve keys must match the trained table (postflop). Without these the
               # pluribus hero gets ~0% postflop blueprint coverage (silent uniform fallback). Net is
               # invariant (raw cards) ⇒ harmless for the production hero.
               "POKERBOT_POSTFLOP_BUCKETS": "1", "POKERBOT_POSTFLOP_MODE": "fast",
               "POKERBOT_POSTFLOP_NUM_BUCKETS": "200",
               **(env_extra or {})}
        log = RUNS / f"{name}.log"
        with open(log, "w", encoding="utf-8") as lf:
            try:
                rc = subprocess.run(cmd, cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT,
                                    timeout=self.a.job_timeout, check=False).returncode
            except subprocess.TimeoutExpired:
                lf.write(f"\nJOB TIMEOUT after {self.a.job_timeout}s\n")
                rc = -1
        if rc == 0 and out.exists():
            self.s3.upload_file(str(out), self.a.bucket, f"{self.rk}/{fn}")
            self._existing.add(fn)
        else:  # keep the log for diagnosis either way
            self.s3.upload_file(str(log), self.a.bucket, f"{self.rk}/FAIL_{name}.log")
        print(f"[{time.strftime('%H:%M:%S')}] {name} rc={rc}", flush=True)

    # ---- phases ----
    def phase_a_jobs(self) -> list[tuple[str, list[str], dict | None]]:
        a, jobs = self.a, []
        for arch in ARCHES:                                   # P0 — the priority work, queued first
            for seed in range(a.seeds):
                for arm, extra in (("A", ("--root-mixer",)),
                                   ("B", ("--root-mixer", "--root-mixer-unopened"))):
                    n = f"eb3000_rm{arm}_{arch}_s{seed}"
                    jobs.append((n, battery_cmd(RUNS / f"{n}.json", hero="pluribus", arch=arch,
                                                seed=seed, hands=a.hands, bp=a.bp, extra=extra), None))
        for vals in itertools.product(*SWEEP_GRID.values()):  # G4 sweep (heuristic, fast)
            cfg = dict(zip(SWEEP_GRID.keys(), vals))
            tag = f"P{cfg['POKERBOT_PRIOR_STRENGTH']}_K{cfg['POKERBOT_K_RAMP']}_G{cfg['POKERBOT_GAMMA_FAST']}"
            for seed in range(6):
                n = f"sw_{tag}_s{seed}"
                jobs.append((n, battery_cmd(RUNS / f"{n}.json", hero="heuristic",
                                            arch="hs_station,hs_tag", seed=seed, hands=a.hands,
                                            bp=a.bp, extra=("--gear-before", "hs_station",
                                                            "--gear-after", "hs_lag")), cfg))
        for arm in ("old", "fitted"):                         # G3 prior A/B (heuristic)
            env = ({"POKERBOT_PRIOR_PATH": "data/player_profiles/irc_population_prior_fitted.json"}
                   if arm == "fitted" else None)
            for seed in range(a.seeds):
                n = f"pab_{arm}_s{seed}"
                jobs.append((n, battery_cmd(RUNS / f"{n}.json", hero="heuristic",
                                            arch=",".join(ARCHES), seed=seed, hands=a.hands,
                                            bp=a.bp), env))
        return jobs

    def pool(self, pattern: str, arch: str) -> dict | None:
        import numpy as np

        ups, ses = [], []
        for f in RUNS.glob(pattern):
            try:
                r = json.loads(f.read_text(encoding="utf-8"))["results"].get(arch)
            except (ValueError, KeyError, OSError):
                continue
            if r:
                ups.append(r["uplift_bb_per_hand"])
                ses.append(r["uplift_se"])
        if not ups:
            return None
        k = len(ups)
        pooled = float(np.mean(ups))
        se = math.sqrt(float(np.sum(np.square(ses)))) / k
        return {"k": k, "pooled": pooled, "se": se,
                "ci": [pooled - 1.96 * se, pooled + 1.96 * se],
                "sign": int(np.sum(np.array(ups) > 0))}

    def pool_absolute(self, pattern: str, key: str) -> dict | None:
        """Pool the ABSOLUTE per-arm AIVAT EV across seeds for one (base, cell): off_aivat = the pure
        base's bb/hand (base true strength), on_aivat = base+exploit. Works for both the archetype
        results shape (results[arch]) and the field shape (results[field]) — both carry off/on_aivat."""
        import numpy as np

        offs, ons = [], []
        for f in RUNS.glob(pattern):
            try:
                r = json.loads(f.read_text(encoding="utf-8"))["results"].get(key)
            except (ValueError, KeyError, OSError):
                continue
            if r and isinstance(r.get("off_aivat"), dict) and isinstance(r.get("on_aivat"), dict):
                offs.append(float(r["off_aivat"]["adj_mean_bb"]))
                ons.append(float(r["on_aivat"]["adj_mean_bb"]))
        if not offs:
            return None
        k = len(offs)
        off_m, on_m = float(np.mean(offs)), float(np.mean(ons))
        off_se = float(np.std(offs, ddof=1) / np.sqrt(k)) if k > 1 else 0.0
        on_se = float(np.std(ons, ddof=1) / np.sqrt(k)) if k > 1 else 0.0
        return {"k": k,
                "base_bb": off_m, "base_se": off_se,
                "base_ci": [off_m - 1.96 * off_se, off_m + 1.96 * off_se],
                "exploit_bb": on_m, "exploit_se": on_se,
                "uplift_bb": on_m - off_m}

    # ---- base-strength mode (task (a)/(b) measurement + Stage-1 strong-config sweep) ----
    _BS_HOMOG = ("hs_station", "hs_nit", "hs_lag", "hs_tag", "weak_mixture", "blueprint")
    _BS_HETERO = ("realistic_table", "soft_table", "tough_table")

    def _bs_configs(self) -> list[tuple[int, str]]:
        """Parse the pluribus strong-config sweep: '<iters>:<leaf>,...' (default floor '10:uniform')."""
        raw = getattr(self.a, "bs_pluribus_configs", None) or "10:uniform"
        out = []
        for tok in raw.split(","):
            tok = tok.strip()
            if not tok:
                continue
            it, _, leaf = tok.partition(":")
            out.append((int(it), leaf or "uniform"))
        return out

    def _bs_bases(self) -> list[tuple[str, str, str | None, int, str]]:
        """(tag, hero, ckpt, iters, leaf) per base arm. The SFT/production base (iters/leaf ignored)
        plus one pluribus arm per swept (iters, leaf) config — so Stage 1a compares search configs as
        distinct 'bases' (bp_i<iters>_<leaf>) under the same absolute-EV pooling."""
        bases: list[tuple[str, str, str | None, int, str]] = []
        if not getattr(self.a, "bs_skip_sft", False):
            bases.append(("sft", "production", self.a.checkpoint, 10, "uniform"))
        for it, leaf in self._bs_configs():
            bases.append((f"bp_i{it}_{leaf}", "pluribus", None, it, leaf))
        return bases

    def basestrength_jobs(self) -> list[tuple[str, list[str], dict | None]]:
        """Measure ABSOLUTE base strength for each base arm (SFT net + pluribus config sweep) vs
        weak/strong/realistic fields. Each cell's OFF arm is the pure base (the 'true strength' figure);
        the ON arm adds the validated exploit. Homogeneous cells run the archetype A/B; heterogeneous
        cells run run_field_absolute on a mixed table. POKERBOT_BP_COVERAGE=1 logs per-street blueprint
        hit-rate (Stage 1b) into each result JSON."""
        a, jobs = self.a, []
        # Blueprint villains / the pluribus base need the snapshot; coverage diagnostic on for pluribus.
        bpenv = {"POKERBOT_BLUEPRINT_SNAPSHOT": a.bp, "POKERBOT_BP_COVERAGE": "1"}
        for btag, hero, ck, it, leaf in self._bs_bases():
            for cell in self._BS_HOMOG:
                for seed in range(a.seeds):
                    n = f"bs_{btag}_{cell}_s{seed}"
                    jobs.append((n, battery_cmd(RUNS / f"{n}.json", hero=hero, arch=cell, seed=seed,
                                                hands=a.hands, bp=a.bp, ckpt=ck, iters=it, leaf=leaf),
                                 bpenv))
            for cell in self._BS_HETERO:
                for seed in range(a.seeds):
                    n = f"bs_{btag}_{cell}_s{seed}"
                    jobs.append((n, battery_cmd(RUNS / f"{n}.json", hero=hero, arch=cell, seed=seed,
                                                hands=a.hands, bp=a.bp, ckpt=ck, iters=it, leaf=leaf,
                                                is_field=True,
                                                per_seat=not a.bs_pooled_profiles), bpenv))
        return jobs

    def _pool_coverage(self, pattern: str) -> dict | None:
        """Sum the Stage-1b per-street blueprint hit/miss counts across a base arm's result JSONs."""
        agg: dict[str, list[int]] = {}
        for f in RUNS.glob(pattern):
            try:
                cov = json.loads(f.read_text(encoding="utf-8")).get("bp_coverage") or {}
            except (ValueError, OSError):
                continue
            for street, d in cov.items():
                rec = agg.setdefault(street, [0, 0])
                rec[0] += int(d.get("hits", 0))
                rec[1] += int(d.get("misses", 0))
        if not agg:
            return None
        return {s: {"hits": h, "misses": m, "hit_rate": (h / (h + m) if (h + m) else 0.0)}
                for s, (h, m) in agg.items()}

    def run_basestrength(self) -> None:
        a = self.a
        jobs = self.basestrength_jobs()
        print(f"basestrength: {len(jobs)} jobs ({len(self._existing)} already in S3)", flush=True)
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            list(ex.map(lambda j: self.run_job(*j), jobs))
        summary = {"mode": "basestrength", "hands": a.hands, "seeds": a.seeds,
                   "checkpoint": a.checkpoint, "configs": [f"{i}:{l}" for i, l in self._bs_configs()],
                   "bases": {}, "coverage": {}}
        for btag, hero, _ck, _it, _leaf in self._bs_bases():
            cells = {}
            for cell in self._BS_HOMOG + self._BS_HETERO:
                p = self.pool_absolute(f"bs_{btag}_{cell}_s*.json", cell)
                if p:
                    cells[cell] = p
            summary["bases"][btag] = cells
            if hero == "pluribus":  # Stage-1b coverage gate (per pluribus config)
                cov = self._pool_coverage(f"bs_{btag}_*_s*.json")
                if cov:
                    summary["coverage"][btag] = cov
        body = json.dumps(summary, indent=1, default=str)
        self.s3.put_object(Bucket=a.bucket, Key=f"{self.rk}/SUMMARY.json", Body=body.encode())
        if any(summary["bases"].values()):
            self.s3.put_object(Bucket=a.bucket, Key=f"{a.prefix}/eval_suite_DONE.marker",
                               Body=time.strftime("%Y-%m-%d %H:%M:%S").encode())
            print("BASESTRENGTH DONE\n" + body, flush=True)
        else:
            print("BASESTRENGTH ALL-FAILED — no DONE marker (check FAIL_*.log)\n" + body, flush=True)

    # ---- Phase 2: value-extraction gauntlet (unopened exploit vs realistic IRC villains) ----
    _VE_FIELDS = ("irc_soft", "irc_station", "irc_mixed", "irc_tough", "irc_lag", "irc_pros")
    _VE_WEAK = ("irc_soft", "irc_station", "irc_mixed")
    _VE_TOUGH = ("irc_tough", "irc_lag", "irc_pros")

    def _ve_config(self) -> tuple[int, str]:
        raw = getattr(self.a, "ve_config", None) or "10:uniform"
        it, leaf = raw.split(":")
        return int(it), leaf

    def valueextract_jobs(self) -> list[tuple[str, list[str], dict | None]]:
        """One gauntlet cell per (field, seed): OFF=pure base vs ON=base+unopened-exploit, per-seat
        profiles, vs the realistic IRC villains. Each cell self-uploads ve_<field>_s<seed>.json."""
        a, jobs = self.a, []
        it, leaf = self._ve_config()
        for field in self._VE_FIELDS:
            for seed in range(a.seeds):
                n = f"ve_{field}_s{seed}"
                cmd = [sys.executable, "scripts/value_extraction_gauntlet.py",
                       "--blueprint-snapshot", a.bp, "--fields", field, "--seeds", "1",
                       "--seed", str(seed), "--hands", str(a.hands),
                       "--iterations", str(it), "--leaf-select", leaf,
                       "--out", str(RUNS / f"{n}.json")]
                jobs.append((n, cmd, None))
        return jobs

    def _pool_ve(self, field: str) -> dict | None:
        offs, ons = [], []
        for f in RUNS.glob(f"ve_{field}_s*.json"):
            try:
                fd = (json.loads(f.read_text(encoding="utf-8")).get("fields") or {}).get(field)
            except (ValueError, OSError):
                continue
            if fd:
                offs.append(float(fd["off"]))
                ons.append(float(fd["on"]))
        if not offs:
            return None
        import math
        def _ms(xs):
            m = sum(xs) / len(xs)
            se = (math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) / math.sqrt(len(xs))
                  ) if len(xs) > 1 else 0.0
            return m, se
        om, ose = _ms(offs)
        nm, nse = _ms(ons)
        return {"off": om, "off_se": ose, "on": nm, "on_se": nse, "value": nm - om, "k": len(offs)}

    def run_valueextract(self) -> None:
        a = self.a
        jobs = self.valueextract_jobs()
        print(f"valueextract: {len(jobs)} jobs ({len(self._existing)} already in S3)", flush=True)
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            list(ex.map(lambda j: self.run_job(*j), jobs))
        it, leaf = self._ve_config()
        fields = {f: self._pool_ve(f) for f in self._VE_FIELDS}
        fields = {f: v for f, v in fields.items() if v}
        weak = [v["value"] for f, v in fields.items() if f in self._VE_WEAK]
        tough = [v["value"] for f, v in fields.items() if f in self._VE_TOUGH]
        summary = {"mode": "valueextract", "hands": a.hands, "seeds": a.seeds,
                   "config": f"{it}:{leaf}", "fields": fields,
                   "value_vs_weak": (sum(weak) / len(weak)) if weak else None,
                   "harm_vs_tough": (min(tough)) if tough else None}
        body = json.dumps(summary, indent=1, default=str)
        self.s3.put_object(Bucket=a.bucket, Key=f"{self.rk}/SUMMARY.json", Body=body.encode())
        if fields:
            self.s3.put_object(Bucket=a.bucket, Key=f"{a.prefix}/eval_suite_DONE.marker",
                               Body=time.strftime("%Y-%m-%d %H:%M:%S").encode())
            print("VALUEEXTRACT DONE\n" + body, flush=True)
        else:
            print("VALUEEXTRACT ALL-FAILED — no DONE marker (check FAIL_*.log)\n" + body, flush=True)

    # ---- LBR6: certified 6-max best-response exploitability of the blueprint foundation ----
    # GRADED Lisy-Bowling protocol: sweep the action menu (fc floor / fcpa / full) to map HOW the exploit
    # grows with the adversary's betting options, with a per-seat deviation-gain breakdown. One cell per
    # (menu, seed) self-uploads lbr6_<menu>_s<seed>.json; pooling independent seeds shrinks the CI like a
    # single large-N run. Two calibration cells (foldbot, station) prove the adversary detects leaks.
    _LBR6_MENUS = ("fc", "fcpa", "full")

    def lbr6_jobs(self) -> list[tuple[str, list[str], dict | None]]:
        a, jobs = self.a, []
        extra = (["--rollout"] if getattr(a, "lbr6_passive", False) else [])  # legacy escape hatch only
        menus = ((getattr(a, "lbr6_menu", "fcpa"),) if getattr(a, "lbr6_single_menu", False)
                 else self._LBR6_MENUS if getattr(a, "lbr6_graded", True)
                 else (getattr(a, "lbr6_menu", "fcpa"),))
        # Target: the RAW blueprint (fast lookup) or the DEPLOYED blueprint+SEARCH base (what the bot really
        # plays). For search the seats run Pluribus per decision (~30x slower) while the LBR range-tracks
        # against the cheap blueprint -- so the same 3-menu sweep costs the same wall-clock as 1 menu (the
        # opponent search cost is per-decision, independent of the LBR's own action menu).
        target = getattr(a, "lbr6_target", "blueprint")
        search = (["--search-leaf", getattr(a, "lbr6_search_leaf", "uniform"),
                   "--search-iters", str(getattr(a, "lbr6_search_iters", 10))]
                  + (["--search-preflop"] if getattr(a, "lbr6_search_preflop", False) else [])
                  if target == "blueprint-search" else [])
        for tgt in ("foldbot", "station"):
            n = f"lbr6_cal_{tgt}_s0"
            jobs.append((n, [sys.executable, "scripts/lbr6_audit.py", "--target", tgt, "--menu", "full",
                             "--hands", str(min(a.hands, 400)), "--seed", "0", "--unpaired",
                             "--out", str(RUNS / f"{n}.json")], None))
        for menu in menus:
            for seed in range(a.seeds):
                n = f"lbr6_{menu}_s{seed}"
                jobs.append((n, [sys.executable, "scripts/lbr6_audit.py", "--target", target,
                                 "--blueprint-snapshot", a.bp, "--menu", menu, "--hands", str(a.hands),
                                 "--seed", str(seed), "--unpaired", "--out", str(RUNS / f"{n}.json")]
                             + search + extra, None))
        return jobs

    @staticmethod
    def _pool_stats(pairs: list[tuple[float, float, int]]) -> dict | None:
        """Pool (mean_bb_per_hand, std_bb_per_hand, n) triples -> bb/100 + 95% CI via total-variance."""
        pairs = [p for p in pairs if p[2]]
        if not pairs:
            return None
        N = sum(n for _, _, n in pairs)
        M = sum(m * n for m, _, n in pairs) / N
        ss = sum(n * (sd ** 2) + n * (m - M) ** 2 for m, sd, n in pairs)
        stderr = math.sqrt((ss / N) / N)
        return {"bb_per_100": M * 100, "ci_lo": (M - 1.96 * stderr) * 100,
                "ci_hi": (M + 1.96 * stderr) * 100, "stderr_bb_per_100": stderr * 100,
                "total_hands": N, "k_seeds": len(pairs)}

    def _pool_lbr6_menu(self, menu: str) -> dict | None:
        """Pool one menu's seed cells -> overall + per-seat deviation gain."""
        cells = []
        for f in RUNS.glob(f"lbr6_{menu}_s*.json"):
            try:
                cells.append(json.loads(f.read_text(encoding="utf-8")))
            except (ValueError, OSError):
                continue
        cells = [c for c in cells if c.get("hands")]
        if not cells:
            return None
        overall = self._pool_stats([(c["ev_bb_per_hand"], c["std_bb_per_hand"], c["hands"]) for c in cells])
        per_seat = {}
        for s in range(6):
            trip = [(ps["bb_per_100"] / 100.0, ps.get("std_bb_per_hand", 0.0), ps["n"])
                    for c in cells for sk, ps in (c.get("detail", {}).get("per_seat", {}) or {}).items()
                    if int(sk) == s]
            pooled_s = self._pool_stats(trip)
            if pooled_s:
                per_seat[s] = {"bb_per_100": round(pooled_s["bb_per_100"], 1),
                               "ci_lo": round(pooled_s["ci_lo"], 1), "ci_hi": round(pooled_s["ci_hi"], 1)}
        return {**overall, "per_seat": per_seat}

    def _read_lbr6_one(self, fn: str) -> dict | None:
        f = RUNS / fn
        if not f.exists():
            return None
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
            return {"bb_per_100": r["bb_per_100"], "hands": r["hands"]}
        except (ValueError, OSError, KeyError):
            return None

    def run_lbr6(self) -> None:
        a = self.a
        jobs = self.lbr6_jobs()
        print(f"lbr6 graded: {len(jobs)} jobs ({len(self._existing)} already in S3)", flush=True)
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            list(ex.map(lambda j: self.run_job(*j), jobs))
        menus = {m: self._pool_lbr6_menu(m) for m in self._LBR6_MENUS}
        menus = {m: v for m, v in menus.items() if v}
        cal = {t: self._read_lbr6_one(f"lbr6_cal_{t}_s0.json") for t in ("foldbot", "station")}
        cal_ok = all(c and c["bb_per_100"] > 50 for c in cal.values())
        # BEATABLE if ANY menu's pooled CI lower bound clears zero (the strongest menu defines the bound).
        beatable = any(v["ci_lo"] > 0 for v in menus.values())
        worst = max((v["bb_per_100"] for v in menus.values()), default=0.0)
        target = getattr(a, "lbr6_target", "blueprint")
        tgt_lbl = ("blueprint+search (DEPLOYED base)" if target == "blueprint-search"
                   else "raw blueprint (lookup only)")
        verdict = ("CALIBRATION-FAILED (adversary not detecting leaks)" if not cal_ok else
                   (f"BEATABLE: best-responder profits vs {tgt_lbl} by up to {worst:+.1f} bb/100 "
                    f"(a menu CI excludes 0)" if beatable else
                    f"ROBUST at this CI: no menu's best-responder proves a positive exploit of {tgt_lbl}"))
        summary = {"mode": "lbr6", "method": "analytic_1ply_lbr", "target": target,
                   "search": ({"leaf": getattr(a, "lbr6_search_leaf", "uniform"),
                               "iters": getattr(a, "lbr6_search_iters", 10)}
                              if target == "blueprint-search" else None),
                   "hands": a.hands, "seeds": a.seeds,
                   "menus": menus, "calibration": cal, "verdict": verdict}
        body = json.dumps(summary, indent=1, default=str)
        self.s3.put_object(Bucket=a.bucket, Key=f"{self.rk}/SUMMARY.json", Body=body.encode())
        if menus:
            self.s3.put_object(Bucket=a.bucket, Key=f"{a.prefix}/eval_suite_DONE.marker",
                               Body=time.strftime("%Y-%m-%d %H:%M:%S").encode())
            print("LBR6 DONE\n" + body, flush=True)
        else:
            print("LBR6 ALL-FAILED — no DONE marker (check FAIL_*.log)\n" + body, flush=True)

    def gate(self) -> tuple[str | None, dict]:
        """The pre-registered arm rule. Returns (winning arm or None, full table)."""
        table: dict = {}
        verdicts: dict[str, dict] = {}
        for arm in ("A", "B"):
            table[arm] = {arch: self.pool(f"eb3000_rm{arm}_{arch}_s*.json", arch) for arch in ARCHES}
            t = table[arm]
            weak_pass = [a for a in WEAK if t[a] and t[a]["ci"][0] > 0]
            tag_ok = bool(t["hs_tag"]) and t["hs_tag"]["ci"][0] > -0.05
            verdicts[arm] = {"weak_pass": weak_pass, "tag_ok": tag_ok,
                             "weak_sum": sum(t[a]["pooled"] for a in WEAK if t[a]),
                             "passes": bool(weak_pass) and tag_ok}
        table["verdicts"] = verdicts
        passing = [arm for arm in ("A", "B") if verdicts[arm]["passes"]]
        if not passing:
            return None, table
        return max(passing, key=lambda arm: verdicts[arm]["weak_sum"]), table

    def phase_b_jobs(self, arm: str) -> list[tuple[str, list[str], dict | None]]:
        a = self.a
        flags: tuple[str, ...] = ("--root-mixer",) + (("--root-mixer-unopened",) if arm == "B" else ())
        jobs = []
        for arch in ARCHES:                                   # P2 bluff model
            for seed in range(a.seeds):
                n = f"eb3000_bluff{arm}_{arch}_s{seed}"
                jobs.append((n, battery_cmd(RUNS / f"{n}.json", hero="pluribus", arch=arch,
                                            seed=seed, hands=a.hands, bp=a.bp,
                                            extra=flags + ("--bluff-model",)), None))
        for villain, tag in (("weak_mixture", "weakmix"),     # G3 gauntlet + P5 do-no-harm
                             ("counter_exploiter", "counter"), ("blueprint", "bpbot")):
            for seed in range(a.seeds):
                n = f"eb3000_{tag}{arm}_s{seed}"
                env = {"POKERBOT_BLUEPRINT_SNAPSHOT": a.bp} if villain == "blueprint" else None
                jobs.append((n, battery_cmd(RUNS / f"{n}.json", hero="pluribus", arch=villain,
                                            seed=seed, hands=a.hands, bp=a.bp, extra=flags), env))
        return jobs

    def confirm_jobs(self) -> list[tuple[str, list[str], dict | None]]:
        """Focused read-gated-mixer confirm: gated arm B (ON) vs pure search (OFF), same seeds,
        on station (keep the win) / lag (must not worsen vs OFF) / tag (control). nit excluded —
        the unopened steal vs a nit triggers all-in-EV timeouts (unreliable subsamples)."""
        a, jobs = self.a, []
        for arch in ("hs_station", "hs_lag", "hs_tag"):
            for seed in range(a.seeds):
                jobs.append((f"cg_{arch}_s{seed}",
                             battery_cmd(RUNS / f"cg_{arch}_s{seed}.json", hero="pluribus",
                                         arch=arch, seed=seed, hands=a.hands, bp=a.bp,
                                         extra=("--root-mixer", "--root-mixer-unopened")), None))
                jobs.append((f"co_{arch}_s{seed}",
                             battery_cmd(RUNS / f"co_{arch}_s{seed}.json", hero="pluribus",
                                         arch=arch, seed=seed, hands=a.hands, bp=a.bp), None))
        return jobs

    def run_confirm(self) -> None:
        import numpy as np

        a = self.a
        jobs = self.confirm_jobs()
        print(f"confirm mode: {len(jobs)} jobs ({len(self._existing)} already in S3)", flush=True)
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            list(ex.map(lambda j: self.run_job(*j), jobs))
        summary = {"mode": "confirm", "pools": {}, "paired_lag_delta": None}
        for arch in ("hs_station", "hs_lag", "hs_tag"):
            for arm in ("cg", "co"):
                p = self.pool(f"{arm}_{arch}_s*.json", arch)
                if p:
                    summary["pools"][f"{arm}_{arch}"] = p
        # Paired (per-seed) gated-minus-OFF delta vs lag: the do-no-harm number that matters.
        dg, do = {}, {}
        for arm, d in (("cg", dg), ("co", do)):
            for f in RUNS.glob(f"{arm}_hs_lag_s*.json"):
                try:
                    s = int(f.stem.split("_s")[-1])
                    d[s] = json.loads(f.read_text())["results"]["hs_lag"]["uplift_bb_per_hand"]
                except (ValueError, KeyError, OSError):
                    pass
        common = sorted(set(dg) & set(do))
        if common:
            deltas = np.array([dg[s] - do[s] for s in common])
            se = float(deltas.std(ddof=1) / np.sqrt(len(deltas))) if len(deltas) > 1 else 0.0
            summary["paired_lag_delta"] = {
                "k": len(common), "mean": float(deltas.mean()), "se": se,
                "ci": [float(deltas.mean() - 1.96 * se), float(deltas.mean() + 1.96 * se)]}
        body = json.dumps(summary, indent=1, default=str)
        self.s3.put_object(Bucket=a.bucket, Key=f"{self.rk}/SUMMARY.json", Body=body.encode())
        if summary["pools"]:  # only mark DONE if SOMETHING succeeded — else let the babysitter relaunch
            self.s3.put_object(Bucket=a.bucket, Key=f"{a.prefix}/eval_suite_DONE.marker",
                               Body=time.strftime("%Y-%m-%d %H:%M:%S").encode())
            print("CONFIRM DONE\n" + body, flush=True)
        else:
            print("CONFIRM ALL-FAILED — no DONE marker (check FAIL_*.log)\n" + body, flush=True)

    def hybrid_jobs(self) -> list[tuple[str, list[str], dict | None]]:
        """HybridAgent (production hero) validation: clean exploit ON vs pure-GTO OFF (the
        set_exploit_enabled toggle) vs the 4 fair archetypes; + do-no-harm vs the counter-exploiter
        and the near-Nash blueprint villain. The CRN A/B isolates the mixer's contribution on the
        neural base. This is the candidate LIVE exploitation vehicle (the search base could not be
        made do-no-harm)."""
        a, jobs = self.a, []
        ck = a.checkpoint
        for arch in ("hs_station", "hs_nit", "hs_lag", "hs_tag"):
            for seed in range(a.seeds):
                jobs.append((f"hy_{arch}_s{seed}",
                             battery_cmd(RUNS / f"hy_{arch}_s{seed}.json", hero="production",
                                         arch=arch, seed=seed, hands=a.hands, bp=a.bp, ckpt=ck), None))
        for villain, tag in (("counter_exploiter", "hycounter"), ("blueprint", "hybpbot")):
            for seed in range(a.seeds):
                env = {"POKERBOT_BLUEPRINT_SNAPSHOT": a.bp} if villain == "blueprint" else None
                jobs.append((f"{tag}_s{seed}",
                             battery_cmd(RUNS / f"{tag}_s{seed}.json", hero="production",
                                         arch=villain, seed=seed, hands=a.hands, bp=a.bp,
                                         ckpt=ck), env))
        return jobs

    def run_hybrid(self) -> None:
        a = self.a
        jobs = self.hybrid_jobs()
        print(f"hybrid mode: {len(jobs)} jobs ({len(self._existing)} already in S3)", flush=True)
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            list(ex.map(lambda j: self.run_job(*j), jobs))
        summary = {"mode": "hybrid", "checkpoint": a.checkpoint, "pools": {}}
        for arch in ("hs_station", "hs_nit", "hs_lag", "hs_tag"):
            p = self.pool(f"hy_{arch}_s*.json", arch)
            if p:
                summary["pools"][f"hy_{arch}"] = p
        for tag, villain in (("hycounter", "counter_exploiter"), ("hybpbot", "blueprint")):
            p = self.pool(f"{tag}_s*.json", villain)
            if p:
                summary["pools"][tag] = p
        body = json.dumps(summary, indent=1, default=str)
        self.s3.put_object(Bucket=a.bucket, Key=f"{self.rk}/SUMMARY.json", Body=body.encode())
        if summary["pools"]:  # only mark DONE if SOMETHING succeeded — else let the babysitter relaunch
            self.s3.put_object(Bucket=a.bucket, Key=f"{a.prefix}/eval_suite_DONE.marker",
                               Body=time.strftime("%Y-%m-%d %H:%M:%S").encode())
            print("HYBRID DONE\n" + body, flush=True)
        else:
            print("HYBRID ALL-FAILED — no DONE marker (check FAIL_*.log)\n" + body, flush=True)

    def run(self) -> None:
        if getattr(self.a, "mode", "full") == "confirm":
            self.run_confirm()
            return
        if getattr(self.a, "mode", "full") == "hybrid":
            self.run_hybrid()
            return
        if getattr(self.a, "mode", "full") == "basestrength":
            self.run_basestrength()
            return
        if getattr(self.a, "mode", "full") == "valueextract":
            self.run_valueextract()
            return
        if getattr(self.a, "mode", "full") == "lbr6":
            self.run_lbr6()
            return
        a = self.a
        jobs = self.phase_a_jobs()
        print(f"phase A: {len(jobs)} jobs ({len(self._existing)} results already in S3)", flush=True)
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            list(ex.map(lambda j: self.run_job(*j), jobs))
        arm, table = self.gate()
        print(f"GATE: winner={arm} verdicts={json.dumps(table.get('verdicts'), default=str)}", flush=True)
        if arm is not None:
            jobs_b = self.phase_b_jobs(arm)
            print(f"phase B ({len(jobs_b)} jobs, arm {arm})", flush=True)
            with ThreadPoolExecutor(max_workers=a.workers) as ex:
                list(ex.map(lambda j: self.run_job(*j), jobs_b))
        # final summary: pool everything that exists
        summary = {"gate": {k: v for k, v in table.items() if k == "verdicts"}, "winner": arm,
                   "pools": {}}
        for pat, arch_list in (("eb3000_rmA_{a}_s*.json", ARCHES), ("eb3000_rmB_{a}_s*.json", ARCHES),
                               (f"eb3000_bluff{arm}_{{a}}_s*.json", ARCHES) if arm else ("", ()),
                               ):
            for arch in arch_list:
                p = self.pool(pat.format(a=arch), arch)
                if p:
                    summary["pools"][f"{pat.format(a=arch)}"] = p
        if arm:
            for tag, villain in (("weakmix", "weak_mixture"), ("counter", "counter_exploiter"),
                                 ("bpbot", "blueprint")):
                p = self.pool(f"eb3000_{tag}{arm}_s*.json", villain)
                if p:
                    summary["pools"][f"{tag}{arm}"] = p
        body = json.dumps(summary, indent=1, default=str)
        self.s3.put_object(Bucket=a.bucket, Key=f"{self.rk}/SUMMARY.json", Body=body.encode())
        self.s3.put_object(Bucket=a.bucket, Key=f"{a.prefix}/eval_suite_DONE.marker",
                           Body=time.strftime("%Y-%m-%d %H:%M:%S").encode())
        print("SUITE DONE\n" + body, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--prefix", required=True, help="run prefix, e.g. pokerbot/prod6000/eval_suite/run1")
    ap.add_argument("--region", default="eu-central-1")
    ap.add_argument("--workers", type=int, default=58)
    ap.add_argument("--seeds", type=int, default=18)
    ap.add_argument("--hands", type=int, default=3000)
    ap.add_argument("--bp", default="/tmp/bp6max_v2_iter200.pkl")
    ap.add_argument("--mode",
                    choices=("full", "confirm", "hybrid", "basestrength", "valueextract", "lbr6"),
                    default="full",
                    help="full = P0+sweep+priorAB+gate+phaseB; confirm = read-gated mixer ON vs "
                         "pure-search OFF (G1 verification); hybrid = HybridAgent production hero "
                         "exploit ON vs pure-GTO OFF vs the 4 fair archetypes + do-no-harm "
                         "(the live exploitation-vehicle validation); basestrength = ABSOLUTE AIVAT EV "
                         "of BOTH bases (SFT net + blueprint+search) vs weak/strong/realistic mixed "
                         "fields (the base-true-strength measurement for the base-routing decision)")
    ap.add_argument("--checkpoint", default="checkpoints/policy_sft_prod6000.pt",
                    help="SFT policy checkpoint for --mode hybrid (production hero base)")
    ap.add_argument("--bs-pluribus-configs", default="10:uniform",
                    help="basestrength STRONG-CONFIG sweep: comma list of <iters>:<leaf>, e.g. "
                         "'10:uniform,50:gadget,150:gadget'. Each becomes a base arm bp_i<iters>_<leaf>.")
    ap.add_argument("--bs-skip-sft", action="store_true",
                    help="basestrength: skip the SFT/production base (already measured) to halve the run")
    ap.add_argument("--bs-pooled-profiles", action="store_true",
                    help="basestrength: use ONE pooled opponent profile across all seats on the mixed "
                         "(heterogeneous) cells (the legacy artifact that made +exploit regress). Default "
                         "is PER-SEAT targeting (one profile per villain, hero reads the aggressor).")
    ap.add_argument("--ve-config", default="10:uniform",
                    help="valueextract: pluribus '<iters>:<leaf>' for the gauntlet hero (e.g. 10:uniform)")
    ap.add_argument("--lbr6-menu", choices=("fc", "fcpa", "full"), default="fcpa",
                    help="lbr6: graded action menu for the analytic LBR adversary (fc floor / fcpa / full)")
    ap.add_argument("--lbr6-rollouts", type=int, default=14,
                    help="lbr6: (legacy rollout mode) MC rollouts per candidate action per LBR decision")
    ap.add_argument("--lbr6-no-allin", action="store_true",
                    help="lbr6: drop all-in from LBR's menu (lower-variance 'reasonable' adversary)")
    ap.add_argument("--lbr6-passive", action="store_true",
                    help="lbr6: use the passive (weaker, valid) continuation instead of the competent one")
    ap.add_argument("--lbr6-target", choices=("blueprint", "blueprint-search"), default="blueprint",
                    help="lbr6: RAW blueprint lookup (default) or the DEPLOYED blueprint+SEARCH base (what "
                         "the bot actually plays). blueprint-search makes the seats run Pluribus per "
                         "decision while the LBR range-tracks against the cheap blueprint (tractable, valid).")
    ap.add_argument("--lbr6-search-leaf", choices=("uniform", "gadget"), default="uniform",
                    help="lbr6 blueprint-search: Pluribus continuation leaf (uniform = deployed/floor).")
    ap.add_argument("--lbr6-search-iters", type=int, default=10,
                    help="lbr6 blueprint-search: FIXED CFR+ iters per opponent solve (box-independent). "
                         "10 = the config the +9.1 basestrength base was measured at.")
    ap.add_argument("--lbr6-search-preflop", action="store_true",
                    help="lbr6 blueprint-search: ALSO search preflop (default: preflop served from the "
                         "blueprint table). Passes --search-preflop to lbr6_audit.py. Use the matching "
                         "POKERBOT_PREFLOP_LOSSLESS env for the snapshot being served.")
    ap.add_argument("--lbr6-single-menu", action="store_true",
                    help="lbr6: run ONLY --lbr6-menu instead of the graded fc/fcpa/full sweep (the "
                         "fc-first OLD-vs-NEW matrix runs a single menu per cell).")
    ap.add_argument("--job-timeout", type=int, default=4 * 3600)
    args = ap.parse_args()
    Suite(args).run()


if __name__ == "__main__":
    main()
