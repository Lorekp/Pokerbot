"""Regression tests for the presum K-COV gate (scripts/blueprint2/bpv4_kcov.py, design amendment 36).

Pins the pre-registered thresholds against the grounding contrast (artifacts/_kcov_grounding_ab.json):
fullhist raised-pot miss ~95% must EARLY-KILL; presum ~25% miss (~75% coverage) must PASS. Also pins
the COVERAGE-line parser (both the `iter=` ab_train.exe spelling and the `iters_done=` cloud-driver
spelling), the nearest-<= lookup, and the INSUFFICIENT_DATA fail-open.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts", "blueprint2"))
import bpv4_kcov as kcov  # noqa: E402

# A realistic train log: ab_train.exe (`iter=`) + cloud driver (`iters_done=`) lines interleaved,
# fullhist-like early misses trending down to a presum-like converged coverage.
LOG = """\
[ab_train] COVERAGE iter=5000 hands=4000 miss% pre=22.2 flop=90.9 turn=95.7 river=100.0 postflop=91.3 raised_postflop=95.3 raised_flop=94.9 raised_turn=100.0 raised_river=100.0 raised_decisions=3600 pre_allin_runout=84.4 decisions=[32020, 396, 23, 8] wall_s=0.3
COVERAGE iters_done=7500 hands=20000 miss% pre=20.0 flop=40.0 turn=60.0 river=80.0 postflop=44.0 raised_postflop=41.0 raised_flop=38.0 raised_turn=62.0 raised_river=80.0 raised_decisions=5200 pre_allin_runout=60.0 decisions=[160000, 5200, 900, 220]
COVERAGE iters_done=10000 hands=20000 miss% pre=18.0 flop=22.0 turn=40.0 river=70.0 postflop=27.0 raised_postflop=24.8 raised_flop=22.3 raised_turn=41.0 raised_river=70.0 raised_decisions=5400 pre_allin_runout=55.0 decisions=[170000, 5400, 950, 240]
"""


def test_parse_accepts_both_spellings_and_sorts():
    cov = kcov.parse_coverage(LOG)
    assert [c["iters"] for c in cov] == [5000, 7500, 10000]
    assert cov[0]["raised_postflop_miss"] == 95.3
    assert cov[0]["raised_decisions"] == 3600
    assert cov[2]["raised_postflop_miss"] == 24.8
    assert cov[1]["pre_allin_runout"] == 60.0


def test_parse_from_file(tmp_path):
    p = tmp_path / "train.log"
    p.write_text(LOG, encoding="utf-8")
    cov = kcov.parse_coverage(str(p))
    assert len(cov) == 3 and cov[-1]["iters"] == 10000


def test_cov_at_exact_and_nearest():
    cov = kcov.parse_coverage(LOG)
    assert kcov.cov_at(cov, 10000)["iters"] == 10000
    assert kcov.cov_at(cov, 9000)["iters"] == 7500   # nearest <=
    assert kcov.cov_at(cov, 4000) is None            # nothing <= 4000
    assert kcov.cov_at(cov, None)["iters"] == 10000  # latest
    assert kcov.cov_at([], 10000) is None


def test_fullhist_early_kills_at_k1a():
    cov = kcov.parse_coverage(LOG)
    r = kcov.evaluate_kcov(kcov.cov_at(cov, 5000), None, k1a_regime=True)
    assert r["status"] == "KILL" and r["hard"] is True and r["passed"] is False


def test_presum_passes_at_k1a_and_k1():
    cov = kcov.parse_coverage(LOG)
    latest, prior = kcov.cov_at(cov, 10000), kcov.cov_at(cov, 7500)
    # @K1a: 24.8% miss < 80% early-kill and coverage 75.2% >= 50% -> PASS, not a hard clause.
    r_k1a = kcov.evaluate_kcov(latest, prior, k1a_regime=True)
    assert r_k1a["status"] == "PASS" and r_k1a["hard"] is False
    assert r_k1a["detail"]["drift_ok"] is True  # 41.0 -> 24.8 fell 16.2pt >= 5pt
    # @K1: cap is hard; 24.8 <= 50 -> PASS hard.
    r_k1 = kcov.evaluate_kcov(latest, prior, k1a_regime=False)
    assert r_k1["status"] == "PASS" and r_k1["hard"] is True and r_k1["passed"] is True


def test_fullhist_fails_binding_k1():
    cov = kcov.parse_coverage(LOG)
    r = kcov.evaluate_kcov(kcov.cov_at(cov, 5000), None, k1a_regime=False)
    assert r["status"] == "KILL" and r["hard"] is True and r["passed"] is False


def test_insufficient_sample_never_folds():
    low = {"iters": 10000, "raised_postflop_miss": 95.0, "raised_decisions": 100,
           "postflop_miss": 95.0, "pre_allin_runout": 80.0}
    r = kcov.evaluate_kcov(low, None, k1a_regime=True)
    assert r["status"] == "INSUFFICIENT_DATA" and r["hard"] is False and r["passed"] is None


def test_no_data_is_advisory():
    r = kcov.evaluate_kcov(None, None, k1a_regime=False)
    assert r["status"] == "INSUFFICIENT_DATA" and r["hard"] is False


def test_k1a_warn_band_never_folds():
    # cap < miss < early_kill (design 36: a converging-but-not-yet-there run). @K1a this must be
    # advisory (hard=False, passed=True) so a converging run is NOT spuriously early-killed.
    mid = {"iters": 10000, "raised_postflop_miss": 60.0, "raised_decisions": 5000,
           "postflop_miss": 62.0, "pre_allin_runout": 55.0}
    r = kcov.evaluate_kcov(mid, None, k1a_regime=True)
    assert r["status"] == "WARN" and r["hard"] is False and r["passed"] is True
    # but the SAME 60% miss at K1 (binding cap 50%) is a hard KILL.
    r_k1 = kcov.evaluate_kcov(mid, None, k1a_regime=False)
    assert r_k1["status"] == "KILL" and r_k1["hard"] is True and r_k1["passed"] is False


def test_raised_decisions_does_not_false_match_iters():
    # The `\b` anchor in _ITERS_RE must not let `raised_decisions=` be read as the iters token. A line
    # with no real iters token parses to nothing; a canonical line's iters is the real value, not 5400.
    assert kcov.parse_coverage("COVERAGE raised_postflop=95.3 raised_decisions=5400") == []
    one = kcov.parse_coverage(
        "COVERAGE iters_done=10000 hands=20000 miss% pre=1 flop=1 turn=1 river=1 postflop=1 "
        "raised_postflop=24.8 raised_flop=1 raised_turn=1 raised_river=1 raised_decisions=5400 "
        "pre_allin_runout=1 decisions=[1]")
    assert len(one) == 1 and one[0]["iters"] == 10000 and one[0]["raised_decisions"] == 5400


def test_min_sample_sizing_calibration():
    # design-36 calibration: raised-pot density is ~0.07/hand on jam-heavy runs, so COV_HANDS must be
    # ~40000 (not 20000) to clear MIN_SAMPLE=2000. Pin the boundary that motivates the recommendation:
    # the 20000-hand projection (~1430 raised decisions) goes INSUFFICIENT_DATA even at a broken 90%
    # miss (=> gate cannot bind => the recommended sample MUST be larger); the 40000-hand projection
    # (~2860) binds and KILLS the same broken run.
    broken_20k = {"iters": 10000, "raised_postflop_miss": 90.0, "raised_decisions": 1430,
                  "postflop_miss": 90.0, "pre_allin_runout": 84.0}
    r20 = kcov.evaluate_kcov(broken_20k, None, k1a_regime=False)
    assert r20["status"] == "INSUFFICIENT_DATA" and r20["hard"] is False  # dark: cannot kill
    broken_40k = {**broken_20k, "raised_decisions": 2860}
    r40 = kcov.evaluate_kcov(broken_40k, None, k1a_regime=False)
    assert r40["status"] == "KILL" and r40["hard"] is True and r40["passed"] is False  # binds


def test_thresholds_are_overridable():
    # A stricter cap flips a marginal presum read from PASS to KILL at K1.
    latest = {"iters": 10000, "raised_postflop_miss": 45.0, "raised_decisions": 5000,
              "postflop_miss": 47.0, "pre_allin_runout": 55.0}
    assert kcov.evaluate_kcov(latest, None, k1a_regime=False)["passed"] is True
    assert kcov.evaluate_kcov(latest, None, k1a_regime=False, cap_pct=40.0)["passed"] is False
