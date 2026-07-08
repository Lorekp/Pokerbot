//! blueprint-v4 blocker D: the standalone K-COV (opener-faced raised-pot on-policy coverage)
//! instrument. Loads a checkpoint read-only and runs `trainer::coverage_probe` — the SAME on-policy
//! probe the trainer emits at checkpoint cadence (the `[ab_train] COVERAGE ...` line) — then writes a
//! JSON with the raised-pot ("opener-faced", preflop n_raises>=1) postflop coverage subclass. Use it
//! to characterize an EXISTING checkpoint offline (e.g. the fullhist baseline snapshot, to pin the
//! pathology on the real run) or ad-hoc; the funded-run gate reads the trainer's COVERAGE line
//! instead, so this needs no live trainer.
//!
//! The key space and abstraction MUST match the run being probed — set the SAME process envs as the
//! run (POKERBOT_FULLHIST_KEY / POKERBOT_PRESUM_KEY / POKERBOT_MRR / POKERBOT_AB_NUM_BUCKETS /
//! POKERBOT_AB_EMD_FROZEN / POKERBOT_AB_RIVER_OCHS / POKERBOT_PRE_KEEP / POKERBOT_POST_KEEP), exactly
//! as `ab_train` / `row_census` do; a mismatch reads a wrong key space and every lookup misses.
//! coverage_probe plays on the CURRENT strategy (cum_regret), so this expects a training checkpoint
//! (.nlcfr with regrets), not a final average-only snapshot (.nlsnap) which carries no cum_regret.
//!
//! Usage: kcov_probe.exe <ckpt.nlcfr> <n_hands> <out.json> [seed]

use battery_native::emd;
use battery_native::river_ochs::RiverOchs;
use battery_native::trainer::{coverage_probe, deserialize, keep_mask_from_env, TrainerCfg};
use std::sync::Arc;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let ckpt = args
        .get(1)
        .expect("usage: kcov_probe <ckpt.nlcfr> <n_hands> <out.json> [seed]");
    let n_hands: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4000);
    let out_path = args.get(3).map(|s| s.as_str()).unwrap_or("kcov_probe.json");
    let seed: u64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0xC0FFEE);

    let num_buckets: i32 = std::env::var("POKERBOT_AB_NUM_BUCKETS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let emd_frozen = match std::env::var("POKERBOT_AB_EMD_FROZEN") {
        Ok(p) if !p.is_empty() => Some(emd::load_shared(&p).expect("load emd")),
        _ => None,
    };
    let river_ochs = match std::env::var("POKERBOT_AB_RIVER_OCHS") {
        Ok(p) if !p.is_empty() => Some(Arc::new(RiverOchs::load_path(&p).expect("load ochs"))),
        _ => None,
    };
    // Same cfg shape as row_census/ab_train: the arm-3 menu envs (absent => byte-identical legacy),
    // so abstracted_legal in the probe matches the run's menus. The M-RR / key envs are read
    // process-globally inside coverage_probe's fingerprint + abstracted_legal dispatch.
    let cfg = TrainerCfg {
        num_buckets,
        depth_edges: vec![],
        allowed_pre: keep_mask_from_env("POKERBOT_PRE_KEEP", &[4, 5, 6]),
        allowed_post: keep_mask_from_env("POKERBOT_POST_KEEP", &[0, 1, 3, 4, 5]),
        n_deals_per_iter: 200,
        base_seed: 0xC0FFEE,
        n_seats: 6,
        prune_threshold: f64::NEG_INFINITY,
        fast_merge: false,
        emd_frozen,
        batch_deals: 0,
        work_steal: false,
        dcfr: false,
        dcfr_prune_threshold: -300_000.0,
        dcfr_explore_frac: 0.05,
        river_ochs,
    };

    let fullhist = std::env::var("POKERBOT_FULLHIST_KEY").map(|v| v == "1").unwrap_or(false);
    let presum = std::env::var("POKERBOT_PRESUM_KEY").map(|v| v == "1").unwrap_or(false);
    let mrr = std::env::var("POKERBOT_MRR").map(|v| v == "1" || v.eq_ignore_ascii_case("true")).unwrap_or(false);
    eprintln!(
        "[kcov_probe] loading {ckpt} ... fullhist={fullhist} presum={presum} mrr={mrr} \
         num_buckets={num_buckets} n_hands={n_hands}"
    );
    let blob = std::fs::read(ckpt).expect("read ckpt");
    let tab = deserialize(&blob).expect("deserialize");
    drop(blob);

    let t0 = std::time::Instant::now();
    let c = coverage_probe(&tab, &cfg, n_hands, seed);
    let wall = t0.elapsed().as_secs_f64();

    // Coverage = 1 - miss. raised_* is the K-COV headline (opener-faced raised pots only).
    let raised_cov = 100.0 * (1.0 - c.raised_postflop_miss_rate());
    let post_cov = 100.0 * (1.0 - c.postflop_miss_rate());
    let json = format!(
        "{{\"checkpoint\":\"{}\",\"n_hands\":{},\"seed\":{},\"fullhist\":{},\"presum\":{},\"mrr\":{},\
         \"num_buckets\":{},\"hands\":{},\"decisions\":[{},{},{},{}],\"misses\":[{},{},{},{}],\
         \"raised_decisions\":[{},{},{},{}],\"raised_misses\":[{},{},{},{}],\
         \"postflop_miss_pct\":{:.2},\"postflop_coverage_pct\":{:.2},\
         \"raised_postflop_miss_pct\":{:.2},\"raised_postflop_coverage_pct\":{:.2},\
         \"raised_flop_miss_pct\":{:.2},\"raised_turn_miss_pct\":{:.2},\"raised_river_miss_pct\":{:.2},\
         \"raised_postflop_decisions\":{},\"preflop_allin_runout_pct\":{:.2},\"wall_s\":{:.1}}}",
        ckpt.replace('\\', "/"),
        n_hands, seed, fullhist, presum, mrr, num_buckets, c.hands,
        c.decisions[0], c.decisions[1], c.decisions[2], c.decisions[3],
        c.misses[0], c.misses[1], c.misses[2], c.misses[3],
        c.raised_decisions[0], c.raised_decisions[1], c.raised_decisions[2], c.raised_decisions[3],
        c.raised_misses[0], c.raised_misses[1], c.raised_misses[2], c.raised_misses[3],
        100.0 * c.postflop_miss_rate(), post_cov,
        100.0 * c.raised_postflop_miss_rate(), raised_cov,
        100.0 * c.raised_miss_rate(1), 100.0 * c.raised_miss_rate(2), 100.0 * c.raised_miss_rate(3),
        c.raised_postflop_decisions(), 100.0 * c.preflop_allin_runout_rate(), wall,
    );
    std::fs::write(out_path, &json).expect("write out");
    eprintln!("[kcov_probe] DONE {json} -> {out_path}");
}
