//! f32_regret strategy-delta A/B harness (local, free). Trains the 6-max CFR+ blueprint for a fixed
//! number of iterations at a FIXED seed and writes the NLCFR001 checkpoint to `argv[1]`. Build it once
//! WITHOUT `--features f32_regret` (f64 baseline) and once WITH it (f32 regret rows): the two runs share
//! an identical deal/RNG schedule and cfg, so the ONLY difference is the persistent regret precision.
//! `scripts/f32_regret_strategy_delta.py` then parses both blobs and reports the per-infoset average-
//! strategy L1 delta against the verdict's ship-gate thresholds (median<=1e-4, max<=1e-2, no edge-bearing
//! line collapse). This is the cheap, real-6-max preliminary to the full cloud T=62000 ship-gate.
//!
//! NOTE: pure-Rust example (no `python` feature) — uses the public trainer API only. The on-disk blob is
//! f64 for BOTH builds (cross-precision-portable), so the Python diff needs no native import.

use battery_native::trainer::{allowed_mask, serialize, NlheTables, TrainerCfg};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out_path = args.get(1).map(|s| s.as_str()).unwrap_or("ab_train_out.bin");
    let iters: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4000);
    let n_deals: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);
    // argv[4] (or env POKERBOT_BATCH_DEALS): batched-MLP traversal width. 0/1 = recursive (default);
    // >=2 = lockstep batched. Used for the local nodes/s A/B (batch_deals 8 vs 1).
    let batch_deals: usize = args
        .get(4)
        .and_then(|s| s.parse().ok())
        .or_else(|| std::env::var("POKERBOT_BATCH_DEALS").ok().and_then(|s| s.parse().ok()))
        .unwrap_or(0);

    // Representative deployed cfg (mirrors trainer::tests::test_cfg) — made-hand bucketer (emd_frozen
    // None) for speed; the regret-precision A/B is abstraction-agnostic (both arms use the identical cfg).
    let cfg = TrainerCfg {
        num_buckets: 200,
        depth_edges: vec![],
        allowed_pre: allowed_mask(&[4, 5, 6]),
        allowed_post: allowed_mask(&[0, 1, 3, 4, 5]),
        n_deals_per_iter: n_deals,
        base_seed: 0xC0FFEE,
        n_seats: 6,
        prune_threshold: f64::NEG_INFINITY,
        fast_merge: false,
        emd_frozen: None,
        batch_deals,
    };

    let precision = if cfg!(feature = "f32_regret") {
        "f32_regret"
    } else {
        "f64"
    };
    eprintln!(
        "[ab_train] precision={precision} iters={iters} n_deals={n_deals} batch_deals={batch_deals} -> {out_path}",
    );

    let mut tab = NlheTables::new();
    let t0 = std::time::Instant::now();
    let mut last = t0;
    for i in 1..=iters {
        tab.train_iteration(&cfg);
        // Heartbeat ~every 2s so a long run is visibly progressing (the user's logging rule).
        if last.elapsed().as_secs_f64() >= 2.0 {
            let done = i as f64 / iters as f64;
            let rate = i as f64 / t0.elapsed().as_secs_f64();
            eprintln!(
                "[ab_train] {i}/{iters} ({:.1}%) {:.1} iters/s strat_rows={}",
                done * 100.0,
                rate,
                tab.n_strat_infosets()
            );
            last = std::time::Instant::now();
        }
    }
    let blob = serialize(&tab);
    std::fs::write(out_path, &blob).expect("write blob");
    eprintln!(
        "[ab_train] DONE precision={precision} iters={iters} strat_rows={} bytes={} in {:.1}s",
        tab.n_strat_infosets(),
        blob.len(),
        t0.elapsed().as_secs_f64()
    );
}
