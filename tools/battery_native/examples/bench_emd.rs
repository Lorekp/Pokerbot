//! EMD-on bench: same hot loop as bench_trainer but with the EMD frozen lookup wired.
//! NOT a real test (the .pyd loader on this box is broken); the bench binary that loads the
//! .pyd via CPython is the production path. This Rust-only bench is a fallback for boxes
//! where the Python loader fails.
use std::env;
use std::sync::Arc;
use std::time::Instant;

use battery_native::emd::FrozenEMDLookup;
use battery_native::trainer::{allowed_mask, NlheTables, TrainerCfg};

fn main() {
    let args: Vec<String> = env::args().collect();
    let iters: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(500);
    let n_deals: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(64);
    let frozen_path = args.get(3).cloned().unwrap_or_default();
    let requested_num_buckets: Option<i32> = args.get(4).and_then(|s| s.parse().ok());

    let (emd_frozen, num_buckets) = if frozen_path.is_empty() {
        (None, requested_num_buckets.unwrap_or(200))
    } else {
        let l = FrozenEMDLookup::load_path(&frozen_path).expect("load emd frozen");
        println!(
            "loaded frozen: K={} n_bins={} n_runouts={} spots={}",
            l.n_buckets,
            l.n_bins,
            l.n_runouts,
            l.rows.len()
        );
        let k = requested_num_buckets.unwrap_or(l.n_buckets as i32);
        assert_eq!(
            k as usize, l.n_buckets,
            "num_buckets argument must match frozen lookup K"
        );
        (Some(Arc::new(l)), k)
    };

    let cfg = TrainerCfg {
        num_buckets,
        depth_edges: vec![],
        allowed_pre: allowed_mask(&[4, 5, 6]),
        allowed_post: allowed_mask(&[0, 1, 3, 4, 5]),
        n_deals_per_iter: n_deals,
        base_seed: 0xC0FFEE_u64,
        n_seats: 6,
        prune_threshold: f64::NEG_INFINITY,
        fast_merge: false,
        emd_frozen: emd_frozen.clone(),
        batch_deals: 0,
    };

    // Warm-up
    let mut tab = NlheTables::new();
    tab.train_iteration(&cfg);

    let mut best_dt = f64::INFINITY;
    let mut infosets = 0;
    for _ in 0..5 {
        let mut t = NlheTables::new();
        t.train_iteration(&cfg);
        let t0 = Instant::now();
        for _ in 0..iters {
            t.train_iteration(&cfg);
        }
        let dt = t0.elapsed().as_secs_f64();
        if dt < best_dt {
            best_dt = dt;
            infosets = t.n_strat_infosets();
        }
    }
    println!(
        "BEST-OF-5 iters={} n_deals={} wall={:.3}s it/s={:.1} infosets={}",
        iters,
        n_deals,
        best_dt,
        iters as f64 / best_dt,
        infosets
    );
}
