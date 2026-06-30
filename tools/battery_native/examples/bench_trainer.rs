//! A/B benchmark + byte-identity harness for the native CFR+ trainer hot path.
//!
//! Trains `NlheTables` to a FIXED config/seed/iters, writes the serialized blob to a file
//! (for an external SHA-256 byte-identity check) and prints wall time + it/s. Not shipped;
//! a dev tool for the optimization pass only.
//!
//! Usage: cargo run --release --example bench_trainer -- <iters> <n_deals> <out_path>

use std::time::Instant;

use battery_native::trainer::{allowed_mask, serialize, NlheTables, TrainerCfg};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let iters: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let n_deals: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(64);
    let out_path = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "bench_blob.bin".to_string());

    // RBP threshold via env (PRUNE_C); default -inf disables (byte-identical no-RBP path).
    let prune_threshold = std::env::var("PRUNE_C")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(f64::NEG_INFINITY);
    // Parallel merge via env (FAST_MERGE=1); default false (byte-identical serial merge).
    let fast_merge = std::env::var("FAST_MERGE")
        .map(|s| s == "1")
        .unwrap_or(false);

    let cfg = TrainerCfg {
        num_buckets: 200,
        depth_edges: vec![],
        allowed_pre: allowed_mask(&[4, 5, 6]),
        allowed_post: allowed_mask(&[0, 1, 3, 4, 5]),
        n_deals_per_iter: n_deals,
        base_seed: 0xC0FFEE_u64,
        n_seats: 6,
        prune_threshold,
        fast_merge,
        emd_frozen: None,
        batch_deals: 0,
        work_steal: false,
        dcfr: false,
        dcfr_prune_threshold: -300_000.0,
        dcfr_explore_frac: 0.05,
    };

    // Warm-up iteration excluded from timing (allocator / table growth transient).
    let mut tab = NlheTables::new();
    tab.train_iteration(&cfg);

    // Best-of-5 wall time over the SAME fixed iters (the trainer is deterministic given the table
    // state, but each repeat continues from a larger table — so we time a fresh table each repeat to
    // keep the per-iter work comparable, and report the best (least-noisy) run).
    let mut best_dt = f64::INFINITY;
    let mut blob = Vec::new();
    let mut infosets = 0usize;
    for _rep in 0..5 {
        let mut t = NlheTables::new();
        t.train_iteration(&cfg); // warm-up (excluded)
        let t0 = Instant::now();
        for _ in 0..iters {
            t.train_iteration(&cfg);
        }
        let dt = t0.elapsed().as_secs_f64();
        if dt < best_dt {
            best_dt = dt;
            blob = serialize(&t);
            infosets = t.n_strat_infosets();
        }
    }
    let dt = best_dt;

    std::fs::write(&out_path, &blob).expect("write blob");

    let total_iters = iters; // not counting warm-up
    let ips = total_iters as f64 / dt;
    let deals = total_iters as f64 * n_deals as f64;
    let dps = deals / dt;
    eprintln!(
        "BEST-OF-5 iters={total_iters} n_deals={n_deals} wall={dt:.3}s  it/s={ips:.2}  deals/s={dps:.0}  infosets={infosets}  blob_bytes={}",
        blob.len()
    );
    println!("{}", out_path);
}
