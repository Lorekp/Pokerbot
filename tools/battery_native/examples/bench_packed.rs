//! EMD-on bench with a packed u64 memo key — alternative to the [i32;7] memo for tighter hot path.
//! NOT a real test. Try: pack (h0, h1, b0, b1, b2, b3) into 8*6=48 bits => u64.
use std::env;
use std::sync::Arc;
use std::time::Instant;

use battery_native::emd::FrozenEMDLookup;
use battery_native::trainer::{allowed_mask, NlheTables, TrainerCfg};
use rustc_hash::FxHashMap as HashMap;
use std::cell::RefCell;

type EmdBucketMemo = HashMap<u64, Option<i32>>;
thread_local! {
    static EMD_BUCKET_MEMO: RefCell<EmdBucketMemo> =
        RefCell::new(EmdBucketMemo::with_capacity_and_hasher(1 << 19, Default::default()));
}

#[inline]
fn pack_card(c: i32) -> u64 {
    if c < 0 {
        0x3F
    } else {
        (c as u64) & 0x3F
    }
}

fn emd_bucket_memoed(lookup: &FrozenEMDLookup, hole: &[i32; 2], board: &[i32]) -> Option<i32> {
    // Pack 7 cards into 48 bits of a u64. Cards are 0..51 (6 bits each).
    let mut key: u64 = 0;
    // Sort hole by id
    let (mut h0, mut h1) = (hole[0], hole[1]);
    if h1 < h0 {
        let t = h0;
        h0 = h1;
        h1 = t;
    }
    // Collect valid board cards
    let mut bs = [0i32; 5];
    let mut bn = 0usize;
    for &c in board {
        if c >= 0 {
            bs[bn] = c;
            bn += 1;
        }
    }
    // Sort valid board cards ascending
    if bn >= 2 {
        let mut b = &mut bs[..bn];
        b.sort_unstable();
    }
    // Pack (h0, h1, b0, b1, b2, b3, -1 pad) into 8 nibbles of 6 bits = 48 bits
    key |= pack_card(h0);
    key |= pack_card(h1) << 6;
    key |= pack_card(bs[0]) << 12;
    key |= pack_card(bs[1]) << 18;
    key |= pack_card(bs[2]) << 24;
    key |= pack_card(bs[3]) << 30;
    key |= pack_card(if bn >= 4 { bs[4] } else { -1 }) << 36;

    EMD_BUCKET_MEMO.with(|m| {
        let map = &mut *m.borrow_mut();
        if let Some(&cached) = map.get(&key) {
            return cached;
        }
        let v = lookup.bucket(&[h0, h1], board);
        map.insert(key, v);
        v
    })
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let iters: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(500);
    let n_deals: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(64);
    let frozen_path = args.get(3).cloned().unwrap_or_default();

    let emd_frozen = if frozen_path.is_empty() {
        None
    } else {
        let l = FrozenEMDLookup::load(&frozen_path).expect("load emd frozen");
        println!("loaded frozen: K={} spots={}", l.num_buckets, l.len());
        Some(Arc::new(l))
    };

    let cfg = TrainerCfg {
        num_buckets: 500,
        depth_edges: vec![],
        allowed_pre: allowed_mask(&[4, 5, 6]),
        allowed_post: allowed_mask(&[0, 1, 3, 4, 5]),
        n_deals_per_iter: n_deals,
        base_seed: 0xC0FFEE_u64,
        n_seats: 6,
        prune_threshold: f64::NEG_INFINITY,
        fast_merge: false,
        emd_frozen,
        batch_deals: 0,
        work_steal: false,
        dcfr: false,
        dcfr_prune_threshold: -300_000.0,
        dcfr_explore_frac: 0.05,
        river_ochs: None,
    };

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
