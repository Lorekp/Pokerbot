//! Bench: cfr_table.rs AoSoA index probe — the per-node hot path of SearchSession's `cfv`.
//!
//! Builds a fixed population of `n_keys` distinct infoset keys (mimicking the SearchSession's
//! working set: ~10k distinct keys for a 6-max postflop subtree), then runs `iters` read+write
//! cycles with FNV-keyed probing. Reports wall + reads/sec + writes/sec.
//!
//! Usage: cargo run --release --example bench_cfr_table -- <n_keys> <iters>

use std::time::Instant;

use battery_native::cfr_table::{infoset_key_u64, CfrTable};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_keys: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10_000);
    let iters: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200);

    // Deterministic key population: n_keys distinct 6-tuple i64 infoset keys, plus their
    // pre-folded u64 scalar keys (so the bench does not measure the FNV fold — it measures the
    // u64-keyed index probe itself, which is the lever).
    let mut vec_keys: Vec<Vec<i64>> = Vec::with_capacity(n_keys);
    let mut u64_keys: Vec<u64> = Vec::with_capacity(n_keys);
    for i in 0..n_keys {
        let mut state = i as u64;
        let mut k = Vec::with_capacity(6);
        for _ in 0..6 {
            state = state
                .wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add(0x123456789ABCDEF0);
            k.push((state >> 33) as i64);
        }
        u64_keys.push(infoset_key_u64(&k));
        vec_keys.push(k);
    }

    // Pre-populate the table so reads HIT (the realistic hot-path case).
    let mut tab = CfrTable::new_with_capacity(n_keys);
    for (i, &u) in u64_keys.iter().enumerate() {
        let mut row = [0.0f64; 11];
        for a in 0..11 {
            row[a] = (i * 11 + a) as f64;
        }
        tab.insert(u, row);
    }
    assert_eq!(tab.n_rows(), n_keys);

    // Warm-up (allocator transient).
    for _ in 0..3 {
        let mut total = 0.0f64;
        for &u in &u64_keys {
            total += tab.get_or_zero(u)[0];
        }
        for &u in &u64_keys {
            let mut row = tab.get_or_zero(u);
            row[0] += 1.0;
            tab.insert(u, row);
        }
        std::hint::black_box(total);
    }

    // Bench: best-of-5 of (iters reads + iters writes) over the working set.
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let t0 = Instant::now();
        let mut total = 0.0f64;
        for it in 0..iters {
            for &u in &u64_keys {
                let row = tab.get_or_zero(u);
                total += row[(it as usize) % 11];
            }
            for &u in &u64_keys {
                let mut row = tab.get_or_zero(u);
                row[0] += 1.0;
                tab.insert(u, row);
            }
        }
        let dt = t0.elapsed().as_secs_f64();
        if dt < best {
            best = dt;
        }
        std::hint::black_box(total);
    }

    let probes = iters as f64 * n_keys as f64;
    let rps = probes / best;
    let wps = probes / best;
    eprintln!(
        "BEST-OF-5 n_keys={n_keys} iters={iters} wall={:.3}s  reads/s={:.2e}  writes/s={:.2e}",
        best, rps, wps
    );

    // Touch the fold so it doesn't get DCE'd (and so we exercise the production fold path).
    let mut hsum: u64 = 0;
    for k in &vec_keys {
        hsum ^= infoset_key_u64(k);
    }
    println!("{:x}", hsum);
}
