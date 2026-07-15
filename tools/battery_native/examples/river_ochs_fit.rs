//! Fit the blueprint-v2 river OCHS abstraction table (`data/abstraction/river_ochs_160.rochs`).
//!
//!   cargo run --release --example river_ochs_fit -- \
//!       --k 160 --spots 250000 --kmeans-iters 30 --seed 7 \
//!       --out ../../data/abstraction/river_ochs_160.rochs
//!
//! Steps (all deterministic given --seed; rayon only parallelizes embarrassingly-parallel maps
//! whose outputs are written to indexed slots, so thread count does not change the result):
//!   1. Per-preflop-class equity vs a random opponent hand + random runout (MC, N per class) —
//!      the 1-D feature Johanson's OCHS clusters opponents by.
//!   2. 1-D k-means (k=8) over the 169 class equities, weighted by combo count; deterministic
//!      quantile init + Lloyd to convergence -> class_to_cluster[169].
//!   3. Sample --spots random river spots (hole 2 + board 5), compute the OCHS 8-vector of each
//!      (exhaustive 990-opponent-combo win rates per cluster; `river_ochs::RiverOchs::ochs_vector`).
//!   4. 8-D k-means (K = --k) over the spot vectors: deterministic spread init (every
//!      spots/K-th distinct vector), Lloyd --kmeans-iters, empty clusters reseeded to the point
//!      farthest from its centroid.
//!   5. Write the ROCHS001 blob and a golden TSV (512 held-out spots re-bucketed through the
//!      FINAL table) for the Rust+Python parity tests. The .sha256 sidecar is written by the
//!      wrapper (scripts/blueprint2/fit_river_ochs.py).
//!
//! Progress: prints % + ETA + heartbeat per the standing job-logging rule.

use battery_native::fingerprint::canonical_preflop_class;
use battery_native::fingerprint::evaluate_cards;
use battery_native::playout::Rng;
use battery_native::river_ochs::{RiverOchs, N_CLUSTERS, N_PREFLOP_CLASSES};
use rayon::prelude::*;
use std::io::Write;
use std::time::Instant;

fn arg<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    for i in 0..args.len() {
        if args[i] == name && i + 1 < args.len() {
            if let Ok(v) = args[i + 1].parse::<T>() {
                return v;
            }
        }
    }
    default
}

fn arg_s(args: &[String], name: &str, default: &str) -> String {
    for i in 0..args.len() {
        if args[i] == name && i + 1 < args.len() {
            return args[i + 1].clone();
        }
    }
    default.to_string()
}

/// All combos (c0 < c1) of a canonical preflop class id.
fn class_combos(cls: i32) -> Vec<[i32; 2]> {
    let mut out = Vec::new();
    for c0 in 0..52i32 {
        for c1 in (c0 + 1)..52i32 {
            if canonical_preflop_class(c0, c1) == cls {
                out.push([c0, c1]);
            }
        }
    }
    out
}

/// MC equity of `hole` vs one random opponent hand on one random full runout.
fn class_equity(cls: i32, n_mc: usize, seed: u64) -> f64 {
    let combos = class_combos(cls);
    let mut rng = Rng::seed_from_u64(seed ^ (cls as u64).wrapping_mul(0x9E37_79B9));
    let mut half: u64 = 0;
    for t in 0..n_mc {
        let hole = combos[t % combos.len()];
        // deal 2 opp + 5 board from the remaining 50
        let mut deck: Vec<i32> = (0..52).filter(|c| *c != hole[0] && *c != hole[1]).collect();
        for i in 0..7 {
            let j = i + rng.next_below((deck.len() - i) as u64) as usize;
            deck.swap(i, j);
        }
        let opp = [deck[0], deck[1]];
        let board = [deck[2], deck[3], deck[4], deck[5], deck[6]];
        let mut h7 = [
            hole[0], hole[1], board[0], board[1], board[2], board[3], board[4],
        ];
        let hr = evaluate_cards(&h7);
        h7[0] = opp[0];
        h7[1] = opp[1];
        let or = evaluate_cards(&h7);
        if hr < or {
            half += 2;
        } else if hr == or {
            half += 1;
        }
    }
    half as f64 / (2 * n_mc) as f64
}

/// Deterministic 1-D k-means (k=8) over class equities, weighted by combo count.
fn cluster_classes(eq: &[f64; N_PREFLOP_CLASSES]) -> [u8; N_PREFLOP_CLASSES] {
    let w: Vec<f64> = (0..N_PREFLOP_CLASSES as i32)
        .map(|c| class_combos(c).len() as f64)
        .collect();
    // init: weighted quantile midpoints over the sorted equities
    let mut order: Vec<usize> = (0..N_PREFLOP_CLASSES).collect();
    order.sort_by(|&a, &b| eq[a].partial_cmp(&eq[b]).unwrap().then(a.cmp(&b)));
    let total_w: f64 = w.iter().sum();
    let mut cents = [0.0f64; N_CLUSTERS];
    for (k, cent) in cents.iter_mut().enumerate() {
        let target = total_w * (k as f64 + 0.5) / N_CLUSTERS as f64;
        let mut acc = 0.0;
        let mut val = eq[order[0]];
        for &i in &order {
            acc += w[i];
            val = eq[i];
            if acc >= target {
                break;
            }
        }
        *cent = val;
    }
    let mut assign = [0u8; N_PREFLOP_CLASSES];
    for _ in 0..200 {
        let mut changed = false;
        for i in 0..N_PREFLOP_CLASSES {
            let mut best = 0usize;
            let mut bd = f64::INFINITY;
            for (k, &c) in cents.iter().enumerate() {
                let d = (eq[i] - c) * (eq[i] - c);
                if d < bd {
                    bd = d;
                    best = k;
                }
            }
            if assign[i] != best as u8 {
                assign[i] = best as u8;
                changed = true;
            }
        }
        let mut num = [0.0f64; N_CLUSTERS];
        let mut den = [0.0f64; N_CLUSTERS];
        for i in 0..N_PREFLOP_CLASSES {
            num[assign[i] as usize] += w[i] * eq[i];
            den[assign[i] as usize] += w[i];
        }
        for k in 0..N_CLUSTERS {
            if den[k] > 0.0 {
                cents[k] = num[k] / den[k];
            }
        }
        if !changed {
            break;
        }
    }
    // Relabel clusters by ascending centroid (stable, human-readable: 0 = weakest octile).
    let mut idx: Vec<usize> = (0..N_CLUSTERS).collect();
    idx.sort_by(|&a, &b| cents[a].partial_cmp(&cents[b]).unwrap().then(a.cmp(&b)));
    let mut relabel = [0u8; N_CLUSTERS];
    for (new, &old) in idx.iter().enumerate() {
        relabel[old] = new as u8;
    }
    let mut out = [0u8; N_PREFLOP_CLASSES];
    for i in 0..N_PREFLOP_CLASSES {
        out[i] = relabel[assign[i] as usize];
    }
    out
}

fn sample_spot(seed: u64) -> ([i32; 2], [i32; 5]) {
    let mut rng = Rng::seed_from_u64(seed);
    let mut deck: [i32; 52] = core::array::from_fn(|i| i as i32);
    for i in 0..7usize {
        let j = i + rng.next_below((52 - i) as u64) as usize;
        deck.swap(i, j);
    }
    (
        [deck[0], deck[1]],
        [deck[2], deck[3], deck[4], deck[5], deck[6]],
    )
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let k: usize = arg(&args, "--k", 160);
    let n_spots: usize = arg(&args, "--spots", 250_000);
    let kmeans_iters: usize = arg(&args, "--kmeans-iters", 30);
    let seed: u64 = arg(&args, "--seed", 7);
    let n_mc_class: usize = arg(&args, "--class-mc", 60_000);
    let out_path = arg_s(&args, "--out", "river_ochs_160.rochs");
    let golden_path = format!("{}.golden.tsv", out_path.trim_end_matches(".rochs"));
    let n_golden: usize = arg(&args, "--goldens", 512);
    let t0 = Instant::now();

    // ---- step 1+2: opponent clusters ------------------------------------------------------
    eprintln!(
        "[fit] step 1/4: per-class equity ({} MC/class, 169 classes)",
        n_mc_class
    );
    let eqs: Vec<f64> = (0..N_PREFLOP_CLASSES as i32)
        .into_par_iter()
        .map(|c| class_equity(c, n_mc_class, seed))
        .collect();
    let mut eq = [0.0f64; N_PREFLOP_CLASSES];
    eq.copy_from_slice(&eqs);
    let class_to_cluster = cluster_classes(&eq);
    eprintln!(
        "[fit] step 2/4: class clusters done ({:.1}s)",
        t0.elapsed().as_secs_f64()
    );

    // ---- step 3: OCHS vectors of sampled river spots ---------------------------------------
    // A provisional table (centroids empty) carries class_to_cluster for ochs_vector.
    let proto = RiverOchs {
        n_buckets: 0,
        class_to_cluster,
        centroids: vec![],
    };
    eprintln!("[fit] step 3/4: OCHS vectors for {} river spots", n_spots);
    let progress = std::sync::atomic::AtomicUsize::new(0);
    let t3 = Instant::now();
    let vecs: Vec<[f64; N_CLUSTERS]> = (0..n_spots)
        .into_par_iter()
        .map(|i| {
            let (hole, board) = sample_spot(seed.wrapping_mul(0x2545F491_4F6CDD1D) ^ (i as u64));
            let v = proto.ochs_vector(&hole, &board);
            let done = progress.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if done % (n_spots / 20).max(1) == 0 {
                let el = t3.elapsed().as_secs_f64();
                eprintln!(
                    "[fit]   {:>3}%  {:.0}s elapsed, ETA {:.0}s",
                    100 * done / n_spots,
                    el,
                    el / done as f64 * (n_spots - done) as f64
                );
            }
            v
        })
        .collect();

    // ---- step 4: K-means over the vectors --------------------------------------------------
    eprintln!(
        "[fit] step 4/4: k-means K={} over {} vectors, {} iters",
        k, n_spots, kmeans_iters
    );
    let mut cents: Vec<[f64; N_CLUSTERS]> = Vec::with_capacity(k);
    let mut seen = std::collections::HashSet::new();
    let stride = (n_spots / k).max(1);
    let mut i = 0usize;
    while cents.len() < k && i < n_spots {
        let key: Vec<u64> = vecs[i].iter().map(|x| x.to_bits()).collect();
        if seen.insert(key) {
            cents.push(vecs[i]);
        }
        i = if cents.len() * stride > i {
            cents.len() * stride
        } else {
            i + 1
        };
    }
    let mut j = 0usize;
    while cents.len() < k && j < n_spots {
        let key: Vec<u64> = vecs[j].iter().map(|x| x.to_bits()).collect();
        if seen.insert(key) {
            cents.push(vecs[j]);
        }
        j += 1;
    }
    assert!(cents.len() == k, "not enough distinct vectors for K={}", k);

    for it in 0..kmeans_iters {
        let assigns: Vec<usize> = vecs
            .par_iter()
            .map(|v| {
                let mut best = 0usize;
                let mut bd = f64::INFINITY;
                for (ci, c) in cents.iter().enumerate() {
                    let mut d = 0.0;
                    for x in 0..N_CLUSTERS {
                        let dx = v[x] - c[x];
                        d += dx * dx;
                    }
                    if d < bd {
                        bd = d;
                        best = ci;
                    }
                }
                best
            })
            .collect();
        let mut num = vec![[0.0f64; N_CLUSTERS]; k];
        let mut den = vec![0usize; k];
        for (v, &a) in vecs.iter().zip(assigns.iter()) {
            for x in 0..N_CLUSTERS {
                num[a][x] += v[x];
            }
            den[a] += 1;
        }
        // empty-cluster reseed: point with max distance to its centroid
        for ci in 0..k {
            if den[ci] == 0 {
                let (far_i, _) = vecs
                    .iter()
                    .enumerate()
                    .map(|(vi, v)| {
                        let a = assigns[vi];
                        let mut d = 0.0;
                        for x in 0..N_CLUSTERS {
                            let dx = v[x] - cents[a][x];
                            d += dx * dx;
                        }
                        (vi, d)
                    })
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then(b.0.cmp(&a.0)))
                    .unwrap();
                cents[ci] = vecs[far_i];
            } else {
                for x in 0..N_CLUSTERS {
                    cents[ci][x] = num[ci][x] / den[ci] as f64;
                }
            }
        }
        if (it + 1) % 5 == 0 || it + 1 == kmeans_iters {
            eprintln!(
                "[fit]   k-means iter {}/{} ({:.0}s)",
                it + 1,
                kmeans_iters,
                t0.elapsed().as_secs_f64()
            );
        }
    }

    let table = RiverOchs {
        n_buckets: k as u32,
        class_to_cluster,
        centroids: cents,
    };
    std::fs::write(&out_path, table.to_bytes()).expect("write blob");
    eprintln!(
        "[fit] wrote {} ({} bytes)",
        out_path,
        table.to_bytes().len()
    );

    // ---- goldens: held-out spots (disjoint seed stream) through the FINAL table ------------
    let mut g = std::fs::File::create(&golden_path).expect("create golden");
    writeln!(
        g,
        "# river_ochs golden parity rows: h0 h1 b0 b1 b2 b3 b4 bucket (card=rank*4+suit)"
    )
    .unwrap();
    for i in 0..n_golden {
        let (hole, board) =
            sample_spot(seed.wrapping_mul(0xD1B5_4A32_D192_ED03) ^ (0xABCD_0000 + i as u64));
        let b = table.bucket(&hole, &board).unwrap();
        writeln!(
            g,
            "{} {} {} {} {} {} {} {}",
            hole[0], hole[1], board[0], board[1], board[2], board[3], board[4], b
        )
        .unwrap();
    }
    eprintln!("[fit] wrote {} ({} rows)", golden_path, n_golden);

    // occupancy report
    let mut occ = vec![0usize; k];
    for v in &vecs {
        occ[table.bucket_of_vector(v) as usize] += 1;
    }
    let nonempty = occ.iter().filter(|&&c| c > 0).count();
    eprintln!(
        "[fit] DONE in {:.0}s: {}/{} buckets occupied on the fit sample; min occ {} max occ {}",
        t0.elapsed().as_secs_f64(),
        nonempty,
        k,
        occ.iter().min().unwrap(),
        occ.iter().max().unwrap()
    );
}
