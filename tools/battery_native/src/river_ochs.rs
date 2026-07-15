//! River OCHS abstraction (Opponent-Cluster Hand Strength, Johanson et al. AAMAS 2013) —
//! the blueprint-v2 replacement for the river's made-hand "mirror fallback" bucket.
//!
//! WHY: the deployed EMD-K160 blueprint keys the river with a SCALAR made-hand rank bucket
//! (`(rank-1)*K/7462` — strength vs a UNIFORM opponent). EMD is degenerate at the river (no
//! runouts left: the strength histogram collapses to a point mass ≡ that scalar). OCHS keeps
//! the river's strategically-relevant structure: an 8-vector of win rates against 8
//! preflop-strength CLUSTERS of opponent holdings — two hands with equal scalar equity but
//! different performance vs strong/weak ranges land in different buckets.
//!
//! PARITY CONTRACT (train ↔ serve, Rust ↔ Python): the bucket must be identical wherever it
//! is computed, because the trainer writes blueprint rows under it and the Python serve path
//! (`pokerbot/encoding/river_ochs.py`) must hit those rows.
//!   * hand ranks: `fingerprint::evaluate_cards` (1 best .. 7462 worst) — a byte-identical
//!     port of the installed python `phevaluator`, so both sides see the same integers.
//!   * OCHS vector: win/tie counts are INTEGERS (half-points accumulated as u32, win=2 tie=1)
//!     normalized once by an integer total — exact in f64, iteration-order independent.
//!   * nearest centroid: squared-L2 accumulated left→right over the 8 dims in f64, strict
//!     `<` keep-first argmin (lowest index wins ties) — fixed summation order on both sides.
//!   * enforced by the shared golden file (`data/abstraction/river_ochs_160.golden.tsv`):
//!     `cargo test` re-derives every golden row natively; the Python twin's test does the same.
//!
//! ARTIFACT (`ROCHS001`, ~10 KB — no 1.2GB river enumeration blob needed; the OCHS vector is
//! computed on the fly, ~990 opponent-combo evals ≈ 20µs native, memoized per rayon thread):
//!     [u8;8]  magic  = b"ROCHS001"
//!     u32 LE  n_buckets (K)
//!     u32 LE  n_clusters (8)
//!     [u8;169] class_to_cluster   (canonical_preflop_class id -> opponent cluster 0..7)
//!     K * 8 * f64 LE centroids (row-major)
//! plus a `.sha256` sidecar (hex digest of the whole blob) written by the fit tool
//! (`examples/river_ochs_fit.rs`) and cross-checked by `scripts/ship_blueprint.py`.
//!
//! Firewall-safe: pure additive module; consumed only by the trainer/fingerprint paths when a
//! table is explicitly configured. `core.rs` and every default path are untouched.

use crate::fingerprint::{canonical_preflop_class, evaluate_cards};
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::fs;

pub const ROCHS_MAGIC: &[u8; 8] = b"ROCHS001";
pub const N_CLUSTERS: usize = 8;
pub const N_PREFLOP_CLASSES: usize = 169;

pub struct RiverOchs {
    pub n_buckets: u32,
    pub class_to_cluster: [u8; N_PREFLOP_CLASSES],
    /// K rows x 8 dims, row-major.
    pub centroids: Vec<[f64; N_CLUSTERS]>,
}

impl RiverOchs {
    pub fn from_bytes(b: &[u8]) -> Result<Self, String> {
        if b.len() < 8 + 4 + 4 + N_PREFLOP_CLASSES {
            return Err("river_ochs: blob too short".into());
        }
        if &b[..8] != ROCHS_MAGIC {
            return Err(format!("river_ochs: bad magic {:?}", &b[..8]));
        }
        let n_buckets = u32::from_le_bytes(b[8..12].try_into().unwrap());
        let n_clusters = u32::from_le_bytes(b[12..16].try_into().unwrap());
        if n_clusters as usize != N_CLUSTERS {
            return Err(format!(
                "river_ochs: n_clusters {} != {}",
                n_clusters, N_CLUSTERS
            ));
        }
        let mut class_to_cluster = [0u8; N_PREFLOP_CLASSES];
        class_to_cluster.copy_from_slice(&b[16..16 + N_PREFLOP_CLASSES]);
        for &c in class_to_cluster.iter() {
            if c as usize >= N_CLUSTERS {
                return Err(format!("river_ochs: cluster id {} out of range", c));
            }
        }
        let expected = 16 + N_PREFLOP_CLASSES + (n_buckets as usize) * N_CLUSTERS * 8;
        if b.len() != expected {
            return Err(format!(
                "river_ochs: blob len {} != expected {}",
                b.len(),
                expected
            ));
        }
        let mut centroids = Vec::with_capacity(n_buckets as usize);
        let mut p = 16 + N_PREFLOP_CLASSES;
        for _ in 0..n_buckets {
            let mut row = [0.0f64; N_CLUSTERS];
            for v in row.iter_mut() {
                *v = f64::from_le_bytes(b[p..p + 8].try_into().unwrap());
                p += 8;
            }
            centroids.push(row);
        }
        Ok(Self {
            n_buckets,
            class_to_cluster,
            centroids,
        })
    }

    pub fn load_path(path: &str) -> Result<Self, String> {
        let b = fs::read(path).map_err(|e| format!("river_ochs read {}: {}", path, e))?;
        Self::from_bytes(&b)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(16 + N_PREFLOP_CLASSES + self.centroids.len() * 64);
        out.extend_from_slice(ROCHS_MAGIC);
        out.extend_from_slice(&self.n_buckets.to_le_bytes());
        out.extend_from_slice(&(N_CLUSTERS as u32).to_le_bytes());
        out.extend_from_slice(&self.class_to_cluster);
        for row in &self.centroids {
            for v in row {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }

    /// OCHS 8-vector for a RIVER spot: per opponent-cluster win rate of `hole` on `board5`.
    ///
    /// Enumerates all C(45,2)=990 opponent combos from the 45 unseen cards; win=2 half-points,
    /// tie=1; `vec[c] = half_points[c] / (2*count[c])` — integer-exact, order-independent.
    /// A cluster with zero live combos (impossible in practice with 45 unseen cards spanning
    /// all 169 classes' clusters, but kept total) reads 0.0.
    pub fn ochs_vector(&self, hole: &[i32; 2], board5: &[i32; 5]) -> [f64; N_CLUSTERS] {
        let mut dead = [false; 52];
        dead[hole[0] as usize] = true;
        dead[hole[1] as usize] = true;
        for &c in board5 {
            dead[c as usize] = true;
        }
        let mut cards7 = [0i32; 7];
        cards7[0] = hole[0];
        cards7[1] = hole[1];
        cards7[2..7].copy_from_slice(board5);
        let hero_rank = evaluate_cards(&cards7);

        let mut half = [0u32; N_CLUSTERS];
        let mut cnt = [0u32; N_CLUSTERS];
        let mut opp7 = [0i32; 7];
        opp7[2..7].copy_from_slice(board5);
        for o1 in 0..52i32 {
            if dead[o1 as usize] {
                continue;
            }
            for o2 in (o1 + 1)..52i32 {
                if dead[o2 as usize] {
                    continue;
                }
                let cls = canonical_preflop_class(o1, o2) as usize;
                let c = self.class_to_cluster[cls] as usize;
                opp7[0] = o1;
                opp7[1] = o2;
                let opp_rank = evaluate_cards(&opp7);
                // ranks: LOWER is better (1 = royal flush).
                if hero_rank < opp_rank {
                    half[c] += 2;
                } else if hero_rank == opp_rank {
                    half[c] += 1;
                }
                cnt[c] += 1;
            }
        }
        let mut v = [0.0f64; N_CLUSTERS];
        for c in 0..N_CLUSTERS {
            if cnt[c] > 0 {
                v[c] = half[c] as f64 / (2 * cnt[c]) as f64;
            }
        }
        v
    }

    /// Nearest-centroid bucket id for an OCHS vector: squared-L2 accumulated left→right,
    /// strict `<` keep-first argmin (the parity-pinned reduction).
    pub fn bucket_of_vector(&self, v: &[f64; N_CLUSTERS]) -> i32 {
        let mut best = 0usize;
        let mut best_d = f64::INFINITY;
        for (k, cent) in self.centroids.iter().enumerate() {
            let mut d = 0.0f64;
            for c in 0..N_CLUSTERS {
                let x = v[c] - cent[c];
                d += x * x;
            }
            if d < best_d {
                best_d = d;
                best = k;
            }
        }
        best as i32
    }

    /// River bucket for (hole, board). `None` unless the board has EXACTLY 5 valid cards and
    /// the holes are valid (the caller's street routing should already guarantee this).
    pub fn bucket(&self, hole: &[i32; 2], board: &[i32]) -> Option<i32> {
        if hole[0] < 0 || hole[1] < 0 {
            return None;
        }
        let mut b5 = [0i32; 5];
        let mut n = 0usize;
        for &c in board {
            if c >= 0 {
                if n == 5 {
                    return None;
                }
                b5[n] = c;
                n += 1;
            }
        }
        if n != 5 {
            return None;
        }
        Some(self.bucket_of_vector(&self.ochs_vector(hole, &b5)))
    }
}

// ---------------------------------------------------------------------------------------------
// Per-rayon-thread memo (the EMD_BUCKET_MEMO pattern): packed sorted 7-card u64 key + table ptr.
// After warmup a river node costs one FxHash probe. ~16 B/entry; river spots are board-diverse
// so the memo is larger than the EMD one — cap-free like the EMD memo (bounded by distinct
// reachable spots per run; a few hundred MB worst case across 64 threads is acceptable on the
// 256 GB train box; serve-side Python has its own bounded memo).
// ---------------------------------------------------------------------------------------------
type RiverMemo = FxHashMap<(usize, u64), i32>;
thread_local! {
    static RIVER_OCHS_MEMO: RefCell<RiverMemo> =
        RefCell::new(RiverMemo::with_capacity_and_hasher(1 << 16, Default::default()));
}

#[inline]
fn pack_key(table: &RiverOchs, hole: &[i32; 2], board5: &[i32; 5]) -> (usize, u64) {
    let (h0, h1) = if hole[0] <= hole[1] {
        (hole[0], hole[1])
    } else {
        (hole[1], hole[0])
    };
    let mut b = *board5;
    b.sort_unstable();
    let mut k: u64 = 0;
    for &c in [h0, h1].iter().chain(b.iter()) {
        k = (k << 6) | (c as u64 & 0x3F);
    }
    (table as *const RiverOchs as usize, k)
}

/// Memoized river bucket (the trainer's hot-path entry). `None` on a non-river spot.
pub fn river_ochs_bucket_memoed(table: &RiverOchs, hole: &[i32; 2], board: &[i32]) -> Option<i32> {
    if hole[0] < 0 || hole[1] < 0 {
        return None;
    }
    let mut b5 = [0i32; 5];
    let mut n = 0usize;
    for &c in board {
        if c >= 0 {
            if n == 5 {
                return None;
            }
            b5[n] = c;
            n += 1;
        }
    }
    if n != 5 {
        return None;
    }
    let key = pack_key(table, hole, &b5);
    RIVER_OCHS_MEMO.with(|m| {
        let map = &mut *m.borrow_mut();
        if let Some(&cached) = map.get(&key) {
            return Some(cached);
        }
        let v = table.bucket_of_vector(&table.ochs_vector(hole, &b5));
        map.insert(key, v);
        Some(v)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_table() -> RiverOchs {
        // 4 centroids spread along the diagonal — enough to test routing determinism.
        let centroids = vec![
            [0.125; N_CLUSTERS],
            [0.375; N_CLUSTERS],
            [0.625; N_CLUSTERS],
            [0.875; N_CLUSTERS],
        ];
        let mut class_to_cluster = [0u8; N_PREFLOP_CLASSES];
        for (i, c) in class_to_cluster.iter_mut().enumerate() {
            *c = (i % N_CLUSTERS) as u8;
        }
        RiverOchs {
            n_buckets: 4,
            class_to_cluster,
            centroids,
        }
    }

    // card id = rank*4 + suit; rank 0=2 .. 12=A; suit 0..3.
    fn cid(rank: i32, suit: i32) -> i32 {
        rank * 4 + suit
    }

    #[test]
    fn royal_flush_wins_every_cluster() {
        let t = tiny_table();
        // Hero As Ks; board Qs Js Ts 2c 3d — hero holds the unique royal flush.
        let hole = [cid(12, 0), cid(11, 0)];
        let board = [cid(10, 0), cid(9, 0), cid(8, 0), cid(0, 1), cid(1, 2)];
        let v = t.ochs_vector(&hole, &board);
        for c in 0..N_CLUSTERS {
            assert!(
                (v[c] - 1.0).abs() < 1e-15,
                "cluster {} win-rate {} != 1.0",
                c,
                v[c]
            );
        }
        assert_eq!(t.bucket(&hole, &board), Some(3)); // nearest to the 0.875 centroid
    }

    #[test]
    fn vector_in_unit_interval_and_bucket_stable() {
        let t = tiny_table();
        let hole = [cid(0, 1), cid(1, 2)]; // 2c 3d — weak
        let board = [cid(12, 0), cid(11, 0), cid(10, 0), cid(9, 1), cid(5, 3)];
        let v = t.ochs_vector(&hole, &board);
        for c in 0..N_CLUSTERS {
            assert!((0.0..=1.0).contains(&v[c]));
        }
        let b1 = t.bucket(&hole, &board).unwrap();
        let b2 = river_ochs_bucket_memoed(&t, &hole, &board).unwrap();
        assert_eq!(b1, b2, "memoized path must match direct path");
        // hole order + board order invariance
        let hole_r = [hole[1], hole[0]];
        let board_r = [board[4], board[3], board[2], board[1], board[0]];
        assert_eq!(t.bucket(&hole_r, &board_r), Some(b1));
    }

    #[test]
    fn roundtrip_bytes() {
        let t = tiny_table();
        let b = t.to_bytes();
        let t2 = RiverOchs::from_bytes(&b).unwrap();
        assert_eq!(t2.n_buckets, 4);
        assert_eq!(t2.class_to_cluster, t.class_to_cluster);
        assert_eq!(t2.centroids, t.centroids);
    }

    #[test]
    fn non_river_spots_are_none() {
        let t = tiny_table();
        let hole = [cid(12, 0), cid(11, 0)];
        assert_eq!(t.bucket(&hole, &[cid(10, 0), cid(9, 0), cid(8, 0)]), None); // flop
        assert_eq!(
            t.bucket(&hole, &[cid(10, 0), cid(9, 0), cid(8, 0), cid(7, 0)]),
            None
        ); // turn
        assert_eq!(t.bucket(&[-1, cid(11, 0)], &[0, 1, 2, 3, 4]), None); // bad hole
    }

    /// Golden-file parity gate (runs iff the fitted artifact + goldens exist — they are
    /// generated by `examples/river_ochs_fit.rs` and checked in small). Re-derives every
    /// golden row natively; the Python twin (`pokerbot/encoding/river_ochs.py`) has the same
    /// test against the same file, closing the cross-language parity chain.
    #[test]
    fn golden_parity_if_present() {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
        let blob = format!("{}/data/abstraction/river_ochs_160.rochs", root);
        let golden = format!("{}/data/abstraction/river_ochs_160.golden.tsv", root);
        if !std::path::Path::new(&blob).exists() || !std::path::Path::new(&golden).exists() {
            eprintln!("river_ochs golden artifacts missing — skipping (fit not run yet)");
            return;
        }
        let t = RiverOchs::load_path(&blob).unwrap();
        let txt = fs::read_to_string(&golden).unwrap();
        let mut n = 0usize;
        for line in txt.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let f: Vec<i32> = line
                .split_whitespace()
                .map(|x| x.parse().unwrap())
                .collect();
            assert_eq!(f.len(), 8, "golden row must be h0 h1 b0 b1 b2 b3 b4 bucket");
            let hole = [f[0], f[1]];
            let board = [f[2], f[3], f[4], f[5], f[6]];
            let got = t.bucket(&hole, &board).unwrap();
            assert_eq!(got, f[7], "golden mismatch at row {}: {:?}", n, f);
            n += 1;
        }
        assert!(n >= 64, "golden file suspiciously small ({} rows)", n);
        eprintln!("river_ochs golden parity: {} rows OK", n);
    }
}
