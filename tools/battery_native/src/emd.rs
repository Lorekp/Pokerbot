//! Native port of `pokerbot.encoding.postflop_bucketer` (potential-aware EMD postflop abstraction)
//! for the cloud-scale offline precompute (≈60k spots × 200 runouts hand evals is slow in Python).
//!
//! Cross-check standard (see `tests/test_native_emd_equiv.py`):
//!   * `strength_histogram` / `strength_histograms_batch`: STATISTICAL EQUIVALENCE + CALIBRATION. We
//!     deliberately use the fast native xoshiro RNG (from `playout::Rng`) instead of replaying numpy's
//!     PCG64 per-showdown (a proven perf dead-end here), so the Monte-Carlo histogram only converges to
//!     Python's at high `n_runouts`, it is NOT bit-identical.
//!   * `emd_bucket_from_hist`: BIT-IDENTICAL to numpy's `argmin(sum(|cumsum(hist)-cumsum(cent)|))`. This
//!     is a pure-float reduction over fixed inputs (centroids + histogram), so it MUST match Python to
//!     the bit (same summation order: cumulative sums left→right in f64, abs-diff accumulated left→right,
//!     strict-less keep-first argmin).
//!   * `emd_fit`: native k-means under EMD (the precompute speedup). Deterministic given a seed; NOT
//!     required to be bit-identical to Python (independent RNG/init), only calibration-equivalent.
//!
//! Card-id convention matches the encoder + `fingerprint::evaluate_cards`: `card_id = rank*4 + suit`
//! (rank 0..12, suit 0..3). Hand ranks come from `fingerprint::evaluate_cards` (1 best .. 7462 worst);
//! evaluation is NOT reimplemented here.

use crate::fingerprint::evaluate_cards;
use crate::playout::Rng;
use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;
use std::fs;
use std::path::Path;

const EMDFRZN_MAGIC: &[u8; 8] = b"EMDFRZN1";
const SUIT_PERMS: [[i32; 4]; 24] = [
    [0, 1, 2, 3],
    [0, 1, 3, 2],
    [0, 2, 1, 3],
    [0, 2, 3, 1],
    [0, 3, 1, 2],
    [0, 3, 2, 1],
    [1, 0, 2, 3],
    [1, 0, 3, 2],
    [1, 2, 0, 3],
    [1, 2, 3, 0],
    [1, 3, 0, 2],
    [1, 3, 2, 0],
    [2, 0, 1, 3],
    [2, 0, 3, 1],
    [2, 1, 0, 3],
    [2, 1, 3, 0],
    [2, 3, 0, 1],
    [2, 3, 1, 0],
    [3, 0, 1, 2],
    [3, 0, 2, 1],
    [3, 1, 0, 2],
    [3, 1, 2, 0],
    [3, 2, 0, 1],
    [3, 2, 1, 0],
];

/// Percentile-of-made-hand-strength histogram (length `n_bins`) over sampled river completions.
///
/// Mirrors `postflop_bucketer.strength_histogram`:
///   * `hole` = exactly 2 card ids; `board` = 3..5 card ids (padding ids `< 0` are filtered, like Python).
///   * Percentile: `pct = 1 - (rank-1)/7461` with `rank` in 1..7462; bin = `min(n_bins-1, int(pct*n_bins))`.
///   * River (5-card board => `need <= 0`): a single deterministic spike (no runouts).
///   * Otherwise: draw `need = 5 - board_len` cards per runout (partial Fisher–Yates from the remaining
///     deck), evaluate the 7-card hand, bin it; then normalize to sum 1 (if the sum is > 0).
///
/// Returns a zero vector for malformed inputs (board < 3 valid cards or hole != 2), like Python.
pub fn strength_histogram(
    hole: &[i32; 2],
    board: &[i32],
    n_bins: usize,
    n_runouts: usize,
    seed: u64,
) -> Vec<f64> {
    let mut hist = vec![0.0f64; n_bins];
    if n_bins == 0 {
        return hist;
    }
    // Filter board padding (ids < 0), like Python's `int(c) >= 0` guard.
    let mut bcards: Vec<i32> = Vec::with_capacity(5);
    for &c in board {
        if c >= 0 {
            bcards.push(c);
        }
    }
    if bcards.len() < 3 || hole[0] < 0 || hole[1] < 0 {
        return hist;
    }

    // Bin a single hand rank by percentile (shared by the river spike + each runout).
    let bin_of = |rank: i32| -> usize {
        let pct = 1.0 - (rank as f64 - 1.0) / 7461.0;
        // Python: min(n_bins-1, int(pct*n_bins)). int() truncates toward zero; pct in [0,1] so >= 0.
        let raw = (pct * n_bins as f64) as i64; // truncation, matches Python int()
        let idx = if raw < 0 { 0 } else { raw as usize };
        idx.min(n_bins - 1)
    };

    let need = 5i32 - bcards.len() as i32;
    if need <= 0 {
        // River: one spike at the made-hand percentile (no runouts).
        let mut cards: Vec<i32> = Vec::with_capacity(7);
        cards.push(hole[0]);
        cards.push(hole[1]);
        cards.extend_from_slice(&bcards);
        let rank = evaluate_cards(&cards);
        hist[bin_of(rank)] += 1.0;
        return hist;
    }
    let need = need as usize;

    // Remaining deck = 52 minus known (hole + board) cards.
    let mut known = [false; 52];
    known[hole[0] as usize] = true;
    known[hole[1] as usize] = true;
    for &c in &bcards {
        known[c as usize] = true;
    }
    let deck: Vec<i32> = (0..52).filter(|&c| !known[c as usize]).collect();
    let dn = deck.len();
    if dn < need {
        return hist; // not enough cards (Python would raise; the batch path guards upstream)
    }

    let mut rng = Rng::seed_from_u64(seed);
    // Scratch: the 7-card hand (2 hole + up-to-5 board + `need` draws) reused per runout.
    let mut cards: Vec<i32> = Vec::with_capacity(7);
    // A reusable draw buffer holding the deck; we do a partial Fisher–Yates of the first `need` slots.
    let mut draw_deck = deck.clone();
    for _ in 0..n_runouts {
        // Partial Fisher–Yates: draw `need` distinct cards uniformly without replacement.
        // (Statistical equivalence to numpy's `rng.choice(len, size=need, replace=False)`: same uniform
        // without-replacement distribution, different RNG stream.) We restore swapped slots each runout
        // so `draw_deck` stays a permutation of the full remaining deck.
        let mut swapped: [(usize, usize); 5] = [(0, 0); 5];
        for i in 0..need {
            // pick j in [i, dn)
            let j = i + rng.next_below((dn - i) as u64) as usize;
            draw_deck.swap(i, j);
            swapped[i] = (i, j);
        }
        cards.clear();
        cards.push(hole[0]);
        cards.push(hole[1]);
        cards.extend_from_slice(&bcards);
        for i in 0..need {
            cards.push(draw_deck[i]);
        }
        let rank = evaluate_cards(&cards);
        hist[bin_of(rank)] += 1.0;
        // Restore the deck to a clean permutation for the next runout (reverse the swaps).
        for i in (0..need).rev() {
            let (a, b) = swapped[i];
            draw_deck.swap(a, b);
        }
    }
    // Normalize to sum 1 (if sum > 0), matching `hist / s if s > 0 else hist`. Sum left→right.
    let mut s = 0.0f64;
    for &v in &hist {
        s += v;
    }
    if s > 0.0 {
        for v in hist.iter_mut() {
            *v /= s;
        }
    }
    hist
}

/// 1-D EMD = L1 of the CDFs: `sum(|cumsum(a) - cumsum(b)|)`. Cumulative sums left→right, abs-diff
/// accumulated left→right — the SAME order numpy uses, so this is bit-identical to `_emd_1d`.
#[inline]
fn emd_1d(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut ca = 0.0f64;
    let mut cb = 0.0f64;
    let mut acc = 0.0f64;
    for i in 0..a.len() {
        ca += a[i];
        cb += b[i];
        acc += (ca - cb).abs();
    }
    acc
}

/// Deterministic argmin of L1-of-CDF over centroids: `argmin_k sum(|cumsum(hist) - cumsum(cent_k)|)`.
///
/// BIT-IDENTICAL to the Python `bucket()` argmin: numpy's `argmin` returns the FIRST minimal index, so
/// we use a strict `<` comparison and keep the first (lowest-index) winner. The per-centroid distance is
/// computed by `emd_1d`, which sums in numpy's order. `centroids` rows must all have `hist.len()` bins.
pub fn emd_bucket_from_hist(hist: &[f64], centroids: &[Vec<f64>]) -> usize {
    let mut best = 0usize;
    let mut best_d = f64::INFINITY;
    for (k, cent) in centroids.iter().enumerate() {
        let d = emd_1d(hist, cent);
        if d < best_d {
            best_d = d;
            best = k;
        }
    }
    best
}

/// Native k-means under EMD over a sample of pre-computed strength histograms (the precompute speedup).
///
/// Mirrors `EMDBucketer.fit` semantics (NOT bit-identity):
///   * Init: pick `num_buckets` DISTINCT rows of `histograms` at random (partial Fisher–Yates over the
///     row indices via the native RNG seeded by `seed`).
///   * Assign each row to the nearest centroid by EMD (`emd_bucket_from_hist`).
///   * Update: each centroid becomes the mean histogram of its assigned members; an EMPTY cluster keeps
///     its old centroid.
///   * Repeat `iters` times. Deterministic given `seed`.
pub fn emd_fit(
    histograms: &[Vec<f64>],
    num_buckets: usize,
    iters: usize,
    seed: u64,
) -> Vec<Vec<f64>> {
    let n = histograms.len();
    assert!(
        num_buckets >= 1 && n >= num_buckets,
        "need >= num_buckets ({num_buckets}) sample histograms, got {n}"
    );
    let n_bins = histograms[0].len();

    // Random DISTINCT init rows: partial Fisher–Yates over [0, n).
    let mut idx: Vec<usize> = (0..n).collect();
    let mut rng = Rng::seed_from_u64(seed);
    for i in 0..num_buckets {
        let j = i + rng.next_below((n - i) as u64) as usize;
        idx.swap(i, j);
    }
    let mut cent: Vec<Vec<f64>> = (0..num_buckets)
        .map(|k| histograms[idx[k]].clone())
        .collect();

    for _ in 0..iters {
        // Assign every row to its nearest centroid.
        let assign: Vec<usize> = histograms
            .par_iter()
            .map(|x| emd_bucket_from_hist(x, &cent))
            .collect();
        // Update: mean of members; empty cluster keeps old centroid.
        let mut sums: Vec<Vec<f64>> = vec![vec![0.0f64; n_bins]; num_buckets];
        let mut counts: Vec<usize> = vec![0usize; num_buckets];
        for (i, &k) in assign.iter().enumerate() {
            counts[k] += 1;
            let row = &histograms[i];
            let s = &mut sums[k];
            for b in 0..n_bins {
                s[b] += row[b];
            }
        }
        for k in 0..num_buckets {
            if counts[k] > 0 {
                let inv = 1.0 / counts[k] as f64;
                for b in 0..n_bins {
                    cent[k][b] = sums[k][b] * inv;
                }
            }
            // else: keep the old centroid (no-op).
        }
    }
    cent
}

/// Mean within-cluster EMD inertia of a histogram set under fixed centroids: the mean over all rows of
/// `EMD(row, nearest_centroid)`. A clustering-quality scalar for the calibration cross-check (compared
/// native-vs-Python within ~10%).
pub fn mean_inertia(histograms: &[Vec<f64>], centroids: &[Vec<f64>]) -> f64 {
    if histograms.is_empty() {
        return 0.0;
    }
    let total: f64 = histograms
        .par_iter()
        .map(|x| {
            let k = emd_bucket_from_hist(x, centroids);
            emd_1d(x, &centroids[k])
        })
        .sum();
    total / histograms.len() as f64
}

/// Rayon-parallel batch of `strength_histogram` over many spots, for the cloud-scale precompute.
///
/// Each spot is `(hole[2], board)`. The per-spot seed is `seed_base ^ (index as u64)`, so results are
/// DETERMINISTIC and reproducible per index regardless of scheduling order.
pub fn strength_histograms_batch(
    spots: &[([i32; 2], Vec<i32>)],
    n_bins: usize,
    n_runouts: usize,
    seed_base: u64,
) -> Vec<Vec<f64>> {
    spots
        .par_iter()
        .enumerate()
        .map(|(i, (hole, board))| {
            strength_histogram(hole, board, n_bins, n_runouts, seed_base ^ (i as u64))
        })
        .collect()
}

#[inline]
fn map_suit(card: i32, perm: &[i32; 4]) -> i32 {
    let rank = card / 4;
    let suit = card % 4;
    rank * 4 + perm[suit as usize]
}

fn pack_cards(hole: &[i32; 2], board: &[i32; 5]) -> u64 {
    let pack = |c: i32| -> u64 {
        if c < 0 {
            0x3F
        } else {
            (c as u64) & 0x3F
        }
    };
    pack(hole[0])
        | (pack(hole[1]) << 6)
        | (pack(board[0]) << 12)
        | (pack(board[1]) << 18)
        | (pack(board[2]) << 24)
        | (pack(board[3]) << 30)
        | (pack(board[4]) << 36)
}

/// Suit-isomorphic canonical key for flop/turn EMD frozen lookup rows.
///
/// The key is the lexicographically-minimal packed `(sorted hole, sorted board)`
/// over all 24 suit relabelings. Board length must be 3 or 4; river is made-hand
/// equivalent and intentionally not stored in the frozen table.
pub fn canonical_frozen_key(hole: &[i32], board: &[i32]) -> Option<u64> {
    if hole.len() < 2 {
        return None;
    }
    let bcards: Vec<i32> = board.iter().copied().filter(|&c| c >= 0).collect();
    if !(bcards.len() == 3 || bcards.len() == 4) {
        return None;
    }
    if hole[0] < 0 || hole[1] < 0 {
        return None;
    }
    let mut best = u64::MAX;
    for perm in SUIT_PERMS.iter() {
        let mut h = [map_suit(hole[0], perm), map_suit(hole[1], perm)];
        if h[1] < h[0] {
            h.swap(0, 1);
        }
        let mut b = [-1i32; 5];
        for (i, &c) in bcards.iter().enumerate() {
            b[i] = map_suit(c, perm);
        }
        b[..bcards.len()].sort_unstable();
        let key = pack_cards(&h, &b);
        if key < best {
            best = key;
        }
    }
    Some(best)
}

#[derive(Clone)]
pub struct FrozenEMDLookup {
    pub n_buckets: usize,
    pub n_bins: usize,
    pub n_runouts: usize,
    pub rows: HashMap<u64, u16>,
}

impl FrozenEMDLookup {
    pub fn load_path(path: &str) -> Result<Self, String> {
        let blob = fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
        if blob.len() < 28 || &blob[0..8] != EMDFRZN_MAGIC {
            return Err(format!("bad EMDFRZN1 header: {path}"));
        }
        let u32_at = |off: usize| -> u32 {
            u32::from_le_bytes([blob[off], blob[off + 1], blob[off + 2], blob[off + 3]])
        };
        let u64_at = |off: usize| -> u64 {
            u64::from_le_bytes([
                blob[off],
                blob[off + 1],
                blob[off + 2],
                blob[off + 3],
                blob[off + 4],
                blob[off + 5],
                blob[off + 6],
                blob[off + 7],
            ])
        };
        let n_buckets = u32_at(8) as usize;
        let n_bins = u32_at(12) as usize;
        let n_runouts = u32_at(16) as usize;
        let n_rows = u64_at(20) as usize;
        let expected = 28usize
            .checked_add(n_rows.checked_mul(10).ok_or("row byte count overflow")?)
            .ok_or("blob byte count overflow")?;
        if blob.len() != expected {
            return Err(format!(
                "bad EMDFRZN1 length: got {}, expected {}",
                blob.len(),
                expected
            ));
        }
        let mut rows = HashMap::with_capacity_and_hasher(n_rows, Default::default());
        let mut off = 28usize;
        for _ in 0..n_rows {
            let key = u64_at(off);
            let bucket = u16::from_le_bytes([blob[off + 8], blob[off + 9]]);
            rows.insert(key, bucket);
            off += 10;
        }
        Ok(Self {
            n_buckets,
            n_bins,
            n_runouts,
            rows,
        })
    }

    pub fn bucket(&self, hole: &[i32], board: &[i32]) -> Option<i32> {
        let key = canonical_frozen_key(hole, board)?;
        self.rows.get(&key).map(|&b| b as i32)
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn push_spot(
    map: &mut HashMap<u64, u16>,
    centroids: &[Vec<f64>],
    hole: [i32; 2],
    board: &[i32],
    n_bins: usize,
    n_runouts: usize,
) {
    if let Some(key) = canonical_frozen_key(&hole, board) {
        if map.contains_key(&key) {
            return;
        }
        let hist = strength_histogram(&hole, board, n_bins, n_runouts, key);
        let bucket = emd_bucket_from_hist(&hist, centroids) as u16;
        map.insert(key, bucket);
    }
}

/// Build an EMDFRZN1 blob for every canonical flop+turn `(hole, board)` spot.
///
/// Heavy by design; Python calls this only from `scripts/precompute_emd_frozen_lookup.py`.
pub fn build_frozen_lookup(
    centroids: &[Vec<f64>],
    out_path: &str,
    n_bins: usize,
    n_runouts: usize,
) -> Result<(u64, String), String> {
    let n_buckets = centroids.len();
    if n_buckets == 0 {
        return Err("centroids must not be empty".to_string());
    }
    let mut map: HashMap<u64, u16> = HashMap::default();
    for h0 in 0..52i32 {
        for h1 in (h0 + 1)..52i32 {
            let hole = [h0, h1];
            let mut dead = [false; 52];
            dead[h0 as usize] = true;
            dead[h1 as usize] = true;
            let deck: Vec<i32> = (0..52i32).filter(|&c| !dead[c as usize]).collect();
            let dn = deck.len();
            for a in 0..dn {
                for b in (a + 1)..dn {
                    for c in (b + 1)..dn {
                        let flop = [deck[a], deck[b], deck[c]];
                        push_spot(&mut map, centroids, hole, &flop, n_bins, n_runouts);
                        for d in (c + 1)..dn {
                            let turn = [deck[a], deck[b], deck[c], deck[d]];
                            push_spot(&mut map, centroids, hole, &turn, n_bins, n_runouts);
                        }
                    }
                }
            }
        }
    }

    let mut rows: Vec<(u64, u16)> = map.into_iter().collect();
    rows.sort_unstable_by_key(|&(k, _)| k);
    let mut blob: Vec<u8> = Vec::with_capacity(28 + rows.len() * 10);
    blob.extend_from_slice(EMDFRZN_MAGIC);
    blob.extend_from_slice(&(n_buckets as u32).to_le_bytes());
    blob.extend_from_slice(&(n_bins as u32).to_le_bytes());
    blob.extend_from_slice(&(n_runouts as u32).to_le_bytes());
    blob.extend_from_slice(&(rows.len() as u64).to_le_bytes());
    for (key, bucket) in rows {
        blob.extend_from_slice(&key.to_le_bytes());
        blob.extend_from_slice(&bucket.to_le_bytes());
    }
    if let Some(parent) = Path::new(out_path).parent() {
        fs::create_dir_all(parent).map_err(|e| format!("create parent for {out_path}: {e}"))?;
    }
    fs::write(out_path, &blob).map_err(|e| format!("write {out_path}: {e}"))?;
    Ok((
        (blob.len() - 28) as u64 / 10,
        format!("{:016x}", fnv1a64(&blob[8..])),
    ))
}

// ---------------------------------------------------------------------------
// Pure-Rust unit tests (no Python). Bit-identity vs Python is proven by the
// cross-check in tests/test_native_emd_equiv.py.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn card(rank: i32, suit: i32) -> i32 {
        rank * 4 + suit
    }

    #[test]
    fn river_is_single_spike_summing_to_one() {
        // 5-card board => need <= 0 => one deterministic spike. Use a concrete made hand.
        // hole: Ah Kh ; board: Qh Jh Th (royal flush in hearts, suit=2). rank should be 1 (best).
        let hole = [card(12, 2), card(11, 2)]; // A=12, K=11 (rank index 0..12), suit hearts=2
        let board = vec![card(10, 2), card(9, 2), card(8, 2)]; // Q,J,T hearts
        let h = strength_histogram(&hole, &board, 20, 200, 0);
        let nonzero: Vec<(usize, f64)> = h
            .iter()
            .enumerate()
            .filter(|(_, &v)| v != 0.0)
            .map(|(i, &v)| (i, v))
            .collect();
        assert_eq!(
            nonzero.len(),
            1,
            "river must be exactly one nonzero bin, got {nonzero:?}"
        );
        let s: f64 = h.iter().sum();
        assert!(
            (s - 1.0).abs() < 1e-12,
            "river spike must sum to 1, got {s}"
        );
        // Best possible hand => percentile ~1.0 => top bin (index 19 for n_bins=20).
        assert_eq!(nonzero[0].0, 19, "royal flush should land in the top bin");
    }

    #[test]
    fn flop_histogram_normalizes_to_one() {
        // 3-card board => 2 runouts each. Histogram must normalize to sum 1.
        let hole = [card(12, 0), card(11, 0)]; // Ac Kc
        let board = vec![card(10, 1), card(2, 2), card(5, 3)]; // Qd 4h 7s
        let h = strength_histogram(&hole, &board, 20, 500, 12345);
        let s: f64 = h.iter().sum();
        assert!(
            (s - 1.0).abs() < 1e-9,
            "flop histogram must normalize to 1, got {s}"
        );
        // All bins non-negative, none exceeds 1.
        for &v in &h {
            assert!((0.0..=1.0).contains(&v), "bin out of [0,1]: {v}");
        }
    }

    #[test]
    fn malformed_inputs_return_zero_vector() {
        // < 3 valid board cards => zero vector.
        let hole = [card(12, 0), card(11, 0)];
        let board = vec![card(10, 1), -1, -1]; // only 1 valid board card
        let h = strength_histogram(&hole, &board, 20, 100, 0);
        assert!(
            h.iter().all(|&v| v == 0.0),
            "malformed board must give zero vector"
        );
        assert_eq!(h.len(), 20);
    }

    #[test]
    fn board_padding_is_filtered() {
        // A river board given with trailing -1 padding still collapses to a single spike.
        let hole = [card(12, 2), card(11, 2)];
        let board = vec![card(10, 2), card(9, 2), card(8, 2), -1, -1];
        let h = strength_histogram(&hole, &board, 20, 200, 0);
        let nz = h.iter().filter(|&&v| v != 0.0).count();
        assert_eq!(nz, 1, "padded river must still be a single spike");
    }

    #[test]
    fn emd_1d_hand_computed() {
        // a = [1,0,0], b = [0,0,1]. cumsum(a)=[1,1,1], cumsum(b)=[0,0,1]. |diff|=[1,1,0] -> 2.
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 0.0, 1.0];
        assert!((emd_1d(&a, &b) - 2.0).abs() < 1e-15);
        // identical -> 0.
        assert!((emd_1d(&a, &a)).abs() < 1e-15);
    }

    #[test]
    fn bucket_argmin_hand_computed_keep_first() {
        // hist = [1,0,0]. centroids: c0=[1,0,0] (EMD 0), c1=[0,1,0] (EMD 1), c2=[1,0,0] (EMD 0 again).
        // Strict-less keep-first => index 0 (the first zero-distance centroid).
        let hist = vec![1.0, 0.0, 0.0];
        let cents = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![1.0, 0.0, 0.0],
        ];
        assert_eq!(emd_bucket_from_hist(&hist, &cents), 0);

        // Now make c1 the unique minimum.
        let hist2 = vec![0.0, 1.0, 0.0];
        assert_eq!(emd_bucket_from_hist(&hist2, &cents), 1);
    }

    #[test]
    fn emd_fit_is_deterministic() {
        // Build a deterministic set of histograms (no RNG dependence in construction).
        let n_bins = 8;
        let n = 50;
        let mut hists: Vec<Vec<f64>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = vec![0.0f64; n_bins];
            // spread mass across a couple of bins deterministically.
            let a = i % n_bins;
            let b = (i * 3 + 1) % n_bins;
            row[a] += 0.6;
            row[b] += 0.4;
            // normalize (already sums to 1 unless a==b, then it's 1.0 in one bin).
            let s: f64 = row.iter().sum();
            for v in row.iter_mut() {
                *v /= s;
            }
            hists.push(row);
        }
        let c1 = emd_fit(&hists, 5, 8, 999);
        let c2 = emd_fit(&hists, 5, 8, 999);
        assert_eq!(c1.len(), c2.len());
        for k in 0..c1.len() {
            assert_eq!(c1[k], c2[k], "same seed must give identical centroids");
        }
        // A different seed should (almost surely) give a different init/result.
        let c3 = emd_fit(&hists, 5, 8, 1000);
        let same = c1 == c3;
        assert!(
            !same,
            "different seed should change the fit (statistically)"
        );
    }

    #[test]
    fn batch_matches_per_spot_determinism() {
        let spots = vec![
            (
                [card(12, 0), card(11, 0)],
                vec![card(10, 1), card(2, 2), card(5, 3)],
            ),
            (
                [card(8, 0), card(8, 1)],
                vec![card(3, 2), card(3, 3), card(9, 0), card(1, 1)],
            ),
        ];
        let b1 = strength_histograms_batch(&spots, 20, 300, 42);
        let b2 = strength_histograms_batch(&spots, 20, 300, 42);
        assert_eq!(b1, b2, "batch must be reproducible for a fixed seed_base");
        // And each entry equals the direct call with the same per-index seed.
        for (i, (hole, board)) in spots.iter().enumerate() {
            let direct = strength_histogram(hole, board, 20, 300, 42 ^ (i as u64));
            assert_eq!(b1[i], direct, "batch entry {i} must equal the direct call");
        }
    }
}
