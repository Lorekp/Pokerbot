//! Native port of `pokerbot.search.pluribus_rollout._playout` (UNIFORM continuation path only).
//!
//! Replicates, per step:
//!   1. is_chance_node -> sample_chance, continue.
//!   2. seat = action_seat.
//!   3. continuation_set(env, seat): full legal mask -> abstraction mask -> blueprint row (fingerprint
//!      lookup, legal-renorm) -> 4 bias_continuation rows.
//!   4. row = cont[choice(seat)]; a = inverse-CDF sample over the FULL legal mask.
//!   5. env = apply_action(env, a).
//! Returns hero terminal value in bb: rewards[hero] / CHIP_UNIT.
//!
//! Continuation choice (uniform mode): each seat draws cont ~ U{0,NC} once (lazily), fixed for the
//! playout. Action sampling and the continuation draw use a fast Rust RNG seeded from a passed
//! integer (statistical equivalence, NOT a replay of numpy's PCG64).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use crate::core::{self, EnvStateN, BET_BUCKET_START, CALL, CHECK, FOLD, NUM_ACTIONS};
use crate::emd::FrozenEMDLookup;
use crate::fingerprint::CHIP_UNIT;

// ---------------------------------------------------------------------------
// SETTLE_SHOWDOWN_LEAF consumer (gate lives in core.rs; env POKERBOT_SETTLE_SHOWDOWN_LEAF,
// default OFF = byte-identical shipped behavior).
//
// The engine NEVER settles a showdown reached by street-advance: `sample_chance` past RIVER leaves
// street=STREET_TERMINAL with done=false and STALE rewards=[0;N] (settle happens only inside
// `apply_action` — fold terminals + pre-river all-in run-outs; the river all-in+call lands in the
// same unsettled class). The trainer detects this and calls its own `terminal_return`
// (trainer.rs FIX 1, commit 7cc9298), but `playout`/`playout_fixed` below break on
// `action_seat < 0` and return `rewards[hero] == 0` — every check/call-to-showdown leaf (and every
// river all-in+call leaf) contributes 0 EV to the depth-limited search (measured: 100% of river-
// subgame leaf mass, 79-98% turn, 4-14% flop).
//
// Gate ON: value such leaves with the REAL showdown returns (`core::showdown_leaf_return`, the
// exact mirror of `trainer::terminal_return`). The counters instrument how often each playout
// terminal class is reached and how much |bb| mass the unsettled class carries, so reach frequency
// and the A/B are measured by the same build. Stats collection is itself gated
// (`leaf_settle_stats_set`, default OFF) so the rayon-parallel hot paths don't contend on the
// shared atomics outside audit runs; the settle decision needs no counter.
//
// COUPLING NOTE (world-freeze): a settled leaf value is HOLE-DEPENDENT (it is the showdown). The
// subgame leaf cache keys on the PUBLIC leaf, so with leaf_cache_worlds=1 the cached value freezes
// the first-visit hole-world -> heavy EV-loss tails (measured in the audit A/B). Deploy the gate
// together with leaf_cache_worlds>1 (subgame.rs / pluribus_search.py running-mean over M visiting
// worlds) so the cached leaf integrates M sampled showdown worlds.
// ---------------------------------------------------------------------------
static LEAF_STATS_ON: AtomicBool = AtomicBool::new(false);
static LEAF_N_DONE: AtomicU64 = AtomicU64::new(0); // env.done: engine-settled (fold / pre-river all-in)
static LEAF_N_UNSETTLED_SHOWDOWN: AtomicU64 = AtomicU64::new(0); // street>=TERMINAL, done=false (the bug class)
static LEAF_N_OTHER: AtomicU64 = AtomicU64::new(0); // depth-capped mid-hand (rewards also stale-0)
static LEAF_ABS_MILLIBB_DONE: AtomicU64 = AtomicU64::new(0); // sum |v_bb|*1000 over settled leaves
static LEAF_ABS_MILLIBB_UNSETTLED: AtomicU64 = AtomicU64::new(0); // sum |settled-would-be v_bb|*1000 over unsettled leaves

/// Terminal value of a finished playout. Shipped behavior (gate OFF): `env.rewards[hero]/CHIP_UNIT`
/// (stale 0 at street-terminal showdowns). Gate ON: real `core::showdown_leaf_return` at the
/// unsettled-showdown class. Depth-capped mid-hand exits keep the stale value under both settings.
fn playout_terminal_value<const N: usize>(env: &EnvStateN<N>, hero: usize) -> f64 {
    playout_terminal_value_gated(env, hero, core::settle_showdown_leaf_enabled())
}

/// Explicit-flag twin of `playout_terminal_value` (the global gate resolved by the caller); lets
/// tests exercise both settings without racing the process-global gate across parallel test threads.
pub(crate) fn playout_terminal_value_gated<const N: usize>(
    env: &EnvStateN<N>,
    hero: usize,
    settle_on: bool,
) -> f64 {
    let stale = env.rewards[hero] as f64 / CHIP_UNIT;
    let stats = LEAF_STATS_ON.load(Ordering::Relaxed);
    if env.done {
        if stats {
            LEAF_N_DONE.fetch_add(1, Ordering::Relaxed);
            LEAF_ABS_MILLIBB_DONE.fetch_add((stale.abs() * 1000.0) as u64, Ordering::Relaxed);
        }
        return stale;
    }
    if core::is_unsettled_showdown(env) {
        // Board is complete (the only route here is the river street-close -> sample_chance
        // advance), so the settle sees 5 board cards. Mirror of trainer FIX 1.
        if stats {
            let real = core::showdown_leaf_return(env, hero);
            LEAF_N_UNSETTLED_SHOWDOWN.fetch_add(1, Ordering::Relaxed);
            LEAF_ABS_MILLIBB_UNSETTLED.fetch_add((real.abs() * 1000.0) as u64, Ordering::Relaxed);
        }
        if settle_on {
            return core::showdown_leaf_return(env, hero);
        }
        return stale; // shipped behavior: 0 for hero at every such leaf
    }
    if stats {
        LEAF_N_OTHER.fetch_add(1, Ordering::Relaxed);
    }
    stale
}

/// Enable/disable the playout terminal-class counters (audit instrumentation; default OFF).
pub fn leaf_settle_stats_set(on: bool) {
    LEAF_STATS_ON.store(on, Ordering::Relaxed);
}

pub fn leaf_settle_stats_reset() {
    LEAF_N_DONE.store(0, Ordering::Relaxed);
    LEAF_N_UNSETTLED_SHOWDOWN.store(0, Ordering::Relaxed);
    LEAF_N_OTHER.store(0, Ordering::Relaxed);
    LEAF_ABS_MILLIBB_DONE.store(0, Ordering::Relaxed);
    LEAF_ABS_MILLIBB_UNSETTLED.store(0, Ordering::Relaxed);
}

/// (n_done, n_unsettled_showdown, n_other, abs_bb_done, abs_bb_unsettled)
pub fn leaf_settle_stats() -> (u64, u64, u64, f64, f64) {
    (
        LEAF_N_DONE.load(Ordering::Relaxed),
        LEAF_N_UNSETTLED_SHOWDOWN.load(Ordering::Relaxed),
        LEAF_N_OTHER.load(Ordering::Relaxed),
        LEAF_ABS_MILLIBB_DONE.load(Ordering::Relaxed) as f64 / 1000.0,
        LEAF_ABS_MILLIBB_UNSETTLED.load(Ordering::Relaxed) as f64 / 1000.0,
    )
}

pub const NUM_CONTINUATIONS: usize = 4;

/// Blueprint snapshot: fingerprint(16 bytes) -> strat row over the 11 actions.
///
/// TIER-1 SERVE FLIP (2026-07-12): the certified tier-1 126k serve object is a 538,848,758-row
/// FULL table — far beyond Python-dict marshaling range — served as the nlcfr_key_index.py
/// `--with-rows` SORTED MMAP INDEX (`keys_sorted.bin`: n*16 B lexicographically sorted digests;
/// `rows_sorted.f32.bin`: n*11 f32 LE, row-aligned with the keys). `Index` memory-maps both files
/// and binary-searches per lookup (~30 probes, ~1 µs page-cache-warm) — the Rust twin of the
/// Python serve object (pokerbot/eval/snapshot_dir.SortedSnapshotDir), so the native subgame CFR
/// reads the SAME table the Python serve path reads (f32 rows upcast to f64, exactly like numpy).
/// `Dict` is the historical HashMap; every existing constructor still builds it, byte-identical.
pub enum Blueprint {
    Dict(HashMap<[u8; 16], [f64; NUM_ACTIONS]>),
    Index(std::sync::Arc<SortedIndex>),
}

impl Clone for Blueprint {
    fn clone(&self) -> Self {
        match self {
            // Historical semantics: a full HashMap copy (the pre-flip type was a bare HashMap).
            Blueprint::Dict(m) => Blueprint::Dict(m.clone()),
            // The mmap index is immutable + Arc-shared: clone = refcount bump (cheap by design).
            Blueprint::Index(ix) => Blueprint::Index(std::sync::Arc::clone(ix)),
        }
    }
}

/// Memory-mapped sorted key/row store (see `Blueprint::Index`). Send+Sync (read-only mmaps), so
/// `Arc<Blueprint>` sharing across rayon workers keeps working.
pub struct SortedIndex {
    keys: memmap2::Mmap,
    rows: memmap2::Mmap,
    n: usize,
}

impl SortedIndex {
    pub fn open(dir: &str) -> std::io::Result<Self> {
        let kp = std::path::Path::new(dir).join("keys_sorted.bin");
        let rp = std::path::Path::new(dir).join("rows_sorted.f32.bin");
        let kf = std::fs::File::open(&kp)?;
        let rf = std::fs::File::open(&rp)?;
        let keys = unsafe { memmap2::Mmap::map(&kf)? };
        let rows = unsafe { memmap2::Mmap::map(&rf)? };
        if keys.len() % 16 != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{}: size {} not a multiple of 16", kp.display(), keys.len()),
            ));
        }
        let n = keys.len() / 16;
        if rows.len() != n * NUM_ACTIONS * 4 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{}: rows size {} != n({}) * {}", rp.display(), rows.len(), n,
                        NUM_ACTIONS * 4),
            ));
        }
        Ok(SortedIndex { keys, rows, n })
    }

    #[inline]
    fn key_at(&self, i: usize) -> &[u8] {
        &self.keys[i * 16..i * 16 + 16]
    }

    /// Binary search over the sorted raw 16-byte keys. Raw memcmp order == numpy 'S16' sort order
    /// for fixed-width 16-byte data (NUL is the smallest byte and S16 pads with NULs), so the
    /// Python-side searchsorted and this search agree on every key.
    pub fn get(&self, key: &[u8; 16]) -> Option<[f64; NUM_ACTIONS]> {
        let (mut lo, mut hi) = (0usize, self.n);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            match self.key_at(mid).cmp(&key[..]) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => {
                    let off = mid * NUM_ACTIONS * 4;
                    let mut row = [0.0f64; NUM_ACTIONS];
                    for (a, slot) in row.iter_mut().enumerate() {
                        let b = off + a * 4;
                        *slot = f32::from_le_bytes([
                            self.rows[b], self.rows[b + 1], self.rows[b + 2], self.rows[b + 3],
                        ]) as f64;
                    }
                    return Some(row);
                }
            }
        }
        None
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.n
    }
}

impl Blueprint {
    pub fn new() -> Self {
        Blueprint::Dict(HashMap::new())
    }

    pub fn with_capacity(n: usize) -> Self {
        Blueprint::Dict(HashMap::with_capacity(n))
    }

    /// Open the nlcfr_key_index.py `--with-rows` output dir as an immutable mmap blueprint.
    pub fn from_index_dir(dir: &str) -> std::io::Result<Self> {
        Ok(Blueprint::Index(std::sync::Arc::new(SortedIndex::open(dir)?)))
    }

    /// Insert (Dict only). The mmap index is immutable by construction; every constructor that
    /// inserts marshals a Python dict and starts from `Dict`, so this is unreachable for `Index`.
    pub fn insert(&mut self, k: [u8; 16], v: [f64; NUM_ACTIONS]) {
        match self {
            Blueprint::Dict(m) => {
                m.insert(k, v);
            }
            Blueprint::Index(_) => panic!("Blueprint::Index is immutable (mmap serve object)"),
        }
    }

    /// Row lookup BY VALUE (11 f64). Dict rows are copied (88 B — same cost as the old deref +
    /// element reads at the call sites); Index rows are decoded f32 LE -> f64.
    #[inline]
    pub fn get(&self, key: &[u8; 16]) -> Option<[f64; NUM_ACTIONS]> {
        match self {
            Blueprint::Dict(m) => m.get(key).copied(),
            Blueprint::Index(ix) => ix.get(key),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Blueprint::Dict(m) => m.len(),
            Blueprint::Index(ix) => ix.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for Blueprint {
    fn default() -> Self {
        Blueprint::new()
    }
}

/// EMD frozen-lookup keying config for the serve fast paths. `None` on `PlayoutConfig.emd` (the
/// default for every pre-existing caller) => the made-hand `fingerprint_fast` key, byte-identical
/// to the shipped binary. `Some` => flop/turn keys come from the shared frozen EMD lookup (the
/// abstraction the promoted EMD-K160 blueprint rows were written under; see
/// `fingerprint::fingerprint_fast_depth_emd` for the exact key + miss semantics).
#[derive(Clone)]
pub struct EmdKeyCfg {
    pub lookup: Arc<FrozenEMDLookup>,
    /// Frozen-miss semantics, mirroring `jax_weva_encode.compute_postflop_bucket`: `true` = made-
    /// hand fallback (mirror/silent fallback — the deployed serve regime, and the trainer's river
    /// rule); `false` = fail-on-miss (bucket -1 -> the preflop `b"w"`-branch key).
    pub miss_made_hand: bool,
}

/// Per-worker immutable config for the native playout.
pub struct PlayoutConfig {
    pub allowed_pre: [bool; NUM_ACTIONS],
    pub allowed_post: [bool; NUM_ACTIONS],
    pub bias_mult: f64,
    pub bias_floor: f64,
    pub max_depth: i32,
    pub num_buckets: i32,
    /// EMD frozen-lookup keying (None = made-hand fast path, byte-identical to the shipped binary).
    pub emd: Option<EmdKeyCfg>,
}

impl PlayoutConfig {
    /// The blueprint dict key for (state, seat) under THIS config's abstraction: made-hand
    /// (`emd: None`, unchanged bytes) or EMD frozen-lookup keying. Every blueprint lookup on the
    /// native subgame/rollout path routes through here so the abstraction is decided in ONE place.
    #[inline]
    pub fn fp_key<const N: usize>(&self, s: &EnvStateN<N>, seat: usize) -> [u8; 16] {
        #[cfg(feature = "prof")]
        let _prof_scope = crate::prof::Scope::new(crate::prof::P_FP_KEY);
        #[cfg(feature = "prof")]
        crate::prof::count(crate::prof::C_BP_GET);
        // OPT-6: memoized digest twins (byte-identical; thread-local pure message->digest cache).
        match &self.emd {
            None => crate::fingerprint::fingerprint_fast_servememo(s, seat, self.num_buckets),
            Some(k) => crate::fingerprint::fingerprint_fast_depth_emd_servememo(
                s,
                seat,
                self.num_buckets,
                &[],
                &k.lookup,
                k.miss_made_hand,
            ),
        }
    }
}

// _BIAS_GROUPS: None(blueprint), fold={FOLD}, passive={CHECK,CALL}, aggro={BET_BUCKET_START..NUM_ACTIONS}
#[inline]
fn group_member(group: usize, a: usize) -> bool {
    match group {
        0 => false, // blueprint (no bias)
        1 => a == FOLD,
        2 => a == CHECK || a == CALL,
        3 => a >= BET_BUCKET_START && a < NUM_ACTIONS,
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// SplitMix64 -> xoshiro256** : fast, well-distributed, deterministic from a seed.
// ---------------------------------------------------------------------------
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn seed_from_u64(mut x: u64) -> Self {
        let mut sm = || {
            x = x.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        };
        Rng {
            s: [sm(), sm(), sm(), sm()],
        }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform float in [0, 1) with 53-bit mantissa (matches the resolution of a typical rng.random()).
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform integer in [0, n).
    #[inline]
    pub fn next_below(&mut self, n: u64) -> u64 {
        // Lemire-ish: simple modulo is fine for tiny n (NUM_CONTINUATIONS=4); bias negligible.
        self.next_u64() % n
    }
}

// ---------------------------------------------------------------------------
// continuation_set: legal mask -> abstraction -> blueprint -> 4 bias rows.
// ---------------------------------------------------------------------------

#[inline]
fn renorm_legal(p: &mut [f64; NUM_ACTIONS], mask: &[bool; NUM_ACTIONS]) {
    let mut s = 0.0;
    for a in 0..NUM_ACTIONS {
        if !mask[a] {
            p[a] = 0.0;
        }
        s += p[a];
    }
    if s > 1e-12 {
        for a in 0..NUM_ACTIONS {
            p[a] /= s;
        }
    } else {
        let mut mc = 0.0f64;
        for a in 0..NUM_ACTIONS {
            if mask[a] {
                mc += 1.0;
            }
        }
        let denom = mc.max(1.0);
        for a in 0..NUM_ACTIONS {
            p[a] = if mask[a] { 1.0 / denom } else { 0.0 };
        }
    }
}

/// Blueprint policy lookup at (env, seat): fingerprint -> row, legal-masked + renormalized; uniform
/// over the abstracted legal mask on miss. Mirrors make_blueprint_policy with abstraction applied.
fn blueprint_policy<const N: usize>(
    s: &EnvStateN<N>,
    seat: usize,
    abst_mask: &[bool; NUM_ACTIONS],
    bp: &Blueprint,
    cfg: &PlayoutConfig,
) -> [f64; NUM_ACTIONS] {
    let key = cfg.fp_key(s, seat);
    let mut p = [0.0f64; NUM_ACTIONS];
    if let Some(row) = bp.get(&key) {
        let mut tot = 0.0;
        for a in 0..NUM_ACTIONS {
            let v = if abst_mask[a] { row[a] } else { 0.0 };
            p[a] = v;
            tot += v;
        }
        if tot > 1e-12 {
            for a in 0..NUM_ACTIONS {
                p[a] /= tot;
            }
            return p;
        }
    }
    // miss -> uniform over abstracted legal mask
    let mut mc = 0.0f64;
    for a in 0..NUM_ACTIONS {
        if abst_mask[a] {
            mc += 1.0;
        }
    }
    let denom = mc.max(1.0);
    for a in 0..NUM_ACTIONS {
        p[a] = if abst_mask[a] { 1.0 / denom } else { 0.0 };
    }
    p
}

/// bias_continuation for a single group over the abstracted mask. Mirrors the Python:
/// p = bp*mask; if group None -> renorm; else optional floor on legal target, *bias_mult on legal
/// target, renorm. Delegates to the explicit-multiplier variant with `cfg.bias_mult` (byte-identical
/// to the pre-refactor body: same ops, same order).
fn bias_continuation(
    bp: &[f64; NUM_ACTIONS],
    group: usize,
    mask: &[bool; NUM_ACTIONS],
    cfg: &PlayoutConfig,
) -> [f64; NUM_ACTIONS] {
    bias_continuation_mult(bp, group, mask, cfg, cfg.bias_mult)
}

/// `bias_continuation` with an EXPLICIT multiplier (Pluribus biased-continuation leaves use the
/// solve-level `leaf_bias_mult` knob — default 5.0, Pluribus's value — instead of the adapter-level
/// `cfg.bias_mult`). The floor still comes from `cfg.bias_floor` (0.0 = faithful, unchanged).
fn bias_continuation_mult(
    bp: &[f64; NUM_ACTIONS],
    group: usize,
    mask: &[bool; NUM_ACTIONS],
    cfg: &PlayoutConfig,
    mult: f64,
) -> [f64; NUM_ACTIONS] {
    let mut p = [0.0f64; NUM_ACTIONS];
    for a in 0..NUM_ACTIONS {
        p[a] = if mask[a] { bp[a] } else { 0.0 };
    }
    if group == 0 {
        renorm_legal(&mut p, mask);
        return p;
    }
    if cfg.bias_floor > 0.0 {
        for a in 0..NUM_ACTIONS {
            if mask[a] && group_member(group, a) {
                p[a] += cfg.bias_floor;
            }
        }
    }
    for a in 0..NUM_ACTIONS {
        if mask[a] && group_member(group, a) {
            p[a] *= mult;
        }
    }
    renorm_legal(&mut p, mask);
    p
}

/// The abstracted mask for a seat: full legal mask AND (allowed_pre if preflop else allowed_post).
#[inline]
fn abstracted_mask(
    full: &[bool; NUM_ACTIONS],
    street: i32,
    cfg: &PlayoutConfig,
) -> [bool; NUM_ACTIONS] {
    let allowed = if street == 0 {
        &cfg.allowed_pre
    } else {
        &cfg.allowed_post
    };
    let mut m = [false; NUM_ACTIONS];
    for a in 0..NUM_ACTIONS {
        m[a] = full[a] && allowed[a];
    }
    m
}

/// Inverse-CDF sample over the FULL legal mask given a continuation row (which already has abstracted
/// support). Mirrors `_sample_action(row, mask, u)`.
fn sample_action(row: &[f64; NUM_ACTIONS], full_mask: &[bool; NUM_ACTIONS], u: f64) -> usize {
    // legal = flatnonzero(mask); p = row[legal]; s = p.sum()
    let mut legal: [usize; NUM_ACTIONS] = [0; NUM_ACTIONS];
    let mut nl = 0usize;
    for a in 0..NUM_ACTIONS {
        if full_mask[a] {
            legal[nl] = a;
            nl += 1;
        }
    }
    if nl == 0 {
        return 0; // shouldn't happen (engine guarantees CHECK legal fallback)
    }
    let mut s = 0.0;
    for i in 0..nl {
        s += row[legal[i]];
    }
    if s <= 1e-12 {
        return legal[0];
    }
    // pick = np.searchsorted(np.cumsum(p), u, side="right") = count of cumsum entries <= u.
    // We must build the cumsum exactly as numpy (sequential f64 accumulation of normalized p) so the
    // comparison boundary matches bit-for-bit.
    let mut cum = 0.0f64;
    let mut pick = 0usize;
    for i in 0..nl {
        cum += row[legal[i]] / s;
        if cum <= u {
            pick = i + 1;
        }
    }
    if pick > nl - 1 {
        pick = nl - 1;
    }
    legal[pick]
}

/// Port of `_reshuffle_undealt`: rebuild the UNDEALT deck tail as a shuffled permutation of the
/// UNSEEN cards (52 minus every set hole/board card). Uses the native RNG (statistical equivalence:
/// integrates board luck; not a replay of numpy's shuffle).
pub fn reshuffle_undealt<const N: usize>(env: &mut EnvStateN<N>, rng: &mut Rng) {
    let mut known = [false; 52];
    for i in 0..N {
        for k in 0..2 {
            let c = env.holes[i][k];
            if c >= 0 {
                known[c as usize] = true;
            }
        }
    }
    let mut nb = 0usize;
    for i in 0..crate::core::MAX_BOARD {
        let c = env.board[i];
        if c >= 0 {
            known[c as usize] = true;
            nb += 1;
        }
    }
    // OPT-8 (2026-07-10): only the next (MAX_BOARD - n_board) deck slots can EVER be consumed —
    // the deck's sole readers are the board deals (`deal_n_cards` / `run_out`; holes are dealt at
    // hand init only). Drawing an ordered k-card sample without replacement (partial front
    // Fisher-Yates) is distribution-identical to the legacy full shuffle + take-first-k, at k
    // draws instead of ~45 (k = 0 on a full board: no draws, deck untouched — the tail is dead).
    // Stack buffer replaces the per-call Vec. rng draw PATTERN differs from the legacy full
    // shuffle (statistical equivalence; the callers' reshuffle rngs are all derived/decoupled
    // streams, so no other draw shifts).
    let k = crate::core::MAX_BOARD - nb;
    if k == 0 {
        return;
    }
    let mut unseen = [0i32; 52];
    let mut n = 0usize;
    for c in 0..52i32 {
        if !known[c as usize] {
            unseen[n] = c;
            n += 1;
        }
    }
    let ptr = env.deck_ptr as usize;
    let cap = if ptr <= 52 { 52 - ptr } else { 0 };
    let take = k.min(n).min(cap);
    for i in 0..take {
        let j = i + rng.next_below((n - i) as u64) as usize;
        unseen.swap(i, j);
        env.deck[ptr + i] = unseen[i];
    }
}

/// Native uniform-mode leaf value: mean over `rollout_samples` runouts of `playout`. Mirrors the
/// uniform branch of `continuation_leaf_value` (cont_weights=None). Seeds are derived from base_seed
/// so the arm is internally reproducible; runouts integrate board luck (statistical equivalence).
pub fn uniform_leaf_value<const N: usize>(
    env: &EnvStateN<N>,
    hero: usize,
    bp: &Blueprint,
    cfg: &PlayoutConfig,
    base_seed: u64,
    rollout_samples: i32,
    resample_board: bool,
) -> f64 {
    if env.done {
        return env.rewards[hero] as f64 / CHIP_UNIT;
    }
    let mut total = 0.0f64;
    for r in 0..rollout_samples {
        let rr = (r as u64).wrapping_add(1);
        let mut env_r = *env;
        if resample_board {
            let mut brng = Rng::seed_from_u64(base_seed.wrapping_add(7919u64.wrapping_mul(rr)));
            reshuffle_undealt(&mut env_r, &mut brng);
        }
        let mut arng = Rng::seed_from_u64(base_seed.wrapping_add(1_000_003u64.wrapping_mul(rr)));
        total += playout(&env_r, hero, bp, cfg, &mut arng);
    }
    total / rollout_samples as f64
}

/// PLURIBUS BIASED-CONTINUATION LEAF (2026-07-02): leaf value under a FIXED per-seat continuation
/// PROFILE with an explicit bias multiplier. Mean over `rollout_samples` runouts of `playout_fixed`
/// — same runout/seed schedule as `uniform_leaf_value` (7919/1000003 offsets), so the two paths draw
/// statistically-identical board integration; only the continuation choices differ (fixed by the
/// subgame CFR's choice nodes instead of drawn uniformly per playout).
#[allow(clippy::too_many_arguments)]
pub fn biased_leaf_value<const N: usize>(
    env: &EnvStateN<N>,
    hero: usize,
    bp: &Blueprint,
    cfg: &PlayoutConfig,
    base_seed: u64,
    rollout_samples: i32,
    resample_board: bool,
    choices: &[i32; N],
    bias_mult: f64,
) -> f64 {
    if env.done {
        return env.rewards[hero] as f64 / CHIP_UNIT;
    }
    let mut total = 0.0f64;
    for r in 0..rollout_samples {
        let rr = (r as u64).wrapping_add(1);
        let mut env_r = *env;
        if resample_board {
            let mut brng = Rng::seed_from_u64(base_seed.wrapping_add(7919u64.wrapping_mul(rr)));
            reshuffle_undealt(&mut env_r, &mut brng);
        }
        let mut arng = Rng::seed_from_u64(base_seed.wrapping_add(1_000_003u64.wrapping_mul(rr)));
        total += playout_fixed(&env_r, hero, bp, cfg, &mut arng, choices, bias_mult);
    }
    total / rollout_samples as f64
}

/// `_playout` under a FIXED per-seat continuation profile: `choices[seat]` in 0..NUM_CONTINUATIONS
/// selects that seat's continuation for the whole playout; a seat with `choices[seat] < 0` (a seat
/// that could not act at the leaf — folded/all-in — and therefore was given no choice node) falls
/// back to group 0 (the unaltered blueprint). Such a seat never acts in a well-formed playout, so
/// the fallback is defensive only. The bias multiplier is the explicit `bias_mult` (the solve-level
/// `leaf_bias_mult` knob), NOT `cfg.bias_mult`.
pub fn playout_fixed<const N: usize>(
    env_in: &EnvStateN<N>,
    hero: usize,
    bp: &Blueprint,
    cfg: &PlayoutConfig,
    rng: &mut Rng,
    choices: &[i32; N],
    bias_mult: f64,
) -> f64 {
    let mut env = *env_in;
    let mut depth = 0;
    while depth < cfg.max_depth && !env.done {
        if core::is_chance_node(&env) {
            env = core::sample_chance(&env);
            continue;
        }
        let seat_i = env.action_seat;
        if seat_i < 0 {
            break;
        }
        let seat = seat_i as usize;
        let full = core::legal_action_mask(&env, seat_i);
        let abst = abstracted_mask(&full, env.street, cfg);
        let bp_row = blueprint_policy(&env, seat, &abst, bp, cfg);
        let group = if choices[seat] >= 0 { choices[seat] as usize } else { 0 };
        let row = bias_continuation_mult(&bp_row, group, &abst, cfg, bias_mult);
        let u = rng.next_f64();
        let a = sample_action(&row, &full, u);
        // OPT-3 (perfect-hash settle, chip-identical) + gated real-NLHE street closure
        // (POKERBOT_CLOSE_CALLED_STREET; default OFF => identity): leaf rollouts must walk the
        // same closed game as the closure-trained blueprint they query.
        env = crate::closure::maybe_close_called_street(crate::subgame::subgame_apply(&env, a));
        depth += 1;
    }
    playout_terminal_value(&env, hero)
}

/// Native uniform-mode `_playout`. choices[seat] is the fixed continuation index for that seat
/// (drawn lazily on first action via `rng`); -1 means "not yet drawn".
pub fn playout<const N: usize>(
    env_in: &EnvStateN<N>,
    hero: usize,
    bp: &Blueprint,
    cfg: &PlayoutConfig,
    rng: &mut Rng,
) -> f64 {
    let mut env = *env_in;
    let mut choices: [i32; N] = [-1; N];
    let mut depth = 0;
    while depth < cfg.max_depth && !env.done {
        if core::is_chance_node(&env) {
            env = core::sample_chance(&env);
            continue;
        }
        let seat_i = env.action_seat;
        if seat_i < 0 {
            break;
        }
        let seat = seat_i as usize;
        #[cfg(feature = "prof")]
        crate::prof::count(crate::prof::C_PLAYOUT_STEP);
        let full = core::legal_action_mask(&env, seat_i);
        let abst = abstracted_mask(&full, env.street, cfg);
        let bp_row = blueprint_policy(&env, seat, &abst, bp, cfg);
        // Continuation choice for this seat (uniform U{0,NC}), fixed across the playout.
        if choices[seat] < 0 {
            choices[seat] = rng.next_below(NUM_CONTINUATIONS as u64) as i32;
        }
        let group = choices[seat] as usize;
        let row = bias_continuation(&bp_row, group, &abst, cfg);
        let u = rng.next_f64();
        let a = sample_action(&row, &full, u);
        // OPT-3 (perfect-hash settle, chip-identical) + gated real-NLHE street closure
        // (POKERBOT_CLOSE_CALLED_STREET; default OFF => identity): leaf rollouts must walk the
        // same closed game as the closure-trained blueprint they query.
        env = crate::closure::maybe_close_called_street(crate::subgame::subgame_apply(&env, a));
        depth += 1;
    }
    playout_terminal_value(&env, hero)
}
