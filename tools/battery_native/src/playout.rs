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

use crate::core::{
    self, EnvState, BET_BUCKET_START, CALL, CHECK, FOLD, NUM_ACTIONS, NUM_PLAYERS,
};
use crate::fingerprint::{fingerprint_fast, CHIP_UNIT};

pub const NUM_CONTINUATIONS: usize = 4;

/// Blueprint snapshot: fingerprint(16 bytes) -> strat row over the 11 actions.
pub type Blueprint = HashMap<[u8; 16], [f64; NUM_ACTIONS]>;

/// Per-worker immutable config for the native playout.
pub struct PlayoutConfig {
    pub allowed_pre: [bool; NUM_ACTIONS],
    pub allowed_post: [bool; NUM_ACTIONS],
    pub bias_mult: f64,
    pub bias_floor: f64,
    pub max_depth: i32,
    pub num_buckets: i32,
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
        let result = self.s[1]
            .wrapping_mul(5)
            .rotate_left(7)
            .wrapping_mul(9);
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
fn blueprint_policy(
    s: &EnvState,
    seat: usize,
    abst_mask: &[bool; NUM_ACTIONS],
    bp: &Blueprint,
    cfg: &PlayoutConfig,
) -> [f64; NUM_ACTIONS] {
    let key = fingerprint_fast(s, seat, cfg.num_buckets);
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
/// target, renorm.
fn bias_continuation(
    bp: &[f64; NUM_ACTIONS],
    group: usize,
    mask: &[bool; NUM_ACTIONS],
    cfg: &PlayoutConfig,
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
            p[a] *= cfg.bias_mult;
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
fn reshuffle_undealt(env: &mut EnvState, rng: &mut Rng) {
    let mut known = [false; 52];
    for i in 0..NUM_PLAYERS {
        for k in 0..2 {
            let c = env.holes[i][k];
            if c >= 0 {
                known[c as usize] = true;
            }
        }
    }
    for i in 0..crate::core::MAX_BOARD {
        let c = env.board[i];
        if c >= 0 {
            known[c as usize] = true;
        }
    }
    let mut unseen: Vec<i32> = (0..52).filter(|&c| !known[c as usize]).collect();
    // Fisher-Yates shuffle.
    let n = unseen.len();
    if n > 1 {
        for i in (1..n).rev() {
            let j = rng.next_below((i + 1) as u64) as usize;
            unseen.swap(i, j);
        }
    }
    let ptr = env.deck_ptr as usize;
    let cap = if ptr <= 52 { 52 - ptr } else { 0 };
    let take = unseen.len().min(cap);
    for i in 0..take {
        env.deck[ptr + i] = unseen[i];
    }
}

/// Native uniform-mode leaf value: mean over `rollout_samples` runouts of `playout`. Mirrors the
/// uniform branch of `continuation_leaf_value` (cont_weights=None). Seeds are derived from base_seed
/// so the arm is internally reproducible; runouts integrate board luck (statistical equivalence).
pub fn uniform_leaf_value(
    env: &EnvState,
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

/// Native uniform-mode `_playout`. choices[seat] is the fixed continuation index for that seat
/// (drawn lazily on first action via `rng`); -1 means "not yet drawn".
pub fn playout(
    env_in: &EnvState,
    hero: usize,
    bp: &Blueprint,
    cfg: &PlayoutConfig,
    rng: &mut Rng,
) -> f64 {
    let mut env = *env_in;
    let mut choices: [i32; NUM_PLAYERS] = [-1; NUM_PLAYERS];
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
        // Continuation choice for this seat (uniform U{0,NC}), fixed across the playout.
        if choices[seat] < 0 {
            choices[seat] = rng.next_below(NUM_CONTINUATIONS as u64) as i32;
        }
        let group = choices[seat] as usize;
        let row = bias_continuation(&bp_row, group, &abst, cfg);
        let u = rng.next_f64();
        let a = sample_action(&row, &full, u);
        env = core::apply_action(&env, a);
        depth += 1;
    }
    env.rewards[hero] as f64 / CHIP_UNIT
}
