//! Fully-native, rayon-parallel Local Best Response (LBR) exploitability harness.
//!
//! This is a faithful port of `pokerbot/eval/lbr6.py` (the analytic 1-ply LBR of Lisy & Bowling 2017,
//! multiway-generalised) and its HU specialisation `pokerbot/eval/lbr.py`. The Python harness is a
//! GIL-bound serial loop that pays a Python call PER belief combo PER decision (range_cap ~ 120 policy
//! queries x ~10 decisions x N hands), capping it at ~1-2 hands/s. Everything here -- engine
//! transitions (`core`), the blueprint dict-key (`fingerprint`), the Bayesian belief, the showdown
//! equity (`core::evaluate_seven`), AND the policy serve -- runs natively, and hands fan out across
//! rayon workers (each hand is independent given its seed). That is the ~100x throughput that turns the
//! wide, inconclusive Python CIs into tight ones.
//!
//! CORRECTNESS MODEL: statistical, NOT bit, equivalence (the RNG is the fast xoshiro of `playout.rs`,
//! not a replay of numpy's PCG64; the weighted belief subsample is Efraimidis-Spirakis, not numpy's
//! `choice`). That is exactly right for an independent Monte-Carlo validator: LBR's realised EV is a
//! lower bound on exploitability whose value is invariant to the RNG stream, and the native estimator
//! carries its own CI. The acceptance gates are (a) a CI-overlap equivalence test vs the Python harness
//! on a shared snapshot/cfg, and (b) the calibration anchors (always-fold ~ +750 mbb/g HU /
//! ~ +128 bb/100 6-max). See `pokerbot/eval/lbr_native.py` + `tests/test_lbr_native_equiv.py`.
//!
//! The hand evaluator is HIGHER = BETTER (`core::evaluate_seven`, base-15 scoring); the Python harness
//! uses phevaluator (LOWER = better). Every showdown comparison here is flipped accordingly.

use crate::core::{
    self, EnvState, BET_BUCKET_START, CALL, CHECK, FOLD, NUM_ACTIONS, NUM_PLAYERS, SEAT_ALLIN,
    SEAT_LIVE,
};
use crate::fingerprint::{fingerprint_fast_depth, fingerprint_fast_depth_emd, CHIP_UNIT};
use crate::playout::{Blueprint, EmdKeyCfg, Rng};
use rayon::prelude::*;

pub const HALF_POT_INDEX: usize = BET_BUCKET_START + 1; // frac 0.5
pub const POT_BET_INDEX: usize = BET_BUCKET_START + 4; // frac 1.0

const SETTLE_SALT: u64 = 0x5A17_5A17_5A17_5A17; // distinct settle RNG stream per hand

// ---------------------------------------------------------------------------
// Policy interface: the one seam that differs between the blueprint and neural validators.
// ---------------------------------------------------------------------------
/// A serveable strategy: `probs(state, seat)` -> distribution over the 11 actions, normalised over the
/// legal (and, for the tabular blueprint, abstracted) support. Must be `Sync` for the rayon fan-out.
pub trait LbrPolicy: Sync {
    fn probs(&self, s: &EnvState, seat: usize) -> [f64; NUM_ACTIONS];
}

/// Calibration / blueprint serve mode.
pub const MODE_BLUEPRINT: u8 = 0;
pub const MODE_FOLD: u8 = 1;
pub const MODE_NIT: u8 = 2;
pub const MODE_UNIFORM: u8 = 3;
pub const MODE_STATION: u8 = 4;

/// Tabular blueprint policy (also the home of the fixed calibration policies, so calibration needs no
/// fake dict). Mirrors `make_blueprint_policy` in `pokerbot/eval/lbr.py`: full legal mask -> abstraction
/// mask (`allowed_pre`/`allowed_post`) -> blueprint row (`fingerprint_fast` key) -> legal-renorm, with a
/// uniform-over-abstracted-mask fallback on a dict miss.
pub struct TabularPolicy {
    pub bp: Blueprint,
    pub allowed_pre: [bool; NUM_ACTIONS],
    pub allowed_post: [bool; NUM_ACTIONS],
    pub num_buckets: i32,
    /// Depth-bucket edges for the depth-aware blueprint key (empty => non-depth blueprint).
    pub depth_edges: Vec<f64>,
    /// EMD frozen-lookup keying (None = made-hand key, byte-identical to the shipped binary).
    pub emd: Option<EmdKeyCfg>,
    pub mode: u8,
}

#[inline]
fn fixed_single(mask: &[bool; NUM_ACTIONS], a: usize) -> [f64; NUM_ACTIONS] {
    let mut p = [0.0f64; NUM_ACTIONS];
    if mask[a] {
        p[a] = 1.0;
    } else {
        // argmax(mask): first legal action.
        for (i, &m) in mask.iter().enumerate() {
            if m {
                p[i] = 1.0;
                break;
            }
        }
    }
    p
}

#[inline]
fn uniform_over(mask: &[bool; NUM_ACTIONS]) -> [f64; NUM_ACTIONS] {
    let mut p = [0.0f64; NUM_ACTIONS];
    let mut n = 0.0f64;
    for &m in mask.iter() {
        if m {
            n += 1.0;
        }
    }
    let d = n.max(1.0);
    for a in 0..NUM_ACTIONS {
        p[a] = if mask[a] { 1.0 / d } else { 0.0 };
    }
    p
}

impl TabularPolicy {
    #[inline]
    fn abstracted(&self, full: &[bool; NUM_ACTIONS], street: i32) -> [bool; NUM_ACTIONS] {
        let allowed = if street == 0 {
            &self.allowed_pre
        } else {
            &self.allowed_post
        };
        let mut m = [false; NUM_ACTIONS];
        for a in 0..NUM_ACTIONS {
            m[a] = full[a] && allowed[a];
        }
        m
    }
}

impl LbrPolicy for TabularPolicy {
    fn probs(&self, s: &EnvState, seat: usize) -> [f64; NUM_ACTIONS] {
        let full = core::legal_action_mask(s, seat as i32);
        match self.mode {
            MODE_FOLD => {
                // always_fold_policy: FOLD if legal else CHECK if legal else argmax(mask).
                if full[FOLD] {
                    fixed_single(&full, FOLD)
                } else if full[CHECK] {
                    fixed_single(&full, CHECK)
                } else {
                    fixed_single(&full, FOLD) // fixed_single falls back to first-legal when FOLD illegal
                }
            }
            MODE_UNIFORM => uniform_over(&full),
            MODE_STATION => {
                // _station_policy (scripts/lbr6_audit.py): calling station -- CALL if legal, else
                // CHECK, else FOLD, else argmax(mask). Never folds when it can call, never bets.
                if full[CALL] {
                    fixed_single(&full, CALL)
                } else if full[CHECK] {
                    fixed_single(&full, CHECK)
                } else if full[FOLD] {
                    fixed_single(&full, FOLD)
                } else {
                    fixed_single(&full, CALL) // falls back to first-legal
                }
            }
            MODE_NIT => {
                // make_nit_policy: bet pot / call with strong made hands, else check/fold.
                let h = s.holes[seat];
                let r0 = h[0] / 4 + 2;
                let r1 = h[1] / 4 + 2;
                let strong = (r0 == r1) || (r0.max(r1) >= 13);
                let potbet = BET_BUCKET_START + 4;
                if strong && full[potbet] {
                    fixed_single(&full, potbet)
                } else if strong && full[CALL] {
                    fixed_single(&full, CALL)
                } else if full[CHECK] {
                    fixed_single(&full, CHECK)
                } else if full[FOLD] {
                    fixed_single(&full, FOLD)
                } else {
                    uniform_over(&full)
                }
            }
            _ => {
                // MODE_BLUEPRINT
                let abst = self.abstracted(&full, s.street);
                let key = match &self.emd {
                    None => fingerprint_fast_depth(s, seat, self.num_buckets, &self.depth_edges),
                    Some(k) => fingerprint_fast_depth_emd(
                        s,
                        seat,
                        self.num_buckets,
                        &self.depth_edges,
                        &k.lookup,
                        k.miss_made_hand,
                    ),
                };
                if let Some(row) = self.bp.get(&key) {
                    let mut p = [0.0f64; NUM_ACTIONS];
                    let mut tot = 0.0;
                    for a in 0..NUM_ACTIONS {
                        let v = if abst[a] { row[a] } else { 0.0 };
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
                uniform_over(&abst)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
pub struct LbrCfg {
    pub n_seats: usize,
    pub track_ranges: bool,
    pub range_cap: usize,
    pub eq_samples: i32,
    pub eq_runouts: i32,
    pub settle_samples: i32,
    pub max_depth: i32,
    pub bet_candidates: Vec<usize>,
    pub value_only_raises: bool,
}

// ---------------------------------------------------------------------------
// Belief over one opponent's hole cards (mirror lbr6.py::_Belief)
// ---------------------------------------------------------------------------
struct Belief {
    combos: Vec<(i32, i32)>,
    weights: Vec<f64>,
}

#[inline]
fn set_holes(s: &EnvState, seat: usize, c0: i32, c1: i32) -> EnvState {
    let mut e = *s;
    e.holes[seat] = [c0, c1];
    e
}

#[inline]
fn dead_mask_of(cards: &[i32]) -> [bool; 52] {
    let mut d = [false; 52];
    for &c in cards {
        if (0..52).contains(&c) {
            d[c as usize] = true;
        }
    }
    d
}

#[inline]
fn board_into(s: &EnvState, out: &mut Vec<i32>) {
    out.clear();
    for &c in s.board.iter() {
        if c >= 0 {
            out.push(c);
        }
    }
}

fn all_combos(dead: &[bool; 52]) -> Vec<(i32, i32)> {
    let live: Vec<i32> = (0..52).filter(|&c| !dead[c as usize]).collect();
    let mut out = Vec::with_capacity(live.len() * (live.len().saturating_sub(1)) / 2);
    for i in 0..live.len() {
        for j in (i + 1)..live.len() {
            out.push((live[i], live[j]));
        }
    }
    out
}

/// Weighted categorical index (weights need not be normalised; sums internally).
#[inline]
fn weighted_index(weights: &[f64], rng: &mut Rng) -> usize {
    let mut s = 0.0f64;
    for &w in weights {
        s += w;
    }
    if s <= 0.0 {
        return rng.next_below(weights.len().max(1) as u64) as usize;
    }
    let u = rng.next_f64() * s;
    let mut cum = 0.0f64;
    for (i, &w) in weights.iter().enumerate() {
        cum += w;
        if u < cum {
            return i;
        }
    }
    weights.len() - 1
}

impl Belief {
    fn uniform(dead: &[bool; 52], cap: usize, rng: &mut Rng) -> Belief {
        let mut combos = all_combos(dead);
        if combos.len() > cap {
            // uniform subsample without replacement (numpy choice(replace=False), no weights).
            // Partial Fisher-Yates: pick `cap` of len by swapping prefix.
            let n = combos.len();
            for i in 0..cap {
                let j = i + rng.next_below((n - i) as u64) as usize;
                combos.swap(i, j);
            }
            combos.truncate(cap);
        }
        let n = combos.len().max(1);
        let weights = vec![1.0 / n as f64; combos.len()];
        Belief { combos, weights }
    }

    #[inline]
    fn len(&self) -> usize {
        self.combos.len()
    }

    fn renorm(&mut self) {
        let s: f64 = self.weights.iter().sum();
        if s > 1e-12 {
            for w in self.weights.iter_mut() {
                *w /= s;
            }
        } else {
            let n = self.combos.len().max(1);
            for w in self.weights.iter_mut() {
                *w = 1.0 / n as f64;
            }
        }
    }

    fn prune(&mut self, blocked: &[bool; 52]) {
        if self.combos.is_empty() {
            return;
        }
        let mut changed = false;
        let mut nc = Vec::with_capacity(self.combos.len());
        let mut nw = Vec::with_capacity(self.weights.len());
        for (i, &(c0, c1)) in self.combos.iter().enumerate() {
            if !blocked[c0 as usize] && !blocked[c1 as usize] {
                nc.push((c0, c1));
                nw.push(self.weights[i]);
            } else {
                changed = true;
            }
        }
        if changed {
            self.combos = nc;
            self.weights = nw;
            self.renorm();
        }
    }

    fn bayes_update<P: LbrPolicy>(
        &mut self,
        s: &EnvState,
        seat: usize,
        action: usize,
        policy: &P,
        cap: usize,
        rng: &mut Rng,
    ) {
        let n = self.combos.len();
        if n == 0 {
            return;
        }
        for i in 0..n {
            let (c0, c1) = self.combos[i];
            let ec = set_holes(s, seat, c0, c1);
            self.weights[i] *= policy.probs(&ec, seat)[action];
        }
        self.renorm();
        if self.combos.len() > cap {
            self.weighted_subsample(cap, rng);
            self.renorm();
        }
    }

    /// Weighted subsample WITHOUT replacement to `cap`, Efraimidis-Spirakis (key = u^(1/w)); keep the
    /// `cap` largest keys. Distributionally equal to numpy `choice(replace=False, p=weights)`.
    fn weighted_subsample(&mut self, cap: usize, rng: &mut Rng) {
        let n = self.combos.len();
        let mut keyed: Vec<(f64, usize)> = Vec::with_capacity(n);
        for i in 0..n {
            let w = self.weights[i].max(1e-300);
            let u = rng.next_f64().max(1e-300);
            keyed.push((u.powf(1.0 / w), i));
        }
        // select the cap largest by key
        keyed.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        keyed.truncate(cap);
        let mut nc = Vec::with_capacity(cap);
        let mut nw = Vec::with_capacity(cap);
        for &(_, i) in keyed.iter() {
            nc.push(self.combos[i]);
            nw.push(self.weights[i]);
        }
        self.combos = nc;
        self.weights = nw;
    }

    fn sample(&self, blocked: &[bool; 52], rng: &mut Rng) -> Option<(i32, i32)> {
        if self.combos.is_empty() {
            return None;
        }
        for _ in 0..8 {
            let i = weighted_index(&self.weights, rng);
            let (c0, c1) = self.combos[i];
            if !blocked[c0 as usize] && !blocked[c1 as usize] {
                return Some((c0, c1));
            }
        }
        None
    }
}

/// Test/cross-check entry: deterministic river (full board) multiway equity.
pub fn multiway_equity_river_test(lbr: (i32, i32), opps: &[(i32, i32)], board: &[i32]) -> f64 {
    let mut rng = Rng::seed_from_u64(1);
    multiway_equity(lbr, opps, board, 1, &mut rng)
}

type BeliefSet = [Option<Belief>; NUM_PLAYERS];

fn in_hand_opponents(s: &EnvState, lbr_seat: usize) -> Vec<usize> {
    (0..NUM_PLAYERS)
        .filter(|&i| {
            i != lbr_seat && (s.seat_status[i] == SEAT_LIVE || s.seat_status[i] == SEAT_ALLIN)
        })
        .collect()
}

fn init_beliefs(s: &EnvState, lbr_seat: usize, cfg: &LbrCfg, rng: &mut Rng) -> BeliefSet {
    let mut board = Vec::with_capacity(5);
    board_into(s, &mut board);
    let mut dead = dead_mask_of(&board);
    dead[s.holes[lbr_seat][0] as usize] = true;
    dead[s.holes[lbr_seat][1] as usize] = true;
    let mut set: BeliefSet = std::array::from_fn(|_| None);
    for opp in in_hand_opponents(s, lbr_seat) {
        set[opp] = Some(Belief::uniform(&dead, cfg.range_cap, rng));
    }
    set
}

// ---------------------------------------------------------------------------
// Showdown equity (native evaluate_seven; HIGHER = BETTER, opposite of phevaluator)
// ---------------------------------------------------------------------------
#[inline]
fn seven_score(hole: (i32, i32), board5: &[i32; 5]) -> i32 {
    let cards = [
        hole.0, hole.1, board5[0], board5[1], board5[2], board5[3], board5[4],
    ];
    core::evaluate_seven(&cards)
}

/// P(LBR best, ties shared) vs the GIVEN opponent holes over `runouts` board completions. Mirrors
/// `_multiway_equity`. `board` holds the dealt board cards (length 0..5).
fn multiway_equity(
    lbr_hole: (i32, i32),
    opp_holes: &[(i32, i32)],
    board: &[i32],
    runouts: i32,
    rng: &mut Rng,
) -> f64 {
    if opp_holes.is_empty() {
        return 1.0;
    }
    let need = 5 - board.len();
    let mut dead = dead_mask_of(board);
    dead[lbr_hole.0 as usize] = true;
    dead[lbr_hole.1 as usize] = true;
    for &(o0, o1) in opp_holes {
        dead[o0 as usize] = true;
        dead[o1 as usize] = true;
    }

    let eval_once = |full_board: &[i32; 5]| -> f64 {
        let lh = seven_score(lbr_hole, full_board);
        let mut best_opp = i32::MIN;
        let mut ties = 0i32;
        for &oh in opp_holes {
            let os = seven_score(oh, full_board);
            if os > best_opp {
                best_opp = os;
            }
        }
        if lh > best_opp {
            1.0
        } else if lh == best_opp {
            for &oh in opp_holes {
                if seven_score(oh, full_board) == lh {
                    ties += 1;
                }
            }
            1.0 / (1.0 + ties as f64)
        } else {
            0.0
        }
    };

    if need == 0 {
        let mut fb = [0i32; 5];
        fb.copy_from_slice(&board[..5]);
        return eval_once(&fb);
    }

    let pool: Vec<i32> = (0..52).filter(|&c| !dead[c as usize]).collect();
    let rounds = runouts.max(1);
    let mut tot = 0.0f64;
    let mut fb = [0i32; 5];
    for i in 0..board.len() {
        fb[i] = board[i];
    }
    for _ in 0..rounds {
        // sample `need` distinct cards from pool (need is 1..5; rejection sampling is cheap at this size).
        let mut picked: [i32; 5] = [-1; 5];
        let mut got = 0usize;
        while got < need {
            let idx = rng.next_below(pool.len() as u64) as usize;
            let c = pool[idx];
            if (0..got).any(|k| picked[k] == c) {
                continue;
            }
            picked[got] = c;
            got += 1;
        }
        for k in 0..need {
            fb[board.len() + k] = picked[k];
        }
        tot += eval_once(&fb);
    }
    tot / rounds as f64
}

/// LBR's multiway showdown win-prob vs `opps` sampled FROM BELIEF, averaged over `samples` draws.
/// Mirrors `_wp_vs_beliefs`. `opps` lists the seat indices to include.
fn wp_vs_beliefs(
    s: &EnvState,
    lbr_seat: usize,
    beliefs: &BeliefSet,
    opps: &[usize],
    samples: i32,
    runouts: i32,
    rng: &mut Rng,
) -> f64 {
    if opps.is_empty() {
        return 1.0;
    }
    let mut board = Vec::with_capacity(5);
    board_into(s, &mut board);
    let lbr_hole = (s.holes[lbr_seat][0], s.holes[lbr_seat][1]);
    let mut base_block = dead_mask_of(&board);
    base_block[lbr_hole.0 as usize] = true;
    base_block[lbr_hole.1 as usize] = true;

    let rounds = samples.max(1);
    let mut tot = 0.0f64;
    let mut opp_holes: Vec<(i32, i32)> = Vec::with_capacity(opps.len());
    for _ in 0..rounds {
        let mut blocked = base_block;
        opp_holes.clear();
        for &opp in opps {
            let drawn = match &beliefs[opp] {
                Some(b) => b.sample(&blocked, rng),
                None => None,
            };
            let drawn = drawn.or_else(|| sample_uniform_pair(&blocked, rng));
            if let Some((c0, c1)) = drawn {
                opp_holes.push((c0, c1));
                blocked[c0 as usize] = true;
                blocked[c1 as usize] = true;
            }
        }
        tot += multiway_equity(lbr_hole, &opp_holes, &board, runouts, rng);
    }
    tot / rounds as f64
}

fn sample_uniform_pair(blocked: &[bool; 52], rng: &mut Rng) -> Option<(i32, i32)> {
    let pool: Vec<i32> = (0..52).filter(|&c| !blocked[c as usize]).collect();
    if pool.len() < 2 {
        return None;
    }
    let i = rng.next_below(pool.len() as u64) as usize;
    let mut j = rng.next_below(pool.len() as u64) as usize;
    while j == i {
        j = rng.next_below(pool.len() as u64) as usize;
    }
    Some((pool[i], pool[j]))
}

// ---------------------------------------------------------------------------
// Fold equity / calling ranges (mirror _fold_and_calling)
// ---------------------------------------------------------------------------
fn fold_and_calling<P: LbrPolicy>(
    env_raised: &EnvState,
    lbr_seat: usize,
    beliefs: &BeliefSet,
    policy: &P,
) -> (f64, BeliefSet, Vec<usize>) {
    let mut fp = 1.0f64;
    let mut calling: BeliefSet = std::array::from_fn(|_| None);
    let mut calling_seats = Vec::new();
    for opp in in_hand_opponents(env_raised, lbr_seat) {
        match &beliefs[opp] {
            Some(b) if b.len() > 0 => {
                let n = b.len();
                let mut folds = vec![0.0f64; n];
                for i in 0..n {
                    let (c0, c1) = b.combos[i];
                    let ec = set_holes(env_raised, opp, c0, c1);
                    folds[i] = policy.probs(&ec, opp)[FOLD];
                }
                let mut fold_mass = 0.0f64;
                for i in 0..n {
                    fold_mass += b.weights[i] * folds[i];
                }
                fp *= fold_mass;
                let mut cw = vec![0.0f64; n];
                let mut tot = 0.0f64;
                for i in 0..n {
                    cw[i] = b.weights[i] * (1.0 - folds[i]);
                    tot += cw[i];
                }
                if tot > 1e-9 {
                    for w in cw.iter_mut() {
                        *w /= tot;
                    }
                    calling[opp] = Some(Belief {
                        combos: b.combos.clone(),
                        weights: cw,
                    });
                    calling_seats.push(opp);
                }
            }
            _ => {
                fp *= policy.probs(env_raised, opp)[FOLD];
            }
        }
    }
    (fp, calling, calling_seats)
}

// ---------------------------------------------------------------------------
// Analytic 1-ply LBR action (mirror _lbr_action_analytic)
// ---------------------------------------------------------------------------
fn lbr_action_analytic<P: LbrPolicy>(
    s: &EnvState,
    lbr_seat: usize,
    beliefs: &BeliefSet,
    policy: &P,
    cfg: &LbrCfg,
    rng: &mut Rng,
) -> usize {
    let mask = core::legal_action_mask(s, lbr_seat as i32);
    let mut ct = [0.0f64; NUM_PLAYERS];
    let mut sum_ct = 0.0f64;
    for i in 0..NUM_PLAYERS {
        ct[i] = s.committed_total[i] as f64;
        sum_ct += ct[i];
    }
    let pot_others = sum_ct - ct[lbr_seat];
    let asked = (s.current_bet as f64 - s.committed_street[lbr_seat] as f64).max(0.0);

    let all_opps = in_hand_opponents(s, lbr_seat);
    let wp = wp_vs_beliefs(
        s,
        lbr_seat,
        beliefs,
        &all_opps,
        cfg.eq_samples,
        cfg.eq_runouts,
        rng,
    );

    let mut best_a: Option<usize> = None;
    let mut best_v = -1e18f64;
    if mask[FOLD] {
        best_a = Some(FOLD);
        best_v = 0.0;
    }
    if mask[CHECK] {
        let v = wp * pot_others;
        if v > best_v {
            best_v = v;
            best_a = Some(CHECK);
        }
    }
    if mask[CALL] {
        let v = wp * (pot_others + asked) - asked;
        if v > best_v {
            best_v = v;
            best_a = Some(CALL);
        }
    }
    for &a in cfg.bet_candidates.iter() {
        if a >= NUM_ACTIONS || !mask[a] {
            continue;
        }
        // Gated real-NLHE street closure (POKERBOT_CLOSE_CALLED_STREET; default OFF => identity).
        // A raise never triggers the closure predicate, so this is belt-and-braces consistency.
        let env_r = crate::closure::maybe_close_called_street(core::apply_action(s, a));
        let invest = env_r.committed_total[lbr_seat] as f64 - ct[lbr_seat];
        let bp_to_act =
            !env_r.done && !core::is_chance_node(&env_r) && env_r.action_seat != lbr_seat as i32;
        let (fp, wp_called) = if cfg.value_only_raises {
            (0.0, wp)
        } else if bp_to_act {
            let (fp, calling, calling_seats) = fold_and_calling(&env_r, lbr_seat, beliefs, policy);
            let wpc = if !calling_seats.is_empty() {
                wp_vs_beliefs(
                    s,
                    lbr_seat,
                    &calling,
                    &calling_seats,
                    cfg.eq_samples,
                    cfg.eq_runouts,
                    rng,
                )
            } else {
                wp
            };
            (fp, wpc)
        } else {
            (0.0, wp)
        };
        let pot_if_called = pot_others + 2.0 * invest;
        let v = fp * pot_others + (1.0 - fp) * (wp_called * pot_if_called - invest);
        if v > best_v {
            best_v = v;
            best_a = Some(a);
        }
    }
    match best_a {
        Some(a) => a,
        None => {
            // argmax(mask): first legal.
            (0..NUM_ACTIONS).find(|&i| mask[i]).unwrap_or(CHECK)
        }
    }
}

// ---------------------------------------------------------------------------
// Policy action sampling (opponent seats) + all-in settlement
// ---------------------------------------------------------------------------
fn sample_policy_action<P: LbrPolicy>(
    s: &EnvState,
    seat: usize,
    policy: &P,
    rng: &mut Rng,
) -> usize {
    let probs = policy.probs(s, seat);
    let mut support: Vec<usize> = Vec::with_capacity(NUM_ACTIONS);
    let mut w: Vec<f64> = Vec::with_capacity(NUM_ACTIONS);
    for a in 0..NUM_ACTIONS {
        if probs[a] > 0.0 {
            support.push(a);
            w.push(probs[a]);
        }
    }
    if support.is_empty() {
        let mask = core::legal_action_mask(s, seat as i32);
        let legal: Vec<usize> = (0..NUM_ACTIONS).filter(|&a| mask[a]).collect();
        if legal.is_empty() {
            return CHECK;
        }
        return legal[rng.next_below(legal.len() as u64) as usize];
    }
    support[weighted_index(&w, rng)]
}

/// All-in EV settlement: average LBR's terminal reward over `samples` runouts of the undealt deck tail.
/// Mirrors `_board_ev_settle`.
fn board_ev_settle(s: &EnvState, lbr_seat: usize, samples: i32, rng: &mut Rng) -> f64 {
    let ptr = s.deck_ptr as usize;
    if ptr >= 52 {
        return s.rewards[lbr_seat] as f64 / CHIP_UNIT;
    }
    let mut tail: Vec<i32> = s.deck[ptr..52].to_vec();
    if tail.is_empty() {
        return s.rewards[lbr_seat] as f64 / CHIP_UNIT;
    }
    let mut total = 0.0f64;
    for _ in 0..samples {
        // Fisher-Yates shuffle of the tail.
        let n = tail.len();
        for i in (1..n).rev() {
            let j = rng.next_below((i + 1) as u64) as usize;
            tail.swap(i, j);
        }
        let mut e = *s;
        for (k, &c) in tail.iter().enumerate() {
            e.deck[ptr + k] = c;
        }
        let mut guard = 0;
        while !e.done && guard < 8 {
            if core::is_chance_node(&e) {
                e = core::sample_chance(&e);
            } else {
                break;
            }
            guard += 1;
        }
        total += e.rewards[lbr_seat] as f64 / CHIP_UNIT;
    }
    total / samples as f64
}

#[inline]
fn live_only_count(s: &EnvState) -> i32 {
    s.seat_status.iter().filter(|&&st| st == SEAT_LIVE).count() as i32
}

// ---------------------------------------------------------------------------
// One hand (LBR always active; mirror _play_hand with br_active=True)
// ---------------------------------------------------------------------------
fn play_hand<P: LbrPolicy>(
    env0: &EnvState,
    lbr_seat: usize,
    policy: &P,
    cfg: &LbrCfg,
    rng: &mut Rng,
    settle_rng: &mut Rng,
) -> f64 {
    let mut env = *env0;
    let mut beliefs: BeliefSet = if cfg.track_ranges {
        init_beliefs(&env, lbr_seat, cfg, rng)
    } else {
        std::array::from_fn(|_| None)
    };
    let has_beliefs = cfg.track_ranges;
    let mut board_scratch = Vec::with_capacity(5);
    let mut depth = 0;
    while depth < cfg.max_depth && !env.done {
        if core::is_chance_node(&env) {
            if live_only_count(&env) == 0 && cfg.settle_samples > 1 {
                return board_ev_settle(&env, lbr_seat, cfg.settle_samples, settle_rng);
            }
            env = core::sample_chance(&env);
            if has_beliefs {
                board_into(&env, &mut board_scratch);
                let blocked = dead_mask_of(&board_scratch);
                for b in beliefs.iter_mut().flatten() {
                    b.prune(&blocked);
                }
            }
            continue;
        }
        let seat_i = env.action_seat;
        if seat_i < 0 {
            break;
        }
        let seat = seat_i as usize;
        if seat == lbr_seat {
            let a = lbr_action_analytic(&env, lbr_seat, &beliefs, policy, cfg, rng);
            // Gated real-NLHE street closure (POKERBOT_CLOSE_CALLED_STREET; default OFF => identity):
            // a closure-trained blueprint must be probed on the closed game it was trained on.
            env = crate::closure::maybe_close_called_street(core::apply_action(&env, a));
        } else {
            let a = sample_policy_action(&env, seat, policy, rng);
            if has_beliefs {
                if let Some(b) = beliefs[seat].as_mut() {
                    b.bayes_update(&env, seat, a, policy, cfg.range_cap, rng);
                }
            }
            env = crate::closure::maybe_close_called_street(core::apply_action(&env, a));
        }
        depth += 1;
    }
    if env.done {
        env.rewards[lbr_seat] as f64 / CHIP_UNIT
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Driver (rayon across hands) + reduction
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Default)]
pub struct SeatAcc {
    pub n: u64,
    pub sum: f64,
    pub sumsq: f64,
}

pub struct LbrOutcome {
    pub ev_bb_per_hand: f64,
    pub stderr_bb_per_hand: f64,
    pub hands: usize,
    pub per_seat: Vec<SeatAcc>,
}

/// Run the LBR over `roots` (one per hand), with `lbr_seats[h]` the best-responding seat and
/// `seeds[h]` the per-hand RNG seed. Parallel across hands.
pub fn run<P: LbrPolicy>(
    roots: &[EnvState],
    lbr_seats: &[usize],
    seeds: &[u64],
    policy: &P,
    cfg: &LbrCfg,
) -> LbrOutcome {
    let n = roots.len();
    let per: Vec<(usize, f64)> = (0..n)
        .into_par_iter()
        .map(|h| {
            let mut rng = Rng::seed_from_u64(seeds[h]);
            let mut srng = Rng::seed_from_u64(seeds[h] ^ SETTLE_SALT);
            let r = play_hand(&roots[h], lbr_seats[h], policy, cfg, &mut rng, &mut srng);
            (lbr_seats[h], r)
        })
        .collect();

    let mut per_seat = vec![SeatAcc::default(); cfg.n_seats.max(1)];
    let mut gsum = 0.0f64;
    let mut gsumsq = 0.0f64;
    for &(seat, r) in per.iter() {
        gsum += r;
        gsumsq += r * r;
        if seat < per_seat.len() {
            let acc = &mut per_seat[seat];
            acc.n += 1;
            acc.sum += r;
            acc.sumsq += r * r;
        }
    }
    let nf = n as f64;
    let ev = if n > 0 { gsum / nf } else { 0.0 };
    let stderr = if n > 1 {
        let var = (gsumsq - gsum * gsum / nf) / (nf - 1.0);
        (var.max(0.0) / nf).sqrt()
    } else {
        0.0
    };
    LbrOutcome {
        ev_bb_per_hand: ev,
        stderr_bb_per_hand: stderr,
        hands: n,
        per_seat,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_bp() -> Blueprint {
        Blueprint::new()
    }

    fn allowed_all() -> [bool; NUM_ACTIONS] {
        [true; NUM_ACTIONS]
    }

    #[test]
    fn fold_policy_folds_when_facing_bet() {
        // Construct a trivial state where seat 0 faces a bet (to_call > 0).
        let mut s = blank_state();
        s.seat_status = [SEAT_LIVE; NUM_PLAYERS];
        s.stacks = [1000; NUM_PLAYERS];
        s.current_bet = 100;
        s.committed_street = [0; NUM_PLAYERS];
        s.action_seat = 0;
        let pol = TabularPolicy {
            bp: empty_bp(),
            allowed_pre: allowed_all(),
            allowed_post: allowed_all(),
            num_buckets: 200,
            depth_edges: vec![],
            emd: None,
            mode: MODE_FOLD,
        };
        let p = pol.probs(&s, 0);
        assert!(
            p[FOLD] > 0.99,
            "fold policy should fold facing a bet: {:?}",
            p
        );
    }

    #[test]
    fn uniform_pair_is_distinct_and_unblocked() {
        let mut rng = Rng::seed_from_u64(1);
        let mut blocked = [false; 52];
        blocked[0] = true;
        blocked[1] = true;
        for _ in 0..1000 {
            let (a, b) = sample_uniform_pair(&blocked, &mut rng).unwrap();
            assert_ne!(a, b);
            assert!(a != 0 && a != 1 && b != 0 && b != 1);
        }
    }

    #[test]
    fn multiway_equity_sure_win() {
        // LBR holds a made set vs a single opponent with junk, full board -> deterministic.
        let mut rng = Rng::seed_from_u64(7);
        // board (Kc Kd 2c 7h 9s) with LBR holding Ks Kh (quads-ish) vs 3c 4d.
        let board = vec![card(13, 0), card(13, 1), card(2, 0), card(7, 2), card(9, 3)];
        let lbr = (card(13, 2), card(13, 3));
        let opp = vec![(card(3, 0), card(4, 1))];
        let eq = multiway_equity(lbr, &opp, &board, 1, &mut rng);
        assert!(eq > 0.99, "trip/quad kings must beat junk: {}", eq);
    }

    // helpers: card id = (rank-2)*4 + suit, rank in 2..14, suit in 0..3.
    fn card(rank: i32, suit: i32) -> i32 {
        (rank - 2) * 4 + suit
    }
    fn blank_state() -> EnvState {
        EnvState {
            holes: [[card(12, 0), card(12, 1)]; NUM_PLAYERS],
            board: [-1; 5],
            seat_status: [SEAT_LIVE; NUM_PLAYERS],
            stacks: [1000; NUM_PLAYERS],
            committed_street: [0; NUM_PLAYERS],
            committed_total: [0; NUM_PLAYERS],
            street: 0,
            pot_dead: 0,
            current_bet: 0,
            min_raise: 100,
            button: 0,
            action_seat: 0,
            last_aggressor: -1,
            acted: [false; NUM_PLAYERS],
            last_raise_full: true,
            deck: [0; 52],
            deck_ptr: 0,
            rng_key: [0, 0],
            hist_seat: [0; MAX_HISTORY_T],
            hist_action: [0; MAX_HISTORY_T],
            hist_amount: [0; MAX_HISTORY_T],
            hist_len: 0,
            done: false,
            rewards: [0; NUM_PLAYERS],
            pending_deal: false,
        }
    }
    const MAX_HISTORY_T: usize = 64;
}
