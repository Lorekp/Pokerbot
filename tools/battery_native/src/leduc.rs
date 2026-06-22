//! Native Leduc Hold'em CFR+ trainer — the cheapest correct artifact for the Workstream-F
//! "native Rust trainer" acceptance.
//!
//! WHY THIS MODULE: the acceptance (ULTRACODE Workstream F) asks for a native Rust trainer that
//! clears the Leduc exploitability gate (~0.76 mbb @ 1000 iters), is equivalence-gated vs the
//! Python/JAX reference, and has durable checkpoint/resume. The expensive option is a neural
//! GEMM+backprop stack (Adam/reservoir in `nnet.rs`, today forward-only). The spec explicitly offers a
//! cheaper OR: "pure-tabular CFR+ in Rust". Leduc is a tiny game (2 players, 3 actions, 2 streets, max
//! 2 raises, 90 deals) whose whole tree enumerates, so a tabular CFR+ solver IS the trainer and the
//! exploitability number is EXACT (full information-set best response, no estimator).
//!
//! FIDELITY: this is a faithful native port of two Python reference files, kept STATISTICALLY/
//! NUMERICALLY identical (the gate asserts it):
//!   * game rules  = `pokerbot/engine/leduc_jax.py` (ANTE=1, RAISE_SIZE=2, MAX_RAISES=2; seat 0 acts
//!     first preflop, seat 1 first on the public round; `_hand_score` pair bonus; split-pot showdown);
//!   * CFR+ scheme = `pokerbot/training/leduc_tabular_cfr.py::tabular_cfr` — regret-matching+ with
//!     ALTERNATING updates and LINEAR averaging (strategy weighted by iteration `t`), the strategy held
//!     fixed across a player's full sweep over all 90 deals and regrets floored to >=0 once per sweep.
//!     This is the exact OpenSpiel `CFRPlusSolver` scheme that hits <1 mbb @1000 iters;
//!   * exploitability = information-set best response (`leduc_tabular_cfr.py::infoset_exploitability_mbb`).
//!
//! It REUSES the CFR+ primitives proven in `subgame.rs`: positive-regret matching (`subgame.rs:91`) and
//! linear strategy averaging by `(t+1)`/`t` (`subgame.rs:429-453`). It does NOT fork that NLHE solver —
//! Leduc has its own (tiny, enumerable) tree, so a shared `EnvStateN` recursion would be the wrong tool.
//!
//! This module is pure Rust (no Python, no PyO3) so `cargo test` exercises the full gate offline; the
//! PyO3 surface in `lib.rs` exposes it for the Python equivalence gate.

use std::collections::HashMap;

pub const NUM_ACTIONS: usize = 3; // fold, call/check, raise
pub const NUM_PLAYERS: usize = 2;
const ANTE: i32 = 1;
const RAISE_SIZE: i32 = 2;
const MAX_RAISES: i32 = 2;

const FOLD: usize = 0;
const CALL: usize = 1;
const RAISE: usize = 2;

const ROUND_PRIVATE: i32 = 0;
const ROUND_PUBLIC: i32 = 1;

const MAX_HIST: usize = 16;

/// Exact native mirror of `leduc_jax.LeducState` (only the fields the CFR tree needs). `cards` are the
/// two private ranks 0..2; `public` is -1 until dealt; `fixed_public` pins the public card for the
/// enumerated tree (deal-determined, never sampled).
#[derive(Clone)]
pub struct State {
    pub cards: [i32; 2],
    pub public: i32,
    pub pot: i32,
    pub stacks: [i32; 2],
    pub committed: [i32; 2],
    pub invested: [i32; 2],
    pub current_bet: i32,
    pub raises: i32,
    pub round: i32,
    pub acting: i32,
    pub acted: [bool; 2],
    pub done: bool,
    pub rewards: [i32; 2],
    pub hist: [i8; MAX_HIST],
    pub hist_len: usize,
    pub fixed_public: i32,
}

impl State {
    /// Mirror of `init_from_deal(cards, fixed_public)`.
    pub fn from_deal(cards: [i32; 2], fixed_public: i32) -> State {
        State {
            cards,
            public: -1,
            pot: ANTE * 2,
            stacks: [100 - ANTE, 100 - ANTE],
            committed: [ANTE, ANTE],
            invested: [ANTE, ANTE],
            current_bet: ANTE,
            raises: 0,
            round: ROUND_PRIVATE,
            acting: 0,
            acted: [false, false],
            done: false,
            rewards: [0, 0],
            hist: [-1; MAX_HIST],
            hist_len: 0,
            fixed_public,
        }
    }
}

/// `legal_actions`: fold + call always; raise iff under the cap and the seat can cover it.
#[inline]
pub fn legal(s: &State) -> [bool; NUM_ACTIONS] {
    let seat = s.acting as usize;
    let to_call = (s.current_bet - s.committed[seat]).max(0);
    let can_raise = (s.raises < MAX_RAISES) && (s.stacks[seat] > to_call);
    [true, true, can_raise]
}

/// `_hand_score`: paired-with-board ranks beat all unpaired (10 + rank), else the bare rank.
#[inline]
fn hand_score(card: i32, public: i32) -> i32 {
    if card == public {
        10 + card
    } else {
        card
    }
}

/// `_resolve_showdown`: split pot with odd-chip remainder to seat 0 (matches the JAX `half + rem`).
fn resolve_showdown(s: &mut State) {
    let s0 = hand_score(s.cards[0], s.public);
    let s1 = hand_score(s.cards[1], s.public);
    let half = s.pot / 2;
    let rem = s.pot - half * 2;
    let inv = s.invested;
    let r0 = if s0 > s1 {
        s.pot - inv[0]
    } else if s0 == s1 {
        half + rem - inv[0]
    } else {
        -inv[0]
    };
    let r1 = s.pot - inv[0] - inv[1] - r0;
    s.done = true;
    s.rewards = [r0, r1];
}

#[inline]
fn round_complete(s: &State) -> bool {
    s.committed[0] == s.committed[1] && s.acted[0] && s.acted[1]
}

/// `_start_public_round`: deal the (fixed) public card, reset street betting, seat 1 acts first.
fn start_public_round(s: &mut State) {
    s.public = s.fixed_public; // enumerated tree: always pinned
    s.round = ROUND_PUBLIC;
    s.current_bet = 0;
    s.committed = [0, 0];
    s.raises = 0;
    s.acted = [false, false];
    s.acting = 1;
}

/// `apply_action`: returns the child state. Mirrors the JAX `do_fold/do_call/do_raise` + the
/// private->public transition + public->showdown resolution + history append.
pub fn apply_action(s: &State, action: usize) -> State {
    let mut ns = s.clone();
    let seat = s.acting as usize;
    let to_call = (s.current_bet - s.committed[seat]).max(0);
    match action {
        FOLD => {
            let winner = 1 - seat;
            let mut r = [-s.invested[0], -s.invested[1]];
            r[winner] += s.pot;
            ns.done = true;
            ns.rewards = r;
        }
        CALL => {
            let pay = to_call.min(s.stacks[seat]);
            ns.stacks[seat] -= pay;
            ns.committed[seat] += pay;
            ns.invested[seat] += pay;
            ns.pot += pay;
            ns.acting = (1 - seat) as i32;
            ns.acted[seat] = true;
        }
        RAISE => {
            let target = s.current_bet + RAISE_SIZE;
            let pay = (target - s.committed[seat]).min(s.stacks[seat]);
            ns.stacks[seat] -= pay;
            ns.committed[seat] += pay;
            ns.invested[seat] += pay;
            ns.pot += pay;
            ns.current_bet = ns.committed[seat];
            ns.raises += 1;
            ns.acting = (1 - seat) as i32;
            ns.acted = [false, false];
            ns.acted[seat] = true;
        }
        _ => unreachable!("illegal action"),
    }
    if !ns.done && round_complete(&ns) && ns.round == ROUND_PRIVATE {
        start_public_round(&mut ns);
    }
    if !ns.done && round_complete(&ns) && ns.round == ROUND_PUBLIC {
        resolve_showdown(&mut ns);
    }
    if ns.hist_len < MAX_HIST {
        ns.hist[ns.hist_len] = action as i8;
    }
    ns.hist_len += 1;
    ns
}

/// `py_info_key(state, seat)` packed into a collision-free i64 vector: the acting seat's view —
/// [seat, own_card, public(-1 if hidden), round, to_call, raises, hist_len, hist..]. Injective over the
/// suit-isomorphic Leduc encoding (the public card identity + own rank + betting history are sufficient).
pub type InfoKey = Vec<i64>;

pub fn info_key(s: &State, seat: usize) -> InfoKey {
    let pub_c = if s.public >= 0 { s.public } else { -1 };
    let to_call = (s.current_bet - s.committed[seat]).max(0);
    let mut k: Vec<i64> = Vec::with_capacity(7 + s.hist_len);
    k.push(seat as i64);
    k.push(s.cards[seat] as i64);
    k.push(pub_c as i64);
    k.push(s.round as i64);
    k.push(to_call as i64);
    k.push(s.raises as i64);
    k.push(s.hist_len as i64);
    for i in 0..s.hist_len.min(MAX_HIST) {
        k.push(s.hist[i] as i64);
    }
    k
}

/// All 90 deck permutations of (0,0,1,1,2,2) -> ((p0,p1), public). Sorted-unique, matching
/// `all_leduc_deals()` so the chance enumeration (and thus the floor) is identical.
pub fn all_deals() -> Vec<([i32; 2], i32)> {
    let deck = [0i32, 0, 1, 1, 2, 2];
    // Mirror `all_leduc_deals`: sorted(set(permutations(deck))) over the FULL 6-tuple, then project to
    // ((p0,p1), public) = (perm[0], perm[1], perm[2]). Deduping the full 6-tuple (not the projected
    // triple) is what yields the 90 distinct deals — the two cards of each rank are distinguishable in
    // the permutation set, so e.g. (0,1,2,..) appears with different tails that collapse to the same
    // first-3 projection but are still counted once per distinct full permutation.
    let mut seen6: Vec<[i32; 6]> = Vec::new();
    let idx = [0usize, 1, 2, 3, 4, 5];
    permute(&idx, 0, &mut |p| {
        let tup = [
            deck[p[0]], deck[p[1]], deck[p[2]], deck[p[3]], deck[p[4]], deck[p[5]],
        ];
        if !seen6.contains(&tup) {
            seen6.push(tup);
        }
    });
    seen6.sort();
    // project to ((p0,p1), public) PER DISTINCT 6-tuple — 90 entries, projections repeat. This is the
    // chance enumeration `tuple(((perm[0],perm[1]),perm[2]) for perm in sorted(set(permutations(deck))))`
    // verbatim: the duplicate projections are real distinct chance outcomes (different unseen burn cards),
    // so the chance weight is uniform 1/90 across them, NOT 1/24.
    seen6.iter().map(|t| ([t[0], t[1]], t[2])).collect()
}

fn permute<F: FnMut(&[usize; 6])>(arr: &[usize; 6], k: usize, f: &mut F) {
    if k == arr.len() {
        f(arr);
        return;
    }
    let mut a = *arr;
    for i in k..a.len() {
        a.swap(k, i);
        permute(&a, k + 1, f);
        // no need to swap back: we recurse on the local copy
    }
}

// --------------------------------------------------------------------------------------------------
// CFR+ tables (the trainable state; checkpoint-serialized)
// --------------------------------------------------------------------------------------------------

/// CFR+ regret / strategy tables — the durable trainer state. Mirrors `tabular_cfr`'s `cum_regret`
/// (regret-matching+, floored once per sweep) and `cum_policy` (linear-averaged strategy sum).
#[derive(Default, Clone)]
pub struct Tables {
    pub cum_regret: HashMap<InfoKey, [f64; NUM_ACTIONS]>,
    pub cum_policy: HashMap<InfoKey, [f64; NUM_ACTIONS]>,
    /// Current (fixed-for-the-sweep) strategy, refreshed from regrets after each sweep.
    pub cur_policy: HashMap<InfoKey, [f64; NUM_ACTIONS]>,
    pub legal_mask: HashMap<InfoKey, [bool; NUM_ACTIONS]>,
    /// Iterations completed (for resume). The next `train` call continues at `iters_done+1`.
    pub iters_done: u64,
}

/// `policy_for`: the fixed current strategy, or uniform-over-legal on a miss.
fn policy_for(tab: &Tables, key: &InfoKey, mask: &[bool; NUM_ACTIONS]) -> [f64; NUM_ACTIONS] {
    if let Some(p) = tab.cur_policy.get(key) {
        return *p;
    }
    uniform_over_legal(mask)
}

#[inline]
fn uniform_over_legal(mask: &[bool; NUM_ACTIONS]) -> [f64; NUM_ACTIONS] {
    let n = mask.iter().filter(|&&x| x).count().max(1) as f64;
    let mut p = [0.0f64; NUM_ACTIONS];
    for a in 0..NUM_ACTIONS {
        p[a] = if mask[a] { 1.0 / n } else { 0.0 };
    }
    p
}

/// One CFR recursion for `player` (the traverser). `reach = [pi_0, pi_1, pi_chance]`. Returns the
/// per-seat value vector. Mirrors `leduc_tabular_cfr.py::cfr` exactly: regrets/strategy accumulate as
/// deltas (no per-node strategy recompute), the strategy is `cur_policy` held fixed for the sweep.
fn cfr(
    s: &State,
    player: usize,
    reach: [f64; 3],
    weight: f64,
    tab: &mut Tables,
) -> [f64; NUM_PLAYERS] {
    if s.done {
        return [s.rewards[0] as f64, s.rewards[1] as f64];
    }
    let seat = s.acting as usize;
    let mask = legal(s);
    let key = info_key(s, seat);
    tab.legal_mask.insert(key.clone(), mask);
    let sigma = policy_for(tab, &key, &mask);

    let mut value = [0.0f64; NUM_PLAYERS];
    let mut child_value: [[f64; NUM_PLAYERS]; NUM_ACTIONS] = [[0.0; NUM_PLAYERS]; NUM_ACTIONS];
    for a in 0..NUM_ACTIONS {
        if mask[a] {
            let mut child_reach = reach;
            child_reach[seat] *= sigma[a];
            let cv = cfr(&apply_action(s, a), player, child_reach, weight, tab);
            for p in 0..NUM_PLAYERS {
                value[p] += sigma[a] * cv[p];
            }
            child_value[a] = cv;
        }
    }
    if seat != player {
        return value;
    }
    // counterfactual reach = product of every reach except the acting player's (opp reach * chance).
    let cf_reach = reach[1 - seat] * reach[NUM_PLAYERS];
    let reach_prob = reach[seat];
    let regret = tab.cum_regret.entry(key.clone()).or_insert([0.0; NUM_ACTIONS]);
    for a in 0..NUM_ACTIONS {
        if mask[a] {
            regret[a] += cf_reach * (child_value[a][seat] - value[seat]);
        }
    }
    let strat = tab.cum_policy.entry(key).or_insert([0.0; NUM_ACTIONS]);
    for a in 0..NUM_ACTIONS {
        if mask[a] {
            strat[a] += weight * reach_prob * sigma[a];
        }
    }
    value
}

/// Refresh `cur_policy` from `cum_regret` (regret-matching+ over the masked positive regrets) and floor
/// regrets to >=0 — done once per sweep, exactly as the Python loop does after each player's traversal.
fn refresh_strategy(tab: &mut Tables) {
    for regret in tab.cum_regret.values_mut() {
        for a in 0..NUM_ACTIONS {
            if regret[a] < 0.0 {
                regret[a] = 0.0;
            }
        }
    }
    let keys: Vec<InfoKey> = tab.cum_regret.keys().cloned().collect();
    for key in keys {
        let regret = tab.cum_regret[&key];
        let mask = tab.legal_mask[&key];
        let mut pos = [0.0f64; NUM_ACTIONS];
        let mut total = 0.0;
        for a in 0..NUM_ACTIONS {
            let v = if mask[a] { regret[a].max(0.0) } else { 0.0 };
            pos[a] = v;
            total += v;
        }
        let cur = if total > 0.0 {
            for a in 0..NUM_ACTIONS {
                pos[a] /= total;
            }
            pos
        } else {
            uniform_over_legal(&mask)
        };
        tab.cur_policy.insert(key, cur);
    }
}

/// Run `iterations` CFR+ sweeps (continuing from `tab.iters_done`). Alternating updates + linear
/// averaging (`weight = t`). Mutates `tab` in place so a checkpoint can be taken between calls.
pub fn train(tab: &mut Tables, iterations: u64) {
    let deals = all_deals();
    let chance = 1.0 / deals.len() as f64;
    let start = tab.iters_done + 1;
    let end = tab.iters_done + iterations;
    for t in start..=end {
        let weight = t as f64; // linear averaging
        for player in 0..NUM_PLAYERS {
            for &(cards, public) in &deals {
                let root = State::from_deal(cards, public);
                cfr(&root, player, [1.0, 1.0, chance], weight, tab);
            }
            refresh_strategy(tab);
        }
        tab.iters_done = t;
    }
}

/// Average strategy = `cum_policy` row-normalized (uniform-over-legal fallback for unseen keys at BR
/// time). Mirrors `tabular_cfr`'s final `avg` and `_policy_for_key`.
pub fn average_strategy(tab: &Tables) -> HashMap<InfoKey, [f64; NUM_ACTIONS]> {
    let mut avg = HashMap::with_capacity(tab.cum_policy.len());
    for (k, strat) in &tab.cum_policy {
        let total: f64 = strat.iter().sum();
        let mut row = *strat;
        if total > 0.0 {
            for a in 0..NUM_ACTIONS {
                row[a] /= total;
            }
        }
        avg.insert(k.clone(), row);
    }
    avg
}

// --------------------------------------------------------------------------------------------------
// information-set best response exploitability (EXACT — the gate ruler)
// --------------------------------------------------------------------------------------------------

fn policy_for_key(
    avg: &HashMap<InfoKey, [f64; NUM_ACTIONS]>,
    key: &InfoKey,
    mask: &[bool; NUM_ACTIONS],
) -> [f64; NUM_ACTIONS] {
    match avg.get(key) {
        None => uniform_over_legal(mask),
        Some(raw) => {
            let mut sigma = *raw;
            let mut s = 0.0;
            for a in 0..NUM_ACTIONS {
                if !mask[a] {
                    sigma[a] = 0.0;
                }
                s += sigma[a];
            }
            if s > 0.0 {
                for a in 0..NUM_ACTIONS {
                    sigma[a] /= s;
                }
                sigma
            } else {
                uniform_over_legal(mask)
            }
        }
    }
}

/// Profile value for `player` under the fixed average strategy (full tree expectation).
fn profile_value(
    s: &State,
    player: usize,
    avg: &HashMap<InfoKey, [f64; NUM_ACTIONS]>,
) -> f64 {
    if s.done {
        return s.rewards[player] as f64;
    }
    let seat = s.acting as usize;
    let mask = legal(s);
    let sigma = policy_for_key(avg, &info_key(s, seat), &mask);
    let mut val = 0.0;
    for a in 0..NUM_ACTIONS {
        if mask[a] {
            val += sigma[a] * profile_value(&apply_action(s, a), player, avg);
        }
    }
    val
}

/// Information-set best response for `br_seat` against the fixed `avg`. Two passes (collect infosets by
/// counterfactual reach, then pick the per-infoset action maximizing the cf-weighted action value),
/// mirroring `_InfosetBestResponse`. Returns the BR seat's root value averaged over the deals.
struct InfosetBR<'a> {
    br_seat: usize,
    avg: &'a HashMap<InfoKey, [f64; NUM_ACTIONS]>,
    infosets: HashMap<InfoKey, Vec<(State, f64)>>,
    br_action: HashMap<InfoKey, usize>,
}

impl<'a> InfosetBR<'a> {
    fn new(br_seat: usize, avg: &'a HashMap<InfoKey, [f64; NUM_ACTIONS]>) -> Self {
        InfosetBR { br_seat, avg, infosets: HashMap::new(), br_action: HashMap::new() }
    }

    fn collect(&mut self, s: &State, cf_prob: f64) {
        if s.done {
            return;
        }
        let seat = s.acting as usize;
        let mask = legal(s);
        if seat == self.br_seat {
            self.infosets
                .entry(info_key(s, seat))
                .or_default()
                .push((s.clone(), cf_prob));
            for a in 0..NUM_ACTIONS {
                if mask[a] {
                    self.collect(&apply_action(s, a), cf_prob);
                }
            }
        } else {
            let sigma = policy_for_key(self.avg, &info_key(s, seat), &mask);
            for a in 0..NUM_ACTIONS {
                if mask[a] {
                    self.collect(&apply_action(s, a), cf_prob * sigma[a]);
                }
            }
        }
    }

    fn br_action_at(&mut self, s: &State) -> usize {
        let key = info_key(s, self.br_seat);
        if let Some(&a) = self.br_action.get(&key) {
            return a;
        }
        let entries = self.infosets[&key].clone();
        let mask = legal(&entries[0].0);
        let mut best_a = 0usize;
        let mut best_v = f64::NEG_INFINITY;
        for a in 0..NUM_ACTIONS {
            if mask[a] {
                let mut q = 0.0;
                for (st, cf_p) in &entries {
                    q += cf_p * self.value(&apply_action(st, a));
                }
                if q > best_v {
                    best_v = q;
                    best_a = a;
                }
            }
        }
        self.br_action.insert(key, best_a);
        best_a
    }

    fn value(&mut self, s: &State) -> f64 {
        if s.done {
            return s.rewards[self.br_seat] as f64;
        }
        let seat = s.acting as usize;
        let mask = legal(s);
        if seat == self.br_seat {
            let a = self.br_action_at(s);
            self.value(&apply_action(s, a))
        } else {
            let sigma = policy_for_key(self.avg, &info_key(s, seat), &mask);
            let mut v = 0.0;
            for a in 0..NUM_ACTIONS {
                if mask[a] {
                    v += sigma[a] * self.value(&apply_action(s, a));
                }
            }
            v
        }
    }
}

/// Exact information-set BR exploitability in mbb (milli-big-blinds, ×1000), matching
/// `infoset_exploitability_mbb`. Returns (expl_mbb, u0, u1, br0, br1).
pub fn infoset_exploitability_mbb(
    avg: &HashMap<InfoKey, [f64; NUM_ACTIONS]>,
) -> (f64, f64, f64, f64, f64) {
    let deals = all_deals();
    let roots: Vec<State> = deals.iter().map(|&(c, p)| State::from_deal(c, p)).collect();

    let mut br0_calc = InfosetBR::new(0, avg);
    let mut br1_calc = InfosetBR::new(1, avg);
    for r in &roots {
        br0_calc.collect(r, 1.0);
        br1_calc.collect(r, 1.0);
    }
    let mut br0 = 0.0;
    let mut br1 = 0.0;
    let mut u0 = 0.0;
    let mut u1 = 0.0;
    for r in &roots {
        br0 += br0_calc.value(r);
        br1 += br1_calc.value(r);
        u0 += profile_value(r, 0, avg);
        u1 += profile_value(r, 1, avg);
    }
    let n = roots.len() as f64;
    br0 /= n;
    br1 /= n;
    u0 /= n;
    u1 /= n;
    let expl = ((br0 - u0).max(0.0) + (br1 - u1).max(0.0)) * 1000.0;
    (expl, u0, u1, br0, br1)
}

/// Convenience: train from scratch for `iterations` and return exploitability (mbb).
pub fn solve_and_exploitability(iterations: u64) -> (f64, f64, f64, f64, f64) {
    let mut tab = Tables::default();
    train(&mut tab, iterations);
    let avg = average_strategy(&tab);
    infoset_exploitability_mbb(&avg)
}

// --------------------------------------------------------------------------------------------------
// atomic checkpoint / resume (durable trainer state)
// --------------------------------------------------------------------------------------------------

/// Serialize the CFR+ tables to a self-describing byte blob (versioned, length-prefixed). The format is
/// trivially parseable; resume reconstructs `cur_policy`/`legal_mask` from the persisted regrets so a
/// (checkpoint -> resume -> continue) run reproduces a straight-through run bit-for-bit.
pub fn serialize(tab: &Tables) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"LDCFR001");
    out.extend_from_slice(&tab.iters_done.to_le_bytes());
    // We persist cum_regret + cum_policy keyed by InfoKey. cur_policy + legal_mask are DERIVED on resume
    // (legal_mask recomputed from the game tree; cur_policy = refresh_strategy from the floored regrets),
    // so they are intentionally not stored — smaller + impossible to desync.
    write_map(&mut out, &tab.cum_regret);
    write_map(&mut out, &tab.cum_policy);
    out
}

fn write_map(out: &mut Vec<u8>, m: &HashMap<InfoKey, [f64; NUM_ACTIONS]>) {
    out.extend_from_slice(&(m.len() as u64).to_le_bytes());
    // deterministic order so the blob is reproducible across runs with the same tables.
    let mut keys: Vec<&InfoKey> = m.keys().collect();
    keys.sort();
    for key in keys {
        out.extend_from_slice(&(key.len() as u32).to_le_bytes());
        for &x in key {
            out.extend_from_slice(&x.to_le_bytes());
        }
        for &v in &m[key] {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
}

fn read_u64(b: &[u8], p: &mut usize) -> u64 {
    let v = u64::from_le_bytes(b[*p..*p + 8].try_into().unwrap());
    *p += 8;
    v
}
fn read_u32(b: &[u8], p: &mut usize) -> u32 {
    let v = u32::from_le_bytes(b[*p..*p + 4].try_into().unwrap());
    *p += 4;
    v
}
fn read_f64(b: &[u8], p: &mut usize) -> f64 {
    let v = f64::from_le_bytes(b[*p..*p + 8].try_into().unwrap());
    *p += 8;
    v
}
fn read_i64(b: &[u8], p: &mut usize) -> i64 {
    let v = i64::from_le_bytes(b[*p..*p + 8].try_into().unwrap());
    *p += 8;
    v
}

fn read_map(b: &[u8], p: &mut usize) -> HashMap<InfoKey, [f64; NUM_ACTIONS]> {
    let n = read_u64(b, p) as usize;
    let mut m = HashMap::with_capacity(n);
    for _ in 0..n {
        let klen = read_u32(b, p) as usize;
        let mut key = Vec::with_capacity(klen);
        for _ in 0..klen {
            key.push(read_i64(b, p));
        }
        let mut row = [0.0f64; NUM_ACTIONS];
        for a in 0..NUM_ACTIONS {
            row[a] = read_f64(b, p);
        }
        m.insert(key, row);
    }
    m
}

/// Reconstruct `Tables` from a blob and re-derive `cur_policy`/`legal_mask` (so a resumed solve is
/// identical to one that never stopped). Returns None on a bad magic/version.
pub fn deserialize(b: &[u8]) -> Option<Tables> {
    if b.len() < 16 || &b[0..8] != b"LDCFR001" {
        return None;
    }
    let mut p = 8usize;
    let iters_done = read_u64(b, &mut p);
    let cum_regret = read_map(b, &mut p);
    let cum_policy = read_map(b, &mut p);
    let mut tab = Tables {
        cum_regret,
        cum_policy,
        cur_policy: HashMap::new(),
        legal_mask: HashMap::new(),
        iters_done,
    };
    // Re-derive legal_mask by walking the tree once, then refresh cur_policy from the (already floored)
    // regrets — exactly the post-sweep state of a straight-through run.
    rebuild_legal_masks(&mut tab);
    refresh_strategy(&mut tab);
    Some(tab)
}

/// Walk the full game tree, populating `legal_mask` for every reachable infoset key (needed before a
/// resumed `refresh_strategy`/`train`).
fn rebuild_legal_masks(tab: &mut Tables) {
    let deals = all_deals();
    for &(cards, public) in &deals {
        walk_masks(&State::from_deal(cards, public), tab);
    }
}

fn walk_masks(s: &State, tab: &mut Tables) {
    if s.done {
        return;
    }
    let seat = s.acting as usize;
    let mask = legal(s);
    tab.legal_mask.insert(info_key(s, seat), mask);
    for a in 0..NUM_ACTIONS {
        if mask[a] {
            walk_masks(&apply_action(s, a), tab);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deal_enumeration_is_90() {
        let d = all_deals();
        // 6!/(2!*2!*2!) = 90 distinct deck permutations (projections repeat — matches Python).
        assert_eq!(d.len(), 90);
        // every card rank is in 0..3 and public is a valid rank.
        for (c, p) in &d {
            assert!((0..3).contains(&c[0]) && (0..3).contains(&c[1]) && (0..3).contains(p));
        }
    }

    #[test]
    fn cfr_plus_clears_leduc_gate_at_1000_iters() {
        // The OpenSpiel-validated anchor: tabular CFR+ on Leduc is < ~1 mbb @ 1000 iters.
        let (expl, u0, u1, _, _) = solve_and_exploitability(1000);
        assert!((u0 + u1).abs() < 1e-6, "profile not zero-sum: {u0} {u1}");
        assert!(expl < 1.0, "exploitability {expl} mbb not below the 1.0 mbb gate @1000 iters");
    }

    #[test]
    fn exploitability_monotone_decreasing() {
        let (e100, ..) = solve_and_exploitability(100);
        let (e1000, ..) = solve_and_exploitability(1000);
        assert!(e1000 < e100, "more iters must not increase exploitability: {e100} -> {e1000}");
    }

    #[test]
    fn checkpoint_resume_reproduces_straight_run() {
        // straight-through 1000 iters
        let mut straight = Tables::default();
        train(&mut straight, 1000);
        let avg_straight = average_strategy(&straight);
        let (expl_straight, ..) = infoset_exploitability_mbb(&avg_straight);

        // 600 iters -> serialize -> deserialize -> 400 more iters
        let mut part = Tables::default();
        train(&mut part, 600);
        let blob = serialize(&part);
        let mut resumed = deserialize(&blob).expect("deserialize");
        assert_eq!(resumed.iters_done, 600);
        train(&mut resumed, 400);
        assert_eq!(resumed.iters_done, 1000);
        let avg_resumed = average_strategy(&resumed);
        let (expl_resumed, ..) = infoset_exploitability_mbb(&avg_resumed);

        // exact reproduction: the resumed tables must equal the straight-through tables.
        assert_eq!(resumed.cum_regret.len(), straight.cum_regret.len());
        for (k, v) in &straight.cum_regret {
            let r = resumed.cum_regret.get(k).expect("missing regret key after resume");
            for a in 0..NUM_ACTIONS {
                assert!((r[a] - v[a]).abs() < 1e-9, "regret mismatch at {k:?}[{a}]: {} vs {}", r[a], v[a]);
            }
        }
        for (k, v) in &straight.cum_policy {
            let r = resumed.cum_policy.get(k).expect("missing policy key after resume");
            for a in 0..NUM_ACTIONS {
                assert!((r[a] - v[a]).abs() < 1e-9, "policy mismatch at {k:?}[{a}]");
            }
        }
        assert!((expl_straight - expl_resumed).abs() < 1e-9,
                "resume changed exploitability: {expl_straight} vs {expl_resumed}");
    }
}
