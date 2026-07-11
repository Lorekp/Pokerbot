//! SERVE-side OFF-MENU ACTION TRANSLATION for blueprint KEY construction
//! (`POKERBOT_ACTION_TRANSLATE=1`, default OFF => byte-identical everywhere).
//!
//! WHY (2026-07-11 serve-coverage root-cause): a fullhist/presum-keyed blueprint bakes the exact
//! abstracted action-id sequence into every lookup key. Any OPPONENT action outside the TRAINED
//! menu (the trainer's `abstracted_legal_menus` thinning: PRE_KEEP / PRE_RR_KEEP / M-RR keep-lists)
//! lands the history — and therefore EVERY subsequent key of the hand — outside the trained key
//! space, degrading the serve to uniform-over-menu for the rest of the hand. Pluribus/Libratus
//! answered this with ACTION TRANSLATION (pseudo-harmonic mapping, Ganzfried & Sandholm 2013):
//! map each observed off-menu action to the nearest in-menu size BEFORE key construction.
//!
//! WHAT: `translate_state_for_key(s)` replays the recorded hand history from a reconstructed root
//! through the RAW frozen `core::apply_action` (the closure-injected CHECK is already RECORDED in
//! `hist_action` — same convention as `fingerprint::n_preflop_boundary`), substituting every
//! recorded action that is OFF the trained menu at its replayed decision point with the
//! pseudo-harmonic-nearest trained-menu action, and returns the fully replayed TRANSLATED state
//! (betting scalars, presum boundary inputs, and the 6-byte S-segment all come from the translated
//! world, so the key lands in trained key space). Board/hole cards are reproduced exactly by
//! replaying with the REAL deck. In-menu histories replay to a state field-identical to the input,
//! and the function returns the INPUT state whenever nothing was translated — byte-identity by
//! construction when there is nothing to translate.
//!
//! Deterministic pseudo-harmonic choice: sizes are compared as raise-to totals (the seat's
//! committed_street AFTER the action) in exact chips; between the two bracketing menu sizes A <= x
//! <= B the observed x maps to A iff f_{A,B}(x) >= 1/2 where f is the pseudo-harmonic mapping
//! f = ((B-x)(P+A)) / ((B-A)(P+x)) with P the sizing pot — evaluated in EXACT integer arithmetic
//! (i128 cross-multiplication), so the Rust and Python twins can never diverge on float rounding.
//! Below the smallest menu size we CLAMP UP to it (never down to a call — misrepresenting a bet as
//! a call flips aggression semantics); above the largest bet bucket the upper bracket is ALL-IN.
//!
//! FAIL-OPEN: any replay divergence (recorded seat mismatch, invalid action id, early terminal,
//! street mismatch, translated-action illegality) returns the ORIGINAL state untouched — exactly
//! the pre-translation behavior. The Python twin (`pokerbot/engine/action_translate.py`) implements
//! the identical decision procedure; parity is gated by scripts/probe_fullhist_key_parity.py
//! --translate.
//!
//! SCOPE: consulted ONLY by the fingerprint.rs SERVE entry points (`fingerprint_fast`,
//! `fingerprint_fast_depth`, `fingerprint_fast_depth_emd`). The trainer's own keying
//! (trainer.rs::fingerprint_fast_depth_memo_emd) is untouched — training generates in-menu actions
//! only, where translation is the identity anyway.

use crate::core::{self, EnvStateN, NUM_ACTIONS};
use crate::trainer::{allowed_mask, keep_mask_from_env, parse_keep_list};

/// `POKERBOT_ACTION_TRANSLATE=1` (default OFF). Read once per process (OnceLock, the standing
/// env-before-launch configuration convention).
#[inline]
pub fn enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("POKERBOT_ACTION_TRANSLATE")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// Trained-menu configuration, latched once from the SAME env knobs the blueprint was trained
/// with (trainer.rs ctor defaults when unset — never hardcoded sizes).
struct MenuCfg {
    allowed_pre: [bool; NUM_ACTIONS],
    allowed_post: [bool; NUM_ACTIONS],
    pre_rr: Option<[bool; NUM_ACTIONS]>,
    flop_rr: Option<[bool; NUM_ACTIONS]>,
    mrr: bool,
}

fn opt_keep_env(var: &str) -> Option<[bool; NUM_ACTIONS]> {
    match std::env::var(var) {
        Ok(s) if !s.trim().is_empty() => {
            let keep = parse_keep_list(&s).unwrap_or_else(|e| panic!("{var}={s:?} invalid: {e}"));
            Some(allowed_mask(&keep))
        }
        _ => None,
    }
}

fn menu_cfg() -> &'static MenuCfg {
    use std::sync::OnceLock;
    static CFG: OnceLock<MenuCfg> = OnceLock::new();
    CFG.get_or_init(|| MenuCfg {
        // Same env vars + compiled defaults as the trainer ctor sites (trainer.rs
        // keep_mask_from_env("POKERBOT_PRE_KEEP", &[4,5,6]) / ("POKERBOT_POST_KEEP", &[0,1,3,4,5])).
        allowed_pre: keep_mask_from_env("POKERBOT_PRE_KEEP", &[4, 5, 6]),
        allowed_post: keep_mask_from_env("POKERBOT_POST_KEEP", &[0, 1, 3, 4, 5]),
        pre_rr: opt_keep_env("POKERBOT_PRE_RR_KEEP"),
        flop_rr: opt_keep_env("POKERBOT_FLOP_RR_KEEP"),
        mrr: std::env::var("POKERBOT_MRR")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false),
    })
}

/// M-RR postflop re-raise keep {1.0x pot} and turn/river first-raise keep {0.5x, 1.0x pot} —
/// the same compiled lists as trainer.rs `mrr_reraise_keep` / `mrr_tr_first_keep`.
fn mrr_reraise_keep() -> [bool; NUM_ACTIONS] {
    allowed_mask(&[4])
}
fn mrr_tr_first_keep() -> [bool; NUM_ACTIONS] {
    allowed_mask(&[1, 4])
}

/// TRAINED-MENU legal mask at a (replayed) state — the serve twin of
/// trainer.rs::abstracted_legal_menus with the TrainerCfg keep-lists replaced by the env-latched
/// `MenuCfg` (same node classification, byte-for-byte the same keep-list choices).
pub fn trained_legal<const N: usize>(s: &EnvStateN<N>, seat: usize) -> [bool; NUM_ACTIONS] {
    let cfg = menu_cfg();
    let full = core::legal_action_mask(s, seat as i32);
    let allowed: [bool; NUM_ACTIONS] = if s.street == 0 {
        if s.current_bet > core::BB_CHIPS {
            cfg.pre_rr.unwrap_or(cfg.allowed_pre)
        } else {
            cfg.allowed_pre
        }
    } else if cfg.mrr {
        let facing_bet = s.current_bet > s.committed_street[seat];
        if facing_bet {
            if s.street == 1 {
                cfg.flop_rr.unwrap_or_else(mrr_reraise_keep)
            } else {
                mrr_reraise_keep()
            }
        } else if s.street >= 2 {
            mrr_tr_first_keep()
        } else {
            cfg.allowed_post
        }
    } else {
        cfg.allowed_post
    };
    let mut m = [false; NUM_ACTIONS];
    for a in 0..NUM_ACTIONS {
        m[a] = full[a] && allowed[a];
    }
    m
}

/// Sizing pot — twin of the private `core::pot_for_sizing` (pot_dead + committed_street sum).
#[inline]
fn pot_for_sizing<const N: usize>(s: &EnvStateN<N>) -> i32 {
    let mut p = s.pot_dead;
    for i in 0..N {
        p += s.committed_street[i];
    }
    p
}

/// Twin of the private `core::bucket_prod` (f32 product truncated toward zero — np.int32(f32)).
#[inline]
fn bucket_prod(pot: i32, frac: f32) -> i32 {
    ((pot as f32) * frac) as i32
}

/// Chip amount the engine would move for `action` at this state — twin of the private
/// `core::action_amount` (pinned by `translate_amount_matches_apply_action` against the observable
/// committed delta of the frozen `core::apply_action`).
fn action_amount<const N: usize>(s: &EnvStateN<N>, seat: usize, action: usize) -> i32 {
    let committed = s.committed_street[seat];
    let to_call = (s.current_bet - committed).max(0);
    let stack = s.stacks[seat];
    let pot = pot_for_sizing(s);
    if action == core::CALL {
        return to_call;
    }
    if action == core::ALL_IN {
        return stack;
    }
    if (core::BET_BUCKET_START..core::ALL_IN).contains(&action) {
        let frac = core::BET_BUCKETS_F32[action - core::BET_BUCKET_START];
        let prod = bucket_prod(pot, frac);
        let target = s.current_bet.max(prod);
        let bucket_val = (target - committed).max(0);
        return to_call + bucket_val;
    }
    0
}

/// Raise-to total (the seat's committed_street AFTER the action) — the exact-chip "size" scalar
/// the pseudo-harmonic mapping compares.
#[inline]
fn raise_to_total<const N: usize>(s: &EnvStateN<N>, seat: usize, action: usize) -> i64 {
    s.committed_street[seat] as i64 + action_amount(s, seat, action) as i64
}

/// Deterministic pseudo-harmonic choice between bracketing sizes `a_tot <= x_tot <= b_tot`
/// (raise-to totals, sizing pot `pot`): map to A iff f_{A,B}(x) >= 1/2, i.e.
/// 2*(B-x)*(P+A) >= (B-A)*(P+x), evaluated exactly in i128.
#[inline]
fn pseudo_harmonic_pick_lower(a_tot: i64, b_tot: i64, x_tot: i64, pot: i64) -> bool {
    let (a, b, x, p) = (a_tot as i128, b_tot as i128, x_tot as i128, pot as i128);
    2 * (b - x) * (p + a) >= (b - a) * (p + x)
}

/// Translate ONE off-menu bet-bucket/oversize action at the replayed state to the nearest trained-
/// menu action. Returns None when no deterministic in-menu betting substitute exists (caller then
/// falls back to CALL/CHECK or fails open).
fn translate_bet<const N: usize>(
    s: &EnvStateN<N>,
    seat: usize,
    action: usize,
    menu: &[bool; NUM_ACTIONS],
) -> Option<usize> {
    let pot = pot_for_sizing(s).max(1) as i64;
    let x_tot = raise_to_total(s, seat, action);
    // Candidates: trained-menu bet buckets that keep the seat LIVE after the action
    // (raise_by < stack — a bucket that consumes the whole stack would flip the seat to all-in and
    // desynchronize the recorded actor sequence), plus ALL-IN when in menu. Ascending action id
    // gives non-decreasing totals for buckets; all-in appended last (its total is >= any live
    // bucket total by the stack bound).
    let mut cand: Vec<(usize, i64)> = Vec::with_capacity(8);
    for b in core::BET_BUCKET_START..core::ALL_IN {
        if menu[b] && action_amount(s, seat, b) < s.stacks[seat] {
            cand.push((b, raise_to_total(s, seat, b)));
        }
    }
    if menu[core::ALL_IN] {
        cand.push((core::ALL_IN, raise_to_total(s, seat, core::ALL_IN)));
    }
    if cand.is_empty() {
        return None;
    }
    // Clamp below the smallest / above the largest.
    if x_tot <= cand[0].1 {
        return Some(cand[0].0);
    }
    if x_tot >= cand[cand.len() - 1].1 {
        return Some(cand[cand.len() - 1].0);
    }
    // Bracket: first candidate with total >= x (dedupe-safe: equal totals fall out via the clamp
    // and the strict A < x < B arithmetic below).
    let hi = cand.iter().position(|&(_, t)| t >= x_tot).unwrap();
    let (b_act, b_tot) = cand[hi];
    let (a_act, a_tot) = cand[hi - 1];
    if a_tot == b_tot {
        return Some(a_act);
    }
    Some(if pseudo_harmonic_pick_lower(a_tot, b_tot, x_tot, pot) {
        a_act
    } else {
        b_act
    })
}

/// Replay-translate the state's recorded history into the trained action abstraction and return
/// the fully replayed TRANSLATED state for key construction. Returns the ORIGINAL state whenever
/// nothing needed translating OR the replay could not be completed faithfully (fail-open).
pub fn translate_state_for_key<const N: usize>(s: &EnvStateN<N>) -> EnvStateN<N> {
    match translate_impl(s) {
        Some(t) => t,
        None => *s,
    }
}

fn translate_impl<const N: usize>(s: &EnvStateN<N>) -> Option<EnvStateN<N>> {
    let hist_len = (s.hist_len.max(0) as usize).min(core::MAX_HISTORY);
    if hist_len == 0 {
        return None;
    }
    // Root reconstruction — same recipe as fingerprint::n_preflop_boundary (starting stack =
    // stacks + committed_total, occupancy = not-EMPTY, ante=0) but with the REAL deck so chance
    // deals reproduce the exact board/hole cards.
    let mut start_stacks = [0i32; N];
    let mut occupancy = [false; N];
    for i in 0..N {
        start_stacks[i] = s.stacks[i] + s.committed_total[i];
        occupancy[i] = s.seat_status[i] != core::SEAT_EMPTY;
    }
    let mut r = core::init_hand_n::<N>(s.button, &start_stacks, &occupancy, 0, &s.deck, s.rng_key);
    let mut changed = false;
    let mut injected: usize = 0;
    // Recorded seats are a CONSISTENCY SIGNAL, not a replay driver: the production GameState engine
    // (pool evals / live bridge, engine/game.py play_hand) advances past `_first_postflop_actor`
    // unconditionally, so its postflop action order starts one live seat LATE (BB-first) vs the
    // trainer core's real-NLHE SB-first order — recorded seats cannot be matched against the
    // trainer engine step-for-step. The replay therefore follows the TRAINER engine's own actor
    // rotation and replays the recorded ACTION BYTES (the only history component the key hashes);
    // `seats_consistent` (np-engine self-play streams, which come from this very engine) selects
    // the exact recorded-closure-CHECK consumption below.
    let mut seats_consistent = true;
    for i in 0..hist_len {
        while !r.done && core::is_chance_node(&r) {
            r = core::sample_chance(&r);
        }
        if r.done || r.action_seat < 0 {
            return None; // replay diverged — fail open
        }
        seats_consistent = seats_consistent && r.action_seat == s.hist_seat[i];
        let recorded = s.hist_action[i];
        if !(0..NUM_ACTIONS as i32).contains(&recorded) {
            return None;
        }
        let mut a = recorded as usize;
        let seat = r.action_seat as usize;
        let menu = trained_legal(&r, seat);
        if !menu[a] {
            if (core::BET_BUCKET_START..=core::ALL_IN).contains(&a) {
                match translate_bet(&r, seat, a, &menu) {
                    Some(t) => a = t,
                    None => {
                        // No in-menu betting substitute: deterministic non-bet fallback.
                        if menu[core::CALL] {
                            a = core::CALL;
                        } else if menu[core::CHECK] {
                            a = core::CHECK;
                        } else {
                            return None;
                        }
                    }
                }
                changed = changed || a != recorded as usize;
            } else {
                // F/X/C/AI are in every keep-list, so an off-menu non-bet recorded action means the
                // replayed world genuinely diverged from the recorded one — fail open.
                return None;
            }
        }
        changed = changed || !seats_consistent;
        // RAW frozen step (closure-injected CHECKs are recorded tokens; replaying them raw
        // reproduces the trainer's transitions — the d33ba74 presum-boundary convention).
        r = core::apply_action::<N>(&r, a);
        // CLOSURE-CHECK SYNTHESIS (train/serve parity, 2026-07-11 root-cause): under
        // POKERBOT_CLOSE_CALLED_STREET=1 the TRAINER records the closure-injected closing CHECK in
        // hist_action, so it is baked into every trained postflop key. Histories bridged from REAL
        // tables / the pool harness (state_bridge::game_state_to_env copies GameState.action_history
        // verbatim) do NOT contain that token — real NLHE closes the street on the call — so every
        // called bet/raise line missed on every later street (measured: pool flop 56% / turn 92% /
        // river 98% miss with all-in-menu tokens). When the replayed child is the aggressor-reopen
        // node (closure.rs predicate): np-engine self-play streams (seats consistent so far) carry
        // the recorded closing CHECK as their next token — consume it naturally; any other stream
        // gets the CHECK synthesized exactly like the trainer did.
        if crate::closure::enabled() {
            let ag = r.action_seat;
            let reopen = !r.done
                && !r.pending_deal
                && r.street < core::STREET_TERMINAL
                && ag >= 0
                && ag == r.last_aggressor
                && r.current_bet == r.committed_street[ag as usize];
            if reopen {
                let consume_recorded = seats_consistent
                    && i + 1 < hist_len
                    && s.hist_seat[i + 1] == ag
                    && s.hist_action[i + 1] == core::CHECK as i32;
                if !consume_recorded {
                    r = core::apply_action::<N>(&r, core::CHECK);
                    injected += 1;
                    changed = true;
                }
            }
        }
    }
    if !changed {
        return None; // in-menu history: the input state IS the translated state (byte-identity).
    }
    while !r.done && core::is_chance_node(&r) {
        r = core::sample_chance(&r);
    }
    // The translated world must land on the same street/decision structure as the real one —
    // otherwise the key would describe a different node class than the state served. The acting
    // seat must ALSO be the hero's: the key's S-segment (to_call, facing_agg) and hole/weva bytes
    // are hero-relative, so a replay whose actor rotation diverged from the recorded game (e.g. a
    // legacy BB-first-postflop GameState history, engine/game.py pre-POSTFLOP_SB_FIRST) would key
    // a DIFFERENT seat's decision — fail open instead.
    if r.done
        || r.street != s.street
        || r.hist_len as usize != hist_len + injected
        || r.action_seat != s.action_seat
    {
        return None;
    }
    // CARD FIELDS come from the ORIGINAL state: a live/pool-bridged state's `deck` is a canonical
    // placeholder (state_bridge fills [0..52)), so the replay's dealt cards are bookkeeping only.
    // The betting replay never reads card VALUES; the key's card segments (postflop bucket, weva)
    // must read the REAL observed cards.
    r.holes = s.holes;
    r.board = s.board;
    r.deck = s.deck;
    r.deck_ptr = s.deck_ptr;
    Some(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{
        apply_action, init_hand_n, is_chance_node, legal_action_mask, sample_chance, EnvState,
        NUM_PLAYERS,
    };

    struct TestRng(u64);
    impl TestRng {
        fn new(seed: u64) -> Self {
            TestRng(seed ^ 0x9E37_79B9_7F4A_7C15)
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next_u64() % n as u64) as usize
        }
    }

    fn shuffled_deck(rng: &mut TestRng) -> [i32; 52] {
        let mut deck: [i32; 52] = std::array::from_fn(|i| i as i32);
        for i in (1..52).rev() {
            let j = rng.below(i + 1);
            deck.swap(i, j);
        }
        deck
    }

    fn play_random_hand(rng: &mut TestRng, button: i32) -> Vec<EnvState> {
        let deck = shuffled_deck(rng);
        let stacks = [100 * core::CHIP_UNIT; NUM_PLAYERS];
        let occ = [true; NUM_PLAYERS];
        let mut s = init_hand_n::<NUM_PLAYERS>(button, &stacks, &occ, 0, &deck, [7, 9]);
        let mut states = Vec::new();
        let mut guard = 0;
        while !s.done && guard < 200 {
            guard += 1;
            if is_chance_node(&s) {
                s = sample_chance(&s);
                continue;
            }
            if s.action_seat < 0 {
                break;
            }
            states.push(s);
            let mask = legal_action_mask(&s, s.action_seat);
            let legal: Vec<usize> = (0..NUM_ACTIONS).filter(|&a| mask[a]).collect();
            // Check/call bias so hands reach postflop.
            let mut pool = Vec::new();
            for &a in &legal {
                let w = if a == core::CHECK || a == core::CALL { 6 } else { 1 };
                for _ in 0..w {
                    pool.push(a);
                }
            }
            s = apply_action(&s, pool[rng.below(pool.len())]);
        }
        states
    }

    /// The private `action_amount` twin matches the frozen engine's observable committed delta.
    #[test]
    fn translate_amount_matches_apply_action() {
        let mut rng = TestRng::new(11);
        let mut checked = 0usize;
        for h in 0..200 {
            for s in play_random_hand(&mut rng, (h % 6) as i32) {
                let seat = s.action_seat as usize;
                let mask = legal_action_mask(&s, s.action_seat);
                for a in 0..NUM_ACTIONS {
                    if !mask[a] || a == core::FOLD || a == core::CHECK {
                        continue;
                    }
                    let child = apply_action(&s, a);
                    // The chips the seat put in = its stack delta (committed_street resets when the
                    // action closes the street, so the stack is the persistent observable).
                    let want = s.stacks[seat] - child.stacks[seat];
                    // apply_action clamps the amount to the stack; our twin reproduces the raw
                    // amount, so compare through the same clamp.
                    let got = action_amount(&s, seat, a).min(s.stacks[seat]);
                    assert_eq!(got, want, "action {a} at hand {h}");
                    checked += 1;
                }
            }
        }
        assert!(checked > 2000, "coverage too thin: {checked}");
    }

    /// In-menu histories are the identity: replaying a hand whose every action is drawn from the
    /// trained menu returns the INPUT state (fail-open/no-change path), for every decision state.
    #[test]
    fn translate_identity_on_in_menu_histories() {
        let mut rng = TestRng::new(23);
        let mut n = 0usize;
        for h in 0..100 {
            let deck = shuffled_deck(&mut rng);
            let stacks = [100 * core::CHIP_UNIT; NUM_PLAYERS];
            let occ = [true; NUM_PLAYERS];
            let mut s = init_hand_n::<NUM_PLAYERS>((h % 6) as i32, &stacks, &occ, 0, &deck, [3, 5]);
            let mut guard = 0;
            while !s.done && guard < 200 {
                guard += 1;
                if is_chance_node(&s) {
                    s = sample_chance(&s);
                    continue;
                }
                if s.action_seat < 0 {
                    break;
                }
                // Decision state: translation of an in-menu history must return the input.
                let t = translate_state_for_key(&s);
                assert_eq!(
                    format!("{:?}", t),
                    format!("{:?}", s),
                    "identity violated at hand {h} guard {guard}"
                );
                n += 1;
                let menu = trained_legal(&s, s.action_seat as usize);
                let legal: Vec<usize> = (0..NUM_ACTIONS).filter(|&a| menu[a]).collect();
                let mut pool = Vec::new();
                for &a in &legal {
                    let w = if a == core::CHECK || a == core::CALL { 6 } else { 1 };
                    for _ in 0..w {
                        pool.push(a);
                    }
                }
                s = apply_action(&s, pool[rng.below(pool.len())]);
            }
        }
        assert!(n > 400, "coverage too thin: {n}");
    }

    /// An off-menu opponent bet is remapped INTO the trained menu: the translated state's history
    /// contains only trained-menu actions at every replayed decision point.
    #[test]
    fn translate_lands_in_menu_on_off_menu_histories() {
        let mut rng = TestRng::new(31);
        let mut translated = 0usize;
        for h in 0..300 {
            for s in play_random_hand(&mut rng, (h % 6) as i32) {
                let t = translate_state_for_key(&s);
                if format!("{:?}", t) == format!("{:?}", s) {
                    continue; // identity (in-menu) or fail-open
                }
                translated += 1;
                // Re-translate: a translated state must be a fixed point (all tokens in-menu).
                let t2 = translate_state_for_key(&t);
                assert_eq!(
                    format!("{:?}", t2),
                    format!("{:?}", t),
                    "translated state not a fixed point at hand {h}"
                );
                // Same public node class.
                assert_eq!(t.street, s.street);
                assert_eq!(t.action_seat, s.action_seat);
                assert_eq!(t.hist_len, s.hist_len);
                assert_eq!(t.board, s.board);
                assert_eq!(t.holes, s.holes);
            }
        }
        assert!(translated > 50, "off-menu coverage too thin: {translated}");
    }
}
