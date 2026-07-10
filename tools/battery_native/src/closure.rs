//! Gated REAL-NLHE street closure (`POKERBOT_CLOSE_CALLED_STREET=1`, default OFF => byte-identical
//! everywhere). SINGLE shared implementation used by every engine-stepping consumer (trainer, subgame
//! solve, playout rollouts, search_session, live solver, LBR) so train and serve play the SAME game.
//!
//! THE BUG (verified 2026-07-10, artifacts_wt/_close_called_street/step0_*.log): the FROZEN
//! `core::street_complete` requires `back_to_aggressor` — action must physically RETURN to the last
//! aggressor — but it is evaluated BEFORE `apply_action` advances `action_seat`, so it can never be
//! satisfied on the closing CALL itself. Every called bet/raise therefore deals the AGGRESSOR an extra
//! decision node with re-open rights (raise your own called bet), on every street. Real NLHE
//! (Robert's Rules; cross-checked live vs third_party/pypokerengine `__is_everyone_agreed`) closes the
//! betting round on the last call: the aggressor gets NO further node unless re-raised. The asymmetry
//! lands exactly on open/bet lines (a called equal-stack jam closes via `n_can_act == 0`; limped pots
//! close via `no_aggressor`), so under unconverged play the extra nodes systematically burn the
//! aggressor's EV — the root cause of the degenerate EP premium-jam preflop.
//!
//! THE FIX (core.rs is FROZEN — no `street_complete` edit): auto-inject the closing CHECK whenever a
//! child state is the extra aggressor-reopen node (action back at `last_aggressor` with nothing to
//! call). Detection is exact: mid-street, action returns to the aggressor only after every other live
//! seat called or folded (any re-raise would have replaced `last_aggressor`), so the predicate below
//! identifies precisely the node real NLHE would not deal. `sample_chance` resets
//! `last_aggressor = -1` + `acted` each street and blinds are not aggression (`init_hand` sets
//! `last_aggressor = -1`), so the BB option node (a LEGAL real-NLHE node: `no_aggressor` + `acted`)
//! is never touched. The injected CHECK is always legal (`to_call == 0`) and never settles a terminal
//! inside `apply_action` (it closes the street -> chance node), so the scalar `core::apply_action` is
//! byte-safe here. NOTE the injected CHECK is recorded in `hist_*` like any action; blueprint keying
//! reads state scalars (playout::fp_key / fingerprint), never the hist arrays, and the live-bridge
//! replay (`subgame::replay_recorded` with `insert_phantom_checks`) already synthesizes exactly this
//! check when reconstructing real-table histories — so closure-ON is strictly CLOSER to live reality.
//!
//! TRAIN/SERVE CONSISTENCY: the flag is read ONCE per process (OnceLock, same convention as
//! POKERBOT_MRR); the Python mirrors (np_engine / jax_engine) read the same variable, so one env var
//! flips the whole stack coherently. Default OFF => `maybe_close_called_street` is the identity and
//! every golden / parity gate stays byte-identical.

use crate::core::{self, EnvStateN};

/// `POKERBOT_CLOSE_CALLED_STREET=1` (default OFF). Read once per process.
#[inline]
pub fn enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("POKERBOT_CLOSE_CALLED_STREET")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// Identify the extra aggressor-reopen node on `child` and, when the gate is ON, inject the closing
/// CHECK so the played game tree matches deployed real NLHE. Identity when the gate is OFF.
#[inline]
pub fn maybe_close_called_street<const N: usize>(child: EnvStateN<N>) -> EnvStateN<N> {
    if !enabled() {
        return child;
    }
    close_called_street(child)
}

/// The unconditional (gate-independent) closure step, split out so tests can exercise the predicate
/// without touching process env. Injects the closing CHECK iff `child` is the aggressor-reopen node.
#[inline]
pub fn close_called_street<const N: usize>(child: EnvStateN<N>) -> EnvStateN<N> {
    let a = child.action_seat;
    if !child.done
        && !child.pending_deal
        && child.street < core::STREET_TERMINAL
        && a >= 0
        && a == child.last_aggressor
        && child.current_bet == child.committed_street[a as usize]
    {
        // Extra aggressor-reopen node: real NLHE closed the street on the last call — inject the
        // closing CHECK so the game tree matches deployed NLHE.
        return core::apply_action(&child, core::CHECK);
    }
    child
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::NUM_PLAYERS;

    fn root(seed: u64) -> EnvStateN<NUM_PLAYERS> {
        let stacks = [100 * core::CHIP_UNIT; NUM_PLAYERS]; // 100bb, same as trainer::START_STACK_CHIPS
        let occupancy = [true; NUM_PLAYERS];
        let mut deck: Vec<i32> = (0..52).collect();
        // xorshift shuffle (deterministic, seed-dependent)
        let mut x = seed | 1;
        for i in (1..52usize).rev() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            deck.swap(i, (x % (i as u64 + 1)) as usize);
        }
        let deck: [i32; 52] = deck.try_into().unwrap();
        core::init_hand_n::<NUM_PLAYERS>(0, &stacks, &occupancy, 0, &deck, [seed as u32, 7])
    }

    /// The predicate fires exactly on the called-open reopen node and the injected CHECK closes the
    /// street; and it does NOT fire on the legal BB-option node of a limped pot.
    #[test]
    fn closure_predicate_exact_on_called_open_and_inert_on_bb_option() {
        let s0 = root(42);
        // UTG opens (first legal raise bucket), everyone folds to BB, BB calls.
        let utg = s0.action_seat as usize;
        let mask = core::legal_action_mask(&s0, s0.action_seat);
        let open = (core::BET_BUCKET_START..core::NUM_ACTIONS)
            .find(|&a| mask[a])
            .expect("no legal raise at root");
        let mut s = core::apply_action(&s0, open);
        assert_eq!(s.last_aggressor, utg as i32);
        // fold everyone except the BB (the seat already in for BB_CHIPS)
        loop {
            let seat = s.action_seat as usize;
            if s.committed_street[seat] == core::BB_CHIPS {
                break; // BB to act
            }
            s = core::apply_action(&s, core::FOLD);
        }
        let called = core::apply_action(&s, core::CALL);
        // VARIANT ENGINE: the called open leaves preflop OPEN with the aggressor to act again.
        assert!(!called.pending_deal && called.street == core::STREET_PREFLOP);
        assert_eq!(called.action_seat, utg as i32, "aggressor dealt the reopen node");
        // CLOSURE: the injected check closes preflop -> chance node (flop pending).
        let closed = close_called_street(called);
        assert!(closed.pending_deal, "closure must close the street");
        assert_eq!(closed.street, core::STREET_PREFLOP); // close_street leaves street; sample_chance advances
        // Idempotent on the closed state.
        let again = close_called_street(closed);
        assert_eq!(again.pending_deal, closed.pending_deal);
        assert_eq!(again.hist_len, closed.hist_len);

        // CONTROL: limped pot — everyone calls to the BB; the BB option node must NOT be closed.
        let mut l = root(42);
        loop {
            let seat = l.action_seat as usize;
            if l.committed_street[seat] == core::BB_CHIPS && l.current_bet == core::BB_CHIPS {
                break; // BB option node
            }
            l = core::apply_action(&l, core::CALL);
        }
        assert!(l.last_aggressor < 0, "limped pot has no aggressor");
        let kept = close_called_street(l);
        assert_eq!(kept.action_seat, l.action_seat, "BB option node must survive closure");
        assert_eq!(kept.hist_len, l.hist_len);
    }

    /// Random-walk invariant over many deals: whenever the walk (with closure applied after every
    /// step) is at a decision node, that node is NEVER the aggressor-reopen node — i.e. the closed
    /// game contains no self-reopen decisions; and the hand still terminates with zero-sum rewards.
    #[test]
    fn closed_game_contains_no_reopen_nodes() {
        let mut x: u64 = 0x9E3779B97F4A7C15;
        let mut rand = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for seed in 0..300u64 {
            let mut s = root(seed * 31 + 7);
            let mut steps = 0;
            while !s.done {
                steps += 1;
                assert!(steps < 400, "runaway hand");
                if core::is_chance_node(&s) {
                    s = core::sample_chance(&s);
                    continue;
                }
                if s.action_seat < 0 {
                    break;
                }
                let a = s.action_seat;
                let reopen = a == s.last_aggressor
                    && s.current_bet == s.committed_street[a as usize]
                    && s.street < core::STREET_TERMINAL;
                assert!(!reopen, "closed game dealt an aggressor-reopen node (seed {seed})");
                let mask = core::legal_action_mask(&s, a);
                let legal: Vec<usize> = (0..core::NUM_ACTIONS).filter(|&i| mask[i]).collect();
                let pick = legal[(rand() % legal.len() as u64) as usize];
                s = close_called_street(core::apply_action(&s, pick));
            }
            if s.done {
                let sum: i32 = s.rewards.iter().sum();
                assert_eq!(sum, 0, "rewards not zero-sum (seed {seed})");
            }
        }
    }
}
