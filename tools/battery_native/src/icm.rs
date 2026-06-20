//! Native Independent Chip Model (ICM) — bit-faithful port of `pokerbot/tournament/icm.py`.
//!
//! The ICM converts a snapshot of tournament chip stacks into real-money ($) equity given the
//! remaining payout structure: the bridge that makes a tournament a $-EV game rather than the
//! chip-EV game the cash engine solves. Three public functions, all unit-agnostic in the
//! stack/payout currency (only ratios matter for equity; payouts set the $ scale):
//!
//!   * [`icm_equity`]   — Malmuth-Harville finish-probability ICM; conserves the prize pool.
//!   * [`bubble_factor`]— pairwise risk premium (how many $ hero risks per $ hero can gain).
//!   * [`icm_delta`]    — maps a hand's chip result to a $-equity delta (busts clamped to 0).
//!
//! WHY native: the ICM terminal is the in-loop hot path for tournament-aware solving in
//! `subgame.rs` (chip-EV solve, ICM bend near the money) and for the push/fold Nash tables. The
//! Python `icm.py` is the certified oracle; this module is cross-checked against it by
//! `tests/test_native_icm_equiv.py` to float tolerance (not bit-identity — it is a float recursion,
//! but the arithmetic order mirrors the numpy oracle so agreement is ~1e-12).
//!
//! Memoization mirrors `icm.py`'s `@lru_cache` on `(stk, pay)`: the recursion is keyed on the exact
//! f64 bit-patterns of the remaining stacks and payouts, so identical remaining-field subproblems
//! are solved once (a real final table re-hits constantly). Pure Rust, no engine dependency — the
//! whole module is `cargo test`-certifiable in isolation.

use std::collections::HashMap;

/// Bit-pattern memo key: identical f64 values (the lru_cache contract) hash identically.
type MhKey = (Vec<u64>, Vec<u64>);

fn key_of(stk: &[f64], pay: &[f64]) -> MhKey {
    (
        stk.iter().map(|x| x.to_bits()).collect(),
        pay.iter().map(|x| x.to_bits()).collect(),
    )
}

/// Malmuth-Harville equity for `m` players with strictly-positive `stk`, competing for the `m`
/// payout ranks `pay` (`pay[0]` = best). Returns equities parallel to `stk`. Memoized on the exact
/// `(stk, pay)` bit-patterns, matching the Python `_mh` `lru_cache`.
fn mh(stk: &[f64], pay: &[f64], memo: &mut HashMap<MhKey, Vec<f64>>) -> Vec<f64> {
    let m = stk.len();
    if m == 0 {
        return Vec::new();
    }
    let key = key_of(stk, pay);
    if let Some(v) = memo.get(&key) {
        return v.clone();
    }

    let total: f64 = stk.iter().sum();
    if total <= 0.0 {
        // defensive: caller passes positives, but never divide by zero
        let s = if !pay.is_empty() {
            pay.iter().sum::<f64>() / m as f64
        } else {
            0.0
        };
        let out = vec![s; m];
        memo.insert(key, out.clone());
        return out;
    }

    let mut out = vec![0.0f64; m];
    let rank0 = if pay.is_empty() { 0.0 } else { pay[0] };
    let deeper = m > 1 && pay.len() > 1;
    for k in 0..m {
        let p_first = stk[k] / total; // P(player k finishes first among the remaining field)
        out[k] += p_first * rank0;
        if deeper {
            // conditional on k first, the rest compete for pay[1:]
            let mut sub_stk = Vec::with_capacity(m - 1);
            sub_stk.extend_from_slice(&stk[..k]);
            sub_stk.extend_from_slice(&stk[k + 1..]);
            let sub = mh(&sub_stk, &pay[1..], memo);
            let mut si = 0usize;
            for j in 0..m {
                if j == k {
                    continue;
                }
                out[j] += p_first * sub[si];
                si += 1;
            }
        }
    }
    memo.insert(key, out.clone());
    out
}

/// Pay vector padded/truncated to length `n` (unpaid finishes -> 0.0).
fn normalize_payouts(payouts: &[f64], n: usize) -> Vec<f64> {
    let mut pay: Vec<f64> = payouts.iter().take(n).copied().collect();
    if pay.len() < n {
        pay.resize(n, 0.0);
    }
    pay
}

/// $-equity of each seat under Malmuth-Harville ICM.
///
/// Seats with stack `<= 0` are treated as already out and take the worst available ranks (split
/// equally on a simultaneous bust). The surviving field competes for the top ranks. The result
/// conserves the prize pool: `sum(equity) == sum(payouts[:n])`.
pub fn icm_equity(stacks: &[f64], payouts: &[f64]) -> Vec<f64> {
    let n = stacks.len();
    if n == 0 {
        return Vec::new();
    }
    let pay = normalize_payouts(payouts, n);
    let mut eq = vec![0.0f64; n];

    let pos: Vec<usize> = (0..n).filter(|&i| stacks[i] > 0.0).collect();
    let zero: Vec<usize> = (0..n).filter(|&i| stacks[i] <= 0.0).collect();
    let z = zero.len();
    if z > 0 {
        // the worst z ranks go to the eliminated seats; simultaneous-bust convention: split equally
        let bottom: f64 = pay[n - z..].iter().sum();
        let share = bottom / z as f64;
        for &i in &zero {
            eq[i] = share;
        }
    }

    let top_pay: Vec<f64> = pay[..n - z].to_vec();
    let pos_stk: Vec<f64> = pos.iter().map(|&i| stacks[i]).collect();
    let mut memo: HashMap<MhKey, Vec<f64>> = HashMap::new();
    let sub = mh(&pos_stk, &top_pay, &mut memo);
    for (r, &i) in pos.iter().enumerate() {
        eq[i] = sub[r];
    }
    eq
}

/// Pairwise ICM risk premium for `hero` stacking off against `villain` for the smaller stack.
///
/// `BF = (equity hero risks by busting) / (equity hero gains by doubling)`. `1.0` => chips are
/// $-linear here (deep / far from the money -> play chip-EV); `> 1` => busting costs more than
/// doubling gains (ICM pressure -> demand more equity to call). `+inf` if hero cannot gain.
pub fn bubble_factor(stacks: &[f64], payouts: &[f64], hero: usize, villain: usize) -> f64 {
    let h = stacks[hero];
    let v = stacks[villain];
    let risked = h.min(v); // an all-in is for the smaller stack
    let base = icm_equity(stacks, payouts)[hero];

    let mut win = stacks.to_vec();
    win[hero] = h + risked;
    win[villain] = v - risked;
    let mut lose = stacks.to_vec();
    lose[hero] = h - risked;
    lose[villain] = v + risked;

    let eq_win = icm_equity(&win, payouts)[hero];
    let eq_lose = icm_equity(&lose, payouts)[hero];

    let reward = eq_win - base;
    let risk = base - eq_lose;
    if reward <= 1e-12 {
        return f64::INFINITY;
    }
    risk / reward
}

/// $-equity change from a hand: `ICM(pre + chip_delta) - ICM(pre)`.
///
/// The single wrap point that turns a chip-EV terminal (the engine's `compute_returns`) into the
/// tournament's $-EV signal. `chip_delta` is the per-seat net chips (sum ~= 0); post-hand stacks
/// are clamped at 0 so a bust realizes the bottom payout.
pub fn icm_delta(pre_stacks: &[f64], chip_delta: &[f64], payouts: &[f64]) -> Vec<f64> {
    assert_eq!(
        chip_delta.len(),
        pre_stacks.len(),
        "chip_delta len != stacks len"
    );
    let post: Vec<f64> = (0..pre_stacks.len())
        .map(|i| (pre_stacks[i] + chip_delta[i]).max(0.0))
        .collect();
    let a = icm_equity(&post, payouts);
    let b = icm_equity(pre_stacks, payouts);
    (0..a.len()).map(|i| a[i] - b[i]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn two_equal_stacks_winner_take_all() {
        // WTA [1,0]: equal stacks -> 0.5/0.5; pool conserved.
        let eq = icm_equity(&[100.0, 100.0], &[1.0, 0.0]);
        assert!(close(eq[0], 0.5, 1e-12) && close(eq[1], 0.5, 1e-12));
        assert!(close(eq.iter().sum(), 1.0, 1e-12));
    }

    #[test]
    fn dominant_stack_more_equity() {
        let eq = icm_equity(&[300.0, 100.0, 100.0], &[5.0, 3.0, 2.0]);
        assert!(eq[0] > eq[1] && eq[1] > 1e-9);
        // prize pool conserved
        assert!(close(eq.iter().sum(), 10.0, 1e-9));
    }

    #[test]
    fn busted_seat_takes_bottom_rank() {
        // one seat at 0 -> takes the worst rank (pay[2]=2), survivors split the top two.
        let eq = icm_equity(&[100.0, 100.0, 0.0], &[5.0, 3.0, 2.0]);
        assert!(close(eq[2], 2.0, 1e-12));
        assert!(close(eq.iter().sum(), 10.0, 1e-9));
    }

    #[test]
    fn bubble_factor_above_one_near_money() {
        // 3 left, top 2 paid, hero medium vs a covering villain -> ICM pressure (BF > 1).
        let bf = bubble_factor(&[100.0, 200.0, 50.0], &[65.0, 35.0], 0, 1);
        assert!(bf > 1.0, "expected ICM pressure, got {bf}");
    }

    #[test]
    fn icm_delta_conserves_and_signs() {
        // hero wins villain's at-risk chips: hero $-gain, villain $-loss, bystander moves too.
        let pre = [100.0, 100.0, 100.0];
        let d = icm_delta(&pre, &[100.0, -100.0, 0.0], &[5.0, 3.0, 2.0]);
        assert!(d[0] > 0.0 && d[1] < 0.0);
        assert!(close(d.iter().sum(), 0.0, 1e-9)); // ICM is pool-conserving on both sides
    }

    #[test]
    fn deeper_payouts_three_ranks() {
        // all three paid -> everyone's equity strictly positive, ordered by stack, pool conserved.
        let eq = icm_equity(&[120.0, 100.0, 80.0], &[50.0, 30.0, 20.0]);
        assert!(eq[0] > eq[1] && eq[1] > eq[2] && eq[2] > 0.0);
        assert!(close(eq.iter().sum(), 100.0, 1e-9));
    }
}
