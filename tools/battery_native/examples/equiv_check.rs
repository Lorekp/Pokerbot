//! Statistical-equivalence harness for the relaxed trainer levers.
//!
//! For the float-order / RNG-relaxed levers (regret_match SIMD, FMA kernels, RBP, parallel merge)
//! the produced blueprint is NOT byte-identical — the acceptance gate is statistical equivalence.
//! This tool trains a table to a fixed config/seed/iters and emits two diff-able fingerprints:
//!   (1) selfplay_mae: the EV (mean-abs per-seat chips/hand) of the average-strategy blueprint in a
//!       fixed-seed 6-seat self-play tournament over many rotated deals. Invariant to internal float
//!       order; equivalent solvers land on the same value within self-play sampling noise.
//!   (2) strat_checksum: an order-independent sum over all blueprint rows of dot(row, PRIMES) — a
//!       commutative (key-iteration-order-free) distributional fingerprint sensitive to any drift.
//!
//! Usage: cargo run --release --example equiv_check -- <iters> <n_deals>

use battery_native::core::{self, EnvStateN};
use battery_native::playout::Rng;
use battery_native::trainer::{
    abstracted_legal_pub, allowed_mask, deal_root_pub, sample_action_pub, snapshot,
    uniform_over_legal_pub, NlheTables, TrainerCfg,
};
fn serve_action<const N: usize, S: std::hash::BuildHasher>(
    env: &EnvStateN<N>,
    seat: usize,
    cfg: &TrainerCfg,
    snap: &std::collections::HashMap<[u8; 16], [f64; 11], S>,
    rng: &mut Rng,
) -> usize {
    let legal = abstracted_legal_pub::<N>(env, seat, cfg);
    if !legal.iter().any(|&x| x) {
        return core::FOLD;
    }
    let key = battery_native::fingerprint::fingerprint_fast_depth(
        env,
        seat,
        cfg.num_buckets,
        &cfg.depth_edges,
    );
    let row = match snap.get(&key) {
        Some(r) => {
            // legal-renorm over positive mass; uniform fallback if none survives.
            let mut out = [0.0f64; 11];
            let mut s = 0.0;
            for a in 0..11 {
                if legal[a] && r[a] > 0.0 {
                    out[a] = r[a];
                    s += r[a];
                }
            }
            if s > 0.0 {
                for a in 0..11 {
                    out[a] /= s;
                }
                out
            } else {
                uniform_over_legal_pub(&legal)
            }
        }
        None => uniform_over_legal_pub(&legal),
    };
    sample_action_pub(&row, &legal, rng.next_f64())
}

fn play_hand<const N: usize, F>(mut env: EnvStateN<N>, mut seat_policy: F) -> [i32; N]
where
    F: FnMut(&EnvStateN<N>, usize) -> usize,
{
    let mut guard = 0;
    loop {
        guard += 1;
        if guard > 8192 {
            break;
        }
        if env.done {
            return env.rewards;
        }
        if core::is_chance_node(&env) {
            env = core::sample_chance(&env);
            continue;
        }
        if env.street >= core::STREET_TERMINAL {
            break;
        }
        let actor_i = env.action_seat;
        if actor_i < 0 {
            break;
        }
        let a = seat_policy(&env, actor_i as usize);
        env = core::apply_action(&env, a);
    }
    let n_show = env
        .seat_status
        .iter()
        .filter(|&&s| s == core::SEAT_LIVE || s == core::SEAT_ALLIN)
        .count() as i32;
    core::compute_returns(
        &env.holes,
        &env.board,
        &env.seat_status,
        &env.committed_total,
        env.button,
        n_show,
    )
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let iters: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(120);
    let n_deals: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(128);
    // Optional base_seed override (default 0xC0FFEE) so a caller can measure seed-to-seed noise.
    let base_seed: u64 = args
        .get(3)
        .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok())
        .unwrap_or(0xC0FFEE_u64);

    let prune_threshold = std::env::var("PRUNE_C")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(f64::NEG_INFINITY);
    let fast_merge = std::env::var("FAST_MERGE")
        .map(|s| s == "1")
        .unwrap_or(false);

    let cfg = TrainerCfg {
        num_buckets: 200,
        depth_edges: vec![],
        allowed_pre: allowed_mask(&[4, 5, 6]),
        allowed_post: allowed_mask(&[0, 1, 3, 4, 5]),
        n_deals_per_iter: n_deals,
        base_seed,
        n_seats: 6,
        prune_threshold,
        fast_merge,
        emd_frozen: None,
        batch_deals: 0,
        work_steal: false,
        dcfr: false,
        dcfr_prune_threshold: -300_000.0,
        dcfr_explore_frac: 0.05,
    };

    let mut tab = NlheTables::new();
    for _ in 0..iters {
        tab.train_iteration(&cfg);
    }
    let snap = snapshot(&tab);
    let infosets = snap.len();

    const PRIMES: [f64; 11] = [2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0, 29.0, 31.0];
    let mut strat_checksum = 0.0f64;
    for row in snap.values() {
        let mut d = 0.0;
        for a in 0..11 {
            d += row[a] * PRIMES[a];
        }
        strat_checksum += d;
    }

    let mut serve_rng = Rng::seed_from_u64(0x5EED_5EED);
    let mut sum_abs: f64 = 0.0;
    let mut n_hands: u64 = 0;
    for d in 0..4000u64 {
        let mut deal_rng = Rng::seed_from_u64(0xA11CE ^ d.wrapping_mul(0x9E3779B97F4A7C15));
        let root = deal_root_pub::<6>(&mut deal_rng, d);
        let rets = play_hand::<6, _>(root, |env, seat| {
            serve_action::<6, _>(env, seat, &cfg, &snap, &mut serve_rng)
        });
        for &r in &rets {
            sum_abs += (r as f64).abs();
        }
        n_hands += 1;
    }
    let selfplay_mae = sum_abs / (n_hands as f64 * 6.0);

    println!(
        "EQUIV iters={iters} n_deals={n_deals} infosets={infosets} selfplay_mae={selfplay_mae:.6} strat_checksum={strat_checksum:.6}"
    );
}
