//! Quality lever 1: more rollouts per leaf — benchmark `uniform_leaf_value` at N=10/50/100/200.
//!
//! Measures ns/leaf + variance/leaf for a representative 6-max leaf and prints the table
//! used to find the rollout-count elbow. Run via:
//!     cargo run --release --example bench_leaf_n
//!
//! Hot loop under test (battery_native/src/playout.rs:320-329):
//!     for r in 0..rollout_samples {
//!         let rr = (r as u64).wrapping_add(1);
//!         let mut env_r = *env;
//!         if resample_board { reshuffle_undealt(&mut env_r, &mut brng); }
//!         let mut arng = Rng::seed_from_u64(base_seed.wrapping_add(1_000_003 * rr));
//!         total += playout(&env_r, hero, bp, cfg, &mut arng, lcw);
//!     }
//!     total / rollout_samples as f64

use battery_native::core::{apply_action, init_hand_n, EnvStateN, NUM_ACTIONS};
use battery_native::playout::{uniform_leaf_value, PlayoutConfig};

use std::collections::HashMap;
use std::time::Instant;

fn allowed_all() -> [bool; NUM_ACTIONS] {
    [true; NUM_ACTIONS]
}

fn main() {
    // ---- Representative 6-max preflop root leaf ----
    // Mirror of tests.rs init_hand_7 logic but for 6 seats; leaf hot loop is the same.
    let stack_bb: i32 = 100;
    let stack0 = stack_bb * 100; // BB_CHIPS = 100 (battery_native/src/core.rs:18)
    let stacks = [stack0; 6];
    let occupancy = [true; 6];
    let base_deck: [i32; 52] = {
        let mut d = [0i32; 52];
        for (i, x) in d.iter_mut().enumerate() {
            *x = i as i32;
        }
        d
    };
    // Build several different DEALS, drive to a non-trivial postflop leaf (everyone limps
    // preflop, flop dealt) so the leaf value varies with both deal and board luck.
    let n_leaves: usize = 256; // independent leaves (different deals)
    let mut deals: Vec<EnvStateN<6>> = Vec::with_capacity(n_leaves);
    for li in 0..n_leaves {
        let ps = [(li as u32).wrapping_mul(0x9E3779B9), 0xDEADBEEF];
        let mut env = init_hand_n::<6>(0, &stacks, &occupancy, 0, &base_deck, ps);
        // UTG seat 3 -> all 5 limpers CALL the BB (action 1 == CALL). After all calls, deal the flop.
        // Note: the search runs CHECKDOWN-style on the postflop leaf (playout runs to terminal).
        // We force a non-trivial scenario: open-2.5bb from UTG, all callers, board runs.
        // The cheap way: open from UTG with action 4 (BET_BUCKET_START = 3 + small bucket = 3..10).
        // We'll just limp (CALL=1) from every seat to keep the leaf simple but non-zero.
        for _ in 0..6 {
            if env.action_seat < 0 || env.done {
                break;
            }
            // action 1 = CALL; safe at every preflop seat
            env = apply_action(&env, 1);
        }
        // Advance through chance (flop deal) by repeatedly sampling chance until street>0.
        let mut safety = 0;
        while !env.done && env.action_seat < 0 && safety < 8 {
            env = battery_native::core::sample_chance(&env);
            safety += 1;
        }
        deals.push(env);
    }
    let hero: usize = deals[0].action_seat.max(0) as usize;

    let bp: HashMap<[u8; 16], [f64; NUM_ACTIONS]> = HashMap::new(); // uniform -> chip-EV base
    let cfg = PlayoutConfig {
        allowed_pre: allowed_all(),
        allowed_post: allowed_all(),
        bias_mult: 5.0,
        bias_floor: 0.0,
        max_depth: 24, // same as tests.rs pcfg
        num_buckets: 200,
        emd: None,
    };
    let base_seed: u64 = 0xC0FFEE_DEADBEEFu64;
    let resample_board = true; // matches search config (battery_native/src/playout.rs:323)

    // ---- Sweep N across 10/50/100/200 ----
    let ns_list = [10i32, 50, 100, 200];
    let reps = 32; // reps per leaf (different rollout RNG seed)

    println!("quality-lever-1 leaf-rollout sweep (battery_native::playout::uniform_leaf_value)");
    println!("env: 6-max, button=0, {stack_bb}bb stacks, root leaf (action_seat={hero})");
    println!("playout: max_depth=24, bias_mult=5.0, num_buckets=200, resample_board=true");
    println!(
        "leaves: {n_leaves} distinct deals × reps {reps} -> {} leaf-evals per N",
        n_leaves * reps
    );
    println!();
    println!("  N    |  ns/leaf  |  variance/leaf  |  stdev/leaf  |  var*N     | 1/sqrt(N) | mc-noise/leaf | time*var");

    let mut prev_var: Option<f64> = None;
    let mut prev_time: Option<f64> = None;

    for &n in &ns_list {
        let n_i32 = n as i32;
        let mut all_means: Vec<f64> = Vec::with_capacity(n_leaves * reps);

        // Warmup.
        let _ = uniform_leaf_value::<6>(
            &deals[0],
            hero,
            &bp,
            &cfg,
            base_seed,
            n_i32,
            resample_board,
        );

        let t0 = Instant::now();
        for leaf_i in 0..n_leaves {
            let env = &deals[leaf_i];
            let seed = base_seed.wrapping_add(0x9E3779B97F4A7C15u64.wrapping_mul(leaf_i as u64));
            for _rep in 0..reps {
                let v = uniform_leaf_value::<6>(
                    env,
                    hero,
                    &bp,
                    &cfg,
                    seed,
                    n_i32,
                    resample_board,
                );
                all_means.push(v);
            }
        }
        let dt = t0.elapsed();
        let total: usize = n_leaves * reps;
        let ns_per_leaf: f64 = dt.as_nanos() as f64 / total as f64;

        let mean: f64 = all_means.iter().sum::<f64>() / all_means.len() as f64;
        let var: f64 = all_means.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
            / all_means.len().max(1) as f64;
        let stdev = var.sqrt();
        let var_n = var * n as f64;

        // Also compute the cross-leaf mean MC-noise variance: average the per-leaf rep-variance.
        // (decompose total variance into leaf-luck + per-leaf MC noise)
        let leaf_var_of_rep: f64 = {
            let mut sum = 0.0f64;
            let mut count = 0usize;
            for leaf_i in 0..n_leaves {
                let lo = leaf_i * reps;
                let hi = lo + reps;
                let m: f64 = all_means[lo..hi].iter().sum::<f64>() / reps as f64;
                let s: f64 = all_means[lo..hi]
                    .iter()
                    .map(|v| (v - m).powi(2))
                    .sum::<f64>()
                    / reps.max(1) as f64;
                sum += s;
                count += 1;
            }
            sum / count.max(1) as f64
        };
        let mc_stdev = leaf_var_of_rep.sqrt();

        let one_over_sqrt_n = 1.0 / (n as f64).sqrt();

        let extra = match (prev_var, prev_time) {
            (Some(pv), Some(pt)) => {
                let var_ratio_sqrt = (var / pv).sqrt();
                let time_ratio = ns_per_leaf / pt;
                format!(
                    "  [time x{:.2}, 1/sqrt(var) x{:.2}]",
                    time_ratio,
                    1.0 / var_ratio_sqrt
                )
            }
            _ => String::new(),
        };

        let time_x_var = ns_per_leaf * var;
        println!("{:5}  | {:8.0}  |   {:10.6}    |   {:8.4}   |  {:8.4}  | {:6.4}  |   {:7.3}  | {:10.0}{}",
                 n, ns_per_leaf, var, stdev, var_n, one_over_sqrt_n, mc_stdev, time_x_var, extra);

        prev_var = Some(var);
        prev_time = Some(ns_per_leaf);
    }

    println!();
    println!("Notes:");
    println!("  * var*N should be ~constant if variance is dominated by per-rollout MC noise.");
    println!(
        "  * If var*N RISES past some N, the per-call overhead is dominating the per-rollout work."
    );
    println!("  * Elbow: doubling N is worthwhile iff time grows < 2x AND 1/sqrt(var) gain >= 1/sqrt(2).");
}
