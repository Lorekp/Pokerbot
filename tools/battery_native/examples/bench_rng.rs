//! RNG microbenchmark for the native playout RNG (xoshiro256**).
//!
//! Compares three RNGs in tight, no-allocation loops:
//!   * XOSHIRO256**: the production RNG in `playout::Rng` (SplitMix64-seeded).
//!   * XOROSHIRO128+: a hand-tuned 128-bit variant (3 adds + 3 xors + 1 rot per step) for reference.
//!   * STD_RAND:     the `fastrand` crate (TinyMT-derived) for an off-the-shelf comparison.
//!
//! Then runs a representative 6-max `uniform_leaf_value` (postflop leaf, all CHECKDOWN to terminal)
//! with the production RNG so we can compute ns/leaf-eval and what fraction of wall the RNG itself
//! burns. The leaf-eval numbers ARE the headline (they include ALL per-step RNG draws, mask lookups,
//! apply_action, sample_action and terminal settle) — no separate profiling needed.
//!
//! Run via:
//!     cargo run --release --example bench_rng

use std::time::Instant;

use battery_native::core::{apply_action, init_hand_n};
use battery_native::playout::{uniform_leaf_value, PlayoutConfig, Rng as PlayRng};

use std::collections::HashMap;

// --------------------------------------------------------------------------------
// Hand-tuned xoroshiro128+ (Vigna) — used as a reference point for "small, fast, deterministic".
// --------------------------------------------------------------------------------
#[derive(Clone)]
struct Xoroshiro128P {
    s0: u64,
    s1: u64,
}
impl Xoroshiro128P {
    #[inline]
    fn seed_from(x: u64) -> Self {
        // SplitMix64 (same as in playout.rs) to spread the seed.
        let mut sm = |mut z: u64| {
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        };
        Self {
            s0: sm(x),
            s1: sm(x.wrapping_add(0x9E3779B97F4A7C15)),
        }
    }
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let s0 = self.s0;
        let mut s1 = self.s1;
        let result = s0.wrapping_add(s1);
        s1 ^= s0;
        self.s0 = s0.rotate_left(24) ^ s1 ^ (s1 << 16);
        self.s1 = s1.rotate_left(37);
        result
    }
}

// --------------------------------------------------------------------------------
// Timing harness: tight loop with no per-iter overhead beyond a single RNG call.
// --------------------------------------------------------------------------------
fn time_it<F: FnMut() -> u64>(name: &str, iters: u64, mut f: F) -> f64 {
    // Warm up.
    let mut acc = 0u64;
    for _ in 0..(iters / 4) {
        acc ^= f();
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        acc ^= f();
    }
    let dt = t0.elapsed();
    let ns = dt.as_nanos() as f64 / iters as f64;
    println!(
        "  {:<32} {:>10} iters | {:>11.3?} | {:7.2} ns/op | acc={:#x}",
        name, iters, dt, ns, acc
    );
    ns
}

// --------------------------------------------------------------------------------
// Leaf-eval helper (representative 6-max postflop CHECKDOWN to terminal).
// --------------------------------------------------------------------------------
fn build_postflop_leaves_6max(n_leaves: usize) -> Vec<battery_native::core::EnvStateN<6>> {
    let stack0 = 100 * 100; // 100bb in chips (BB_CHIPS = 100)
    let stacks = [stack0; 6];
    let occupancy = [true; 6];
    let mut base_deck = [0i32; 52];
    for (i, x) in base_deck.iter_mut().enumerate() {
        *x = i as i32;
    }
    let mut out = Vec::with_capacity(n_leaves);
    for li in 0..n_leaves {
        let ps = [(li as u32).wrapping_mul(0x9E3779B9), 0xDEADBEEF];
        let mut env = init_hand_n::<6>(0, &stacks, &occupancy, 0, &base_deck, ps);
        // Limp around (CALL=1) so all 6 are live preflop.
        for _ in 0..8 {
            if env.action_seat < 0 || env.done {
                break;
            }
            env = apply_action(&env, 1);
        }
        // Burn the flop chance nodes.
        let mut safety = 0;
        while !env.done && env.action_seat < 0 && safety < 8 {
            env = battery_native::core::sample_chance(&env);
            safety += 1;
        }
        if !env.done {
            out.push(env);
        }
    }
    out
}

fn time_leaf_eval() {
    println!(
        "\n=== Leaf-eval timing (6-max postflop CHECKDOWN to terminal, N=10/50/200 rollouts) ==="
    );
    let leaves = build_postflop_leaves_6max(64);
    let bp: HashMap<[u8; 16], [f64; 11]> = HashMap::new();
    let cfg = PlayoutConfig {
        allowed_pre: [true; 11],
        allowed_post: [true; 11],
        bias_mult: 1.0,
        bias_floor: 0.0,
        max_depth: 24,
        num_buckets: 256,
        emd: None,
    };
    let hero = 0usize;
    let rounds = 4u32;
    let mut leaf_call_ns: Vec<f64> = Vec::new();
    let mut leaf_eval_ns: Vec<f64> = Vec::new();
    for &n in &[10u32, 50, 200] {
        let iters = (leaves.len() as u64) * rounds as u64;
        let mut acc = 0.0f64;
        let t0 = Instant::now();
        for _r in 0..rounds {
            for env in &leaves {
                acc +=
                    uniform_leaf_value::<6>(env, hero, &bp, &cfg, 0x12345, n as i32, false, false);
            }
        }
        let dt = t0.elapsed();
        let total_leaf_evals = iters * n as u64; // = leaves x rollouts x rounds
        let ns_per_leaf_eval = dt.as_nanos() as f64 / total_leaf_evals as f64;
        let ns_per_leaf_call = dt.as_nanos() as f64 / iters as f64;
        leaf_call_ns.push(ns_per_leaf_call);
        leaf_eval_ns.push(ns_per_leaf_eval);
        println!(
            "  N_rollouts={n:>3} | leaves={leaves} | rounds={rounds} | {dt:>11.3?} total | \
             {ns_per_leaf_call:>7.1} ns/leaf-CALL | {ns_per_leaf_eval:>7.1} ns/leaf-EVAL | acc={acc}",
            leaves = leaves.len(),
        );
    }
    // Single-rollout number = ns/leaf-CALL when N=1 (each leaf-CALL = one playout).
    println!(
        "\n  ns/playout (one rollout = one playout to terminal) ~ {:?}",
        leaf_call_ns
    );
    println!("  ns/leaf-eval (= ns/leaf-CALL / N_rollouts, includes rng+mask+apply+settle)");
    let ns_list: Vec<u32> = vec![10, 50, 200];
    for (i, n) in ns_list.iter().enumerate() {
        println!("    N={n}: {:.1} ns/eval", leaf_eval_ns[i]);
    }
}

// --------------------------------------------------------------------------------
// Main.
// --------------------------------------------------------------------------------
fn main() {
    println!("=== RNG microbenchmark (n = 10M draws, no per-iter overhead) ===");
    let iters: u64 = 10_000_000;

    // 1) XOSHIRO256** (production) via next_f64 (the path rollouts use).
    let mut rng = PlayRng::seed_from_u64(0xC0FFEE);
    let ns_xoshiro_f64 = time_it("xoshiro256** next_f64", iters, || rng.next_f64().to_bits());

    // 2) Hand-tuned xoroshiro128+ reference (next_u64 directly).
    let mut rng = Xoroshiro128P::seed_from(0xC0FFEE);
    let ns_xoroshiro128 = time_it("xoroshiro128+ next_u64", iters, || rng.next_u64());

    // 3) Off-the-shelf: fastrand.
    let mut rng = fastrand::Rng::with_seed(0xC0FFEE);
    let ns_fastrand = time_it("fastrand    next_u64", iters, || rng.u64(..));

    println!("\n=== RNG speed comparison ===");
    println!(
        "  xoshiro256**  : {:.2} ns/f64        (production, includes (u64>>11)*1.0/2^53)",
        ns_xoshiro_f64
    );
    println!(
        "  xoroshiro128+ : {:.2} ns/random (hand-tuned 128-bit)",
        ns_xoroshiro128
    );
    println!(
        "  fastrand      : {:.2} ns/random (off-the-shelf, TinyMT-derived)",
        ns_fastrand
    );
    println!(
        "  Ratio: production xoshiro256** is {:.2}x xoroshiro128+ and {:.2}x fastrand",
        ns_xoshiro_f64 / ns_xoroshiro128,
        ns_xoshiro_f64 / ns_fastrand
    );

    // ---- Leaf eval ----
    time_leaf_eval();
}
