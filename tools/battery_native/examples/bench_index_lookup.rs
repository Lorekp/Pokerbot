//! BOUNDED profiler for the tier-1 32GB mmap blueprint index (`Blueprint::Index` / `SortedIndex`).
//!
//! Answers: "on a 32GB-RAM box serving the 32GB tier-1 mmap index, how much of a native subgame
//! solve's wall time is mmap/page-fault STALL in the blueprint-index lookups vs actual CFR compute?"
//!
//! Two modes:
//!   --mode keys   Open the real index the SAME way the serve does (`Blueprint::from_index_dir` ->
//!                 `SortedIndex::open`, both files mmapped). Sample N random keys FROM
//!                 `keys_sorted.bin` itself (sparse seeks, never streaming the whole file) and time
//!                 each `Blueprint::get`: FIRST-TOUCH (cold) vs an IMMEDIATE repeat (warm). Reports
//!                 count, hit-rate, p50/p90/p99/max ns (cold + warm), and the cold/warm ratio. Also
//!                 probes M random ABSENT keys (a sampled key with one byte perturbed) to measure the
//!                 miss-path (full ~30-probe binary search that returns None) latency.
//!
//!   --mode solve  Run N full native subgame solves against the mmap index at the DEPLOYED config
//!                 (streams=1, leaf_cache_worlds=4, K via --iters). Built `--features prof`, prints
//!                 per-solve wall + the prof phase dump so `bp_lookup` vs `solve_total` vs
//!                 `leaf_playout` attribution is visible. States are sourced GAME-FLOW-SAFELY
//!                 (init_hand -> play a hand forward -> solve at naturally-reached decision points),
//!                 the same class of states bench_subgame.rs reads from its exported corpus. A
//!                 `--states-file` (bench_subgame states.bin format) may be supplied instead.
//!
//! Both modes: `--limit` caps work, a progress line prints every few seconds, and a final
//! machine-parsable `JSON\t{...}` summary line is emitted.
//!
//! Usage:
//!   bench_index_lookup --mode keys  --index-dir DIR [--n 1000] [--absent 200] [--seed S] [--limit L]
//!   bench_index_lookup --mode solve --index-dir DIR [--iters K] [--n 3] [--config config.txt]
//!                      [--states-file states.bin] [--stack-bb 100] [--seed S] [--limit L]
//!
//! SAFETY: bounded smoke only touches the real index at tiny N; the live bot shares this box's page
//! cache. NEVER run with RAYON_NUM_THREADS=1 (nested-rayon deadlock in the base solve).

use std::collections::HashMap;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::time::{Duration, Instant};

use battery_native::core::{
    self, EnvState, BB_CHIPS, CALL, CHECK, FOLD, NUM_ACTIONS, NUM_PLAYERS, STREET_TERMINAL,
};
use battery_native::emd;
use battery_native::flat;
use battery_native::playout::{Blueprint, EmdKeyCfg, PlayoutConfig, Rng};
use battery_native::subgame::{self, SubgameConfig};

// --------------------------------------------------------------------------------------------------
// small helpers
// --------------------------------------------------------------------------------------------------

fn percentile(sorted: &[u64], q: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = (((sorted.len() - 1) as f64) * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn stats_line(label: &str, ns: &mut Vec<u64>) -> (u64, u64, u64, u64) {
    ns.sort_unstable();
    let p50 = percentile(ns, 0.50);
    let p90 = percentile(ns, 0.90);
    let p99 = percentile(ns, 0.99);
    let mx = *ns.last().unwrap_or(&0);
    eprintln!(
        "[{label}] n={} p50={}ns p90={}ns p99={}ns max={}ns",
        ns.len(),
        p50,
        p90,
        p99,
        mx
    );
    (p50, p90, p99, mx)
}

/// Approx invariant-TSC cycles-per-ns (== GHz). Only used to render the rdtsc phase dump in ns.
#[cfg(feature = "prof")]
fn calibrate_tsc_ghz() -> f64 {
    let t0 = Instant::now();
    let c0 = battery_native::prof::tsc();
    while t0.elapsed() < Duration::from_millis(120) {
        std::hint::spin_loop();
    }
    let c1 = battery_native::prof::tsc();
    let ns = t0.elapsed().as_nanos() as f64;
    let cyc = c1.wrapping_sub(c0) as f64;
    if ns > 0.0 {
        cyc / ns
    } else {
        1.0
    }
}

// --------------------------------------------------------------------------------------------------
// config.txt (bench_subgame format) parsing
// --------------------------------------------------------------------------------------------------

fn load_config(path: &Path) -> HashMap<String, String> {
    let txt = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read config {}: {e}", path.display()));
    let mut m = HashMap::new();
    for line in txt.lines() {
        if let Some((k, v)) = line.split_once('=') {
            m.insert(k.trim().to_string(), v.trim().to_string());
        }
    }
    m
}

fn mask11(s: &str) -> [bool; NUM_ACTIONS] {
    let mut m = [false; NUM_ACTIONS];
    for (i, tok) in s.split(',').enumerate().take(NUM_ACTIONS) {
        m[i] = tok.trim() == "1";
    }
    m
}

// --------------------------------------------------------------------------------------------------
// GAME-FLOW-SAFE state sourcing (init_hand -> walk forward -> collect natural decision nodes).
//
// This deliberately AVOIDS the hang/panic path (manual postflop construction: blueprint-play preflop
// + manual chance sample then solve at street>0 -> seat=-1 into frozen core.rs). We only ever advance
// through the real engine: is_chance_node -> sample_chance, and legal-masked apply_action. Every state
// we hand to `solve` is a naturally-reached decision node (action_seat >= 0), hero = the seat to act.
// --------------------------------------------------------------------------------------------------

fn shuffled_deck(rng: &mut Rng) -> [i32; 52] {
    let mut d = [0i32; 52];
    for (i, c) in d.iter_mut().enumerate() {
        *c = i as i32;
    }
    // full Fisher-Yates over the 52-card identity deck
    for i in (1..52).rev() {
        let j = rng.next_below((i + 1) as u64) as usize;
        d.swap(i, j);
    }
    d
}

/// Passive-biased legal action: CHECK when free, else mostly CALL with an occasional FOLD (to vary
/// pot width). Never raises (keeps limped multiway pots -> many blueprint lookups per solve, and
/// steers clear of all-in edge lines) — the state realism we need is "a valid decision node the
/// solver actually faces", not a specific action distribution.
fn pick_action(mask: &[bool; NUM_ACTIONS], rng: &mut Rng) -> usize {
    if mask[CHECK] {
        return CHECK;
    }
    let u = rng.next_f64();
    if mask[CALL] && u < 0.9 {
        return CALL;
    }
    if mask[FOLD] {
        return FOLD;
    }
    if mask[CALL] {
        return CALL;
    }
    (0..NUM_ACTIONS).find(|&a| mask[a]).unwrap_or(CHECK)
}

/// Collect up to `want` decision-node states. `min_street` biases collection toward postflop nodes
/// (street>=1) — the heavier subgames with the most index lookups; if too few are found it also
/// takes preflop nodes so the caller always gets `want` states when possible.
fn gen_states(want: usize, seed: u64, stack_bb: i32, min_street: i32) -> Vec<(EnvState, usize)> {
    assert_eq!(NUM_PLAYERS, 6, "state gen assumes the 6-max root");
    let mut rng = Rng::seed_from_u64(seed);
    let mut post: Vec<(EnvState, usize)> = Vec::new();
    let mut pre: Vec<(EnvState, usize)> = Vec::new();
    let stacks = [stack_bb * BB_CHIPS; NUM_PLAYERS];
    let occ = [true; NUM_PLAYERS];
    let max_hands = (want as u64) * 400 + 2000;
    let mut hand = 0u64;
    while post.len() < want && hand < max_hands {
        hand += 1;
        let deck = shuffled_deck(&mut rng);
        let button = (hand % NUM_PLAYERS as u64) as i32;
        let rng_key = [(hand & 0xffff_ffff) as u32, (hand >> 32) as u32];
        let mut env = core::init_hand(button, &stacks, &occ, 0, &deck, rng_key);
        for _ in 0..400 {
            if env.done || env.street >= STREET_TERMINAL {
                break;
            }
            if core::is_chance_node(&env) {
                env = core::sample_chance(&env);
                continue;
            }
            let seat = env.action_seat;
            if seat < 0 {
                break;
            }
            if env.street >= min_street {
                post.push((env, seat as usize));
                if post.len() >= want {
                    break;
                }
            } else {
                pre.push((env, seat as usize));
            }
            let mask = core::legal_action_mask(&env, seat);
            let a = pick_action(&mask, &mut rng);
            env = core::apply_action(&env, a);
        }
    }
    // Prefer postflop; backfill from preflop only if postflop came up short.
    let mut out = post;
    if out.len() < want {
        for s in pre.into_iter() {
            if out.len() >= want {
                break;
            }
            out.push(s);
        }
    }
    out.truncate(want);
    out
}

/// Load states from a bench_subgame `states.bin` (u64 n, then n * (u8 cls, u8 hero, FLAT_LEN i64 LE)).
fn load_states_file(path: &Path) -> Vec<(EnvState, usize)> {
    let data = fs::read(path).unwrap_or_else(|e| panic!("states.bin {}: {e}", path.display()));
    let n = u64::from_le_bytes(data[..8].try_into().unwrap()) as usize;
    let flat_len = flat::FLAT_LEN;
    let rec = 2 + flat_len * 8;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let off = 8 + i * rec;
        let hero = data[off + 1] as usize;
        let mut buf = vec![0i64; flat_len];
        for (j, slot) in buf.iter_mut().enumerate() {
            let b = off + 2 + j * 8;
            *slot = i64::from_le_bytes(data[b..b + 8].try_into().unwrap());
        }
        out.push((flat::from_flat(&buf), hero));
    }
    out
}

// --------------------------------------------------------------------------------------------------
// KEYS MODE
// --------------------------------------------------------------------------------------------------

fn read_key_at(f: &mut fs::File, idx: usize) -> [u8; 16] {
    let mut key = [0u8; 16];
    f.seek(SeekFrom::Start((idx as u64) * 16))
        .expect("seek keys_sorted.bin");
    f.read_exact(&mut key).expect("read 16B key");
    key
}

/// OPT-BP status of an opened Blueprint (fence sidecar loaded? row cache on?). Dict => (false, false).
fn optbp_status(bp: &Blueprint) -> (bool, bool) {
    match bp {
        Blueprint::Index(ix) => (ix.fence_active(), ix.cache_active()),
        _ => (false, false),
    }
}

fn run_keys(index_dir: &str, n_present: usize, n_absent: usize, seed: u64) {
    let t_open = Instant::now();
    let bp = Blueprint::from_index_dir(index_dir)
        .unwrap_or_else(|e| panic!("from_index_dir({index_dir}): {e}"));
    let n = bp.len();
    let (fence_on, cache_on) = optbp_status(&bp);
    eprintln!(
        "[keys] opened index dir={index_dir} n_rows={n} in {:.2}s (mmap; pages fault in on demand) \
fence={fence_on} rowcache={cache_on} (env POKERBOT_BP_FENCE/POKERBOT_BP_ROWCACHE=0 to disable; \
with rowcache ON the WARM repeat is a cache hit, not a re-search)",
        t_open.elapsed().as_secs_f64()
    );
    let keys_path = Path::new(index_dir).join("keys_sorted.bin");
    let mut kf = fs::File::open(&keys_path).expect("open keys_sorted.bin");
    let mut rng = Rng::seed_from_u64(seed);

    // --- present keys: cold (first touch) vs warm (immediate repeat) ---
    let mut cold: Vec<u64> = Vec::with_capacity(n_present);
    let mut warm: Vec<u64> = Vec::with_capacity(n_present);
    let mut hits = 0usize;
    let mut last = Instant::now();
    for i in 0..n_present {
        let idx = rng.next_below(n as u64) as usize;
        let key = read_key_at(&mut kf, idx);
        // COLD: first `get` after the sample read. The matched-key page is warm (we just read it),
        // but the ~30 binary-search MIDPOINT probe pages + the rows page fault in here — that stall
        // is exactly what we measure. (Upper-tree midpoints warm across the run, like the real serve.)
        let t0 = Instant::now();
        let r1 = bp.get(&key);
        let c = t0.elapsed().as_nanos() as u64;
        // WARM: identical lookup, pages now resident -> pure binary-search compute. black_box keeps
        // the optimizer from eliding (or CSE-ing) the second lookup even though its value is unused.
        let t1 = Instant::now();
        let r2 = bp.get(&key);
        let w = t1.elapsed().as_nanos() as u64;
        std::hint::black_box(&r2);
        if r1.is_some() {
            hits += 1;
        }
        cold.push(c);
        warm.push(w);
        if last.elapsed() >= Duration::from_secs(3) {
            eprintln!("[keys] present {}/{} hits={hits}", i + 1, n_present);
            last = Instant::now();
        }
    }

    // --- absent keys: perturb a sampled key so the search returns None (miss-path latency) ---
    let mut miss: Vec<u64> = Vec::with_capacity(n_absent);
    let mut absent_false_hits = 0usize;
    for i in 0..n_absent {
        let idx = rng.next_below(n as u64) as usize;
        let mut key = read_key_at(&mut kf, idx);
        // flip a middle byte to a fresh random value (a 128-bit digest with 1 byte changed is absent
        // w.p. ~1 - n/2^128 ~= 1).
        let b = (rng.next_below(16)) as usize;
        key[b] ^= (rng.next_below(255) as u8).wrapping_add(1);
        let t0 = Instant::now();
        let r = bp.get(&key);
        miss.push(t0.elapsed().as_nanos() as u64);
        if r.is_some() {
            absent_false_hits += 1;
        }
        if last.elapsed() >= Duration::from_secs(3) {
            eprintln!("[keys] absent {}/{}", i + 1, n_absent);
            last = Instant::now();
        }
    }

    let (cp50, cp90, cp99, cmax) = stats_line("cold", &mut cold);
    let (wp50, wp90, wp99, wmax) = stats_line("warm", &mut warm);
    let (mp50, mp90, mp99, mmax) = stats_line("miss", &mut miss);
    let hit_rate = if n_present > 0 {
        hits as f64 / n_present as f64
    } else {
        0.0
    };
    let ratio = if wp50 > 0 {
        cp50 as f64 / wp50 as f64
    } else {
        0.0
    };
    eprintln!(
        "[keys] hit_rate={hit_rate:.4} cold/warm(p50)={ratio:.1}x absent_false_hits={absent_false_hits}"
    );
    println!(
        "JSON\t{{\"mode\":\"keys\",\"index_dir\":\"{}\",\"n_rows\":{n},\
\"fence_active\":{fence_on},\"cache_active\":{cache_on},\"present\":{n_present},\
\"hits\":{hits},\"hit_rate\":{hit_rate:.6},\
\"cold_ns\":{{\"p50\":{cp50},\"p90\":{cp90},\"p99\":{cp99},\"max\":{cmax}}},\
\"warm_ns\":{{\"p50\":{wp50},\"p90\":{wp90},\"p99\":{wp99},\"max\":{wmax}}},\
\"cold_over_warm_p50\":{ratio:.3},\
\"absent\":{n_absent},\"absent_false_hits\":{absent_false_hits},\
\"miss_ns\":{{\"p50\":{mp50},\"p90\":{mp90},\"p99\":{mp99},\"max\":{mmax}}}}}",
        index_dir.replace('\\', "\\\\")
    );
}

// --------------------------------------------------------------------------------------------------
// SOLVE MODE
// --------------------------------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn run_solve(
    index_dir: &str,
    cfg_path: &Path,
    iters: i32,
    n_states: usize,
    states_file: Option<&Path>,
    stack_bb: i32,
    seed_base: u64,
) {
    let cfgm = load_config(cfg_path);
    let t_open = Instant::now();
    let bp = Blueprint::from_index_dir(index_dir)
        .unwrap_or_else(|e| panic!("from_index_dir({index_dir}): {e}"));
    let emd_path = cfgm["emd_frozen_path"].clone();
    let lookup = emd::load_shared(&emd_path).expect("emd frozen blob");
    let (fence_on, cache_on) = optbp_status(&bp);
    eprintln!(
        "[solve] index n_rows={} emd={} loaded in {:.2}s fence={fence_on} rowcache={cache_on}",
        bp.len(),
        emd_path,
        t_open.elapsed().as_secs_f64()
    );

    let pcfg = PlayoutConfig {
        allowed_pre: mask11(&cfgm["allowed_pre"]),
        allowed_post: mask11(&cfgm["allowed_post"]),
        bias_mult: cfgm["bias_mult"].parse().unwrap(),
        bias_floor: cfgm["bias_floor"].parse().unwrap(),
        max_depth: cfgm["leaf_max_playout_depth"].parse().unwrap(),
        num_buckets: cfgm["num_buckets"].parse().unwrap(),
        emd: Some(EmdKeyCfg {
            lookup,
            miss_made_hand: cfgm["emd_miss_made_hand"] == "1",
        }),
    };
    // Deployed config, with K overridden and streams forced to 1 (single-stream, as specified).
    let streams: i32 = cfgm.get("ensemble_streams").map(|s| s.parse().unwrap()).unwrap_or(1);
    if streams != 1 {
        eprintln!("[solve] WARNING config ensemble_streams={streams}; forcing 1 (single-stream bench)");
    }
    let scfg = SubgameConfig {
        max_iterations: iters,
        max_depth: cfgm["max_depth"].parse().unwrap(),
        blueprint_mix: cfgm["blueprint_mix"].parse().unwrap(),
        rollout_samples: cfgm["rollout_samples"].parse().unwrap(),
        resample_board: cfgm["resample_board"] == "1",
        linear_averaging: cfgm["linear_averaging"] == "1",
        repair_deck: cfgm["repair_deck"] == "1",
        two_sided: cfgm["two_sided"] == "1",
        decouple_sampler: cfgm["decouple_sampler"] == "1",
        range_aware: cfgm["range_aware"] == "1",
        range_uniform_floor: cfgm["range_uniform_floor"].parse().unwrap(),
        range_reach_full: cfgm["range_reach_full"] == "1",
        ensemble_streams: 1,
        leaf_cache_worlds: cfgm["leaf_cache_worlds"].parse().unwrap(),
        avg_skip_frac: cfgm["avg_skip_frac"].parse().unwrap(),
        leaf_biased_continuations: cfgm["leaf_biased_continuations"] == "1",
        leaf_bias_mult: cfgm["leaf_bias_mult"].parse().unwrap(),
        serve_final_iterate: cfgm["serve_final_iterate"] == "1",
        root_unmask: 0,
    };

    // States: from file, or generated game-flow-safely.
    let t_gen = Instant::now();
    let states = match states_file {
        Some(p) => {
            let mut v = load_states_file(p);
            v.truncate(n_states);
            eprintln!("[solve] loaded {} states from {}", v.len(), p.display());
            v
        }
        None => {
            let v = gen_states(n_states, seed_base ^ 0xA5A5_5A5A, stack_bb, 1);
            eprintln!(
                "[solve] generated {} game-flow states ({:.2}s); streets={:?}",
                v.len(),
                t_gen.elapsed().as_secs_f64(),
                v.iter().map(|(e, _)| e.street).collect::<Vec<_>>()
            );
            v
        }
    };
    if states.is_empty() {
        eprintln!("[solve] no states to solve; aborting");
        return;
    }

    #[cfg(feature = "prof")]
    let ghz = calibrate_tsc_ghz();
    #[cfg(feature = "prof")]
    eprintln!("[solve] tsc ~{ghz:.3} GHz (rdtsc phase-dump ns conversion)");
    #[cfg(not(feature = "prof"))]
    eprintln!(
        "[solve] NOTE built WITHOUT --features prof: no phase attribution (wall only). \
Rebuild with `--features prof` for bp_lookup vs solve_total."
    );

    let mut walls: Vec<u64> = Vec::new();
    #[cfg(feature = "prof")]
    let mut per_solve_json: Vec<String> = Vec::new();
    #[cfg(feature = "prof")]
    let (mut agg_bp_get, mut agg_bp_miss, mut agg_bp_chit) = (0u64, 0u64, 0u64);

    for (i, (env, hero)) in states.iter().enumerate() {
        let seed = seed_base.wrapping_add(i as u64 * 1_000_003);
        // Zero the global prof accumulators so this solve's dump is isolated.
        #[cfg(feature = "prof")]
        let _ = battery_native::prof::dump_and_reset();
        let t = Instant::now();
        let probs = subgame::solve(env, *hero, &bp, &pcfg, &scfg, seed);
        let wall_ns = t.elapsed().as_nanos() as u64;
        walls.push(wall_ns);
        let top = probs
            .iter()
            .enumerate()
            .fold((0usize, 0.0f64), |acc, (a, &p)| if p > acc.1 { (a, p) } else { acc });
        eprintln!(
            "[solve] state {}/{} street={} hero={hero} wall_ms={:.2} top_action={} p={:.3}",
            i + 1,
            states.len(),
            env.street,
            wall_ns as f64 / 1e6,
            top.0,
            top.1
        );
        #[cfg(feature = "prof")]
        {
            let dump = battery_native::prof::dump_and_reset();
            let m: HashMap<&str, u64> = dump
                .split_whitespace()
                .filter_map(|kv| kv.split_once('='))
                .map(|(k, v)| (k, v.parse::<u64>().unwrap_or(0)))
                .collect();
            let g = |k: &str| *m.get(k).unwrap_or(&0);
            let solve_c = g("solve_total").max(1);
            let bp_c = g("bp_lookup");
            let leaf_c = g("leaf_playout").max(1);
            let fpk_c = g("fp_key");
            let bp_get = g("bp_get");
            let bp_miss = g("bp_miss");
            let bp_chit = g("bp_cache_hit");
            agg_bp_get += bp_get;
            agg_bp_miss += bp_miss;
            agg_bp_chit += bp_chit;
            let f_solve = bp_c as f64 / solve_c as f64;
            let f_leaf = bp_c as f64 / leaf_c as f64;
            let to_ns = |c: u64| (c as f64 / ghz) as u64;
            eprintln!(
                "  PROF cyc solve_total={solve_c} leaf_playout={leaf_c} fp_key={fpk_c} \
bp_lookup={bp_c} | bp_lookup={:.1}% of solve, {:.1}% of leaf | bp_get={bp_get} bp_miss={bp_miss} \
bp_cache_hit={bp_chit} (distinct mmap lookups={}) | bp_lookup~{}us ({} lookups, ~{}ns each)",
                100.0 * f_solve,
                100.0 * f_leaf,
                bp_get.saturating_sub(bp_chit),
                to_ns(bp_c) / 1000,
                bp_get,
                if bp_get > 0 { to_ns(bp_c) / bp_get } else { 0 },
            );
            per_solve_json.push(format!(
                "{{\"i\":{i},\"street\":{},\"wall_ns\":{wall_ns},\"solve_total_cyc\":{solve_c},\
\"leaf_playout_cyc\":{leaf_c},\"fp_key_cyc\":{fpk_c},\"bp_lookup_cyc\":{bp_c},\
\"bp_lookup_frac_solve\":{f_solve:.4},\"bp_lookup_frac_leaf\":{f_leaf:.4},\
\"bp_get\":{bp_get},\"bp_miss\":{bp_miss},\"bp_cache_hit\":{bp_chit}}}",
                env.street
            ));
        }
    }

    walls.sort_unstable();
    let wp50 = percentile(&walls, 0.5);
    #[cfg(feature = "prof")]
    {
        let hr = if agg_bp_get > 0 {
            (agg_bp_get - agg_bp_miss) as f64 / agg_bp_get as f64
        } else {
            0.0
        };
        println!(
            "JSON\t{{\"mode\":\"solve\",\"index_dir\":\"{}\",\"iters\":{iters},\"n_states\":{},\
\"fence_active\":{fence_on},\"cache_active\":{cache_on},\
\"tsc_ghz\":{ghz:.4},\"wall_ns_p50\":{wp50},\"bp_get_total\":{agg_bp_get},\
\"bp_miss_total\":{agg_bp_miss},\"bp_cache_hit_total\":{agg_bp_chit},\
\"hit_rate\":{hr:.6},\"per_solve\":[{}]}}",
            index_dir.replace('\\', "\\\\"),
            states.len(),
            per_solve_json.join(",")
        );
    }
    #[cfg(not(feature = "prof"))]
    println!(
        "JSON\t{{\"mode\":\"solve\",\"index_dir\":\"{}\",\"iters\":{iters},\"n_states\":{},\
\"fence_active\":{fence_on},\"cache_active\":{cache_on},\"wall_ns_p50\":{wp50},\"prof\":false}}",
        index_dir.replace('\\', "\\\\"),
        states.len()
    );
}

// --------------------------------------------------------------------------------------------------
// arg parsing / main
// --------------------------------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut mode = String::new();
    let mut index_dir = String::new();
    let mut n: Option<usize> = None;
    let mut absent: usize = 200;
    let mut iters: i32 = 2000;
    let mut seed: u64 = 4242;
    let mut limit: Option<usize> = None;
    let mut config: Option<String> = None;
    let mut states_file: Option<String> = None;
    let mut stack_bb: i32 = 100;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--mode" => {
                mode = args[i + 1].clone();
                i += 2;
            }
            "--index-dir" => {
                index_dir = args[i + 1].clone();
                i += 2;
            }
            "--n" => {
                n = Some(args[i + 1].parse().unwrap());
                i += 2;
            }
            "--absent" => {
                absent = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--iters" => {
                iters = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--seed" => {
                seed = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--limit" => {
                limit = Some(args[i + 1].parse().unwrap());
                i += 2;
            }
            "--config" => {
                config = Some(args[i + 1].clone());
                i += 2;
            }
            "--states-file" => {
                states_file = Some(args[i + 1].clone());
                i += 2;
            }
            "--stack-bb" => {
                stack_bb = args[i + 1].parse().unwrap();
                i += 2;
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(2);
            }
        }
    }

    if index_dir.is_empty() {
        eprintln!(
            "usage:\n  bench_index_lookup --mode keys  --index-dir DIR [--n 1000] [--absent 200] [--seed S] [--limit L]\n  bench_index_lookup --mode solve --index-dir DIR [--iters K] [--n 3] [--config config.txt] [--states-file states.bin] [--stack-bb 100] [--seed S] [--limit L]"
        );
        std::process::exit(2);
    }

    match mode.as_str() {
        "keys" => {
            let mut n_present = n.unwrap_or(1000);
            let mut n_absent = absent;
            if let Some(l) = limit {
                n_present = n_present.min(l);
                n_absent = n_absent.min(l);
            }
            run_keys(&index_dir, n_present, n_absent, seed);
        }
        "solve" => {
            let mut n_states = n.unwrap_or(3);
            if let Some(l) = limit {
                n_states = n_states.min(l);
            }
            // Default config: <crate>/bench_data/config.txt (resolves regardless of cwd).
            let cfg_path = config.unwrap_or_else(|| {
                format!("{}/bench_data/config.txt", env!("CARGO_MANIFEST_DIR"))
            });
            run_solve(
                &index_dir,
                Path::new(&cfg_path),
                iters,
                n_states,
                states_file.as_deref().map(Path::new),
                stack_bb,
                seed,
            );
        }
        other => {
            eprintln!("--mode must be keys|solve (got '{other}')");
            std::process::exit(2);
        }
    }
}
