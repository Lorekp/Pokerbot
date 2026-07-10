//! Native subgame-solve micro-bench (perf worktree harness, 2026-07-10).
//!
//! Loads the exported S1-style corpus (`bench_data/`: blueprint.bin + states.bin + config.txt,
//! see `bench/export_bench_data.py`) and runs `subgame::solve` on every state at a fixed K with
//! deterministic per-(state, repeat) seeds. Designed for INTERLEAVED A/B between two builds of
//! this example (old vs optimized): absolute walls drift with host load, so only relative ratios
//! from alternating invocations are trusted.
//!
//! Output: one `RESULT\t<id>\t<cls>\t<repeat>\t<wall_ns>\t<probs csv>` line per solve (stdout),
//! then a `SUMMARY` block of per-class median wall.
//!
//! Usage: bench_subgame <bench_data_dir> [--k 2000] [--repeats 3] [--seed 4242] [--states 0,5,7]
//!                      [--quiet-probs]

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

use battery_native::core::NUM_PLAYERS;
use battery_native::emd;
use battery_native::flat;
use battery_native::playout::{Blueprint, EmdKeyCfg, PlayoutConfig};
use battery_native::subgame::{self, SubgameConfig};

fn read_u64_le(b: &[u8]) -> u64 {
    u64::from_le_bytes(b[..8].try_into().unwrap())
}

fn load_blueprint(path: &Path) -> Blueprint {
    let data = fs::read(path).expect("blueprint.bin");
    let n = read_u64_le(&data) as usize;
    let mut bp: Blueprint = Blueprint::with_capacity_and_hasher(n, Default::default());
    let rec = 16 + 11 * 8;
    for i in 0..n {
        let off = 8 + i * rec;
        let mut key = [0u8; 16];
        key.copy_from_slice(&data[off..off + 16]);
        let mut row = [0f64; 11];
        for a in 0..11 {
            row[a] = f64::from_le_bytes(data[off + 16 + a * 8..off + 24 + a * 8].try_into().unwrap());
        }
        bp.insert(key, row);
    }
    bp
}

struct BenchState {
    cls: u8,
    hero: usize,
    env: battery_native::core::EnvState,
}

fn load_states(path: &Path) -> Vec<BenchState> {
    let data = fs::read(path).expect("states.bin");
    let n = read_u64_le(&data) as usize;
    let flat_len = flat::FLAT_LEN;
    let rec = 2 + flat_len * 8;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let off = 8 + i * rec;
        let cls = data[off];
        let hero = data[off + 1] as usize;
        let mut buf = vec![0i64; flat_len];
        for j in 0..flat_len {
            buf[j] = i64::from_le_bytes(data[off + 2 + j * 8..off + 10 + j * 8].try_into().unwrap());
        }
        out.push(BenchState {
            cls,
            hero,
            env: flat::from_flat(&buf),
        });
    }
    out
}

fn load_config(path: &Path) -> HashMap<String, String> {
    let txt = fs::read_to_string(path).expect("config.txt");
    let mut m = HashMap::new();
    for line in txt.lines() {
        if let Some((k, v)) = line.split_once('=') {
            m.insert(k.trim().to_string(), v.trim().to_string());
        }
    }
    m
}

fn mask11(s: &str) -> [bool; 11] {
    let mut m = [false; 11];
    for (i, tok) in s.split(',').enumerate().take(11) {
        m[i] = tok.trim() == "1";
    }
    m
}

fn cls_name(c: u8) -> &'static str {
    match c {
        0 => "river",
        1 => "live_probe",
        2 => "mw_flop",
        _ => "?",
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: bench_subgame <bench_data_dir> [--k K] [--repeats R] [--seed S] [--states i,j,..] [--quiet-probs]");
        std::process::exit(2);
    }
    let dir = Path::new(&args[1]);
    let mut k: i32 = 2000;
    let mut repeats: usize = 3;
    let mut seed_base: u64 = 4242;
    let mut quiet = false;
    let mut only: Option<Vec<usize>> = None;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--k" => {
                k = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--repeats" => {
                repeats = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--seed" => {
                seed_base = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--states" => {
                only = Some(args[i + 1].split(',').map(|x| x.parse().unwrap()).collect());
                i += 2;
            }
            "--quiet-probs" => {
                quiet = true;
                i += 1;
            }
            other => panic!("unknown arg {other}"),
        }
    }

    let cfgm = load_config(&dir.join("config.txt"));
    let t0 = Instant::now();
    let bp = load_blueprint(&dir.join("blueprint.bin"));
    let states = load_states(&dir.join("states.bin"));
    let ids: Vec<String> = cfgm["ids"].split(',').map(|s| s.to_string()).collect();
    let emd_path = cfgm["emd_frozen_path"].clone();
    let lookup = emd::load_shared(&emd_path).expect("emd blob");
    eprintln!(
        "[bench] loaded bp={} states={} emd in {:.1}s",
        bp.len(),
        states.len(),
        t0.elapsed().as_secs_f64()
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
    let scfg = SubgameConfig {
        max_iterations: k,
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
        ensemble_streams: cfgm["ensemble_streams"].parse().unwrap(),
        leaf_cache_worlds: cfgm["leaf_cache_worlds"].parse().unwrap(),
        avg_skip_frac: cfgm["avg_skip_frac"].parse().unwrap(),
        leaf_biased_continuations: cfgm["leaf_biased_continuations"] == "1",
        leaf_bias_mult: cfgm["leaf_bias_mult"].parse().unwrap(),
        serve_final_iterate: cfgm["serve_final_iterate"] == "1",
        root_unmask: 0,
    };

    let sel: Vec<usize> = only.unwrap_or_else(|| (0..states.len()).collect());
    let mut per_class: HashMap<u8, Vec<f64>> = HashMap::new();
    for rep in 0..repeats {
        for &si in &sel {
            let st = &states[si];
            assert_eq!(NUM_PLAYERS, 6);
            let seed = seed_base
                .wrapping_add(si as u64 * 1_000_003)
                .wrapping_add(rep as u64 * 7_919);
            let t = Instant::now();
            let probs = subgame::solve(&st.env, st.hero, &bp, &pcfg, &scfg, seed);
            let ns = t.elapsed().as_nanos() as u64;
            per_class.entry(st.cls).or_default().push(ns as f64);
            let pcsv = if quiet {
                String::new()
            } else {
                probs
                    .iter()
                    .map(|x| format!("{x:.17e}"))
                    .collect::<Vec<_>>()
                    .join(",")
            };
            println!(
                "RESULT\t{}\t{}\t{}\t{}\t{}",
                ids[si],
                cls_name(st.cls),
                rep,
                ns,
                pcsv
            );
            #[cfg(feature = "prof")]
            println!(
                "PROF\t{}\t{}\twall_ns={} {}",
                ids[si],
                cls_name(st.cls),
                ns,
                battery_native::prof::dump_and_reset()
            );
        }
    }
    let mut classes: Vec<u8> = per_class.keys().copied().collect();
    classes.sort();
    for c in classes {
        let mut v = per_class.remove(&c).unwrap();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = v[v.len() / 2];
        let tot: f64 = v.iter().sum();
        println!(
            "SUMMARY\t{}\tn={}\tmedian_ms={:.2}\tmean_ms={:.2}\titers_per_s_med={:.0}",
            cls_name(c),
            v.len(),
            med / 1e6,
            (tot / v.len() as f64) / 1e6,
            k as f64 / (med / 1e9)
        );
    }
}
