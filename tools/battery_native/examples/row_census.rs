//! blueprint-v4 Stage-0 M1 helper: per-street classification of a checkpoint's rows
//! (artifacts/blueprint_pluribus_grade_design.md §6.1-M1 — measurement, not a gate).
//!
//! Full-history keys are opaque 16-byte digests over an astronomically large message space, so the
//! summary-era exhaustive hash inversion (classify_checkpoint_streets.py) cannot work. Instead we
//! WALK the abstract game (trainer-shaped traversals: hero branches every abstracted-legal action
//! for each of the 6 hero seats, opponents sample an epsilon-uniform mix of the checkpoint's OWN
//! average strategy), compute every visited decision state's key under the checkpoint's key space
//! (env-driven exactly like ab_train: POKERBOT_FULLHIST_KEY / POKERBOT_AB_NUM_BUCKETS /
//! POKERBOT_AB_EMD_FROZEN / POKERBOT_AB_RIVER_OCHS), and tag the matching checkpoint rows with the
//! state's street. Coverage is honest: unmatched rows stay unclassified (reported), matched rows
//! are EXACT (a digest match at a game-flow state pins the street).
//!
//! Usage: row_census.exe <ckpt.nlcfr> <n_deals> <out.json> [seed]

use battery_native::core::{apply_action, is_chance_node, sample_chance, EnvStateN, NUM_PLAYERS};
use battery_native::playout::Rng;
use battery_native::trainer::{
    abstracted_legal_pub, deal_root_pub, deserialize, fingerprint_fast_depth_memo_emd,
    keep_mask_from_env, sample_action_pub, uniform_over_legal_pub, TrainerCfg,
};
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::Arc;

const NUM_ACTIONS: usize = 11;

struct Ctx {
    cfg: TrainerCfg,
    strat: FxHashMap<[u8; 16], [f64; NUM_ACTIONS]>,
    cum_keys: FxHashSet<[u8; 16]>,
    classified: FxHashMap<[u8; 16], u8>,
    memo: battery_native::fingerprint::FpMemo,
    eps: f64,
    nodes: u64,
}

fn walk(s: &EnvStateN<{ NUM_PLAYERS }>, hero: usize, ctx: &mut Ctx, rng: &mut Rng, depth: u32) {
    if s.done || depth > 96 {
        return;
    }
    if is_chance_node(s) {
        let ns = sample_chance(s);
        walk(&ns, hero, ctx, rng, depth + 1);
        return;
    }
    let seat = s.action_seat;
    if seat < 0 {
        return;
    }
    let seat = seat as usize;
    let legal = abstracted_legal_pub(s, seat, &ctx.cfg);
    if !legal.iter().any(|&b| b) {
        return;
    }
    ctx.nodes += 1;
    let key = fingerprint_fast_depth_memo_emd(
        s,
        seat,
        ctx.cfg.num_buckets,
        &ctx.cfg.depth_edges,
        ctx.cfg.emd_frozen.as_deref(),
        ctx.cfg.river_ochs.as_deref(),
        &mut ctx.memo,
    );
    if ctx.strat.contains_key(&key) || ctx.cum_keys.contains(&key) {
        ctx.classified.entry(key).or_insert(s.street as u8);
    }
    if seat == hero {
        for a in 0..NUM_ACTIONS {
            if legal[a] {
                let ns = apply_action(s, a);
                walk(&ns, hero, ctx, rng, depth + 1);
            }
        }
    } else {
        let strat: [f64; NUM_ACTIONS] = if rng.next_f64() < ctx.eps {
            uniform_over_legal_pub(&legal)
        } else if let Some(row) = ctx.strat.get(&key) {
            let mut p = *row;
            let mut tot = 0.0;
            for a in 0..NUM_ACTIONS {
                if !legal[a] {
                    p[a] = 0.0;
                }
                tot += p[a];
            }
            if tot > 0.0 {
                for v in p.iter_mut() {
                    *v /= tot;
                }
                p
            } else {
                uniform_over_legal_pub(&legal)
            }
        } else {
            uniform_over_legal_pub(&legal)
        };
        let a = sample_action_pub(&strat, &legal, rng.next_f64());
        let ns = apply_action(s, a);
        walk(&ns, hero, ctx, rng, depth + 1);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let ckpt = args.get(1).expect("usage: row_census <ckpt> <n_deals> <out.json> [seed]");
    let n_deals: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2000);
    let out_path = args.get(3).map(|s| s.as_str()).unwrap_or("row_census.json");
    let seed: u64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0xBEEF);

    let num_buckets: i32 = std::env::var("POKERBOT_AB_NUM_BUCKETS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let emd_frozen = match std::env::var("POKERBOT_AB_EMD_FROZEN") {
        Ok(p) if !p.is_empty() => {
            Some(battery_native::emd::load_shared(&p).expect("load emd"))
        }
        _ => None,
    };
    let river_ochs = match std::env::var("POKERBOT_AB_RIVER_OCHS") {
        Ok(p) if !p.is_empty() => Some(Arc::new(
            battery_native::river_ochs::RiverOchs::load_path(&p).expect("load ochs"),
        )),
        _ => None,
    };
    let cfg = TrainerCfg {
        num_buckets,
        depth_edges: vec![],
        // Arm-3 menu envs (absent => legacy, byte-identical) — must match the run being censused.
        allowed_pre: keep_mask_from_env("POKERBOT_PRE_KEEP", &[4, 5, 6]),
        allowed_post: keep_mask_from_env("POKERBOT_POST_KEEP", &[0, 1, 3, 4, 5]),
        n_deals_per_iter: 200,
        base_seed: 0xC0FFEE,
        n_seats: 6,
        prune_threshold: f64::NEG_INFINITY,
        fast_merge: false,
        emd_frozen,
        batch_deals: 0,
        work_steal: false,
        dcfr: false,
        dcfr_prune_threshold: -300_000.0,
        dcfr_explore_frac: 0.05,
        river_ochs,
    };

    eprintln!("[row_census] loading {ckpt} ...");
    let blob = std::fs::read(ckpt).expect("read ckpt");
    let tab = deserialize(&blob).expect("deserialize");
    drop(blob);
    let mut strat: FxHashMap<[u8; 16], [f64; NUM_ACTIONS]> = FxHashMap::default();
    let mut cum_keys: FxHashSet<[u8; 16]> = FxHashSet::default();
    for sh in tab.shards.iter() {
        for (k, row) in sh.strat_sum.iter() {
            strat.insert(*k, *row);
        }
        for (k, _) in sh.cum_regret.iter() {
            cum_keys.insert(*k);
        }
    }
    let n_strat = strat.len();
    let n_cum = cum_keys.len();
    drop(tab);
    eprintln!(
        "[row_census] iters=? strat_rows={n_strat} cum_rows={n_cum} fullhist={} num_buckets={num_buckets}",
        std::env::var("POKERBOT_FULLHIST_KEY").map(|v| v == "1").unwrap_or(false)
    );

    let mut ctx = Ctx {
        cfg,
        strat,
        cum_keys,
        classified: FxHashMap::default(),
        memo: battery_native::fingerprint::FpMemo::default(),
        eps: 0.5,
        nodes: 0,
    };
    let t0 = std::time::Instant::now();
    let mut last = t0;
    let mut rng = Rng::seed_from_u64(seed);
    for d in 0..n_deals {
        let root: EnvStateN<{ NUM_PLAYERS }> = deal_root_pub(&mut rng, d ^ seed);
        for hero in 0..NUM_PLAYERS {
            walk(&root, hero, &mut ctx, &mut rng, 0);
        }
        if last.elapsed().as_secs_f64() > 15.0 {
            last = std::time::Instant::now();
            eprintln!(
                "[row_census] deal {d}/{n_deals} classified={}/{} ({:.1}%) nodes={}",
                ctx.classified.len(),
                n_strat,
                100.0 * ctx.classified.len() as f64 / n_strat.max(1) as f64,
                ctx.nodes
            );
        }
        if ctx.memo.len() > 20_000_000 {
            ctx.memo.clear(); // bound the digest memo
        }
    }
    let mut per_street = [0u64; 8];
    let mut strat_per_street = [0u64; 8];
    for (k, st) in ctx.classified.iter() {
        per_street[(*st as usize).min(7)] += 1;
        if ctx.strat.contains_key(k) {
            strat_per_street[(*st as usize).min(7)] += 1;
        }
    }
    let coverage = ctx.classified.len() as f64 / (n_strat.max(1)) as f64;
    let json = format!(
        "{{\"checkpoint\":\"{}\",\"n_deals\":{},\"seed\":{},\"strat_rows\":{},\"cum_rows\":{},\
         \"classified\":{},\"coverage_of_strat\":{:.4},\"nodes_walked\":{},\
         \"classified_per_street\":{{\"preflop\":{},\"flop\":{},\"turn\":{},\"river\":{}}},\
         \"strat_classified_per_street\":{{\"preflop\":{},\"flop\":{},\"turn\":{},\"river\":{}}},\
         \"wall_s\":{:.1}}}",
        ckpt.replace('\\', "/"),
        n_deals,
        seed,
        n_strat,
        n_cum,
        ctx.classified.len(),
        coverage,
        ctx.nodes,
        per_street[0],
        per_street[1],
        per_street[2],
        per_street[3],
        strat_per_street[0],
        strat_per_street[1],
        strat_per_street[2],
        strat_per_street[3],
        t0.elapsed().as_secs_f64()
    );
    std::fs::write(out_path, &json).expect("write out");
    eprintln!("[row_census] DONE {} -> {out_path}", &json);
}
