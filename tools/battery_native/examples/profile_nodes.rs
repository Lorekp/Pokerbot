//! THROWAWAY per-node cost attribution for the native CFR+ trainer hot path (post-memo baseline).
//!
//! Re-implements the EXACT trainer recursion (deal_root + one_traversal) single-threaded using the
//! crate's public API, with (a) node-type + per-function call counters to get calls/iteration, and
//! (b) harvested representative decision states to micro-bench each hot function in isolation (ns/call).
//!
//! % of node time = (ns/call * calls/iteration) / sum over functions. Not shipped; dev profiling only.
//!
//! Usage: cargo run --release --example profile_nodes -- <n_deals> <warm_iters>

use std::time::Instant;

use battery_native::core::{self, EnvStateN, NUM_ACTIONS, NUM_PLAYERS};
use battery_native::fingerprint::{
    self, fingerprint_fast_depth, fingerprint_fast_depth_memo, fingerprint_message, FpMemo,
    CHIP_UNIT,
};
use battery_native::playout::Rng;
use battery_native::trainer::{allowed_mask, NlheTables, TrainerCfg};

const ZEROS: [f64; NUM_ACTIONS] = [0.0; NUM_ACTIONS];
const START_STACK_CHIPS: i32 = 100 * 100;
const ANTE_CHIPS: i32 = 0;

// ----- replicate trainer::deal_root (private) via public init_hand_n -----
fn deal_root<const N: usize>(rng: &mut Rng, deal_idx: u64) -> EnvStateN<N> {
    let mut deck: [i32; 52] = [0; 52];
    for (i, c) in deck.iter_mut().enumerate() {
        *c = i as i32;
    }
    for i in (1..52).rev() {
        let j = rng.next_below((i + 1) as u64) as usize;
        deck.swap(i, j);
    }
    let button = (deal_idx % N as u64) as i32;
    let stacks = [START_STACK_CHIPS; N];
    let occupancy = [true; N];
    core::init_hand_n::<N>(button, &stacks, &occupancy, ANTE_CHIPS, &deck, [0u32, 0u32])
}

fn abstracted_legal<const N: usize>(
    s: &EnvStateN<N>,
    seat: usize,
    cfg: &TrainerCfg,
) -> [bool; NUM_ACTIONS] {
    let full = core::legal_action_mask(s, seat as i32);
    let allowed = if s.street == 0 {
        &cfg.allowed_pre
    } else {
        &cfg.allowed_post
    };
    let mut m = [false; NUM_ACTIONS];
    for a in 0..NUM_ACTIONS {
        m[a] = full[a] && allowed[a];
    }
    m
}
fn any_legal(m: &[bool; NUM_ACTIONS]) -> bool {
    m.iter().any(|&x| x)
}
fn regret_match(regrets: &[f64; NUM_ACTIONS], legal: &[bool; NUM_ACTIONS]) -> [f64; NUM_ACTIONS] {
    let mut pos = [0.0f64; NUM_ACTIONS];
    let mut s = 0.0;
    for a in 0..NUM_ACTIONS {
        if legal[a] {
            let v = regrets[a].max(0.0);
            pos[a] = v;
            s += v;
        }
    }
    if s > 0.0 {
        for a in 0..NUM_ACTIONS {
            pos[a] /= s;
        }
        return pos;
    }
    let nl = legal.iter().filter(|&&x| x).count().max(1) as f64;
    let mut out = [0.0f64; NUM_ACTIONS];
    for a in 0..NUM_ACTIONS {
        out[a] = if legal[a] { 1.0 / nl } else { 0.0 };
    }
    out
}
fn sample_action(strat: &[f64; NUM_ACTIONS], legal: &[bool; NUM_ACTIONS], u: f64) -> usize {
    let mut idx: [usize; NUM_ACTIONS] = [0; NUM_ACTIONS];
    let mut n = 0usize;
    let mut s = 0.0;
    for a in 0..NUM_ACTIONS {
        if legal[a] {
            idx[n] = a;
            n += 1;
            s += strat[a];
        }
    }
    if n == 0 {
        return 0;
    }
    if s <= 1e-12 {
        return idx[0];
    }
    let mut cum = 0.0;
    for i in 0..n {
        cum += strat[idx[i]] / s;
        if u < cum {
            return idx[i];
        }
    }
    idx[n - 1]
}
// terminal_return leaves call compute_returns with n_show = live+allin count. (N always 6 here.)
fn record_cr<const N: usize>(env: &EnvStateN<N>, c: &mut Counters) {
    assert_eq!(N, NUM_PLAYERS);
    let n_show = env
        .seat_status
        .iter()
        .filter(|&&s| s == core::SEAT_LIVE || s == core::SEAT_ALLIN)
        .count() as i32;
    push_cr(env, n_show, c);
}
// apply_action-internal settle calls compute_returns with n_live = live count (see core::apply_action).
fn record_cr_settle<const N: usize>(env: &EnvStateN<N>, c: &mut Counters) {
    assert_eq!(N, NUM_PLAYERS);
    let n_live = env
        .seat_status
        .iter()
        .filter(|&&s| s == core::SEAT_LIVE)
        .count() as i32;
    push_cr(env, n_live, c);
}
fn push_cr<const N: usize>(env: &EnvStateN<N>, n: i32, c: &mut Counters) {
    let mut holes = [[0i32; 2]; NUM_PLAYERS];
    let mut ss = [0i8; NUM_PLAYERS];
    let mut ct = [0i32; NUM_PLAYERS];
    for i in 0..NUM_PLAYERS {
        holes[i] = env.holes[i];
        ss[i] = env.seat_status[i];
        ct[i] = env.committed_total[i];
    }
    c.cr_inputs.push((holes, env.board, ss, ct, env.button, n));
}

fn terminal_return<const N: usize>(env: &EnvStateN<N>, hero: usize) -> f64 {
    let n_show = env
        .seat_status
        .iter()
        .filter(|&&s| s == core::SEAT_LIVE || s == core::SEAT_ALLIN)
        .count() as i32;
    let rets = core::compute_returns(
        &env.holes,
        &env.board,
        &env.seat_status,
        &env.committed_total,
        env.button,
        n_show,
    );
    rets[hero] as f64 / CHIP_UNIT
}

#[derive(Default, Clone)]
struct Counters {
    nodes: u64,        // non-terminal decision nodes visited (fingerprint computed here)
    hero_nodes: u64,   // hero decision nodes (branch all actions)
    opp_nodes: u64,    // opponent decision nodes (sample one)
    chance_nodes: u64, // is_chance_node true -> sample_chance
    terminals: u64,    // terminal leaf returns (compute_returns via terminal_return OR env.rewards)
    apply_action: u64,
    legal_mask: u64, // legal_action_mask calls (one per decision node, inside abstracted_legal)
    fingerprint: u64, // fingerprint_fast_depth_memo calls (one per decision node)
    regret_match: u64,
    sample_action: u64,
    compute_returns: u64, // compute_returns calls (terminal_return + apply_action internal settle)
    // captured REAL terminal settle args (the exact compute_returns inputs the trainer passes), so the
    // compute_returns micro-bench runs on the TRUE terminal distribution (fold-wins cheap, showdowns
    // expensive) — not a forced 6-way showdown on every harvested state.
    cr_inputs: Vec<(
        [[i32; 2]; NUM_PLAYERS],
        [i32; 5],
        [i8; NUM_PLAYERS],
        [i32; NUM_PLAYERS],
        i32,
        i32,
    )>,
}

// Counting + memoized one_traversal, mirroring trainer::one_traversal exactly. `read` empty (cold,
// the warmup phase reflects the true visitation shape). Counts EVERY call to each hot fn.
#[allow(clippy::too_many_arguments)]
fn one_traversal<const N: usize>(
    env: &EnvStateN<N>,
    hero: usize,
    cfg: &TrainerCfg,
    read: &NlheTables,
    local: &mut NlheTables,
    rng: &mut Rng,
    t: u64,
    memo: &mut FpMemo,
    c: &mut Counters,
) -> f64 {
    if env.done {
        c.terminals += 1;
        return env.rewards[hero] as f64 / CHIP_UNIT;
    }
    if core::is_chance_node(env) {
        c.chance_nodes += 1;
        let child = core::sample_chance(env);
        return one_traversal(&child, hero, cfg, read, local, rng, t, memo, c);
    }
    if env.street >= core::STREET_TERMINAL {
        c.terminals += 1;
        c.compute_returns += 1;
        record_cr(env, c);
        return terminal_return(env, hero);
    }
    let actor_i = env.action_seat;
    if actor_i < 0 {
        c.terminals += 1;
        c.compute_returns += 1;
        record_cr(env, c);
        return terminal_return(env, hero);
    }
    let actor = actor_i as usize;
    c.legal_mask += 1;
    let legal = abstracted_legal(env, actor, cfg);
    if !any_legal(&legal) {
        c.terminals += 1;
        c.compute_returns += 1;
        return terminal_return(env, hero);
    }
    c.nodes += 1;
    c.fingerprint += 1;
    let key = fingerprint_fast_depth_memo(env, actor, cfg.num_buckets, &cfg.depth_edges, memo);

    if actor != hero {
        c.opp_nodes += 1;
        let regrets = *read.cum_regret.get(&key).unwrap_or(&ZEROS);
        c.regret_match += 1;
        let strat = regret_match(&regrets, &legal);
        let w = t as f64;
        {
            let ss = local.strat_sum.entry(key).or_insert(ZEROS);
            for a in 0..NUM_ACTIONS {
                if legal[a] {
                    ss[a] += w * strat[a];
                }
            }
        }
        let u = rng.next_f64();
        c.sample_action += 1;
        let a = sample_action(&strat, &legal, u);
        c.apply_action += 1;
        let child = core::apply_action(env, a);
        if child.done {
            record_cr_settle(&child, c);
        }
        return one_traversal(&child, hero, cfg, read, local, rng, t, memo, c);
    }

    let regrets = *read.cum_regret.get(&key).unwrap_or(&ZEROS);
    c.regret_match += 1;
    let strat = regret_match(&regrets, &legal);
    let mut av = [0.0f64; NUM_ACTIONS];
    for a in 0..NUM_ACTIONS {
        if legal[a] {
            c.apply_action += 1;
            let child = core::apply_action(env, a);
            if child.done {
                record_cr_settle(&child, c);
            }
            av[a] = one_traversal(&child, hero, cfg, read, local, rng, t, memo, c);
        }
    }
    c.hero_nodes += 1;
    let mut ev = 0.0;
    for a in 0..NUM_ACTIONS {
        ev += strat[a] * av[a];
    }
    {
        let r = local.cum_regret.entry(key).or_insert(ZEROS);
        for a in 0..NUM_ACTIONS {
            if legal[a] {
                r[a] += av[a] - ev;
            }
        }
    }
    ev
}

// Harvest representative decision states (preflop + postflop) from real traversals, recording the
// (state, actor) at every decision node so we can micro-bench each hot fn over the true mix.
fn harvest<const N: usize>(
    cfg: &TrainerCfg,
    n_deals: usize,
) -> Vec<(EnvStateN<N>, usize, [bool; NUM_ACTIONS])> {
    let mut out = Vec::new();
    let mut walk = |env: &EnvStateN<N>,
                    hero: usize,
                    rng: &mut Rng,
                    memo: &mut FpMemo,
                    out: &mut Vec<(EnvStateN<N>, usize, [bool; NUM_ACTIONS])>| {
        // iterative single-line walk down one sampled path, recording opp-style decision states.
        let mut e = *env;
        let mut guard = 0;
        loop {
            guard += 1;
            if guard > 512 || e.done {
                break;
            }
            if core::is_chance_node(&e) {
                e = core::sample_chance(&e);
                continue;
            }
            if e.street >= core::STREET_TERMINAL || e.action_seat < 0 {
                break;
            }
            let actor = e.action_seat as usize;
            let legal = abstracted_legal(&e, actor, cfg);
            if !any_legal(&legal) {
                break;
            }
            let key =
                fingerprint_fast_depth_memo(&e, actor, cfg.num_buckets, &cfg.depth_edges, memo);
            let _ = key;
            out.push((e, actor, legal));
            // sample one legal action uniformly to advance.
            let strat = regret_match(&ZEROS, &legal);
            let a = sample_action(&strat, &legal, rng.next_f64());
            e = core::apply_action(&e, a);
        }
        let _ = hero;
    };
    let mut memo = FpMemo::default();
    for d in 0..n_deals {
        let mut rng = Rng::seed_from_u64(0xBEEF ^ d as u64);
        let root = deal_root::<N>(&mut rng, d as u64);
        for _ in 0..3 {
            walk(&root, 0, &mut rng, &mut memo, &mut out);
        }
    }
    out
}

fn time_ns<F: FnMut() -> u64>(iters: u64, mut f: F) -> f64 {
    // returns ns/iter; f returns a black-box accumulator to defeat DCE.
    let t0 = Instant::now();
    let mut acc = 0u64;
    for _ in 0..iters {
        acc = acc.wrapping_add(f());
    }
    let dt = t0.elapsed().as_secs_f64();
    std::hint::black_box(acc);
    dt * 1e9 / iters as f64
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_deals: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(64);
    let warm: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(40);

    let cfg = TrainerCfg {
        num_buckets: 200,
        depth_edges: vec![],
        allowed_pre: allowed_mask(&[4, 5, 6]),
        allowed_post: allowed_mask(&[0, 1, 3, 4, 5]),
        n_deals_per_iter: n_deals,
        base_seed: 0xC0FFEE_u64,
        n_seats: 6,
        prune_threshold: f64::NEG_INFINITY,
        fast_merge: false,
        emd_frozen: None,
        batch_deals: 0,
        work_steal: false,
        dcfr: false,
        dcfr_prune_threshold: -300_000.0,
        dcfr_explore_frac: 0.05,
    };

    // ---- (1) call-count attribution: run ONE single-threaded iteration's worth of deals through the
    // counting traversal, on a WARMED read table (so the memo + visitation reflect steady state). ----
    let mut read = NlheTables::new();
    // warm the persistent read table with `warm` real (rayon) iterations so cum_regret is populated.
    for _ in 0..warm {
        read.train_iteration(&cfg);
    }
    let t = read.iters_done + 1;
    let iter_seed = cfg.base_seed ^ t.wrapping_mul(0x9E3779B97F4A7C15);
    let mut memo = FpMemo::default();
    let mut c = Counters::default();
    // memo warm pass (so fingerprint counts reflect the steady-state hit path; count is unaffected).
    for d in 0..n_deals {
        let deal_seed = iter_seed
            .wrapping_add((d as u64).wrapping_mul(0x100000001b3))
            .rotate_left(17);
        let mut rng = Rng::seed_from_u64(deal_seed);
        let root = deal_root::<NUM_PLAYERS>(&mut rng, d as u64);
        let mut local = NlheTables::default();
        for hero in 0..NUM_PLAYERS {
            one_traversal::<NUM_PLAYERS>(
                &root, hero, &cfg, &read, &mut local, &mut rng, t, &mut memo, &mut c,
            );
        }
    }
    // compute_returns also fires inside apply_action on every terminal-settling action. Count those by
    // instrumentation proxy: every apply_action that produces a `done` state calls compute_returns once.
    // We approximate via a second pass counting apply_action->done. (Reported separately below.)
    let mut cr_in_apply = 0u64;
    {
        for d in 0..n_deals {
            let deal_seed = iter_seed
                .wrapping_add((d as u64).wrapping_mul(0x100000001b3))
                .rotate_left(17);
            let mut rng = Rng::seed_from_u64(deal_seed);
            let root = deal_root::<NUM_PLAYERS>(&mut rng, d as u64);
            let mut local = NlheTables::default();
            let mut cc = Counters::default();
            for hero in 0..NUM_PLAYERS {
                count_apply_done::<NUM_PLAYERS>(
                    &root,
                    hero,
                    &cfg,
                    &read,
                    &mut local,
                    &mut rng,
                    t,
                    &mut memo,
                    &mut cc,
                    &mut cr_in_apply,
                );
            }
        }
    }

    println!(
        "EnvStateN<6> size = {} bytes (Copy on every apply_action / sample_chance)",
        std::mem::size_of::<EnvStateN<NUM_PLAYERS>>()
    );
    println!("=== CALL COUNTS for ONE iteration (n_deals={n_deals}, warmed {warm} iters) ===");
    println!("decision_nodes      {}", c.nodes);
    println!("  hero_nodes        {}", c.hero_nodes);
    println!("  opp_nodes         {}", c.opp_nodes);
    println!("chance_nodes        {}", c.chance_nodes);
    println!("terminals           {}", c.terminals);
    println!("apply_action        {}", c.apply_action);
    println!("legal_mask          {}", c.legal_mask);
    println!("fingerprint(memo)   {}", c.fingerprint);
    println!("regret_match        {}", c.regret_match);
    println!("sample_action       {}", c.sample_action);
    println!("compute_returns(term){}", c.compute_returns);
    println!("compute_returns(in apply_action, done) {}", cr_in_apply);
    println!("compute_returns TOTAL {}", c.compute_returns + cr_in_apply);

    // ---- (2) per-function ns/call micro-benches over harvested decision states ----
    let states: Vec<(EnvStateN<NUM_PLAYERS>, usize, [bool; NUM_ACTIONS])> =
        harvest::<NUM_PLAYERS>(&cfg, n_deals.max(64));
    let ns = states.len() as u64;
    println!("\n=== MICRO-BENCH ns/call (over {ns} harvested decision states) ===");
    let reps = 2000u64;

    // legal_action_mask
    let t_legal = time_ns(reps, || {
        let mut acc = 0u64;
        for (s, seat, _) in &states {
            let m = core::legal_action_mask(s, *seat as i32);
            acc = acc.wrapping_add(m[0] as u64 + m[10] as u64);
        }
        acc
    }) / ns as f64;

    // fingerprint MEMO (steady-state hit) — warm a memo first, then time hits.
    let mut hot_memo = FpMemo::default();
    for (s, seat, _) in &states {
        let _ =
            fingerprint_fast_depth_memo(s, *seat, cfg.num_buckets, &cfg.depth_edges, &mut hot_memo);
    }
    let t_fp_memo_hit = time_ns(reps, || {
        let mut acc = 0u64;
        for (s, seat, _) in &states {
            let k = fingerprint_fast_depth_memo(
                s,
                *seat,
                cfg.num_buckets,
                &cfg.depth_edges,
                &mut hot_memo,
            );
            acc = acc.wrapping_add(k[0] as u64);
        }
        acc
    }) / ns as f64;

    // fingerprint MISS (full blake2b each call) — non-memo path.
    let t_fp_miss = time_ns(reps, || {
        let mut acc = 0u64;
        for (s, seat, _) in &states {
            let k = fingerprint_fast_depth(s, *seat, cfg.num_buckets, &cfg.depth_edges);
            acc = acc.wrapping_add(k[0] as u64);
        }
        acc
    }) / ns as f64;

    // fingerprint MESSAGE only (no blake2b): isolates the state-feature extraction cost.
    let t_fp_msg = time_ns(reps, || {
        let mut acc = 0u64;
        for (s, seat, _) in &states {
            let (m, l) = fingerprint_message(s, *seat, cfg.num_buckets, &cfg.depth_edges);
            acc = acc.wrapping_add(m[0] as u64 + l as u64);
        }
        acc
    }) / ns as f64;

    // postflop bucket (phevaluator) only, on the postflop subset.
    let t_bucket = time_ns(reps, || {
        let mut acc = 0u64;
        for (s, seat, _) in &states {
            let b = fingerprint::compute_postflop_bucket(
                s.holes[*seat][0],
                s.holes[*seat][1],
                &s.board,
                cfg.num_buckets,
            );
            acc = acc.wrapping_add((b + 1) as u64);
        }
        acc
    }) / ns as f64;

    // apply_action: apply the FIRST legal action at each state (realistic state Copy + transition).
    let t_apply = time_ns(reps, || {
        let mut acc = 0u64;
        for (s, _seat, legal) in &states {
            let a = (0..NUM_ACTIONS).find(|&a| legal[a]).unwrap_or(1);
            let child = core::apply_action(s, a);
            acc = acc.wrapping_add(child.action_seat as u64 ^ child.done as u64);
        }
        acc
    }) / ns as f64;

    // bare state Copy cost: `let s = *s_in` — the dominant fixed cost of apply_action/sample_chance.
    let t_copy = {
        let s0 = states[0].0;
        time_ns(reps * ns, || {
            let s = std::hint::black_box(s0);
            s.action_seat as u64 ^ s.done as u64
        })
    };

    // apply_action on states whose chosen action does NOT settle to terminal (isolates transition
    // logic + Copy from the internal compute_returns on a done-state).
    let nonterm: Vec<_> = states
        .iter()
        .filter_map(|(s, _seat, legal)| {
            let a = (0..NUM_ACTIONS).find(|&a| legal[a]).unwrap_or(1);
            let child = core::apply_action(s, a);
            if child.done {
                None
            } else {
                Some((*s, a))
            }
        })
        .collect();
    let t_apply_nonterm = if nonterm.is_empty() {
        0.0
    } else {
        let nn = nonterm.len() as u64;
        time_ns(reps, || {
            let mut acc = 0u64;
            for (s, a) in &nonterm {
                let child = core::apply_action(s, *a);
                acc = acc.wrapping_add(child.action_seat as u64);
            }
            acc
        }) / nn as f64
    };

    // compute_returns: over the TRUE terminal distribution captured during the counting traversal
    // (fold-wins with n_show<=1 take the cheap branch; multiway showdowns run the expensive eval).
    let cr_inputs = &c.cr_inputs;
    let n_cr = cr_inputs.len().max(1) as u64;
    let n_showdown = cr_inputs.iter().filter(|(_, _, _, _, _, n)| *n > 1).count();
    let n_foldwin = cr_inputs.len() - n_showdown;
    let t_cr = time_ns(reps, || {
        let mut acc = 0u64;
        for (holes, board, ss, ct, btn, n) in cr_inputs.iter() {
            let r = core::compute_returns(holes, board, ss, ct, *btn, *n);
            acc = acc.wrapping_add(r[0] as u64);
        }
        acc
    }) / n_cr as f64;
    // also bench the two sub-populations to expose the cost split.
    let showdown_states: Vec<_> = cr_inputs
        .iter()
        .filter(|(_, _, _, _, _, n)| *n > 1)
        .cloned()
        .collect();
    let t_cr_showdown = if showdown_states.is_empty() {
        0.0
    } else {
        let nn = showdown_states.len() as u64;
        time_ns(reps, || {
            let mut acc = 0u64;
            for (holes, board, ss, ct, btn, n) in &showdown_states {
                let r = core::compute_returns(holes, board, ss, ct, *btn, *n);
                acc = acc.wrapping_add(r[0] as u64);
            }
            acc
        }) / nn as f64
    };
    let foldwin_states: Vec<_> = cr_inputs
        .iter()
        .filter(|(_, _, _, _, _, n)| *n <= 1)
        .cloned()
        .collect();
    let t_cr_foldwin = if foldwin_states.is_empty() {
        0.0
    } else {
        let nn = foldwin_states.len() as u64;
        time_ns(reps, || {
            let mut acc = 0u64;
            for (holes, board, ss, ct, btn, n) in &foldwin_states {
                let r = core::compute_returns(holes, board, ss, ct, *btn, *n);
                acc = acc.wrapping_add(r[0] as u64);
            }
            acc
        }) / nn as f64
    };

    // sample_chance: only meaningful on chance nodes; bench on states forced pending (advance a copy).
    let t_chance = time_ns(reps, || {
        let mut acc = 0u64;
        for (s, _seat, _) in &states {
            // force a chance node by closing the street artificially is invasive; instead time the
            // common path: is_chance_node check + (if chance) sample_chance. Most harvested states are
            // decision nodes, so this mostly times is_chance_node. We separately bench sample_chance on
            // a constructed pending state below.
            let ch = core::is_chance_node(s);
            acc = acc.wrapping_add(ch as u64);
        }
        acc
    }) / ns as f64;

    // regret_match
    let t_rm = time_ns(reps, || {
        let mut acc = 0u64;
        for (_s, _seat, legal) in &states {
            let r = regret_match(&ZEROS, legal);
            acc = acc.wrapping_add(r[0].to_bits());
        }
        acc
    }) / ns as f64;

    // sample_action
    let t_sa = time_ns(reps, || {
        let mut acc = 0u64;
        for (_s, _seat, legal) in &states {
            let strat = regret_match(&ZEROS, legal);
            let a = sample_action(&strat, legal, 0.42);
            acc = acc.wrapping_add(a as u64);
        }
        acc
    }) / ns as f64;

    // RNG next_f64
    let t_rng = {
        let mut rng = Rng::seed_from_u64(1);
        time_ns(reps * ns, || rng.next_f64().to_bits())
    };

    println!("legal_action_mask    {:8.1} ns/call", t_legal);
    println!("fingerprint memo-hit {:8.1} ns/call", t_fp_memo_hit);
    println!(
        "fingerprint MISS     {:8.1} ns/call  (full blake2b)",
        t_fp_miss
    );
    println!(
        "  fp message-only    {:8.1} ns/call  (feature extract, no hash)",
        t_fp_msg
    );
    println!(
        "  postflop bucket    {:8.1} ns/call  (phevaluator, postflop subset avg)",
        t_bucket
    );
    println!(
        "apply_action         {:8.1} ns/call  (incl state Copy, mixed term/nonterm)",
        t_apply
    );
    println!(
        "  apply nonterminal  {:8.1} ns/call  (transition+Copy, no internal compute_returns)",
        t_apply_nonterm
    );
    println!(
        "  bare state Copy    {:8.1} ns/call  (let s = *s_in, {} bytes)",
        t_copy,
        std::mem::size_of::<EnvStateN<NUM_PLAYERS>>()
    );
    println!(
        "compute_returns      {:8.1} ns/call  (TRUE terminal mix: {} showdown / {} fold-win of {} captured)",
        t_cr, n_showdown, n_foldwin, cr_inputs.len()
    );
    println!(
        "  cr showdown(n>1)   {:8.1} ns/call  (multiway eval)",
        t_cr_showdown
    );
    println!(
        "  cr fold-win(n<=1)  {:8.1} ns/call  (cheap branch)",
        t_cr_foldwin
    );
    println!("is_chance_node       {:8.1} ns/call", t_chance);
    println!("regret_match         {:8.1} ns/call", t_rm);
    println!(
        "sample_action        {:8.1} ns/call  (incl regret_match)",
        t_sa
    );
    println!("rng.next_f64         {:8.3} ns/call", t_rng);

    // ---- (3) combine into % of node time using steady-state call counts ----
    // Per-iteration totals (calls):
    let calls_fp = c.fingerprint as f64;
    let calls_legal = c.legal_mask as f64;
    let calls_apply = c.apply_action as f64;
    // compute_returns calls = exactly the captured terminal-settle population (true mix already timed).
    let calls_cr = cr_inputs.len() as f64;
    let _ = cr_in_apply;
    let calls_rm = c.regret_match as f64;
    let calls_sa = c.sample_action as f64;
    let calls_chance = c.chance_nodes as f64; // sample_chance calls

    // sample_chance cost (constructed): time on a freshly sampled chance node.
    // Build a chance node by walking a deal to street-close.
    let t_samplechance = {
        let mut rng = Rng::seed_from_u64(7);
        let mut chance_states: Vec<EnvStateN<NUM_PLAYERS>> = Vec::new();
        for d in 0..256u64 {
            let mut e = deal_root::<NUM_PLAYERS>(&mut rng, d);
            let mut g = 0;
            while g < 256 && !e.done {
                g += 1;
                if core::is_chance_node(&e) {
                    chance_states.push(e);
                    e = core::sample_chance(&e);
                    continue;
                }
                if e.street >= core::STREET_TERMINAL || e.action_seat < 0 {
                    break;
                }
                let seat = e.action_seat as usize;
                let legal = abstracted_legal(&e, seat, &cfg);
                if !any_legal(&legal) {
                    break;
                }
                let strat = regret_match(&ZEROS, &legal);
                let a = sample_action(&strat, &legal, rng.next_f64());
                e = core::apply_action(&e, a);
            }
        }
        if chance_states.is_empty() {
            0.0
        } else {
            let nch = chance_states.len() as u64;
            time_ns(reps, || {
                let mut acc = 0u64;
                for s in &chance_states {
                    let ch = core::sample_chance(s);
                    acc = acc.wrapping_add(ch.street as u64);
                }
                acc
            }) / nch as f64
        }
    };
    println!(
        "sample_chance        {:8.1} ns/call  (constructed chance nodes)",
        t_samplechance
    );

    // ---- (4) rayon MERGE + CFR+ floor overhead (per-iteration FIXED cost, scales with TABLE size not
    // n_deals). The bench's it/s drop with n_deals while deals/s stays ~flat is this fixed cost being
    // amortized over more deals. Measure: time `warm` more train_iterations and also time a manual
    // merge of one partial + a full floor pass over the warmed cum_regret, to expose the per-iter tax.
    {
        // floor pass over the full cum_regret table (runs once per iteration).
        let tbl_n = read.cum_regret.len();
        let mut floor_clone = read.cum_regret.clone();
        let t_floor = time_ns(50, || {
            let mut acc = 0u64;
            for row in floor_clone.values_mut() {
                for a in 0..NUM_ACTIONS {
                    if row[a] < 0.0 {
                        row[a] = 0.0;
                    }
                }
                acc = acc.wrapping_add(row[0].to_bits());
            }
            acc
        });
        println!("\n=== PER-ITERATION FIXED OVERHEAD (scales with table, not n_deals) ===",);
        println!("cum_regret table rows: {tbl_n}");
        println!("CFR+ floor pass (full table): {:.1} us/iter", t_floor / 1e3);
        // merge: time merging `n_deals` typical partials into a fresh accumulator (what the iteration
        // does serially after the par_iter). Build representative partials by running one counting iter.
        let mut partials: Vec<NlheTables> = Vec::new();
        for d in 0..n_deals {
            let deal_seed = iter_seed
                .wrapping_add((d as u64).wrapping_mul(0x100000001b3))
                .rotate_left(17);
            let mut rng = Rng::seed_from_u64(deal_seed);
            let root = deal_root::<NUM_PLAYERS>(&mut rng, d as u64);
            let mut local = NlheTables::default();
            let mut cc = Counters::default();
            for hero in 0..NUM_PLAYERS {
                one_traversal::<NUM_PLAYERS>(
                    &root, hero, &cfg, &read, &mut local, &mut rng, t, &mut memo, &mut cc,
                );
            }
            partials.push(local);
        }
        let avg_partial_rows: f64 = partials
            .iter()
            .map(|p| p.cum_regret.len() + p.strat_sum.len())
            .sum::<usize>() as f64
            / partials.len().max(1) as f64;
        let t_merge = time_ns(20, || {
            let mut acc = NlheTables::default();
            // replicate NlheTables::merge (private) field-by-field.
            for p in &partials {
                for (k, dr) in &p.cum_regret {
                    let e = acc.cum_regret.entry(*k).or_insert(ZEROS);
                    for a in 0..NUM_ACTIONS {
                        e[a] += dr[a];
                    }
                }
                for (k, ds) in &p.strat_sum {
                    let e = acc.strat_sum.entry(*k).or_insert(ZEROS);
                    for a in 0..NUM_ACTIONS {
                        e[a] += ds[a];
                    }
                }
            }
            acc.cum_regret.len() as u64
        });
        println!(
            "serial merge of {n_deals} partials (avg {:.0} rows each): {:.1} us/iter",
            avg_partial_rows,
            t_merge / 1e3
        );
    }

    // Per-iteration time contribution (ns):
    struct Row {
        name: &'static str,
        ns_call: f64,
        calls: f64,
    }
    // De-double-count: apply_action's MIXED cost (t_apply) already includes the internal compute_returns
    // it runs on a terminal-settling action. Attribute the TRANSITION cost (t_apply_nonterm) to all
    // apply_action calls, and the terminal-settle compute_returns ONCE via the compute_returns row
    // (calls_cr = the captured terminal population, which == the apply-internal settles + the
    // terminal_return leaves). This splits the work additively with no overlap.
    let rows = vec![
        Row {
            name: "apply_action (transition)",
            ns_call: t_apply_nonterm,
            calls: calls_apply,
        },
        Row {
            name: "compute_returns (terminals)",
            ns_call: t_cr,
            calls: calls_cr,
        },
        Row {
            name: "fingerprint (memo-hit)",
            ns_call: t_fp_memo_hit,
            calls: calls_fp,
        },
        Row {
            name: "legal_action_mask",
            ns_call: t_legal,
            calls: calls_legal,
        },
        Row {
            name: "regret_match",
            ns_call: t_rm,
            calls: calls_rm,
        },
        Row {
            name: "sample_action(excl rm)",
            ns_call: (t_sa - t_rm).max(0.0),
            calls: calls_sa,
        },
        Row {
            name: "sample_chance",
            ns_call: t_samplechance,
            calls: calls_chance,
        },
        Row {
            name: "rng.next_f64",
            ns_call: t_rng,
            calls: calls_sa,
        },
    ];
    let _ = t_apply;
    let total: f64 = rows.iter().map(|r| r.ns_call * r.calls).sum();
    println!("\n=== % OF NODE TIME (ns/call * calls/iter, single-thread model) ===");
    println!("modeled total per-iter node work: {:.0} us", total / 1e3);
    let mut sorted: Vec<&Row> = rows.iter().collect();
    sorted.sort_by(|a, b| {
        (b.ns_call * b.calls)
            .partial_cmp(&(a.ns_call * a.calls))
            .unwrap()
    });
    for r in sorted {
        let contrib = r.ns_call * r.calls;
        println!(
            "{:28} {:7.1} ns/call  x {:>10.0} calls = {:6.2}%",
            r.name,
            r.ns_call,
            r.calls,
            100.0 * contrib / total
        );
    }
}

// Second-pass counter: counts compute_returns calls that happen INSIDE apply_action (terminal settle).
#[allow(clippy::too_many_arguments)]
fn count_apply_done<const N: usize>(
    env: &EnvStateN<N>,
    hero: usize,
    cfg: &TrainerCfg,
    read: &NlheTables,
    local: &mut NlheTables,
    rng: &mut Rng,
    t: u64,
    memo: &mut FpMemo,
    _c: &mut Counters,
    cr_in_apply: &mut u64,
) -> f64 {
    if env.done {
        return env.rewards[hero] as f64 / CHIP_UNIT;
    }
    if core::is_chance_node(env) {
        let child = core::sample_chance(env);
        return count_apply_done(
            &child,
            hero,
            cfg,
            read,
            local,
            rng,
            t,
            memo,
            _c,
            cr_in_apply,
        );
    }
    if env.street >= core::STREET_TERMINAL {
        return terminal_return(env, hero);
    }
    let actor_i = env.action_seat;
    if actor_i < 0 {
        return terminal_return(env, hero);
    }
    let actor = actor_i as usize;
    let legal = abstracted_legal(env, actor, cfg);
    if !any_legal(&legal) {
        return terminal_return(env, hero);
    }
    let key = fingerprint_fast_depth_memo(env, actor, cfg.num_buckets, &cfg.depth_edges, memo);
    if actor != hero {
        let regrets = *read.cum_regret.get(&key).unwrap_or(&ZEROS);
        let strat = regret_match(&regrets, &legal);
        let u = rng.next_f64();
        let a = sample_action(&strat, &legal, u);
        let child = core::apply_action(env, a);
        if child.done {
            *cr_in_apply += 1;
        }
        return count_apply_done(
            &child,
            hero,
            cfg,
            read,
            local,
            rng,
            t,
            memo,
            _c,
            cr_in_apply,
        );
    }
    let regrets = *read.cum_regret.get(&key).unwrap_or(&ZEROS);
    let strat = regret_match(&regrets, &legal);
    let mut av = [0.0f64; NUM_ACTIONS];
    for a in 0..NUM_ACTIONS {
        if legal[a] {
            let child = core::apply_action(env, a);
            if child.done {
                *cr_in_apply += 1;
            }
            av[a] = count_apply_done(
                &child,
                hero,
                cfg,
                read,
                local,
                rng,
                t,
                memo,
                _c,
                cr_in_apply,
            );
        }
    }
    let mut ev = 0.0;
    for a in 0..NUM_ACTIONS {
        ev += strat[a] * av[a];
    }
    ev
}
