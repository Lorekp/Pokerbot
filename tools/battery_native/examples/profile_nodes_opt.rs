//! THROWAWAY per-node cost attribution for the OPTIMIZED native CFR+ trainer hot path.
//!
//! Identical methodology to examples/profile_nodes.rs (the BASELINE-path profiler), but it routes the
//! transition + terminal settle through the trainer's ACTUAL optimized primitives:
//!   - `trainer_apply` = `core::apply_action_with::<N,_,false>` (skips the 768-byte dead hist write) +
//!     the phe-fast `showdown_returns` settle  (== src/trainer.rs `trainer_apply`)
//!   - `showdown_returns` = phe 7-card `evaluate_cards` lookup fed into `core::settle_showdown_with_scores`
//!     (== src/trainer.rs `showdown_returns`/`phe_showdown_score`), instead of the 21-combo
//!     `core::compute_returns`.
//! So the ns/call + %-of-node-time here reflect the OPTIMIZED trainer, not the baseline engine path.
//!
//! Usage: cargo run --release --example profile_nodes_opt -- <n_deals> <warm_iters>

use std::time::Instant;

use battery_native::core::{self, EnvStateN, MAX_BOARD, NUM_ACTIONS, NUM_PLAYERS};
use battery_native::fingerprint::{
    self, evaluate_cards, fingerprint_fast_depth, fingerprint_fast_depth_memo, fingerprint_message,
    FpMemo, CHIP_UNIT,
};
use battery_native::playout::Rng;
use battery_native::trainer::{allowed_mask, NlheTables, TrainerCfg};

const ZEROS: [f64; NUM_ACTIONS] = [0.0; NUM_ACTIONS];
const START_STACK_CHIPS: i32 = 100 * 100;
const ANTE_CHIPS: i32 = 0;

// ===== EXACT copies of src/trainer.rs optimized terminal primitives (private there) =====
#[inline]
fn phe_showdown_score(holes: &[i32; 2], board: &[i32; MAX_BOARD]) -> i32 {
    let mut cards = [0i32; 7];
    cards[0] = holes[0];
    cards[1] = holes[1];
    let mut len = 2usize;
    for i in 0..MAX_BOARD {
        if board[i] >= 0 {
            cards[len] = board[i];
            len += 1;
        }
    }
    7463 - evaluate_cards(&cards[..len])
}

#[inline]
fn showdown_returns<const N: usize>(
    holes: &[[i32; 2]; N],
    board: &[i32; MAX_BOARD],
    seat_status: &[i8; N],
    committed_total: &[i32; N],
    button: i32,
    n_showdown: i32,
) -> [i32; N] {
    if n_showdown <= 1 {
        let mut winner = 0usize;
        for i in 0..N {
            if seat_status[i] == core::SEAT_LIVE || seat_status[i] == core::SEAT_ALLIN {
                winner = i;
                break;
            }
        }
        let total: i32 = committed_total.iter().sum();
        let mut out = [0i32; N];
        for i in 0..N {
            out[i] = -committed_total[i];
        }
        out[winner] += total;
        return out;
    }
    let mut scores = [-1i32; N];
    for s in 0..N {
        if seat_status[s] == core::SEAT_LIVE || seat_status[s] == core::SEAT_ALLIN {
            scores[s] = phe_showdown_score(&holes[s], board);
        }
    }
    core::settle_showdown_with_scores(&scores, seat_status, committed_total, button)
}

// trainer_apply: phe settle + WRITE_HIST=false (the trainer's actual transition).
#[inline]
fn trainer_apply<const N: usize>(env: &EnvStateN<N>, action: usize) -> EnvStateN<N> {
    core::apply_action_with::<N, _, false>(env, action, showdown_returns::<N>)
}

// ----- replicate trainer::deal_root via public init_hand_n -----
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
// compact_legal + sample_action_compact: the OPTIMIZED lever-4 sampling (build idx once).
fn compact_legal(legal: &[bool; NUM_ACTIONS]) -> ([usize; NUM_ACTIONS], usize) {
    let mut idx: [usize; NUM_ACTIONS] = [0; NUM_ACTIONS];
    let mut n = 0usize;
    for a in 0..NUM_ACTIONS {
        if legal[a] {
            idx[n] = a;
            n += 1;
        }
    }
    (idx, n)
}
fn sample_action_compact(
    strat: &[f64; NUM_ACTIONS],
    idx: &[usize; NUM_ACTIONS],
    n: usize,
    u: f64,
) -> usize {
    if n == 0 {
        return 0;
    }
    let mut s = 0.0;
    for i in 0..n {
        s += strat[idx[i]];
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

// terminal_return via the OPTIMIZED showdown_returns (phe).
fn record_cr<const N: usize>(env: &EnvStateN<N>, c: &mut Counters) {
    assert_eq!(N, NUM_PLAYERS);
    let n_show = env
        .seat_status
        .iter()
        .filter(|&&s| s == core::SEAT_LIVE || s == core::SEAT_ALLIN)
        .count() as i32;
    push_cr(env, n_show, c);
}
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
    let rets = showdown_returns(
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
    nodes: u64,
    hero_nodes: u64,
    opp_nodes: u64,
    chance_nodes: u64,
    terminals: u64,
    apply_action: u64,
    legal_mask: u64,
    fingerprint: u64,
    regret_match: u64,
    sample_action: u64,
    compute_returns: u64,
    cr_inputs: Vec<(
        [[i32; 2]; NUM_PLAYERS],
        [i32; 5],
        [i8; NUM_PLAYERS],
        [i32; NUM_PLAYERS],
        i32,
        i32,
    )>,
}

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
    let (lidx, ln) = compact_legal(&legal);
    let regrets = *read.cum_regret.get(&key).unwrap_or(&ZEROS);

    if actor != hero {
        c.opp_nodes += 1;
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
        let a = sample_action_compact(&strat, &lidx, ln, u);
        c.apply_action += 1;
        let child = trainer_apply(env, a);
        if child.done {
            record_cr_settle(&child, c);
        }
        return one_traversal(&child, hero, cfg, read, local, rng, t, memo, c);
    }

    c.regret_match += 1;
    let strat = regret_match(&regrets, &legal);
    let mut av = [0.0f64; NUM_ACTIONS];
    for i in 0..ln {
        let a = lidx[i];
        c.apply_action += 1;
        let child = trainer_apply(env, a);
        if child.done {
            record_cr_settle(&child, c);
        }
        av[a] = one_traversal(&child, hero, cfg, read, local, rng, t, memo, c);
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

fn harvest<const N: usize>(
    cfg: &TrainerCfg,
    n_deals: usize,
) -> Vec<(EnvStateN<N>, usize, [bool; NUM_ACTIONS])> {
    let mut out = Vec::new();
    let walk = |env: &EnvStateN<N>,
                rng: &mut Rng,
                memo: &mut FpMemo,
                out: &mut Vec<(EnvStateN<N>, usize, [bool; NUM_ACTIONS])>| {
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
            let strat = regret_match(&ZEROS, &legal);
            let (idx, n) = compact_legal(&legal);
            let a = sample_action_compact(&strat, &idx, n, rng.next_f64());
            e = trainer_apply(&e, a);
        }
    };
    let mut memo = FpMemo::default();
    for d in 0..n_deals {
        let mut rng = Rng::seed_from_u64(0xBEEF ^ d as u64);
        let root = deal_root::<N>(&mut rng, d as u64);
        for _ in 0..3 {
            walk(&root, &mut rng, &mut memo, &mut out);
        }
    }
    out
}

fn time_ns<F: FnMut() -> u64>(iters: u64, mut f: F) -> f64 {
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
    };

    let mut read = NlheTables::new();
    for _ in 0..warm {
        read.train_iteration(&cfg);
    }
    let t = read.iters_done + 1;
    let iter_seed = cfg.base_seed ^ t.wrapping_mul(0x9E3779B97F4A7C15);
    let mut memo = FpMemo::default();
    let mut c = Counters::default();
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
    println!("fingerprint(memo)   {}", c.fingerprint);
    println!(
        "compute_returns TOTAL (== captured cr_inputs) {}",
        c.cr_inputs.len()
    );

    let states: Vec<(EnvStateN<NUM_PLAYERS>, usize, [bool; NUM_ACTIONS])> =
        harvest::<NUM_PLAYERS>(&cfg, n_deals.max(64));
    let ns = states.len() as u64;
    println!("\n=== MICRO-BENCH ns/call (over {ns} harvested decision states) ===");
    let reps = 2000u64;

    let t_legal = time_ns(reps, || {
        let mut acc = 0u64;
        for (s, seat, _) in &states {
            let m = core::legal_action_mask(s, *seat as i32);
            acc = acc.wrapping_add(m[0] as u64 + m[10] as u64);
        }
        acc
    }) / ns as f64;

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

    let t_fp_miss = time_ns(reps, || {
        let mut acc = 0u64;
        for (s, seat, _) in &states {
            let k = fingerprint_fast_depth(s, *seat, cfg.num_buckets, &cfg.depth_edges);
            acc = acc.wrapping_add(k[0] as u64);
        }
        acc
    }) / ns as f64;

    let t_fp_msg = time_ns(reps, || {
        let mut acc = 0u64;
        for (s, seat, _) in &states {
            let (m, l) = fingerprint_message(s, *seat, cfg.num_buckets, &cfg.depth_edges);
            acc = acc.wrapping_add(m[0] as u64 + l as u64);
        }
        acc
    }) / ns as f64;

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

    // OPTIMIZED apply: trainer_apply (WRITE_HIST=false + phe settle).
    let t_apply = time_ns(reps, || {
        let mut acc = 0u64;
        for (s, _seat, legal) in &states {
            let a = (0..NUM_ACTIONS).find(|&a| legal[a]).unwrap_or(1);
            let child = trainer_apply(s, a);
            acc = acc.wrapping_add(child.action_seat as u64 ^ child.done as u64);
        }
        acc
    }) / ns as f64;

    let t_copy = {
        let s0 = states[0].0;
        time_ns(reps * ns, || {
            let s = std::hint::black_box(s0);
            s.action_seat as u64 ^ s.done as u64
        })
    };

    let nonterm: Vec<_> = states
        .iter()
        .filter_map(|(s, _seat, legal)| {
            let a = (0..NUM_ACTIONS).find(|&a| legal[a]).unwrap_or(1);
            let child = trainer_apply(s, a);
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
                let child = trainer_apply(s, *a);
                acc = acc.wrapping_add(child.action_seat as u64);
            }
            acc
        }) / nn as f64
    };

    // OPTIMIZED settle: showdown_returns (phe) over the TRUE terminal distribution.
    let cr_inputs = &c.cr_inputs;
    let n_cr = cr_inputs.len().max(1) as u64;
    let n_showdown = cr_inputs.iter().filter(|(_, _, _, _, _, n)| *n > 1).count();
    let n_foldwin = cr_inputs.len() - n_showdown;
    let t_cr = time_ns(reps, || {
        let mut acc = 0u64;
        for (holes, board, ss, ct, btn, n) in cr_inputs.iter() {
            let r = showdown_returns(holes, board, ss, ct, *btn, *n);
            acc = acc.wrapping_add(r[0] as u64);
        }
        acc
    }) / n_cr as f64;
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
                let r = showdown_returns(holes, board, ss, ct, *btn, *n);
                acc = acc.wrapping_add(r[0] as u64);
            }
            acc
        }) / nn as f64
    };
    // Compare against the BASELINE 21-combo core::compute_returns on the SAME showdown states.
    let t_cr_showdown_baseline = if showdown_states.is_empty() {
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
                let r = showdown_returns(holes, board, ss, ct, *btn, *n);
                acc = acc.wrapping_add(r[0] as u64);
            }
            acc
        }) / nn as f64
    };

    let t_rm = time_ns(reps, || {
        let mut acc = 0u64;
        for (_s, _seat, legal) in &states {
            let r = regret_match(&ZEROS, legal);
            acc = acc.wrapping_add(r[0].to_bits());
        }
        acc
    }) / ns as f64;

    let t_sa = time_ns(reps, || {
        let mut acc = 0u64;
        for (_s, _seat, legal) in &states {
            let strat = regret_match(&ZEROS, legal);
            let (idx, n) = compact_legal(legal);
            let a = sample_action_compact(&strat, &idx, n, 0.42);
            acc = acc.wrapping_add(a as u64);
        }
        acc
    }) / ns as f64;

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
    println!("  fp message-only    {:8.1} ns/call", t_fp_msg);
    println!(
        "  postflop bucket    {:8.1} ns/call  (phe, postflop subset avg)",
        t_bucket
    );
    println!(
        "trainer_apply        {:8.1} ns/call  (WRITE_HIST=false + phe settle, mixed)",
        t_apply
    );
    println!(
        "  apply nonterminal  {:8.1} ns/call  (transition+Copy, no internal settle)",
        t_apply_nonterm
    );
    println!(
        "  bare state Copy    {:8.1} ns/call  ({} bytes)",
        t_copy,
        std::mem::size_of::<EnvStateN<NUM_PLAYERS>>()
    );
    println!(
        "showdown_returns     {:8.1} ns/call  (phe, TRUE mix: {} showdown / {} fold-win of {})",
        t_cr,
        n_showdown,
        n_foldwin,
        cr_inputs.len()
    );
    println!(
        "  sd showdown(n>1)   {:8.1} ns/call  (phe multiway eval)",
        t_cr_showdown
    );
    println!(
        "  [baseline 21-combo {:8.1} ns/call  core::compute_returns on SAME states]",
        t_cr_showdown_baseline
    );
    println!("  sd fold-win(n<=1)  {:8.1} ns/call", t_cr_foldwin);
    println!("regret_match         {:8.1} ns/call", t_rm);
    println!(
        "sample_action        {:8.1} ns/call  (compact, incl regret_match)",
        t_sa
    );
    println!("rng.next_f64         {:8.3} ns/call", t_rng);

    let calls_fp = c.fingerprint as f64;
    let calls_legal = c.legal_mask as f64;
    let calls_apply = c.apply_action as f64;
    let calls_cr = cr_inputs.len() as f64;
    let calls_rm = c.regret_match as f64;
    let calls_sa = c.sample_action as f64;
    let calls_chance = c.chance_nodes as f64;

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
                let (idx, n) = compact_legal(&legal);
                let a = sample_action_compact(&strat, &idx, n, rng.next_f64());
                e = trainer_apply(&e, a);
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
    println!("sample_chance        {:8.1} ns/call", t_samplechance);

    // PER-ITERATION FIXED OVERHEAD (merge + CFR+ floor).
    {
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
        println!("\n=== PER-ITERATION FIXED OVERHEAD (serial, scales with table/n_deals) ===");
        println!("cum_regret table rows: {tbl_n}");
        println!("CFR+ floor pass (full table): {:.1} us/iter", t_floor / 1e3);
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

    struct Row {
        name: &'static str,
        ns_call: f64,
        calls: f64,
    }
    let rows = vec![
        Row {
            name: "trainer_apply (transition)",
            ns_call: t_apply_nonterm,
            calls: calls_apply,
        },
        Row {
            name: "showdown_returns (terminals)",
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
            "{:30} {:7.1} ns/call  x {:>10.0} calls = {:6.2}%",
            r.name,
            r.ns_call,
            r.calls,
            100.0 * contrib / total
        );
    }
}
