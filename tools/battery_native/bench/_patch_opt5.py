# OPT-5: merge regrets + strat_sum into one FxHashMap<u128, Row> (GTO path only). Bit-identical:
# same arithmetic, same update order; just one map (1 probe for opp reads, 2 for hero updates
# instead of 3) and better locality.
import io

p = 'src/subgame.rs'
t = io.open(p, encoding='utf-8').read()
R = []

R.append(("""/// Mutable per-solve tables (regrets + strategy sums keyed by infoset, leaf-value cache keyed by
/// public leaf state). OPT-2: keyed by the 128-bit incremental digests (see `NodeDig`) in FxHashMaps
/// — no per-node key allocation, O(1) hashing. The EXPLOIT path keeps its own legacy-keyed
/// `TablesExploit` (firewall).
struct Tables {
    regrets: FxHashMap<u128, [f64; NUM_ACTIONS]>,
    strat_sum: FxHashMap<u128, [f64; NUM_ACTIONS]>,
    /// Leaf value cache: `(sum, n)` running mean over the first `leaf_cache_worlds` VISITING hole-worlds
    /// (n frozen at that cap). With leaf_cache_worlds=1 (default) this is exactly the legacy first-touch
    /// single-value cache: `(v, 1)` inserted on the miss, `sum / 1` (== v bit-for-bit) on every hit.
    leaf: FxHashMap<u128, (f64, i32)>,
}""",
"""/// One infoset's mutable state: cumulative CFR+ regrets + the (linear-)averaged strategy sum.
/// OPT-5: merged into ONE row so a hero-node update is a single map probe (entry) instead of the
/// legacy regrets.insert + strat_sum.entry pair, and an opponent read touches one map. Arithmetic
/// and update order are unchanged (bit-identical outputs).
#[derive(Clone, Copy)]
struct IsetRow {
    reg: [f64; NUM_ACTIONS],
    ss: [f64; NUM_ACTIONS],
}

const ROW_ZERO: IsetRow = IsetRow {
    reg: [0.0; NUM_ACTIONS],
    ss: [0.0; NUM_ACTIONS],
};

/// Mutable per-solve tables (regrets + strategy sums keyed by infoset, leaf-value cache keyed by
/// public leaf state). OPT-2: keyed by the 128-bit incremental digests (see `NodeDig`) in FxHashMaps
/// — no per-node key allocation, O(1) hashing. The EXPLOIT path keeps its own legacy-keyed
/// `TablesExploit` (firewall).
struct Tables {
    rows: FxHashMap<u128, IsetRow>,
    /// Leaf value cache: `(sum, n)` running mean over the first `leaf_cache_worlds` VISITING hole-worlds
    /// (n frozen at that cap). With leaf_cache_worlds=1 (default) this is exactly the legacy first-touch
    /// single-value cache: `(v, 1)` inserted on the miss, `sum / 1` (== v bit-for-bit) on every hit.
    leaf: FxHashMap<u128, (f64, i32)>,
}"""))

# leaf_value_biased choice sampling
R.append(("""        let ckd = cont_dig(iset_dig(nd, env, s));
        let ck = cont_choice_key(env, s);
        let regs = *tab.regrets.get(&ckd).unwrap_or(&ZEROS);
        choices[s] = sample_cont_choice(&regs, &ck, t, ctx.public_seed) as i32;""",
"""        let ckd = cont_dig(iset_dig(nd, env, s));
        let ck = cont_choice_key(env, s);
        let regs = tab.rows.get(&ckd).map(|r| r.reg).unwrap_or(ZEROS);
        choices[s] = sample_cont_choice(&regs, &ck, t, ctx.public_seed) as i32;"""))

R.append(("""    let ckd = cont_dig(iset_dig(nd, env, hero));
    let regs = *tab.regrets.get(&ckd).unwrap_or(&ZEROS);
    let strat = regret_match(&regs, &CONT_LEGAL);""",
"""    let ckd = cont_dig(iset_dig(nd, env, hero));
    let regs = tab.rows.get(&ckd).map(|r| r.reg).unwrap_or(ZEROS);
    let strat = regret_match(&regs, &CONT_LEGAL);"""))

R.append(("""    tab.regrets.insert(ckd, newr);
    // Strategy-sum accumulation: identical gating/weighting to the action nodes (delayed averaging +
    // CFR+ linear weight), so the choice node is a first-class citizen of the served average.
    if t >= ctx.avg_skip {
        let w = if ctx.scfg.linear_averaging {
            (t + 1) as f64
        } else {
            1.0
        };
        let ss = tab.strat_sum.entry(ckd).or_insert([0.0; NUM_ACTIONS]);
        for c in 0..playout::NUM_CONTINUATIONS {
            ss[c] += w * strat[c];
        }
    }
    ev
}""",
"""    let row = tab.rows.entry(ckd).or_insert(ROW_ZERO);
    row.reg = newr;
    // Strategy-sum accumulation: identical gating/weighting to the action nodes (delayed averaging +
    // CFR+ linear weight), so the choice node is a first-class citizen of the served average.
    if t >= ctx.avg_skip {
        let w = if ctx.scfg.linear_averaging {
            (t + 1) as f64
        } else {
            1.0
        };
        for c in 0..playout::NUM_CONTINUATIONS {
            row.ss[c] += w * strat[c];
        }
    }
    ev
}"""))

# cfv opp node
R.append(("""    if actor != hero {
        prof_count!(crate::prof::C_NODE_OPP);
        let regrets = *tab.regrets.get(&key).unwrap_or(&ZEROS);""",
"""    if actor != hero {
        prof_count!(crate::prof::C_NODE_OPP);
        let regrets = tab.rows.get(&key).map(|r| r.reg).unwrap_or(ZEROS);"""))

# cfv hero node read
R.append(("""    // hero node: branch all (abstracted-)legal actions, CFR+ regret update.
    prof_count!(crate::prof::C_NODE_HERO);
    let regrets = *tab.regrets.get(&key).unwrap_or(&ZEROS);
    let strat = regret_match(&regrets, &legal);""",
"""    // hero node: branch all (abstracted-)legal actions, CFR+ regret update.
    prof_count!(crate::prof::C_NODE_HERO);
    let regrets = tab.rows.get(&key).map(|r| r.reg).unwrap_or(ZEROS);
    let strat = regret_match(&regrets, &legal);"""))

# cfv hero node write
R.append(("""    tab.regrets.insert(key, newr);
    // BUG-5 (partial) — DELAYED AVERAGING: the SERVED average excludes iterations t < avg_skip
    // (floor(avg_skip_frac * K), 0 = legacy). Regret updates above are NEVER skipped.
    if t >= ctx.avg_skip {
        // CFR+ linear averaging: weight iteration t's strategy by (t+1) (matches the Python solver).
        let w = if ctx.scfg.linear_averaging {
            (t + 1) as f64
        } else {
            1.0
        };
        let ss = tab.strat_sum.entry(key).or_insert([0.0; NUM_ACTIONS]);
        for a in 0..NUM_ACTIONS {
            ss[a] += w * strat[a];
        }
    }
    ev
}""",
"""    let row = tab.rows.entry(key).or_insert(ROW_ZERO);
    row.reg = newr;
    // BUG-5 (partial) — DELAYED AVERAGING: the SERVED average excludes iterations t < avg_skip
    // (floor(avg_skip_frac * K), 0 = legacy). Regret updates above are NEVER skipped.
    if t >= ctx.avg_skip {
        // CFR+ linear averaging: weight iteration t's strategy by (t+1) (matches the Python solver).
        let w = if ctx.scfg.linear_averaging {
            (t + 1) as f64
        } else {
            1.0
        };
        for a in 0..NUM_ACTIONS {
            row.ss[a] += w * strat[a];
        }
    }
    ev
}"""))

# solve_stream tables init + extraction
R.append(("""    let mut tab = Tables {
        regrets: FxHashMap::default(),
        strat_sum: FxHashMap::default(),
        leaf: FxHashMap::default(),
    };""",
"""    let mut tab = Tables {
        rows: FxHashMap::default(),
        leaf: FxHashMap::default(),
    };"""))

R.append(("""    // Return the ROOT rows raw (unnormalized) so the ensemble can SUM strat_sum across streams.
    let ss = tab.strat_sum.get(&root_key).copied().unwrap_or(ZEROS);
    let has_avg = ss.iter().sum::<f64>() > 1e-12;
    let rr = tab.regrets.get(&root_key).copied().unwrap_or(ZEROS);
    (ss, rr, has_avg)""",
"""    // Return the ROOT rows raw (unnormalized) so the ensemble can SUM strat_sum across streams.
    let root_row = tab.rows.get(&root_key).copied().unwrap_or(ROW_ZERO);
    let ss = root_row.ss;
    let has_avg = ss.iter().sum::<f64>() > 1e-12;
    (ss, root_row.reg, has_avg)"""))

for old, new in R:
    n = t.count(old)
    assert n == 1, (n, old[:70])
    t = t.replace(old, new)
io.open(p, 'w', encoding='utf-8').write(t)
print('OPT-5 edits applied:', len(R))
