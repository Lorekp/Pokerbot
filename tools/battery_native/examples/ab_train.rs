//! Local blueprint mini-trainer (f32_regret strategy-delta harness, extended for the blueprint-v4
//! Stage-0 causal mini-proof — artifacts/blueprint_pluribus_grade_design.md §7.1).
//!
//! Trains the 6-max CFR+ (or DCFR+RBP) blueprint for a fixed number of iterations at a FIXED seed
//! and writes NLCFR001 checkpoints. The Stage-0 arms drive it purely via env (all default to the
//! ORIGINAL harness behavior when unset — the f32 A/B usage is unchanged):
//!
//!   POKERBOT_FULLHIST_KEY=1      read by fingerprint.rs (OnceLock) — the Stage-0 keying arm.
//!   POKERBOT_AB_NUM_BUCKETS      made-hand postflop bucket count (default 200; K1000 arms).
//!   POKERBOT_AB_DCFR=1           DCFR(1.5,0,2) + negative-regret pruning (threshold -300k bb,
//!                                explore 0.05 — the Pluribus-rule gated trainer mode).
//!   POKERBOT_AB_EMD_FROZEN=path  EMD frozen flop/turn lookup (HE160 arm).
//!   POKERBOT_AB_RIVER_OCHS=path  river OCHS table (HE160 arm).
//!   POKERBOT_AB_CHECKPOINTS      comma list of iteration counts to checkpoint at (e.g.
//!                                "5000,10000,20000,30000,35000,40000,45000"). Files are written
//!                                next to <out> as iter_XXXXXX.nlcfr (the bp3 ladder shape).
//!   POKERBOT_AB_RESUME=1         resume from the LATEST iter_*.nlcfr next to <out> (kill-safe:
//!                                a restarted arm continues from its last checkpoint).
//!   POKERBOT_WORKSTEAL=1         dynamic deal scheduler (statistically equivalent, +33% wall).
//!
//! blueprint-v4 v3 trainer schedule (artifacts/blueprint_v4_amendment_v3.md; all default-off =>
//! byte-identical, read by trainer::V3Cfg::from_env inside the lib):
//!
//!   POKERBOT_SIGNED_REGRETS=1    arm-2 mode: signed regrets + regret floor + hard pruning, NO
//!                                LCFR discount, CFR+ w=t averaging kept (excl. with LCFR).
//!   POKERBOT_LCFR=1              Pluribus Linear-CFR mode: signed regrets + periodic discount +
//!                                unweighted average accumulation.
//!   POKERBOT_LCFR_D / _W         discount interval / window in ITERATIONS (defaults 250 / 10000).
//!   POKERBOT_PRUNE_C             hard negative-regret prune threshold in bb (default off).
//!   POKERBOT_PRUNE_START         first prune iteration (default W/2).
//!   POKERBOT_PRUNE_EXPLORE       explore fraction (default 0.05).
//!   POKERBOT_REGRET_FLOOR        regret floor in bb (default C*31/30).
//!   POKERBOT_SNAPSHOT_AVG=1      no-postflop-strat_sum + NLSNAP01 snapshot emission at every
//!                                checkpoint (offline average via snap_avg.exe).
//!   POKERBOT_SNAP_START          first iteration eligible for snapshot emission (default 0;
//!                                Pluribus discarded pre-800-min snapshots — set ~8% of the target).
//!   POKERBOT_COV_HANDS           on-policy coverage probe hands at each checkpoint (default 0=off).
//!   POKERBOT_REGSTATS=1          per-checkpoint regret-percentile line (prune-C calibration).
//!
//! Usage: ab_train.exe <out.nlcfr> <iters> <n_deals> [batch_deals]
//!
//! NOTE: pure-Rust example (no `python` feature) — uses the public trainer API only. The on-disk
//! blob is f64 for BOTH builds (cross-precision-portable), so Python diffing needs no native import.

use battery_native::trainer::{
    allowed_mask, coverage_probe, deserialize, regret_stats, serialize,
    serialize_current_strategy, NlheTables, TrainerCfg,
};
use std::sync::Arc;

fn env_flag(name: &str) -> bool {
    std::env::var(name).map(|v| v == "1").unwrap_or(false)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out_path = args.get(1).map(|s| s.as_str()).unwrap_or("ab_train_out.bin");
    let iters: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4000);
    let n_deals: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);
    // argv[4] (or env POKERBOT_BATCH_DEALS): batched-MLP traversal width. 0/1 = recursive (default);
    // >=2 = lockstep batched. Used for the local nodes/s A/B (batch_deals 8 vs 1).
    let batch_deals: usize = args
        .get(4)
        .and_then(|s| s.parse().ok())
        .or_else(|| std::env::var("POKERBOT_BATCH_DEALS").ok().and_then(|s| s.parse().ok()))
        .unwrap_or(0);

    let num_buckets: i32 = std::env::var("POKERBOT_AB_NUM_BUCKETS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let dcfr = env_flag("POKERBOT_AB_DCFR");
    let work_steal = env_flag("POKERBOT_WORKSTEAL");
    let emd_frozen = match std::env::var("POKERBOT_AB_EMD_FROZEN") {
        Ok(p) if !p.is_empty() => Some(
            battery_native::emd::load_shared(&p).expect("load POKERBOT_AB_EMD_FROZEN"),
        ),
        _ => None,
    };
    let river_ochs = match std::env::var("POKERBOT_AB_RIVER_OCHS") {
        Ok(p) if !p.is_empty() => Some(Arc::new(
            battery_native::river_ochs::RiverOchs::load_path(&p)
                .expect("load POKERBOT_AB_RIVER_OCHS"),
        )),
        _ => None,
    };
    let mut checkpoints: Vec<u64> = std::env::var("POKERBOT_AB_CHECKPOINTS")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_default();
    checkpoints.sort_unstable();

    // Representative deployed cfg (mirrors trainer::tests::test_cfg); Stage-0 arms override via env.
    let cfg = TrainerCfg {
        num_buckets,
        depth_edges: vec![],
        allowed_pre: allowed_mask(&[4, 5, 6]),
        allowed_post: allowed_mask(&[0, 1, 3, 4, 5]),
        n_deals_per_iter: n_deals,
        base_seed: 0xC0FFEE,
        n_seats: 6,
        prune_threshold: f64::NEG_INFINITY,
        fast_merge: false,
        emd_frozen,
        batch_deals,
        work_steal,
        dcfr,
        dcfr_prune_threshold: -300_000.0,
        dcfr_explore_frac: 0.05,
        river_ochs,
    };

    let precision = if cfg!(feature = "f32_regret") {
        "f32_regret"
    } else {
        "f64"
    };
    let fullhist = env_flag("POKERBOT_FULLHIST_KEY");
    eprintln!(
        "[ab_train] precision={precision} iters={iters} n_deals={n_deals} batch_deals={batch_deals} \
         num_buckets={num_buckets} dcfr={dcfr} work_steal={work_steal} fullhist={fullhist} \
         emd={} ochs={} ckpts={:?} -> {out_path}",
        cfg.emd_frozen.is_some(),
        cfg.river_ochs.is_some(),
        checkpoints,
    );
    // v3 trainer-schedule banner (the trainer itself reads the same env via its OnceLock; this is
    // the human-visible record of what the run trained with — the babysitter greps it).
    let v3 = battery_native::trainer::V3Cfg::from_env();
    let snap_start: u64 = std::env::var("POKERBOT_SNAP_START")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let cov_hands: u64 = std::env::var("POKERBOT_COV_HANDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let regstats = env_flag("POKERBOT_REGSTATS");
    eprintln!(
        "[ab_train] v3: lcfr={} signed_regrets={} lcfr_d={} lcfr_w={} prune_c={} prune_start={} \
         prune_explore={} regret_floor={} snapshot_avg={} snap_start={snap_start} \
         cov_hands={cov_hands} regstats={regstats}",
        v3.lcfr,
        v3.signed_regrets,
        v3.lcfr_d,
        v3.lcfr_w,
        v3.prune_c,
        v3.prune_start,
        v3.prune_explore,
        v3.regret_floor,
        v3.snapshot_avg,
    );

    let out_dir = std::path::Path::new(out_path)
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    if !out_dir.as_os_str().is_empty() {
        let _ = std::fs::create_dir_all(&out_dir);
    }
    let ckpt_path = |it: u64| out_dir.join(format!("iter_{:06}.nlcfr", it));
    let snap_path = |it: u64| out_dir.join(format!("snap_{:06}.nlsnap", it));

    // v3 checkpoint-cadence extras (all read-only on the table; all default-off):
    //   * NLSNAP01 snapshot emission (POKERBOT_SNAPSHOT_AVG=1, from POKERBOT_SNAP_START) — the
    //     rounds-2-4 snapshot-averaging inputs, tmp+rename like checkpoints.
    //   * COVERAGE line (POKERBOT_COV_HANDS>0) — the on-policy per-street miss-rate metric the
    //     coverage kill/health gates pre-register against (Stage-0 verdict §3.1).
    //   * REGSTATS line (POKERBOT_REGSTATS=1) — negative/positive regret percentiles for the
    //     hard-prune-C calibration.
    let ckpt_extras = |tab: &NlheTables, cfg: &TrainerCfg, it: u64| {
        let v3 = battery_native::trainer::V3Cfg::from_env();
        let snap_start: u64 = std::env::var("POKERBOT_SNAP_START")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let cov_hands: u64 = std::env::var("POKERBOT_COV_HANDS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        if v3.snapshot_avg && it >= snap_start {
            let blob = serialize_current_strategy(tab);
            let p = snap_path(it);
            let tmp = p.with_extension("nlsnap.tmp");
            std::fs::write(&tmp, &blob).expect("write snapshot tmp");
            std::fs::rename(&tmp, &p).expect("rename snapshot");
            eprintln!(
                "[ab_train] SNAPSHOT iter={} rows={} bytes={} -> {}",
                it,
                (blob.len() - 24) / 60,
                blob.len(),
                p.display()
            );
        }
        if cov_hands > 0 {
            let t0 = std::time::Instant::now();
            let c = coverage_probe(tab, cfg, cov_hands, 0xC0FFEE ^ it);
            eprintln!(
                "[ab_train] COVERAGE iter={} hands={} miss% pre={:.1} flop={:.1} turn={:.1} \
                 river={:.1} postflop={:.1} pre_allin_runout={:.1} decisions={:?} wall_s={:.1}",
                it,
                c.hands,
                100.0 * c.miss_rate(0),
                100.0 * c.miss_rate(1),
                100.0 * c.miss_rate(2),
                100.0 * c.miss_rate(3),
                100.0 * c.postflop_miss_rate(),
                100.0 * c.preflop_allin_runout_rate(),
                c.decisions,
                t0.elapsed().as_secs_f64(),
            );
        }
        if env_flag("POKERBOT_REGSTATS") {
            let st = regret_stats(tab);
            eprintln!(
                "[ab_train] REGSTATS iter={} rows={} neg_vals={} pos_vals={} neg_p50={:.3e} \
                 neg_p90={:.3e} neg_p99={:.3e} neg_p999={:.3e} neg_max={:.3e} pos_p99={:.3e} \
                 pos_max={:.3e}",
                it,
                st.rows,
                st.neg_vals,
                st.pos_vals,
                st.neg_p50,
                st.neg_p90,
                st.neg_p99,
                st.neg_p999,
                st.neg_max,
                st.pos_p99,
                st.pos_max,
            );
        }
    };

    // Resume: newest iter_*.nlcfr in the out dir (<= iters). The trainer's deal schedule is a pure
    // function of (base_seed, iters_done, deal index), so resume reproduces the exact remaining run.
    let mut tab = NlheTables::new();
    if env_flag("POKERBOT_AB_RESUME") {
        let mut best: Option<(u64, std::path::PathBuf)> = None;
        if let Ok(rd) = std::fs::read_dir(&out_dir) {
            for e in rd.flatten() {
                let name = e.file_name().to_string_lossy().to_string();
                if let Some(numpart) = name
                    .strip_prefix("iter_")
                    .and_then(|s| s.strip_suffix(".nlcfr"))
                {
                    if let Ok(it) = numpart.parse::<u64>() {
                        if it <= iters && best.as_ref().map(|(b, _)| it > *b).unwrap_or(true) {
                            best = Some((it, e.path()));
                        }
                    }
                }
            }
        }
        if let Some((it, p)) = best {
            let blob = std::fs::read(&p).expect("read resume checkpoint");
            tab = deserialize(&blob).expect("deserialize resume checkpoint");
            eprintln!(
                "[ab_train] RESUMED from {} (iters_done={}, expected {})",
                p.display(),
                tab.iters_done,
                it
            );
        } else {
            eprintln!("[ab_train] resume requested but no iter_*.nlcfr found; starting fresh");
        }
    }

    let already = tab.iters_done;
    let t0 = std::time::Instant::now();
    let mut last = t0;
    // In-process memory guard (OPT-IN, OFF the byte-identity path). Under POKERBOT_FULLHIST_KEY the
    // dynamic Shard table grows ~42k rows/iter with no plateau; on a ~31 GB box its committed
    // footprint crosses the OS commit limit and the OS issues an external ExitProcess(1)/rc1 kill
    // BEFORE the runner's 2s RSS poll can react (Windows working-set trimming hides the climb). Row
    // count is a deterministic, allocation-free proxy for committed memory. When POKERBOT_AB_ROW_CAP>0
    // and the table reaches it, write a checkpoint at the CURRENT iter and exit(0) so the arm records
    // furthest-reached partial evidence (design §7.1.2). Default 0 = disabled => byte-identical to the
    // old harness (S200/S1000/production never set it).
    let row_cap: u64 = std::env::var("POKERBOT_AB_ROW_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    for i in (already + 1)..=iters {
        tab.train_iteration(&cfg);
        if checkpoints.binary_search(&i).is_ok() {
            // Time serialize + local write separately (ops instrumentation for the cloud driver's
            // checkpoint-pause budget: serialize+write is the ONLY training pause once the S3 upload
            // is detached — see the arm-2 operational addendum). No effect on the blob bytes.
            let t_ser = std::time::Instant::now();
            let blob = serialize(&tab);
            let ser_s = t_ser.elapsed().as_secs_f64();
            let p = ckpt_path(i);
            let tmp = p.with_extension("nlcfr.tmp");
            let t_wr = std::time::Instant::now();
            std::fs::write(&tmp, &blob).expect("write checkpoint tmp");
            std::fs::rename(&tmp, &p).expect("rename checkpoint");
            let wr_s = t_wr.elapsed().as_secs_f64();
            eprintln!(
                "[ab_train] CHECKPOINT iter={} strat_rows={} bytes={} serialize_s={:.2} \
                 write_s={:.2} -> {}",
                i,
                tab.n_strat_infosets(),
                blob.len(),
                ser_s,
                wr_s,
                p.display()
            );
            ckpt_extras(&tab, &cfg, i);
        }
        // Graceful memory-cap stop (only decides WHEN to stop; never any value — see row_cap above).
        // Writes an already-correct checkpoint at the current iter via the SAME serialize() path,
        // then exit(0) so the run records status at this checkpoint instead of an external rc1 kill.
        // Placed AFTER the fixed-checkpoint block so a stop can land at ANY iter (the crash at iter
        // 2040 fell BETWEEN fixed checkpoints 1000 and 2500).
        // v3 note: under POKERBOT_SNAPSHOT_AVG the strat map is preflop-bounded (~60M) while
        // cum_regret keeps growing — the cap must watch BOTH maps. max == strat for every legacy
        // config (measured strat > cum on all Stage-0 arms), so pre-v3 behavior is unchanged.
        if row_cap > 0 && (tab.n_strat_infosets().max(tab.n_cum_infosets()) as u64) >= row_cap {
            let blob = serialize(&tab);
            let p = ckpt_path(i);
            let tmp = p.with_extension("nlcfr.tmp");
            std::fs::write(&tmp, &blob).expect("write row-cap checkpoint tmp");
            std::fs::rename(&tmp, &p).expect("rename row-cap checkpoint");
            eprintln!(
                "[ab_train] ROW_CAP {} reached at iter={} strat_rows={} bytes={} -> {}; wrote \
                 checkpoint and stopping cleanly (partial evidence, design §7.1.2)",
                row_cap,
                i,
                tab.n_strat_infosets(),
                blob.len(),
                p.display()
            );
            ckpt_extras(&tab, &cfg, i);
            std::process::exit(0);
        }
        // Heartbeat ~every 15s so a long run is visibly progressing (the user's logging rule).
        if last.elapsed().as_secs_f64() >= 15.0 {
            let done_this_run = (i - already) as f64;
            let frac = i as f64 / iters as f64;
            let rate = done_this_run / t0.elapsed().as_secs_f64();
            let eta_s = (iters - i) as f64 / rate.max(1e-9);
            eprintln!(
                "[ab_train] {i}/{iters} ({:.1}%) {:.2} iters/s eta {:.1}m strat_rows={}",
                frac * 100.0,
                rate,
                eta_s / 60.0,
                tab.n_strat_infosets()
            );
            last = std::time::Instant::now();
        }
    }
    let blob = serialize(&tab);
    std::fs::write(out_path, &blob).expect("write blob");
    // Final v3 extras (snapshot at the terminal iteration if it wasn't a fixed checkpoint, plus the
    // final coverage/regstats lines) — only when the run actually trained to `iters`.
    if tab.iters_done == iters && checkpoints.binary_search(&iters).is_err() {
        ckpt_extras(&tab, &cfg, iters);
    }
    eprintln!(
        "[ab_train] DONE precision={precision} iters={iters} strat_rows={} bytes={} in {:.1}s",
        tab.n_strat_infosets(),
        blob.len(),
        t0.elapsed().as_secs_f64()
    );
}
