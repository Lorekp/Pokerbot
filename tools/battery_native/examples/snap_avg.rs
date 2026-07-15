//! Offline snapshot averager — blueprint-v4 v3 (Pluribus rounds-2-4 snapshot averaging,
//! P-supp p.15 / Jackson 2016; amendment doc artifacts/blueprint_v4_amendment_v3.md §3).
//!
//! Merges the NLSNAP01 current-strategy snapshots a POKERBOT_SNAPSHOT_AVG=1 ab_train run emitted
//! (snap_XXXXXX.nlsnap) with the FINAL checkpoint's strat section (the preflop round-1 in-memory
//! average) into the final servable blueprint:
//!
//!   * preflop rows (present in the checkpoint strat section) are taken from the checkpoint
//!     verbatim — the true weighted average, exactly what Pluribus serves for round 1;
//!   * every other row is the uniform mean of its non-zero snapshot rows;
//!   * rows with no positive mass in any snapshot are dropped (serve falls back on the miss).
//!
//! Output is a standard NLCFR001 blob with an EMPTY cum section and normalized-probability strat
//! rows — loadable by every existing consumer (extractors, probes, bpv4_nlcfr_index.py; the
//! row-normalization they apply is the identity on probabilities). Streaming k-way merge: O(1)
//! memory in the row count, so a 100 GB run averages on any box.
//!
//! Usage: snap_avg.exe <out.nlcfr> <final_checkpoint.nlcfr | -> <snap1.nlsnap> [snap2 ...]
//!   "-" for the checkpoint runs a pure snapshot average (no round-1 override — diagnostic only).

use battery_native::trainer::{average_snapshots, NlcfrStratStream, SnapshotStream};
use std::fs::File;
use std::io::{BufReader, BufWriter};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: snap_avg.exe <out.nlcfr> <final_checkpoint.nlcfr | -> <snap1.nlsnap> [snap2 ...]"
        );
        std::process::exit(2);
    }
    let out_path = &args[1];
    let ckpt_path = &args[2];
    let snap_paths = &args[3..];

    let mut snaps: Vec<SnapshotStream<BufReader<File>>> = Vec::with_capacity(snap_paths.len());
    let mut max_snap_iters = 0u64;
    for p in snap_paths {
        let f = File::open(p).unwrap_or_else(|e| panic!("open snapshot {p}: {e}"));
        let s = SnapshotStream::open(BufReader::new(f))
            .unwrap_or_else(|e| panic!("parse snapshot {p}: {e}"));
        eprintln!(
            "[snap_avg] snapshot {} iters={} rows={}",
            p, s.iters_done, s.n_rows
        );
        max_snap_iters = max_snap_iters.max(s.iters_done);
        snaps.push(s);
    }

    let mut ckpt: Option<NlcfrStratStream<BufReader<File>>> = if ckpt_path != "-" {
        let f =
            File::open(ckpt_path).unwrap_or_else(|e| panic!("open checkpoint {ckpt_path}: {e}"));
        let c = NlcfrStratStream::open(BufReader::new(f))
            .unwrap_or_else(|e| panic!("parse checkpoint {ckpt_path}: {e}"));
        eprintln!(
            "[snap_avg] checkpoint {} iters={} strat_rows={}",
            ckpt_path, c.iters_done, c.n_rows
        );
        Some(c)
    } else {
        eprintln!("[snap_avg] no checkpoint (pure snapshot average — diagnostic mode)");
        None
    };

    let iters_done = ckpt
        .as_ref()
        .map(|c| c.iters_done)
        .unwrap_or(max_snap_iters);
    let tmp = format!("{out_path}.tmp");
    let out_f = File::create(&tmp).unwrap_or_else(|e| panic!("create {tmp}: {e}"));
    let mut out = BufWriter::new(out_f);
    let t0 = std::time::Instant::now();
    let stats = average_snapshots(&mut snaps, ckpt.as_mut(), &mut out, iters_done)
        .unwrap_or_else(|e| panic!("average_snapshots: {e}"));
    drop(out);
    std::fs::rename(&tmp, out_path).unwrap_or_else(|e| panic!("rename {tmp} -> {out_path}: {e}"));
    eprintln!(
        "[snap_avg] DONE rows_out={} from_ckpt={} from_snaps={} zero_skipped={} iters_done={} \
         in {:.1}s -> {}",
        stats.rows_out,
        stats.rows_from_ckpt,
        stats.rows_from_snaps,
        stats.zero_rows_skipped,
        iters_done,
        t0.elapsed().as_secs_f64(),
        out_path,
    );
}
