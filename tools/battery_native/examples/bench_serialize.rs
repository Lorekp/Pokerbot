//! Checkpoint-serialize phase profiler + byte-identity gate (checkpoint-speed ops, 2026-07-06).
//!
//! Measures the per-checkpoint TRAINING PAUSE decomposition — the pause is `serialize()` (collect ->
//! par_sort -> parallel fixed-width encode) plus the in-Python PyBytes copy + local disk write. This
//! bench isolates the Rust `serialize` cost and its phases, reports MB/s, and PROVES byte-identity of the
//! parallel encode against an independent serial reference encoder for the same sorted input.
//!
//! Run (from crate dir, worktree-local target):
//!   RAYON_NUM_THREADS=8 cargo run --release --example bench_serialize -- <n_infosets> [reps]
//!
//! Reports serialize-rate (MB/s) at the achievable local size; project to 41 GB / 80 GB with the
//! measured rate (the pause is ~linear in blob size: sort is n log n but dominated by the memcpy-bound
//! encode at these widths).

use std::env;
use std::time::Instant;

use battery_native::core::NUM_ACTIONS;
use battery_native::trainer::{
    apply_delta_checked, bench_insert_row, bench_mark_dirty_fraction, deserialize,
    deserialize_parallel, serialize, serialize_delta, serialize_delta_to_file, serialize_to_file,
    NlheTables,
};

const SER_ROW_BYTES: usize = 16 + 8 * NUM_ACTIONS;

// ------------------------------------------------------------------------------------------------
// OLD-IMPL (pre checkpoint-speed R3) emulation — byte- and perf-faithful replica of the retired
// pipeline so one binary measures BEFORE and AFTER on the same box and PROVES sha equality:
//   gather   SERIAL per-key collect of OWNED (key, [f64;11]) pairs (~104 B/elem, Vec doubling growth)
//   sort     par_sort_unstable over those heavy elements
//   encode   parallel fixed-width scatter into ONE full in-RAM blob
//   write    16 MiB BufWriter stream + flush + UNCONDITIONAL sync_all (wait for device writeback)
// ------------------------------------------------------------------------------------------------

fn old_encode_map(out: &mut Vec<u8>, mut pairs: Vec<([u8; 16], [f64; NUM_ACTIONS])>) {
    use rayon::prelude::*;
    out.extend_from_slice(&(pairs.len() as u64).to_le_bytes());
    pairs.par_sort_unstable_by(|a, b| a.0.cmp(&b.0));
    let n = pairs.len();
    let base = out.len();
    out.resize(base + n * SER_ROW_BYTES, 0u8);
    let dst = &mut out[base..];
    let rows_per_chunk = n.div_ceil(rayon::current_num_threads().max(1) * 4).max(1);
    dst.par_chunks_mut(rows_per_chunk * SER_ROW_BYTES)
        .zip(pairs.par_chunks(rows_per_chunk))
        .for_each(|(byte_chunk, pair_chunk)| {
            for (i, (key, row)) in pair_chunk.iter().enumerate() {
                let off = i * SER_ROW_BYTES;
                byte_chunk[off..off + 16].copy_from_slice(&key[..]);
                let mut o = off + 16;
                for &v in row.iter() {
                    byte_chunk[o..o + 8].copy_from_slice(&v.to_le_bytes());
                    o += 8;
                }
            }
        });
}

/// OLD full-base blob: header + serial-gathered owned rows per map (dump_* is the serial
/// iterate-and-collect the old `write_map_sharded` did), old sort+encode.
fn old_impl_full_blob(tab: &NlheTables) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"NLCFR001");
    out.extend_from_slice(&tab.iters_done().to_le_bytes());
    old_encode_map(&mut out, tab.dump_cum_rows());
    old_encode_map(&mut out, tab.dump_strat_rows());
    out
}

/// OLD delta blob: 32-byte header + SERIAL per-dirty-key gather (HashSet iter + map.get random
/// probes, one thread) + old sort+encode — the exact shape of the retired `write_map_dirty`.
fn old_impl_delta_blob(tab: &NlheTables, base_iter: u64, seq: u64) -> Vec<u8> {
    let dirty = tab.dirty.as_deref().expect("delta tracking armed");
    let mut out = Vec::new();
    out.extend_from_slice(b"NLDLT001");
    out.extend_from_slice(&base_iter.to_le_bytes());
    out.extend_from_slice(&tab.iters_done().to_le_bytes());
    out.extend_from_slice(&seq.to_le_bytes());
    let cum: Vec<([u8; 16], [f64; NUM_ACTIONS])> = tab
        .shards
        .iter()
        .enumerate()
        .flat_map(|(s, sh)| {
            dirty[s].cum.iter().filter_map(move |k| {
                sh.cum_regret.get(k).map(|r| {
                    let mut row = [0.0f64; NUM_ACTIONS];
                    for a in 0..NUM_ACTIONS {
                        row[a] = r[a] as f64;
                    }
                    (*k, row)
                })
            })
        })
        .collect();
    let strat: Vec<([u8; 16], [f64; NUM_ACTIONS])> = tab
        .shards
        .iter()
        .enumerate()
        .flat_map(|(s, sh)| {
            dirty[s]
                .strat
                .iter()
                .filter_map(move |k| sh.strat_sum.get(k).map(|r| (*k, *r)))
        })
        .collect();
    old_encode_map(&mut out, cum);
    old_encode_map(&mut out, strat);
    out
}

/// OLD file write: 16 MiB BufWriter stream + flush + sync_all (the unconditional fsync the old
/// `serialize_to_file`/`serialize_delta_to_file` did inside the training pause).
fn old_impl_write(path: &str, blob: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let f = std::fs::File::create(path)?;
    let mut w = std::io::BufWriter::with_capacity(16 * 1024 * 1024, f);
    w.write_all(blob)?;
    w.flush()?;
    w.into_inner()
        .map_err(|e| std::io::Error::other(e.to_string()))?
        .sync_all()?;
    Ok(())
}

/// Independent SERIAL reference encoder: gather -> sort -> serial byte-by-byte (the pre-optimization
/// shape). Produces the reference NLCFR001 blob to prove the parallel `serialize` is byte-identical.
fn serialize_serial_reference(tab: &NlheTables) -> Vec<u8> {
    fn write_map_serial(out: &mut Vec<u8>, mut pairs: Vec<([u8; 16], [f64; NUM_ACTIONS])>) {
        out.extend_from_slice(&(pairs.len() as u64).to_le_bytes());
        pairs.sort_by(|a, b| a.0.cmp(&b.0));
        for (key, row) in pairs {
            out.extend_from_slice(&key[..]);
            for &v in &row {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    let mut out = Vec::new();
    out.extend_from_slice(b"NLCFR001");
    out.extend_from_slice(&tab.iters_done().to_le_bytes());
    write_map_serial(&mut out, tab.dump_cum_rows());
    write_map_serial(&mut out, tab.dump_strat_rows());
    out
}

fn sha256_hex(b: &[u8]) -> String {
    // Tiny self-contained SHA-256 (no crate dep) — only for the identity gate, not perf-sensitive.
    // FIPS 180-4.
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut msg = b.to_vec();
    let bitlen = (b.len() as u64) * 8;
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bitlen.to_be_bytes());
    for block in msg.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ ((!v[4]) & v[6]);
            let t1 = v[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [
                t1.wrapping_add(t2),
                v[0],
                v[1],
                v[2],
                v[3].wrapping_add(t1),
                v[4],
                v[5],
                v[6],
            ];
        }
        for i in 0..8 {
            h[i] = h[i].wrapping_add(v[i]);
        }
    }
    let mut s = String::with_capacity(64);
    for x in h {
        s.push_str(&format!("{:08x}", x));
    }
    s
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(4_000_000);
    let reps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);

    println!("== SERIALIZE PHASE PROFILE ==");
    println!("rayon threads = {}", rayon::current_num_threads());
    println!("n_infosets    = {}", n);

    // Synthesize a realistically sharded table: each infoset gets a cum_regret AND a strat_sum row,
    // routed to its shard by the real `shard_of` (via `bench_insert_row`).
    let build0 = Instant::now();
    let mut tab = NlheTables::new();
    for i in 0..n {
        let mut key = [0u8; 16];
        let a = (i as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ 0xDEADBEEFCAFEBABE;
        let b = (i as u64).wrapping_mul(0x123456789ABCDEF0) ^ 0x1357911131517191;
        key[..8].copy_from_slice(&a.to_le_bytes());
        key[8..].copy_from_slice(&b.to_le_bytes());
        let mut cum = [0.0f64; NUM_ACTIONS];
        let mut strat = [0.0f64; NUM_ACTIONS];
        for k in 0..NUM_ACTIONS {
            cum[k] = (i as f64) * 1e-6 + (k as f64);
            strat[k] = 0.1 + (k as f64) * 0.01;
        }
        bench_insert_row(&mut tab, key, cum, strat);
    }
    tab.set_iters_done(62_000);
    println!("built table in {:.2}s", build0.elapsed().as_secs_f64());

    // -- byte-identity GATE: parallel serialize vs independent serial reference --------------------
    let blob = serialize(&tab);
    let ref_blob = serialize_serial_reference(&tab);
    let ident = blob == ref_blob;
    println!("\n[GATE] parallel serialize == serial reference : {}", ident);
    println!("[GATE] parallel blob sha256   = {}", sha256_hex(&blob));
    println!("[GATE] serial   blob sha256   = {}", sha256_hex(&ref_blob));
    assert!(ident, "BYTE-IDENTITY GATE FAILED: parallel encode diverged from serial reference");
    drop(ref_blob); // free the 2nd full blob before the memory-heavier phases (large-n RAM headroom)

    // -- round-trip GATE: serialize -> deserialize -> serialize --------------------------------------
    let tab2 = deserialize(&blob).expect("deserialize");
    let blob2 = serialize(&tab2);
    let rt = blob == blob2;
    println!("[GATE] serialize->deserialize->serialize identical : {}", rt);
    assert!(rt, "ROUND-TRIP GATE FAILED");
    drop(tab2);
    drop(blob2);

    let blob_len = blob.len();
    println!(
        "\nblob = {} bytes ({:.3} GB), rows encoded = {}",
        blob_len,
        blob_len as f64 / 1e9,
        blob_len / SER_ROW_BYTES
    );

    // -- timing: full serialize (best-of reps) -------------------------------------------------------
    let mut best = f64::INFINITY;
    let mut sum = 0.0;
    for _ in 0..reps {
        let t0 = Instant::now();
        let out = serialize(&tab);
        let dt = t0.elapsed().as_secs_f64();
        std::hint::black_box(&out);
        best = best.min(dt);
        sum += dt;
    }
    // -- phase breakdown (collect / par_sort / parallel-encode vs serial-encode) on the cum map -----
    {
        use rayon::prelude::*;
        let cum = tab.dump_cum_rows(); // owned (key,row)
        // collect phase (already have owned rows; time a representative clone+ref build)
        let t_c = Instant::now();
        let mut pairs: Vec<(&[u8; 16], &[f64; NUM_ACTIONS])> =
            cum.iter().map(|(k, r)| (k, r)).collect();
        let collect_dt = t_c.elapsed().as_secs_f64();
        let t_s = Instant::now();
        pairs.par_sort_unstable_by(|a, b| a.0.cmp(b.0));
        let sort_dt = t_s.elapsed().as_secs_f64();
        // parallel encode
        let nb = pairs.len() * SER_ROW_BYTES;
        let mut buf = vec![0u8; nb];
        let t_pe = Instant::now();
        let rpc = pairs.len().div_ceil(rayon::current_num_threads().max(1) * 4).max(1);
        buf.par_chunks_mut(rpc * SER_ROW_BYTES)
            .zip(pairs.par_chunks(rpc))
            .for_each(|(bc, pc)| {
                for (i, (k, r)) in pc.iter().enumerate() {
                    let o = i * SER_ROW_BYTES;
                    bc[o..o + 16].copy_from_slice(&k[..]);
                    let mut q = o + 16;
                    for &v in r.iter() {
                        bc[q..q + 8].copy_from_slice(&v.to_le_bytes());
                        q += 8;
                    }
                }
            });
        let penc_dt = t_pe.elapsed().as_secs_f64();
        std::hint::black_box(&buf);
        // serial encode
        let mut buf2 = Vec::with_capacity(nb);
        let t_se = Instant::now();
        for (k, r) in &pairs {
            buf2.extend_from_slice(&k[..]);
            for &v in r.iter() {
                buf2.extend_from_slice(&v.to_le_bytes());
            }
        }
        let senc_dt = t_se.elapsed().as_secs_f64();
        std::hint::black_box(&buf2);
        let cum_gb = nb as f64 / 1e9;
        println!("\n-- PHASE BREAKDOWN (cum map only, {:.3} GB) --", cum_gb);
        println!("  collect refs   : {:.4}s", collect_dt);
        println!("  par_sort       : {:.4}s  ({:.1} MB/s)", sort_dt, nb as f64 / 1e6 / sort_dt);
        println!("  encode PARALLEL: {:.4}s  ({:.1} MB/s)", penc_dt, nb as f64 / 1e6 / penc_dt);
        println!("  encode SERIAL  : {:.4}s  ({:.1} MB/s)", senc_dt, nb as f64 / 1e6 / senc_dt);
        println!("  encode speedup : {:.2}x", senc_dt / penc_dt);
    }

    let avg = sum / reps as f64;
    let mbps_best = blob_len as f64 / 1e6 / best;
    let mbps_avg = blob_len as f64 / 1e6 / avg;
    println!("\n-- serialize (parallel) --");
    println!("best {:.4}s  avg {:.4}s  ({} reps)", best, avg, reps);
    println!("rate best {:.1} MB/s   avg {:.1} MB/s", mbps_best, mbps_avg);

    // -- timing: serial reference for comparison -----------------------------------------------------
    let mut best_ser = f64::INFINITY;
    for _ in 0..reps {
        let t0 = Instant::now();
        let out = serialize_serial_reference(&tab);
        let dt = t0.elapsed().as_secs_f64();
        std::hint::black_box(&out);
        best_ser = best_ser.min(dt);
    }
    let mbps_ser = blob_len as f64 / 1e6 / best_ser;
    println!("\n-- serialize (serial reference: gather+sort_by+serial-encode) --");
    println!("best {:.4}s   rate {:.1} MB/s", best_ser, mbps_ser);
    println!("\nSPEEDUP (serial -> parallel) = {:.2}x", best_ser / best);

    // -- projection to production blob sizes ---------------------------------------------------------
    println!("\n== PROJECTION (at parallel best rate {:.1} MB/s) ==", mbps_best);
    for gb in [41.0f64, 58.0, 80.0] {
        let secs = gb * 1e9 / (mbps_best * 1e6);
        println!("  {:>4.0} GB blob : pause ~{:.1}s", gb, secs);
    }
    println!("\n== PROJECTION (serial reference rate {:.1} MB/s) ==", mbps_ser);
    for gb in [41.0f64, 58.0, 80.0] {
        let secs = gb * 1e9 / (mbps_ser * 1e6);
        println!("  {:>4.0} GB blob : pause ~{:.1}s", gb, secs);
    }

    // ================================================================================================
    // FULL-BASE TO-FILE (the actual `save_bytes_to_file` training pause) — OLD impl emulation
    // (serial gather + heavy sort + full blob + BufWriter + sync_all) vs NEW streaming writer
    // (parallel gather + 24B-ref sort + parallel positioned writes, fsync opt-in). SHA gate: both
    // must produce byte-identical files (and == serialize()).
    // ================================================================================================
    println!("\n\n== FULL-BASE TO-FILE (OLD impl vs NEW streaming writer) ==");
    let file_dir = env::var("BENCH_FILE_DIR").map(std::path::PathBuf::from).unwrap_or_else(|_| env::temp_dir());
    println!("file dir = {}", file_dir.display());
    let p_old = file_dir.join("bench_ser_old.nlcfr");
    let p_new = file_dir.join("bench_ser_new.nlcfr");

    // OLD: encode phases timed separately from the write+fsync.
    let t0 = Instant::now();
    let old_blob = old_impl_full_blob(&tab);
    let old_enc_dt = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    old_impl_write(p_old.to_str().unwrap(), &old_blob).expect("old write");
    let old_wr_dt = t0.elapsed().as_secs_f64();
    let old_total = old_enc_dt + old_wr_dt;
    assert_eq!(old_blob, blob, "[GATE] OLD-impl blob != current serialize() blob");
    println!(
        "  OLD  encode {:.3}s + write+fsync {:.3}s = {:.3}s  ({:.1} MB/s end-to-end)",
        old_enc_dt, old_wr_dt, old_total, blob_len as f64 / 1e6 / old_total
    );
    drop(old_blob);

    // NEW (default: no fsync — page-cache write; kernel writes back while training resumes).
    let mut best_new = f64::INFINITY;
    for _ in 0..2 {
        let t0 = Instant::now();
        let nb = serialize_to_file(&tab, p_new.to_str().unwrap()).expect("serialize_to_file");
        best_new = best_new.min(t0.elapsed().as_secs_f64());
        assert_eq!(nb, blob_len, "byte count mismatch");
    }
    println!(
        "  NEW  (no fsync)  {:.3}s  ({:.1} MB/s end-to-end)",
        best_new, blob_len as f64 / 1e6 / best_new
    );
    // NEW with POKERBOT_CKPT_FSYNC=1 (restores the old durability wait).
    env::set_var("POKERBOT_CKPT_FSYNC", "1");
    let t0 = Instant::now();
    serialize_to_file(&tab, p_new.to_str().unwrap()).expect("serialize_to_file fsync");
    let new_fsync_dt = t0.elapsed().as_secs_f64();
    env::remove_var("POKERBOT_CKPT_FSYNC");
    println!(
        "  NEW  (fsync=1)   {:.3}s  ({:.1} MB/s end-to-end)",
        new_fsync_dt, blob_len as f64 / 1e6 / new_fsync_dt
    );
    let old_bytes = std::fs::read(&p_old).expect("read old file");
    let new_bytes = std::fs::read(&p_new).expect("read new file");
    println!("  [GATE] old-impl file sha256 = {}", sha256_hex(&old_bytes));
    println!("  [GATE] new-impl file sha256 = {}", sha256_hex(&new_bytes));
    assert_eq!(old_bytes, new_bytes, "[GATE] FULL-BASE FILE BYTES DIVERGED old vs new");
    assert_eq!(new_bytes, blob, "[GATE] new file != serialize() blob");
    println!("  [GATE] full-base file bytes identical (old == new == serialize()) : true");
    println!("  SPEEDUP (old -> new, no-fsync default) = {:.2}x", old_total / best_new);
    drop(old_bytes);
    drop(new_bytes);
    let _ = std::fs::remove_file(&p_old);
    let _ = std::fs::remove_file(&p_new);

    // -- DELTA CHECKPOINTS (checkpoint-speed R2, 2026-07-06) ------------------------------------------
    // Measure delta-serialize rate + pause at several touched fractions, and PROVE base+delta resume is
    // byte-identical to a full re-serialize of the same state. The dirty set is 1-in-`every` rows.
    //
    // Production touched fraction: the fullhist table grows ~42k rows/iter (memory note); at a 250-iter
    // cadence a window touches on the order of (250 * touched_per_iter) rows. We sweep a range of
    // fractions and report the per-fraction delta pause + projected S3 upload volume so the report can
    // pick the operating point.
    println!("\n\n== DELTA CHECKPOINT PROFILE ==");
    // Full-base serialize timing (reuse `best` from above as the base pause at this local size).
    let full_pause = best;
    let full_gb = blob_len as f64 / 1e9;
    for &every in &[100usize, 50, 20, 10, 5] {
        // Fresh table so the dirty fraction is clean (arm + mark).
        let mut dt = NlheTables::new();
        for i in 0..n {
            let mut key = [0u8; 16];
            let a = (i as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ 0xDEADBEEFCAFEBABE;
            let b = (i as u64).wrapping_mul(0x123456789ABCDEF0) ^ 0x1357911131517191;
            key[..8].copy_from_slice(&a.to_le_bytes());
            key[8..].copy_from_slice(&b.to_le_bytes());
            let mut cum = [0.0f64; NUM_ACTIONS];
            let mut strat = [0.0f64; NUM_ACTIONS];
            for k in 0..NUM_ACTIONS {
                cum[k] = (i as f64) * 1e-6 + (k as f64);
                strat[k] = 0.1 + (k as f64) * 0.01;
            }
            bench_insert_row(&mut dt, key, cum, strat);
        }
        dt.set_iters_done(62_250);
        let (dc, ds) = bench_mark_dirty_fraction(&mut dt, every);
        let frac = (dc + ds) as f64 / (2 * n) as f64;

        // Delta serialize timing (best-of reps).
        let mut best_d = f64::INFINITY;
        let mut dblob = Vec::new();
        for _ in 0..reps {
            let t0 = Instant::now();
            let out = serialize_delta(&dt, 62_000, 1).expect("armed");
            best_d = best_d.min(t0.elapsed().as_secs_f64());
            dblob = out;
        }
        let dgb = dblob.len() as f64 / 1e9;
        let dmbps = dblob.len() as f64 / 1e6 / best_d;

        // BYTE-IDENTITY GATE: a fresh "base" table (dirty rows perturbed away, then delta re-applied)
        // must re-serialize identically to `dt`. Simplest faithful check: base = dt serialized as
        // NLCFR001 at base_iter; deserialize; apply the delta; compare to serialize(dt).
        // (Here base == full dt, so base+delta trivially == dt; the crate unit test covers the
        // divergent-base case. We still assert to catch a format regression.)
        let want_dt = serialize(&dt);
        let mut base_tab = deserialize(&want_dt).unwrap();
        base_tab.set_iters_done(62_000);
        let applied = apply_delta_checked(&mut base_tab, &dblob, 62_000, 0);
        assert!(applied.is_some(), "delta chain-verify failed at every={every}");
        assert_eq!(
            serialize(&base_tab),
            want_dt,
            "[GATE] base+delta != full re-serialize at every={every}"
        );

        let speedup_vs_full = full_pause / best_d;
        let vol_ratio = dgb / full_gb;
        println!(
            "  every={:>3} (touched {:>5.2}%): dirty=({},{}) delta_blob={:.4} GB  \
             pause={:.4}s ({:.0} MB/s)  vs full: {:.1}x faster, {:.1}% of upload volume  [GATE ok]",
            every,
            frac * 100.0,
            dc,
            ds,
            dgb,
            best_d,
            dmbps,
            speedup_vs_full,
            vol_ratio * 100.0
        );

        // -- DELTA TO-FILE (the actual `save_delta_to_file` training pause): OLD impl emulation
        //    (serial dirty gather + heavy sort + blob + BufWriter + sync_all) vs NEW streaming writer.
        //    SHA gate: identical file bytes.
        let dp_old = file_dir.join(format!("bench_delta_old_{every}.nldlt"));
        let dp_new = file_dir.join(format!("bench_delta_new_{every}.nldlt"));
        let t0 = Instant::now();
        let old_dblob = old_impl_delta_blob(&dt, 62_000, 1);
        let old_enc = t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        old_impl_write(dp_old.to_str().unwrap(), &old_dblob).expect("old delta write");
        let old_wr = t0.elapsed().as_secs_f64();
        assert_eq!(old_dblob, dblob, "[GATE] OLD delta blob != current serialize_delta blob");
        let mut best_nf = f64::INFINITY;
        for _ in 0..2 {
            let t0 = Instant::now();
            let nb = serialize_delta_to_file(&dt, dp_new.to_str().unwrap(), 62_000, 1)
                .expect("delta_to_file")
                .expect("armed");
            best_nf = best_nf.min(t0.elapsed().as_secs_f64());
            assert_eq!(nb, dblob.len());
        }
        let old_file_bytes = std::fs::read(&dp_old).unwrap();
        let new_file_bytes = std::fs::read(&dp_new).unwrap();
        assert_eq!(old_file_bytes, new_file_bytes, "[GATE] DELTA FILE BYTES DIVERGED old vs new");
        assert_eq!(new_file_bytes, dblob, "[GATE] delta file != serialize_delta blob");
        let _ = std::fs::remove_file(&dp_old);
        let _ = std::fs::remove_file(&dp_new);
        println!(
            "        TO-FILE: OLD encode {:.3}s + write+fsync {:.3}s = {:.3}s ({:.0} MB/s)  \
             NEW {:.3}s ({:.0} MB/s)  speedup {:.1}x  [file-sha GATE ok]",
            old_enc,
            old_wr,
            old_enc + old_wr,
            dblob.len() as f64 / 1e6 / (old_enc + old_wr),
            best_nf,
            dblob.len() as f64 / 1e6 / best_nf,
            (old_enc + old_wr) / best_nf
        );
    }
    println!(
        "\n  (full base at this size: {:.4} GB, pause {:.4}s. Delta pause/volume scale ~linearly with\n   \
         the touched fraction; the report projects 41/80/100 GB base sizes with these ratios.)",
        full_gb, full_pause
    );
    // ================================================================================================
    // LOAD (deserialize) phase — the RESUME PROLOGUE cost. This is the round-3 target: the serial
    // `deserialize` routes+inserts one row at a time into rehashing shard maps; `deserialize_parallel`
    // hash-partitions in parallel and pre-sizes each shard map. Same blob => both must round-trip to a
    // re-serialize-identical table (byte-equivalence gate), and we report GB/s + rows/s for each.
    // ================================================================================================
    println!("\n\n== DESERIALIZE (RESUME LOAD) PHASE ==");
    let n_rows_total = blob_len / SER_ROW_BYTES; // cum + strat rows
    let gb = blob_len as f64 / 1e9;
    drop(tab); // free the source table — the load phase only needs `blob` (large-n RAM headroom)

    // Byte-equivalence GATE: serial-load and parallel-load re-serialize to the SAME blob. Done ONE table
    // at a time (serialize each, drop it) to keep peak RAM at ~2 blobs, not 2 tables + 2 blobs.
    let ser_blob = {
        let t_ld = deserialize(&blob).expect("serial deserialize");
        serialize(&t_ld)
    };
    let par_blob = {
        let t_lp = deserialize_parallel(&blob).expect("parallel deserialize");
        serialize(&t_lp)
    };
    let load_ident = ser_blob == par_blob && ser_blob == blob;
    println!("[GATE] serial-load re-serialize == original          : {}", ser_blob == blob);
    println!("[GATE] parallel-load re-serialize == serial-load     : {}", ser_blob == par_blob);
    println!("[GATE] parallel-load re-serialize == original        : {}", par_blob == blob);
    assert!(load_ident, "LOAD BYTE-EQUIVALENCE GATE FAILED");
    drop(ser_blob);
    drop(par_blob);

    // -- timing: serial deserialize (best-of reps) --
    let mut best_ld = f64::INFINITY;
    for _ in 0..reps {
        let t0 = Instant::now();
        let tt = deserialize(&blob).expect("serial deserialize");
        let dt = t0.elapsed().as_secs_f64();
        std::hint::black_box(&tt);
        best_ld = best_ld.min(dt);
    }
    // -- timing: parallel deserialize (best-of reps) --
    let mut best_lp = f64::INFINITY;
    for _ in 0..reps {
        let t0 = Instant::now();
        let tt = deserialize_parallel(&blob).expect("parallel deserialize");
        let dt = t0.elapsed().as_secs_f64();
        std::hint::black_box(&tt);
        best_lp = best_lp.min(dt);
    }
    let gbps_ld = gb / best_ld;
    let gbps_lp = gb / best_lp;
    let rps_ld = n_rows_total as f64 / best_ld;
    let rps_lp = n_rows_total as f64 / best_lp;
    println!("\nload rows (cum+strat) = {}", n_rows_total);
    println!("-- deserialize SERIAL   : {:.4}s  {:.3} GB/s  {:.2}e6 rows/s", best_ld, gbps_ld, rps_ld / 1e6);
    println!("-- deserialize PARALLEL : {:.4}s  {:.3} GB/s  {:.2}e6 rows/s", best_lp, gbps_lp, rps_lp / 1e6);
    println!("LOAD SPEEDUP (serial -> parallel) = {:.2}x", best_ld / best_lp);

    println!("\n== LOAD PROJECTION (serial {:.3} GB/s | parallel {:.3} GB/s) ==", gbps_ld, gbps_lp);
    for pgb in [41.0f64, 55.0, 58.0, 80.0, 100.0] {
        let s_ld = pgb / gbps_ld;
        let s_lp = pgb / gbps_lp;
        println!("  {:>4.0} GB blob : serial ~{:.1}s ({:.1} min)  parallel ~{:.1}s ({:.1} min)",
                 pgb, s_ld, s_ld / 60.0, s_lp, s_lp / 60.0);
    }
}
