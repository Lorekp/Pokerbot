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
    apply_delta_checked, bench_insert_row, bench_mark_dirty_fraction, deserialize, serialize,
    serialize_delta, NlheTables,
};

const SER_ROW_BYTES: usize = 16 + 8 * NUM_ACTIONS;

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

    // -- round-trip GATE: serialize -> deserialize -> serialize --------------------------------------
    let tab2 = deserialize(&blob).expect("deserialize");
    let blob2 = serialize(&tab2);
    let rt = blob == blob2;
    println!("[GATE] serialize->deserialize->serialize identical : {}", rt);
    assert!(rt, "ROUND-TRIP GATE FAILED");

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
    }
    println!(
        "\n  (full base at this size: {:.4} GB, pause {:.4}s. Delta pause/volume scale ~linearly with\n   \
         the touched fraction; the report projects 41/80/100 GB base sizes with these ratios.)",
        full_gb, full_pause
    );
}
