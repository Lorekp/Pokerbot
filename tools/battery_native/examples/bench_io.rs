//! Bench the 3 IO paths the bottleneck hypothesis names:
//!   1. Snapshot save (NlheTables -> bytes -> disk)
//!   2. Snapshot load (disk -> bytes -> NlheTables)
//!   3. Centroid load (EMDFRZN1 blob used by FrozenEMDLookup at runtime)
//!
//! All numbers reported are wall-clock best-of-N. Throughput = bytes / seconds.
//!
//! Run: `cargo run --release --example bench_io -- <emd_json_path> <out_dir> <n_infosets>`

use std::env;
use std::fs;
use std::time::Instant;

use battery_native::core::NUM_ACTIONS;
use battery_native::emd::FrozenEMDLookup;
use battery_native::trainer::{bench_insert_row, serialize, NlheTables};

fn main() {
    let args: Vec<String> = env::args().collect();
    let emd_json = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "data/abstraction/emd_buckets_500.json".to_string());
    let out_dir = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "/tmp/io_bench".to_string());
    let n_infosets: usize = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3_500_000);
    fs::create_dir_all(&out_dir).expect("mkdir out_dir");

    println!("== IO PATH PROFILE ==");
    println!("emd_json = {}", emd_json);
    println!("out_dir  = {}", out_dir);
    println!("n_infosets for snapshot synth = {}", n_infosets);

    // -----------------------------------------------------------------------------------
    // Build a representative NlheTables. The relevant parameters are:
    //   - infoset count (drives serialized size)
    //   - deterministic so the bytes match across runs.
    // We target ~3.5M infosets (~700 MB blob) which matches the deployed iter62k snapshot
    // size reported in prior memory.
    // -----------------------------------------------------------------------------------
    println!("\n-- synth NlheTables with {} infosets --", n_infosets);
    let mut tab = NlheTables::new();
    for i in 0..n_infosets {
        let mut key = [0u8; 16];
        let v = (i as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ 0xDEADBEEFCAFEBABE;
        key[..8].copy_from_slice(&v.to_le_bytes());
        key[8..16].copy_from_slice(&((i as u64).wrapping_mul(0x123456789ABCDEF0)).to_le_bytes());
        let mut row = [0.0f64; NUM_ACTIONS];
        for a in 0..NUM_ACTIONS {
            row[a] = (i as f64 * 1e-9) + (a as f64);
        }
        let mut srow = [0.0f64; NUM_ACTIONS];
        for a in 0..NUM_ACTIONS {
            srow[a] = 0.1 + (a as f64) * 0.01;
        }
        // Route the (cum_regret, strat_sum) pair to the shard the trainer would place it in.
        bench_insert_row(&mut tab, key, row, srow);
    }
    tab.iters_done = 62_000;

    // --- Snapshot SAVE -------------------------------------------------------------------
    let blob = serialize(&tab);
    let blob_len = blob.len();
    println!(
        "\n[1] snapshot SAVE: in-memory blob = {} bytes ({:.2} MB)",
        blob_len,
        blob_len as f64 / 1e6
    );

    let save_ser_dt = bench("serialize only", 5, || -> f64 {
        let t0 = Instant::now();
        let _ = serialize(&tab);
        t0.elapsed().as_secs_f64()
    });
    println!(
        "    serialize only (no I/O)   best={:.3} ms   throughput={:.1} MB/s",
        save_ser_dt * 1e3,
        (blob_len as f64 / 1e6) / save_ser_dt
    );

    // Use a unique path per iteration to avoid Windows file-mapping lock (1224).
    let write_dt = bench("write to disk", 5, || -> f64 {
        let t0 = Instant::now();
        let p = format!("{}/snap_{}.bin", out_dir, std::process::id());
        fs::write(&p, &blob).expect("write snap");
        // Drop the file-mapping window immediately so the next iteration can re-write.
        let _ = fs::remove_file(&p);
        t0.elapsed().as_secs_f64()
    });
    println!(
        "    fs::write only            best={:.3} ms   throughput={:.1} MB/s",
        write_dt * 1e3,
        (blob_len as f64 / 1e6) / write_dt
    );

    let save_total = bench("save end-to-end (ser+write)", 5, || -> f64 {
        let t0 = Instant::now();
        let b = serialize(&tab);
        let p = format!("{}/snap_{}.bin", out_dir, std::process::id());
        fs::write(&p, &b).expect("write");
        let _ = fs::remove_file(&p);
        t0.elapsed().as_secs_f64()
    });
    println!(
        "    save TOTAL (ser+write)    best={:.3} ms   throughput={:.1} MB/s",
        save_total * 1e3,
        (blob_len as f64 / 1e6) / save_total
    );

    // --- Snapshot LOAD -------------------------------------------------------------------
    // Persist one fixed snapshot for the load bench.
    let stable_snap = format!("{}/stable_snap.bin", out_dir);
    fs::write(&stable_snap, &blob).expect("stable write");
    println!("\n[2] snapshot LOAD: read {} bytes", blob_len);
    let preloaded = fs::read(&stable_snap).expect("read snap");

    let read_dt = bench("read from disk", 5, || -> f64 {
        let t0 = Instant::now();
        let _ = fs::read(&stable_snap).expect("read");
        t0.elapsed().as_secs_f64()
    });
    println!(
        "    fs::read only             best={:.3} ms   throughput={:.1} MB/s",
        read_dt * 1e3,
        (blob_len as f64 / 1e6) / read_dt
    );

    let deserialize_dt = bench("deserialize only", 5, || -> f64 {
        let t0 = Instant::now();
        let _ = battery_native::trainer::deserialize(&preloaded);
        t0.elapsed().as_secs_f64()
    });
    println!(
        "    deserialize only          best={:.3} ms   throughput={:.1} MB/s",
        deserialize_dt * 1e3,
        (blob_len as f64 / 1e6) / deserialize_dt
    );

    let load_total = bench("load end-to-end (read+de)", 5, || -> f64 {
        let t0 = Instant::now();
        let b = fs::read(&stable_snap).expect("read");
        let _ = battery_native::trainer::deserialize(&b);
        t0.elapsed().as_secs_f64()
    });
    println!(
        "    load TOTAL (read+de)      best={:.3} ms   throughput={:.1} MB/s",
        load_total * 1e3,
        (blob_len as f64 / 1e6) / load_total
    );
    let _ = fs::remove_file(&stable_snap);

    // --- Centroid load (EMDFRZN blob in native) ------------------------------------------
    println!("\n[3] centroid LOAD");

    // 3a. EMDFRZN1 binary path (the production lookup)
    let emdfzn_path = format!("{}/fake_emdfzn.bin", out_dir);
    let n_emd_spots: usize = 1_100_000;
    write_fake_emdfzn(&emdfzn_path, n_emd_spots);
    let emdfzn_meta = fs::metadata(&emdfzn_path).expect("emdfzn meta");
    let emdfzn_size = emdfzn_meta.len() as usize;
    println!(
        "\n[3a] EMDFRZN1 binary (synth, {} spots, {} bytes = {:.2} MB)",
        n_emd_spots,
        emdfzn_size,
        emdfzn_size as f64 / 1e6
    );

    let emd_read_dt = bench("read EMDFRZN", 5, || -> f64 {
        let t0 = Instant::now();
        let _ = fs::read(&emdfzn_path).expect("read");
        t0.elapsed().as_secs_f64()
    });
    println!(
        "    fs::read only             best={:.3} ms   throughput={:.1} MB/s",
        emd_read_dt * 1e3,
        (emdfzn_size as f64 / 1e6) / emd_read_dt
    );

    // NOTE: the former `FrozenEMDLookup::from_bytes` (parse-only, in-memory) was removed — parsing is
    // now inlined into `load_path` (read+parse), so the parse-only step can no longer be isolated via
    // the public API. Report read-only + end-to-end (load_path); parse ≈ end-to-end − read.
    let emd_total_dt = bench("EMDFRZN load end-to-end", 5, || -> f64 {
        let t0 = Instant::now();
        let _ = FrozenEMDLookup::load_path(&emdfzn_path).expect("load");
        t0.elapsed().as_secs_f64()
    });
    println!(
        "    EMDFRZN load TOTAL        best={:.3} ms   throughput={:.1} MB/s",
        emd_total_dt * 1e3,
        (emdfzn_size as f64 / 1e6) / emd_total_dt
    );
    let _ = fs::remove_file(&emdfzn_path);

    // 3b. The actual emd_buckets_500.json (centroid file used by Python fit step)
    println!("\n[3b] emd_buckets_500.json (Python centroid file, IF present)");
    let emd_json_path = std::path::PathBuf::from(&emd_json);
    if emd_json_path.exists() {
        let md = fs::metadata(&emd_json_path).expect("meta");
        let sz = md.len() as usize;
        println!("    file size = {} bytes ({:.2} MB)", sz, sz as f64 / 1e6);
        let jread_dt = bench("read json", 5, || -> f64 {
            let t0 = Instant::now();
            let _ = fs::read(&emd_json_path).expect("read");
            t0.elapsed().as_secs_f64()
        });
        println!(
            "    fs::read best={:.3} ms   throughput={:.1} MB/s",
            jread_dt * 1e3,
            (sz as f64 / 1e6) / jread_dt
        );
    } else {
        println!("    (file not present: {})", emd_json_path.display());
    }

    // --- Pipeline context ------
    println!("\n== PIPELINE CONTEXT ==");
    let train_iters_per_sec = 1.0_f64;
    println!(
        "(for context) trainer target ~{} it/s  ==> one iter ~{} ms",
        train_iters_per_sec,
        1000.0 / train_iters_per_sec
    );

    let snapshot_at: usize = std::env::var("SNAPSHOT_EVERY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2_000);
    println!(
        "snapshot every {} iters  ==> save cost amortized to ~{:.4} ms/iter",
        snapshot_at,
        (save_total * 1e3) / snapshot_at as f64
    );
    println!(
        "load cost amortized over {}-iter resume = ~{:.4} ms/iter",
        snapshot_at,
        (load_total * 1e3) / snapshot_at as f64
    );
    println!(
        "EMDFRZN load amortized over a full training run ({} iters) = ~{:.6} ms/iter",
        snapshot_at,
        (emd_total_dt * 1e3) / snapshot_at as f64
    );

    println!("\nverdict: see the per-path summary printed above.");
}

/// Run the closure N times and return the BEST wall-clock (s).
fn bench<F: FnMut() -> f64>(_label: &str, n: usize, mut f: F) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..n {
        let dt = f();
        if dt < best {
            best = dt;
        }
    }
    best
}

/// Write a synthetic EMDFRZN1 blob with `n_spots` entries (8-byte key + 2-byte bucket).
fn write_fake_emdfzn(path: &str, n_spots: usize) {
    let mut blob: Vec<u8> = Vec::with_capacity(28 + n_spots * 10);
    blob.extend_from_slice(b"EMDFRZN1");
    blob.extend_from_slice(&500u32.to_le_bytes());
    blob.extend_from_slice(&20u32.to_le_bytes());
    blob.extend_from_slice(&200u32.to_le_bytes());
    blob.extend_from_slice(&(n_spots as u64).to_le_bytes());
    for i in 0..n_spots {
        let k = (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
        let b = (i % 500) as u16;
        blob.extend_from_slice(&k.to_le_bytes());
        blob.extend_from_slice(&b.to_le_bytes());
    }
    fs::write(path, &blob).expect("write emdfzn");
}
