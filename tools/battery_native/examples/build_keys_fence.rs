//! OFFLINE builder for the `keys_fence.bin` sidecar (OPT-BP1 sparse fence index — see
//! `playout.rs::build_keys_fence`). ONE sequential scan of `keys_sorted.bin`, recording the first
//! 16B key of every 4KB page and verifying sort order; writes `<dir>/keys_fence.bin` via a `.tmp` +
//! rename. The serve loads the sidecar automatically at `SortedIndex::open` and FAILS OPEN to the
//! plain binary search when it is absent/invalid, so building it is a pure additive deploy step.
//!
//! Tier-1 index (538.85M keys / 8.6GB): sequential read, ~34MB sidecar, roughly the drive's
//! sequential-read time (tens of seconds on NVMe). It cycles ~8.6GB of page cache — run in a
//! MAINTENANCE WINDOW, never next to the live bot. Progress lines every ~2s.
//!
//! Usage: build_keys_fence --index-dir DIR

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut index_dir = String::new();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--index-dir" => {
                index_dir = args[i + 1].clone();
                i += 2;
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(2);
            }
        }
    }
    if index_dir.is_empty() {
        eprintln!("usage: build_keys_fence --index-dir DIR");
        std::process::exit(2);
    }
    match battery_native::playout::build_keys_fence(&index_dir, true) {
        Ok((n_keys, n_fence)) => {
            println!("JSON\t{{\"mode\":\"build_fence\",\"index_dir\":\"{}\",\"n_keys\":{n_keys},\"n_fence\":{n_fence}}}",
                index_dir.replace('\\', "\\\\"));
        }
        Err(e) => {
            eprintln!("[build_keys_fence] FAILED: {e}");
            std::process::exit(1);
        }
    }
}
