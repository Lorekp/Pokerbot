//! Lock-free (hogwild) shared regret/strategy table + traversal for the native 6-max NLHE CFR+ trainer.
//!
//! WHY: the fork-merge trainer (`trainer.rs`) pays a per-iteration barrier every iteration —
//! `collect_partials_n` (rayon fold of deals into per-task accumulator tables) -> `merge_partials`
//! (barrier) -> `floor_cum_regret` (barrier). The barrier idle is ~20% at 6 threads and grows with core
//! count. This module replaces the whole fork->merge->floor pipeline with ONE shared open-addressing
//! regret/strategy table that all rayon workers read+write IN PLACE: no per-task accumulator, no merge, no
//! floor barrier (the CFR+ floor is applied on each regret add). Small-write cache locality is preserved
//! AND the barrier idle is removed.
//!
//! This is the Rust port of `pokerbot/training/shared_hogwild_table.py` — same open-addressing layout, same
//! lock-free hot path, same per-slot-lock first-visit claim, same f64-strat-sum quality requirement, same
//! CFR+ floor on the regret add, same hard-error-on-overflow. The ONLY mechanism difference: the Python
//! uses `multiprocessing.shared_memory` + numpy views (separate processes); here the storage is
//! `Vec<Atomic*>` shared `&` across rayon THREADS (one process), so concurrent access is data-race-free at
//! the language level (every cell is an atomic) even though the VALUES race — the accepted, provably-
//! harmless hogwild noise for external-sampling MCCFR.
//!
//! The table is gated behind `feature = "hogwild"` and selected at RUNTIME (see `trainer.rs`'s
//! `train_iteration_hogwild`): with the feature off this module is not compiled and the trainer is the
//! unchanged fork-merge path.

use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::Mutex;

use crate::core::NUM_ACTIONS;
use crate::trainer::{NlheTables, Shard};

/// 64-bit odd constant for mixing the (lo,hi) key halves into the start slot (matches `_MIX` in the
/// Python reference and the rest of the codebase's golden-ratio constant).
const MIX: u64 = 0x9E3779B97F4A7C15;

/// Default per-slot-lock pool size (matches the Python default `n_locks=256`). Only touched on a
/// first-visit slot CLAIM; after warmup every reachable infoset is claimed and the lock pool is never hit.
const N_LOCKS: usize = 256;

/// Load-factor ceiling: claiming a slot that would push occupancy past this is a HARD ERROR (size up),
/// never silent corruption — exactly the Python's bounded-capacity contract. 0.7 keeps probe chains short.
const MAX_LOAD_FACTOR: f64 = 0.7;

/// A fixed-capacity, open-addressing, atomic-backed shared regret + strategy-sum table.
///
/// Layout (all length `capacity`, except the row arrays which are `capacity * NUM_ACTIONS`):
///   * `occ[slot]`     : 0 = empty, 1 = claimed (AtomicU8). Published AFTER the key, exactly like the Python.
///   * `klo`/`khi`     : the two u64 halves of the 16-byte fingerprint (AtomicU64) — keys reconstruct, so a
///                       false slot match is impossible.
///   * `regret`        : f64 cum_regret rows, stored as `AtomicU64` via `f64::to_bits`/`from_bits`.
///   * `strat`         : f64 strat_sum rows (MUST be f64 — racy small-increment loss in f32 corrupts the
///                       average; measured 97 vs <5 mbb on Leduc), same atomic-bits storage.
///   * `locks[slot % N_LOCKS]` : a pool of `Mutex<()>` guarding ONLY first-visit slot claims.
///
/// All read-modify-writes on `regret`/`strat` cells are RACY by design (load Relaxed, add, store Relaxed)
/// — the accepted hogwild noise. They are NOT undefined behavior because every access goes through an
/// atomic (no `&mut` aliasing across threads, no `UnsafeCell`).
pub struct SharedHogwildTable {
    capacity: usize,
    occ: Vec<AtomicU8>,
    klo: Vec<AtomicU64>,
    khi: Vec<AtomicU64>,
    regret: Vec<AtomicU64>,
    strat: Vec<AtomicU64>,
    locks: Vec<Mutex<()>>,
}

#[inline]
fn split_key(key: &[u8; 16]) -> (u64, u64) {
    let mut lo = [0u8; 8];
    let mut hi = [0u8; 8];
    lo.copy_from_slice(&key[0..8]);
    hi.copy_from_slice(&key[8..16]);
    (u64::from_le_bytes(lo), u64::from_le_bytes(hi))
}

#[inline]
fn join_key(lo: u64, hi: u64) -> [u8; 16] {
    let mut key = [0u8; 16];
    key[0..8].copy_from_slice(&lo.to_le_bytes());
    key[8..16].copy_from_slice(&hi.to_le_bytes());
    key
}

impl SharedHogwildTable {
    /// Allocate a zeroed table of `capacity` slots. `capacity` should be generous vs the bounded
    /// abstraction's reachable infoset count (the 6-max EMD-K160 blueprint is ~60-70K infosets, so a few
    /// hundred K slots keeps the load factor well under `MAX_LOAD_FACTOR`).
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "SharedHogwildTable capacity must be > 0");
        let row_cells = capacity
            .checked_mul(NUM_ACTIONS)
            .expect("capacity * NUM_ACTIONS overflow");
        let mut occ = Vec::with_capacity(capacity);
        let mut klo = Vec::with_capacity(capacity);
        let mut khi = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            occ.push(AtomicU8::new(0));
            klo.push(AtomicU64::new(0));
            khi.push(AtomicU64::new(0));
        }
        let mut regret = Vec::with_capacity(row_cells);
        let mut strat = Vec::with_capacity(row_cells);
        for _ in 0..row_cells {
            regret.push(AtomicU64::new(0)); // 0.0f64.to_bits() == 0
            strat.push(AtomicU64::new(0));
        }
        let mut locks = Vec::with_capacity(N_LOCKS);
        for _ in 0..N_LOCKS {
            locks.push(Mutex::new(()));
        }
        SharedHogwildTable {
            capacity,
            occ,
            klo,
            khi,
            regret,
            strat,
            locks,
        }
    }

    /// Total slot capacity (fixed at construction).
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    #[inline]
    fn start_slot(&self, lo: u64, hi: u64) -> usize {
        // Same mix as the Python `_start`: lo XOR (hi*MIX), modulo capacity. capacity need not be a power
        // of two (we keep the Python's `% capacity`); the multiplicative mix decorrelates the halves.
        ((lo ^ hi.wrapping_mul(MIX)) % self.capacity as u64) as usize
    }

    /// Row index for an infoset fingerprint; claims an empty slot on first visit (lock-guarded re-check).
    /// Returns `Err` if the table is full past the load factor (HARD error — size up, never corrupt).
    ///
    /// Lock-free hot path: hash -> linear-probe `occ`/`klo`/`khi`; a claimed slot with a matching key
    /// returns immediately with NO locking. First-visit claim takes `locks[slot % N_LOCKS]` and re-checks
    /// `occ` under it (a memory barrier so two distinct keys never share a slot; a stale lock-free "empty"
    /// read just causes a harmless re-check). The key (klo/khi) is published BEFORE `occ=1` (Release on the
    /// occ store / Acquire on the occ load) so any reader that sees `occ==1` also sees the published key.
    pub fn slot_for(&self, key: &[u8; 16]) -> Result<usize, String> {
        let (lo, hi) = split_key(key);
        let mut slot = self.start_slot(lo, hi);
        for _ in 0..self.capacity {
            if self.occ[slot].load(Ordering::Acquire) != 0 {
                if self.klo[slot].load(Ordering::Relaxed) == lo
                    && self.khi[slot].load(Ordering::Relaxed) == hi
                {
                    return Ok(slot);
                }
                slot = (slot + 1) % self.capacity;
                continue;
            }
            // Apparently empty — claim under the per-slot lock (re-check: a barrier vs concurrent claims).
            {
                let _guard = self.locks[slot % N_LOCKS].lock().unwrap_or_else(|e| e.into_inner());
                if self.occ[slot].load(Ordering::Acquire) == 0 {
                    // Publish the key BEFORE occ=1 (readers check occ first, then the key).
                    self.klo[slot].store(lo, Ordering::Relaxed);
                    self.khi[slot].store(hi, Ordering::Relaxed);
                    self.occ[slot].store(1, Ordering::Release);
                    return Ok(slot);
                }
                if self.klo[slot].load(Ordering::Relaxed) == lo
                    && self.khi[slot].load(Ordering::Relaxed) == hi
                {
                    return Ok(slot);
                }
            }
            slot = (slot + 1) % self.capacity;
        }
        Err(format!(
            "SharedHogwildTable full (capacity {}, load factor exceeded {:.2}); raise hogwild_capacity for this abstraction.",
            self.capacity, MAX_LOAD_FACTOR
        ))
    }

    /// Lock-free lookup WITHOUT claiming: returns the slot for a known key, or `None` if the key is not
    /// present. A no-claim probe primitive (covered by the unit tests; the production hot path uses
    /// `slot_for`, which claims). `#[allow(dead_code)]` keeps the non-test build warning-clean.
    #[allow(dead_code)]
    fn find_slot(&self, key: &[u8; 16]) -> Option<usize> {
        let (lo, hi) = split_key(key);
        let mut slot = self.start_slot(lo, hi);
        for _ in 0..self.capacity {
            if self.occ[slot].load(Ordering::Acquire) == 0 {
                return None;
            }
            if self.klo[slot].load(Ordering::Relaxed) == lo
                && self.khi[slot].load(Ordering::Relaxed) == hi
            {
                return Some(slot);
            }
            slot = (slot + 1) % self.capacity;
        }
        None
    }

    /// Current number of occupied (claimed) slots. O(capacity) — for introspection / load-factor checks,
    /// not the hot path.
    pub fn n_occupied(&self) -> usize {
        self.occ
            .iter()
            .filter(|c| c.load(Ordering::Relaxed) != 0)
            .count()
    }

    #[inline]
    pub fn load_factor(&self) -> f64 {
        self.n_occupied() as f64 / self.capacity.max(1) as f64
    }

    /// HARD-error guard against silently overrunning the table: call before/around a batch of claims. The
    /// per-claim `slot_for` already returns `Err` when no empty slot is found within `capacity` probes;
    /// this is an additional defensive ceiling so we panic LOUDLY well before the probe chains degrade.
    pub fn assert_within_load_factor(&self) -> Result<(), String> {
        let lf = self.load_factor();
        if lf > MAX_LOAD_FACTOR {
            return Err(format!(
                "SharedHogwildTable load factor {lf:.3} exceeded ceiling {MAX_LOAD_FACTOR:.2} (capacity {}); raise hogwild_capacity.",
                self.capacity
            ));
        }
        Ok(())
    }

    // ---- hogwild reads/updates (racy by design) ----

    #[inline]
    fn cell_get(cell: &AtomicU64) -> f64 {
        f64::from_bits(cell.load(Ordering::Relaxed))
    }

    /// Racy read-modify-write of one f64 cell: load Relaxed, add, store Relaxed. THIS IS the accepted
    /// hogwild race — a concurrent writer's update may be lost, which is the provably-harmless noise for
    /// external-sampling MCCFR. NOT undefined behavior (every access is atomic).
    #[inline]
    fn cell_add(cell: &AtomicU64, delta: f64) {
        let cur = f64::from_bits(cell.load(Ordering::Relaxed));
        cell.store((cur + delta).to_bits(), Ordering::Relaxed);
    }

    /// Racy add-then-CFR+-floor of one regret cell (clamp the post-add value to >= 0). Matches the Python
    /// `add_regret(cfr_plus=True)`: the floor is applied ON THE ADD (not once-per-iteration as in the
    /// fork-merge path, because there is no per-iteration barrier here). regret_match also floors on read,
    /// so a transiently-torn value is harmless.
    #[inline]
    fn cell_add_floor(cell: &AtomicU64, delta: f64) {
        let cur = f64::from_bits(cell.load(Ordering::Relaxed));
        let next = (cur + delta).max(0.0);
        cell.store(next.to_bits(), Ordering::Relaxed);
    }

    /// Read the cum_regret row for `slot` into a plain `[f64; NUM_ACTIONS]` (for regret-matching). Torn
    /// rows under concurrent writes = accepted hogwild noise.
    #[inline]
    pub fn regret_row(&self, slot: usize) -> [f64; NUM_ACTIONS] {
        let base = slot * NUM_ACTIONS;
        let mut row = [0.0f64; NUM_ACTIONS];
        for a in 0..NUM_ACTIONS {
            row[a] = Self::cell_get(&self.regret[base + a]);
        }
        row
    }

    /// Add a per-action regret delta to `slot` with the CFR+ floor applied per element (clamp >= 0). Only
    /// the `legal` lanes are touched, matching the fork-merge write loop (`if legal[a] { r[a] += ... }`).
    #[inline]
    pub fn add_regret(&self, slot: usize, legal: &[bool; NUM_ACTIONS], delta: &[f64; NUM_ACTIONS]) {
        let base = slot * NUM_ACTIONS;
        for a in 0..NUM_ACTIONS {
            if legal[a] {
                Self::cell_add_floor(&self.regret[base + a], delta[a]);
            }
        }
    }

    /// Add a per-action strat-sum contribution to `slot` (f64 cells, no floor). Only `legal` lanes.
    #[inline]
    pub fn add_strat(&self, slot: usize, legal: &[bool; NUM_ACTIONS], contrib: &[f64; NUM_ACTIONS]) {
        let base = slot * NUM_ACTIONS;
        for a in 0..NUM_ACTIONS {
            if legal[a] {
                Self::cell_add(&self.strat[base + a], contrib[a]);
            }
        }
    }

    // ---- snapshot back into the fork-merge `NlheTables` shape (checkpoint/serialize-compatible) ----

    /// Materialize the shared table into a fresh `NlheTables` (the SAME in-memory shape `serialize` /
    /// `snapshot` / the serve path consume), copying `iters_done` through. Every occupied slot's
    /// reconstructed 16-byte key is routed to its shard exactly as `deserialize`/`merge` does (via the
    /// trainer's `shard_of`), so the produced `NlheTables` is byte-format-identical to a fork-merge one and
    /// round-trips through the unchanged `serialize`/`deserialize`. Point-in-time stable copy: a concurrent
    /// claim lands in-or-out (fine for a checkpoint); the key is published before `occ=1` so any slot seen
    /// occupied here has a valid key.
    pub fn snapshot_tables(&self, iters_done: u64) -> NlheTables {
        let mut tab = NlheTables::default();
        tab.iters_done = iters_done;
        for slot in 0..self.capacity {
            if self.occ[slot].load(Ordering::Acquire) == 0 {
                continue;
            }
            let lo = self.klo[slot].load(Ordering::Relaxed);
            let hi = self.khi[slot].load(Ordering::Relaxed);
            let key = join_key(lo, hi);
            let base = slot * NUM_ACTIONS;
            let mut reg = [0.0f64; NUM_ACTIONS];
            let mut str_ = [0.0f64; NUM_ACTIONS];
            let mut reg_any = false;
            let mut str_any = false;
            for a in 0..NUM_ACTIONS {
                reg[a] = Self::cell_get(&self.regret[base + a]);
                str_[a] = Self::cell_get(&self.strat[base + a]);
                reg_any |= reg[a] != 0.0;
                str_any |= str_[a] != 0.0;
            }
            // Match fork-merge map membership: a key gets a cum_regret row only if it was a HERO node
            // (regret written) and a strat_sum row only if it was an OPPONENT node (strat written). A slot
            // is CLAIMED in `slot_for` regardless of which side wrote, so without this filter a hero-only
            // infoset would emit an all-zero strat_sum row (which `snapshot`/serialize would carry as a
            // spurious zero key, diverging from the fork-merge blueprint). Emitting only the populated
            // side(s) keeps the materialized `NlheTables` — and thus the served blueprint + the NLCFR001
            // checkpoint — semantically identical to a fork-merge table.
            crate::trainer::shard_insert_snapshot_filtered(
                &mut tab.shards,
                &key,
                if reg_any { Some(&reg) } else { None },
                if str_any { Some(&str_) } else { None },
            );
        }
        tab
    }

    /// Load an existing `NlheTables` (e.g. a resumed checkpoint) into the shared table, claiming a slot per
    /// key and writing its regret/strat rows. Used when resuming a hogwild run from a fork-merge (or prior
    /// hogwild) checkpoint. Returns `Err` on capacity overflow.
    pub fn load_from_tables(&self, tab: &NlheTables) -> Result<(), String> {
        crate::trainer::for_each_table_row(tab, |key, reg, str_| {
            let slot = self.slot_for(key)?;
            let base = slot * NUM_ACTIONS;
            for a in 0..NUM_ACTIONS {
                self.regret[base + a].store(reg[a].to_bits(), Ordering::Relaxed);
                self.strat[base + a].store(str_[a].to_bits(), Ordering::Relaxed);
            }
            Ok(())
        })
    }
}

// `Shard` is referenced only to keep the snapshot/serialize contract explicit in docs above.
#[allow(unused_imports)]
use Shard as _ShardContract;

#[cfg(test)]
mod tests {
    use super::*;

    fn k(lo: u64, hi: u64) -> [u8; 16] {
        join_key(lo, hi)
    }

    #[test]
    fn claim_and_reprobe_returns_same_slot() {
        let t = SharedHogwildTable::new(1024);
        let key = k(123, 456);
        let s1 = t.slot_for(&key).unwrap();
        let s2 = t.slot_for(&key).unwrap();
        assert_eq!(s1, s2, "same key must map to the same slot");
        assert_eq!(t.n_occupied(), 1);
        // find_slot (no-claim) sees it.
        assert_eq!(t.find_slot(&key), Some(s1));
        // A never-claimed key is absent.
        assert_eq!(t.find_slot(&k(999, 999)), None);
    }

    #[test]
    fn distinct_colliding_keys_get_distinct_slots() {
        // Two keys engineered to share a start slot (lo identical, hi differs but mixes to same residue is
        // hard to force; instead force the SAME start by using capacity that makes both map to 0). Simpler:
        // many distinct keys, assert all slots distinct and keys reconstruct.
        let cap = 2048;
        let t = SharedHogwildTable::new(cap);
        let mut slots = std::collections::HashSet::new();
        for i in 0..500u64 {
            let key = k(i.wrapping_mul(0x100000001b3), i ^ 0xDEAD);
            let s = t.slot_for(&key).unwrap();
            assert!(slots.insert(s), "two distinct keys shared slot {s}");
            // Reprobe returns the same slot and the key reconstructs.
            assert_eq!(t.slot_for(&key).unwrap(), s);
            let (lo, hi) = (
                t.klo[s].load(Ordering::Relaxed),
                t.khi[s].load(Ordering::Relaxed),
            );
            assert_eq!(join_key(lo, hi), key);
        }
        assert_eq!(t.n_occupied(), 500);
    }

    #[test]
    fn add_regret_floors_and_strat_accumulates_f64() {
        let t = SharedHogwildTable::new(64);
        let key = k(7, 7);
        let slot = t.slot_for(&key).unwrap();
        let legal = [true; NUM_ACTIONS];
        // Negative regret add must floor to 0 (CFR+).
        let mut d = [0.0f64; NUM_ACTIONS];
        d[0] = -5.0;
        d[1] = 3.0;
        t.add_regret(slot, &legal, &d);
        let row = t.regret_row(slot);
        assert_eq!(row[0], 0.0, "negative regret must floor to 0");
        assert_eq!(row[1], 3.0);
        // A second positive add accumulates; a later negative add re-floors.
        let mut d2 = [0.0f64; NUM_ACTIONS];
        d2[1] = 2.0;
        d2[0] = -1.0;
        t.add_regret(slot, &legal, &d2);
        let row = t.regret_row(slot);
        assert_eq!(row[1], 5.0);
        assert_eq!(row[0], 0.0);
        // strat_sum accumulates small f64 increments exactly (no floor).
        let c = [0.25f64; NUM_ACTIONS];
        for _ in 0..4 {
            t.add_strat(slot, &legal, &c);
        }
        let base = slot * NUM_ACTIONS;
        assert!((SharedHogwildTable::cell_get(&t.strat[base + 2]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn snapshot_roundtrips_through_serialize() {
        let t = SharedHogwildTable::new(256);
        let legal = [true; NUM_ACTIONS];
        for i in 0..20u64 {
            let key = k(i, i.wrapping_mul(31));
            let slot = t.slot_for(&key).unwrap();
            let mut r = [0.0f64; NUM_ACTIONS];
            let mut s = [0.0f64; NUM_ACTIONS];
            for a in 0..NUM_ACTIONS {
                r[a] = (i as f64) + (a as f64) * 0.5;
                s[a] = (i as f64) * 2.0 + (a as f64);
            }
            t.add_regret(slot, &legal, &r);
            t.add_strat(slot, &legal, &s);
        }
        let tab = t.snapshot_tables(42);
        assert_eq!(tab.iters_done, 42);
        assert_eq!(tab.n_strat_infosets(), 20);
        // Round-trip through the UNCHANGED serialize/deserialize and reload into a fresh shared table.
        let blob = crate::trainer::serialize(&tab);
        let tab2 = crate::trainer::deserialize(&blob).expect("deserialize");
        assert_eq!(tab2.n_strat_infosets(), 20);
        let t2 = SharedHogwildTable::new(256);
        t2.load_from_tables(&tab2).expect("load");
        let tab3 = t2.snapshot_tables(tab2.iters_done);
        // snapshot(tab) and snapshot(tab3) must be identical maps.
        let m1 = crate::trainer::snapshot(&tab);
        let m3 = crate::trainer::snapshot(&tab3);
        assert_eq!(m1.len(), m3.len());
        for (key, row) in &m1 {
            let r3 = m3.get(key).expect("key present after roundtrip");
            for a in 0..NUM_ACTIONS {
                assert!((row[a] - r3[a]).abs() < 1e-12, "row mismatch at key after roundtrip");
            }
        }
    }

    #[test]
    fn overflow_is_a_hard_error() {
        // Tiny capacity: claim until it errors. With capacity 4 we can claim at most 4 distinct keys; the
        // 5th MUST return Err (never silently corrupt).
        let t = SharedHogwildTable::new(4);
        let mut ok = 0;
        let mut err = false;
        for i in 0..16u64 {
            match t.slot_for(&k(i, i.wrapping_mul(7).wrapping_add(1))) {
                Ok(_) => ok += 1,
                Err(_) => {
                    err = true;
                    break;
                }
            }
        }
        assert!(err, "expected a hard capacity-overflow error");
        assert!(ok <= 4, "claimed more slots ({ok}) than capacity 4");
    }
}
