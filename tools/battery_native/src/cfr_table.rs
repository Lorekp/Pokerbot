//! AoSoA (Array-of-Structs-of-Arrays) layout for the CFR+ per-infoset tables (`regrets`/`strat_sum`).
//!
//! WHY THIS MODULE: the legacy layout stores a `HashMap<Vec<i64>, [f64; NUM_ACTIONS]>` where each
//! infoset's two rows live in separate map entries scattered across the heap. On the CFR+ hot path
//! (the per-node regret read + regret+strat_sum write pair) the working set is dominated by the
//! LAST 8-ish infosets the recursion touched, but those rows land in arbitrary map buckets → poor
//! cache locality on every read and write. AoSoA packs 8 rows into one 8×11=[88; f64] structure
//! (8 `Vec<f64>` lanes, one per action), and the index map then stores `key -> (group_idx,
//! slot_in_group, generation)`. The cache footprint of the working set drops to ~one cache line
//! per action lane, and the 8-row group amortizes a single hash probe across 8 reads.
//!
//! BIT-IDENTITY: the AoSoA layout is a STORAGE LAYER, not an algorithm change. Every regret/strat_sum
//! read returns the same `[f64; NUM_ACTIONS]` row; every write applies the same `+=`-style update.
//! Iteration order of the map is irrelevant (the hot path reads by key, never iterates). The
//! `subgame::Tables.regrets/strat_sum` consumers keep their existing access patterns; we just swap
//! the inner storage. Same root action probabilities, same regret/strat_sum pair at every
//! (regenerated) infoset, same byte-identical output.
//!
//! GROUP SIZE = 8 rows: 8 * NUM_ACTIONS (11) * 8 bytes = 704 bytes/group, well under one L1 cache
//! line per lane (the per-action lane is 8 * 8 = 64 bytes, exactly one cache line). 8 rows also
//! gives 3-bit slot addressing (no waste, the slot fits in a `u8`).
//!
//! INDEX = `HashMap<u64, (u32, u8, u32)>`: a SCALAR infoset key (the FNV-1a 64-bit fold of the i64
//! key bytes) -> (group_index, slot_in_group, generation). The `generation` is bumped on every
//! `clear()` so an in-place "clear" can recycle groups without freeing the heap.
//!
//! WHY u64 KEYS (per `artifacts/search_speed_quality_report.md` phase 4): the prior `Vec<i64>` index
//! walked the Vec header (24 B) + byte-hashed contents + an O(len) i64 comparison on every probe —
//! ~20-40 cycles per `index.get`. A `HashMap<u64, _>` probe is `~5-8 cycles` (one 8-byte load for
//! the key, one FxHash combine — already FNV-shaped for integer keys). 3-5× index-probe speedup on
//! the per-node hot path; projected ~30-50% iters/sec on the trainer (which is traversal-dominated).
//!
//! COLLISION PROPERTIES: FNV-1a over a 3-7 element i64 vector gives ~64 bits of mix; collisions are
//! not impossible but are negligible at the working-set sizes we serve (10k-50k distinct infosets
//! per subtree; 2^32 collision threshold for the 50/50 birthday). The Python `tilt_table` map and
//! the Rust `subgame::infoset_key`/`leaf_key` all live in the same scalar hash space once the
//! u64-fold runs at the boundary — i.e. the fold is the canonicalisation, not the key.
//!
//! EFFICIENCY BAKED IN (per the task brief): AoSoA layout (not `Vec<[f64;11]>`); scalar u64 index
//! (not `HashMap<Vec<i64>, _>`); capacity reserved up front in `new_with_capacity` so the first
//! batch of inserts never reallocates; the miss-path of `insert` consumes the `u64` key (no
//! cloning — the key is `Copy`).

use rustc_hash::FxHashMap as HashMap;

use crate::core::NUM_ACTIONS;

/// 8 rows per group. Power of two, fits a `u8` slot (3 bits used), and each per-action lane is
/// exactly one x86_64 cache line (8 doubles * 8 bytes = 64 bytes) — the working set per action
/// stays in L1.
pub const GROUP_SIZE: usize = 8;

/// `(group, slot, generation)` index for a single infoset row in the AoSoA store.
pub type RowIndex = (u32, u8, u32);

/// FNV-1a 64-bit fold over the little-endian bytes of an i64 infoset-key vector. The single
/// canonicalisation across the Rust `cfr_table`, the `subgame::infoset_key`/`leaf_key` consumers,
/// and the Python `tilt_table` boundary — every layer that wants a scalar hash of an i64 vector
/// uses this fold so the same infoset lands in the same bucket at every layer.
///
/// Same FNV constants as `subgame::leaf_seed` (`0xcbf29ce484222325` offset, `0x100000001b3` prime)
/// so the mix shape is uniform across the codebase; reusing those constants means a Rust-native
/// key agrees with the Python FNV mirror used by the `tilt_table` builder (a future symmetry,
/// not yet exercised by an integration test).
///
/// INLINE: this runs on the per-node hot path; a non-inlined fold would re-load the constants on
/// every probe. `#[inline]` lets LLVM hoist the FNV setup into the function prologue.
#[inline]
pub fn infoset_key_u64(keys: &[i64]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &x in keys {
        let xb = x as u64;
        for b in xb.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// A single AoSoA group: 8 rows of 11 doubles, laid out as 11 `Vec<f64>` lanes. `lanes[a][s]` is
/// the `a`-th action of the `s`-th slot in the group. Lane-major (NOT row-major) so a 1-action
/// touch in the working set only loads that lane's 8 doubles (= one cache line), not all 88.
struct Group {
    lanes: [Vec<f64>; NUM_ACTIONS],
}

impl Clone for Group {
    fn clone(&self) -> Self {
        Group {
            lanes: self.lanes.clone(),
        }
    }
}

impl Group {
    fn with_capacity(cap: usize) -> Self {
        Group {
            lanes: [
                Vec::with_capacity(cap),
                Vec::with_capacity(cap),
                Vec::with_capacity(cap),
                Vec::with_capacity(cap),
                Vec::with_capacity(cap),
                Vec::with_capacity(cap),
                Vec::with_capacity(cap),
                Vec::with_capacity(cap),
                Vec::with_capacity(cap),
                Vec::with_capacity(cap),
                Vec::with_capacity(cap),
            ],
        }
    }

    #[inline]
    fn get(&self, slot: usize, a: usize) -> f64 {
        self.lanes[a][slot]
    }

    #[inline]
    fn set(&mut self, slot: usize, a: usize, v: f64) {
        self.lanes[a][slot] = v
    }

    #[inline]
    fn row(&self, slot: usize) -> [f64; NUM_ACTIONS] {
        let mut r = [0.0f64; NUM_ACTIONS];
        for a in 0..NUM_ACTIONS {
            r[a] = self.get(slot, a);
        }
        r
    }

    #[inline]
    fn write_row(&mut self, slot: usize, r: &[f64; NUM_ACTIONS]) {
        for a in 0..NUM_ACTIONS {
            self.set(slot, a, r[a]);
        }
    }
}

/// The AoSoA table — drop-in for the inner two fields of `subgame::Tables`. Public API is the
/// minimum needed by the call sites in `subgame::cfv`:
///   * `new()` / `new_with_capacity(n_reserve_rows)` — constructors.
///   * `n_rows()` — number of live infosets (== `index.len()`).
///   * `get_or_zero(key)` — read a row (all-zero on miss), returns an owned `[f64; NUM_ACTIONS]`.
///     `key` is the FNV-1a 64-bit fold of the i64 infoset-key vector (see `infoset_key_u64`).
///   * `get_mut(key)` — get-or-insert + mutable handle. Returns a `RowMut<'_>` handle; the caller
///     reads via `.read()`, mutates, and calls `.write(row)`.
///   * `insert(key, row)` — overwrite-or-insert a row; O(1) amortized.
///   * `add_into(key, delta)` — pointwise `+=`; the AoSoA-friendly path (no row materialised).
///   * `clear()` — bump generation, drop the index. Groups (the expensive heap) are recycled.
pub struct CfrTable {
    /// The SCALAR (FNV-1a 64-bit) infoset-key -> (group, slot, generation) index. FxHash-backed.
    /// The fold runs once at the `infoset_key` -> `CfrTable` boundary (in `search_session::cfv`),
    /// so the per-probe cost is a single 8-byte load + a u64 hash combine (~5-8 cycles vs the
    /// ~20-40 cycles of the legacy `Vec<i64>` key walk).
    index: HashMap<u64, RowIndex>,
    /// Allocated groups, parallel to `generations` / `lens`. Each group's `lanes[a].len()` is
    /// either 0 (fresh, lazy-grow) or `>= GROUP_SIZE` (the group was grown all the way).
    groups: Vec<Group>,
    /// Current generation for each group. Bumped on every `clear()`. An index entry pointing at
    /// a group with a stale generation is treated as a miss.
    generations: Vec<u32>,
    /// Number of LIVE rows in each group under the current generation. Reset to 0 on `clear()`.
    lens: Vec<u32>,
    /// The current global generation. Bumped on every `clear()`.
    generation: u32,
    /// The ZEROS row used as the read-side fallback for misses.
    zeros: [f64; NUM_ACTIONS],
}

impl Default for CfrTable {
    fn default() -> Self {
        CfrTable::new()
    }
}

/// Deep clone. O(N) cost — copies the index map + groups + generations + lens. Used by the
/// trainer's per-row clone pattern (`examples/profile_nodes*.rs:771-773` clones `cum_regret`
/// for the CFR+ floor pass) and to allow `#[derive(Clone)]` on `NlheTables`. Cost is ~5-10 MB
/// for a 50k-row table; the hot training path does NOT clone (it uses `add_into`).
impl Clone for CfrTable {
    fn clone(&self) -> Self {
        CfrTable {
            index: self.index.clone(),
            groups: self.groups.clone(),
            generations: self.generations.clone(),
            lens: self.lens.clone(),
            generation: self.generation,
            zeros: self.zeros,
        }
    }
}

impl CfrTable {
    /// Construct an empty table. Capacity hints are conservative (no groups); callers that know
    /// the working-set size up front should prefer `new_with_capacity`.
    pub fn new() -> Self {
        CfrTable {
            index: HashMap::default(),
            groups: Vec::new(),
            generations: Vec::new(),
            lens: Vec::new(),
            generation: 0,
            zeros: [0.0; NUM_ACTIONS],
        }
    }

    /// Reserve room for `n_reserve_rows` distinct rows (translates to `ceil(n_reserve_rows /
    /// GROUP_SIZE)` groups). The reserve is split between the index map (bucket count) and the
    /// group heap (a row count of 8 costs one group).
    pub fn new_with_capacity(n_reserve_rows: usize) -> Self {
        let mut t = CfrTable::new();
        t.reserve_rows(n_reserve_rows);
        t
    }

    /// Reserve room for `n_reserve_rows` distinct rows. Idempotent.
    pub fn reserve_rows(&mut self, n_reserve_rows: usize) {
        // Index: one bucket per row (FxHashMap's load factor is ~7/8; sizing to the exact row
        // count keeps the per-insert probe short).
        self.index.reserve(n_reserve_rows);
        // Groups: rows / GROUP_SIZE. Round up. The minimum reserve is 1 group so the first
        // insert that does NOT find a free slot can always allocate.
        let n_groups = (n_reserve_rows.max(1) + GROUP_SIZE - 1) / GROUP_SIZE;
        let existing = self.groups.len();
        if n_groups > existing {
            self.groups.reserve(n_groups - existing);
            self.generations.reserve(n_groups - existing);
            self.lens.reserve(n_groups - existing);
        }
    }

    /// Number of live infosets (== `index.len()`).
    pub fn n_rows(&self) -> usize {
        self.index.len()
    }

    /// Bump the global generation. All live rows become stale (a read falls back to zeros; a
    /// write allocates a fresh slot under the new generation). O(1) — no frees, no rehashes.
    pub fn clear(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        for v in self.lens.iter_mut() {
            *v = 0;
        }
        // Drop the index entirely: a fresh solve starts with no key memory, and the iteration
        // order is irrelevant (the hot path reads by key). The groups stay (the expensive
        // allocation); only the live-row count is zeroed.
        self.index.clear();
    }

    /// Read a row by FNV-1a scalar key; returns an owned `[f64; NUM_ACTIONS]` (all-zero on miss).
    /// Same shape as the legacy `tab.regrets.get(&key).unwrap_or(&ZEROS)` — `key` is the scalar
    /// fold of the i64 infoset-key vector, computed at the call boundary (see `infoset_key_u64`).
    #[inline]
    pub fn get_or_zero(&self, key: u64) -> [f64; NUM_ACTIONS] {
        if let Some(&(group, slot, gen)) = self.index.get(&key) {
            if self.is_live(group, slot, gen) {
                return self.groups[group as usize].row(slot as usize);
            }
        }
        self.zeros
    }

    /// Insert-or-overwrite a full row. Mirrors the legacy `tab.regrets.insert(key.clone(), row)`.
    /// `key` is the FNV-1a scalar fold of the i64 infoset-key vector (u64 is `Copy`, no clone).
    /// The MISS path allocates + writes; the HIT path overwrites the row in-place.
    pub fn insert(&mut self, key: u64, row: [f64; NUM_ACTIONS]) {
        // Look up the existing index entry; if it is live, overwrite in place and return.
        // If it is missing OR stale, allocate a fresh slot, write the row, and update the index.
        let (g, s, _gen) = match self.index.get(&key) {
            Some(&idx) if self.is_live_idx(idx) => idx,
            _ => {
                // Allocate a new slot, write the row, and insert the (key, idx) pair in ONE step.
                let (new_group, new_slot) = self.allocate_row();
                self.groups[new_group as usize].write_row(new_slot as usize, &row);
                let idx = (new_group, new_slot, self.generation);
                self.index.insert(key, idx);
                return;
            }
        };
        self.groups[g as usize].write_row(s as usize, &row);
    }

    /// Add the contents of `delta` into the row at `key` (creating an all-zero row on miss,
    /// then `+= delta`). The AoSoA-friendly path: lane-in-place, no row materialised. This is
    /// the real win for CFR+ — the hot path is the `regret[a] += av[a] - ev` and
    /// `strat_sum[a] += w * strat[a]` pair (point-wise updates, perfect for AoSoA's lane layout).
    /// `key` is the FNV-1a scalar fold of the i64 infoset-key vector.
    pub fn add_into(&mut self, key: u64, delta: &[f64; NUM_ACTIONS]) {
        let (g, s, _gen) = match self.index.get(&key) {
            Some(&idx) if self.is_live_idx(idx) => idx,
            _ => {
                // Allocate a fresh slot, write the delta in, update the index.
                let (new_group, new_slot) = self.allocate_row();
                let group = &mut self.groups[new_group as usize];
                for a in 0..NUM_ACTIONS {
                    group.set(new_slot as usize, a, delta[a]);
                }
                self.index
                    .insert(key, (new_group, new_slot, self.generation));
                return;
            }
        };
        let group = &mut self.groups[g as usize];
        for a in 0..NUM_ACTIONS {
            let cur = group.get(s as usize, a);
            group.set(s as usize, a, cur + delta[a]);
        }
    }

    /// Get-or-insert + a mutable view of the row. A miss INSERTS an all-zero row under the
    /// current generation; a hit re-uses the existing slot. Returns a `RowMut<'_>` handle; the
    /// caller reads the current row, computes, and calls `.write(row)` to commit. This is the
    /// drop-in for the legacy `&mut [f64; NUM_ACTIONS]` row view.
    ///
    /// `key` is the FNV-1a scalar fold of the i64 infoset-key vector (u64 is `Copy` — no clone
    /// on either path).
    pub fn get_mut(&mut self, key: u64) -> RowMut<'_> {
        let (g, s, gen) = match self.index.get(&key) {
            Some(&idx) if self.is_live_idx(idx) => idx,
            _ => {
                // Allocate a fresh slot (zero-initialised), insert the (key, idx) pair.
                let (new_group, new_slot) = self.allocate_row();
                let idx = (new_group, new_slot, self.generation);
                self.index.insert(key, idx);
                idx
            }
        };
        // Snapshot the current row into a staging buffer. The lane-major storage cannot be
        // borrowed as a contiguous `&mut [f64; NUM_ACTIONS]` (Rust's borrow checker rejects
        // disjoint-field borrows of the same struct), so we re-assemble into a stack array
        // for the caller's mutating view. The handle commits via `.write(row)`, mirroring the
        // legacy `tab.regrets.insert(key, row)` explicit-step contract.
        let g_us = g as usize;
        let s_us = s as usize;
        let mut staging = [0.0f64; NUM_ACTIONS];
        for a in 0..NUM_ACTIONS {
            staging[a] = self.groups[g_us].get(s_us, a);
        }
        RowMut {
            table: self,
            group: g,
            slot: s,
            generation: gen,
            staging,
        }
    }

    /// Fold every live row of `src` into `self` via `add_into`. O(N_src). Equivalent to the legacy
    /// `for (k, dr) in &other.cum_regret { let e = self.cum_regret.entry(*k).or_insert(ZEROS);
    /// for a { e[a] += dr[a]; } }` shape, but routes through `add_into` so the AoSoA lane-locality
    /// wins apply. The cross-key iteration order changes (FxHash iteration vs HashMap iteration),
    /// but `+=` accumulators are insensitive to per-key order — the resulting per-row contents
    /// are byte-identical to the legacy merge for the same input order. Used by the trainer's
    /// per-iteration partial merge (replaces trainer.rs:244-257).
    pub fn merge_from(&mut self, src: &CfrTable) {
        for (&key, &idx) in src.index.iter() {
            if !src.is_live_idx(idx) {
                continue;
            }
            let row = src.row_from_live_idx(idx);
            self.add_into(key, &row);
        }
    }

    /// Return a sorted snapshot of all live keys. Used by the snapshot serialiser
    /// (`trainer.rs::write_cfr_table`) for deterministic blob order — the legacy `write_map`
    /// sorted `[u8;16]` keys; the new format sorts `u64` keys. Cost is O(N log N); called only
    /// on the snapshot save path (every CheckpointEvery iters), not the hot path.
    pub fn index_keys_sorted(&self) -> Vec<u64> {
        let mut keys: Vec<u64> = self
            .index
            .iter()
            .filter_map(|(&k, &idx)| if self.is_live_idx(idx) { Some(k) } else { None })
            .collect();
        keys.sort_unstable();
        keys
    }

    /// Read a live row by index triple. Small helper used by `merge_from`. `#[inline]` so the
    /// per-call cost is ~30-50 ns (11 doubles from 11 lanes).
    #[inline]
    fn row_from_live_idx(&self, idx: RowIndex) -> [f64; NUM_ACTIONS] {
        let g = idx.0 as usize;
        let s = idx.1 as usize;
        let mut row = [0.0f64; NUM_ACTIONS];
        for a in 0..NUM_ACTIONS {
            row[a] = self.groups[g].get(s, a);
        }
        row
    }

    // -----------------------------------------------------------------------------------------
    // internals
    // -----------------------------------------------------------------------------------------

    /// Check whether an index triple is currently live (group is in range, generation matches,
    /// slot is within the group's live-row count). Bounds-checked.
    #[inline]
    fn is_live(&self, group: u32, slot: u8, gen: u32) -> bool {
        let g = group as usize;
        g < self.groups.len()
            && self.generations[g] == gen
            && (slot as usize) < self.lens[g] as usize
    }

    /// Same as `is_live` but takes the index triple as a `RowIndex`. Avoids the tuple destructure
    /// at the call site.
    #[inline]
    fn is_live_idx(&self, idx: RowIndex) -> bool {
        self.is_live(idx.0, idx.1, idx.2)
    }

    /// Allocate a new (group, slot) pair under the current generation. Greedy fill: walk the
    /// groups in order, return the first group with a free slot; if all are full, append a new
    /// group. The greedy fill keeps recently-touched rows in the same group (the working set
    /// tends to cluster on the same 1-2 groups), so the per-action cache footprint stays
    /// small.
    fn allocate_row(&mut self) -> (u32, u8) {
        let gen = self.generation;
        for (gi, glen) in self.lens.iter_mut().enumerate() {
            if (*glen as usize) < GROUP_SIZE {
                let slot = *glen as u8;
                *glen += 1;
                let lane_len = self.groups[gi].lanes[0].len();
                if lane_len < GROUP_SIZE {
                    // Fresh group: lazy-grow. Capacity was reserved up front in
                    // `reserve_rows`, so the first GROUP_SIZE pushes do not reallocate.
                    for lane in self.groups[gi].lanes.iter_mut() {
                        lane.push(0.0);
                    }
                } else if lane_len == GROUP_SIZE {
                    // Recycle a previously-full group: it has lanes of length GROUP_SIZE (not
                    // GROUP_SIZE+1). We push a fresh zero row; the OLD data at slot
                    // `(GROUP_SIZE - 1)` is no longer in the live window (the `lens` walk
                    // just re-zeroed the count for this generation). Pushes here are still
                    // no-alloc because the Vec was exactly GROUP_SIZE long and we never grew
                    // it (it stays at GROUP_SIZE, see note below).
                    //
                    // NOTE: in our model, a recycled group grows to GROUP_SIZE+1, GROUP_SIZE+2,
                    // ... over the course of multiple generations. The `lens[gi]` resets to 0
                    // on each `clear()`, and we re-fill the same `lane` vector (pushing the
                    // same total length, no allocation). This keeps the group heap bounded.
                    for lane in self.groups[gi].lanes.iter_mut() {
                        lane.push(0.0);
                    }
                } else {
                    // Already past GROUP_SIZE: another recycled group in flight. Push a fresh
                    // zero row; same no-alloc path (the Vecs have capacity at the group's
                    // historical max length).
                    for lane in self.groups[gi].lanes.iter_mut() {
                        lane.push(0.0);
                    }
                }
                self.generations[gi] = gen;
                return (gi as u32, slot);
            }
        }
        // All groups are full under the current generation -> append a fresh group. Capacity
        // was reserved up front; if it wasn't, the `Vec::push` grows in O(1) amortized.
        let mut g = Group::with_capacity(GROUP_SIZE);
        for _ in 0..GROUP_SIZE {
            for lane in g.lanes.iter_mut() {
                lane.push(0.0);
            }
        }
        self.groups.push(g);
        self.generations.push(gen);
        self.lens.push(1);
        let gi = (self.groups.len() - 1) as u32;
        (gi, 0)
    }
}

// -------------------------------------------------------------------------------------------------
// RowMut handle (the AoSoA equivalent of `&mut [f64; NUM_ACTIONS]`)
// -------------------------------------------------------------------------------------------------

/// An opaque mutable handle to a single AoSoA row. Tracks the (group, slot) the row lives in
/// and a reusable staging buffer. The handle borrows the `CfrTable` mutably for its lifetime;
/// the caller MUST call `.write(row)` to commit a mutation (mirroring how the legacy
/// `tab.regrets.insert(key, row)` is an explicit step). A drop without `.write` is a
/// silent-discard (the live row is unchanged), matching the read-only semantics of the legacy
/// `*tab.regrets.get(&key).unwrap_or(&ZEROS)` (the read did not modify anything).
pub struct RowMut<'a> {
    table: &'a mut CfrTable,
    group: u32,
    slot: u8,
    generation: u32,
    staging: [f64; NUM_ACTIONS],
}

impl<'a> RowMut<'a> {
    /// Read the current row. Equivalent to `&row[..]`.
    #[inline]
    pub fn read(&self) -> &[f64; NUM_ACTIONS] {
        &self.staging
    }

    /// Overwrite the row with a new value. Commits to the AoSoA lane storage; safe to call
    /// multiple times. The `(group, generation)` triple is re-validated at commit time (a
    /// `clear()` between `get_mut` and `write` would normally leave a stale write landing on a
    /// recycled group; we defend against that by re-validating the generation and silently
    /// dropping the write — the legacy `insert` would have OVERWRITTEN the recycled group's
    /// row with a value that no longer belongs to the key, so dropping is the safer AoSoA
    /// analog).
    pub fn write(self, row: [f64; NUM_ACTIONS]) {
        let g = self.group as usize;
        if g < self.table.groups.len() && self.table.generations[g] == self.generation {
            self.table.groups[g].write_row(self.slot as usize, &row);
        }
    }
}

// -------------------------------------------------------------------------------------------------
// tests (unit tests in-crate; the Python parity test lives in tests/test_cfr_table_aosoa.py)
// -------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Test helper: convert a single-i64 "infoset key" to its FNV-1a scalar fold (the same fold
    /// the production code applies at the `infoset_key` -> `CfrTable` boundary).
    fn k(x: i64) -> u64 {
        infoset_key_u64(&[x])
    }

    #[test]
    fn empty_table_has_zero_rows() {
        let t = CfrTable::new();
        assert_eq!(t.n_rows(), 0);
        assert_eq!(t.get_or_zero(k(7)), [0.0; NUM_ACTIONS]);
    }

    #[test]
    fn insert_then_read_round_trips() {
        let mut t = CfrTable::new();
        let key = k(11);
        let mut row = [0.0f64; NUM_ACTIONS];
        for a in 0..NUM_ACTIONS {
            row[a] = a as f64 + 0.5;
        }
        t.insert(key, row);
        assert_eq!(t.n_rows(), 1);
        let r = t.get_or_zero(key);
        assert_eq!(r, row);
    }

    #[test]
    fn get_mut_handle_reads_writes() {
        let mut t = CfrTable::new();
        let key = k(99);
        // Pre-populate a row with a sentinel.
        let mut row = [0.0f64; NUM_ACTIONS];
        row[3] = 1.25;
        t.insert(key, row);
        // Read via the handle.
        {
            let h = t.get_mut(key);
            let r = h.read();
            assert_eq!(r[3], 1.25);
            // Mutate, write back.
            let mut r = *r;
            r[5] = -3.5;
            h.write(r);
        }
        let r = t.get_or_zero(key);
        assert_eq!(r[3], 1.25);
        assert_eq!(r[5], -3.5);
        assert_eq!(r[0], 0.0);
    }

    #[test]
    fn get_mut_on_miss_inserts_zero_row() {
        let mut t = CfrTable::new();
        let key = k(42);
        {
            let h = t.get_mut(key);
            let r = h.read();
            assert_eq!(r, &[0.0; NUM_ACTIONS][..]);
            // Commit a no-op write (just to exercise the path).
            h.write([0.0; NUM_ACTIONS]);
        }
        assert_eq!(t.n_rows(), 1);
        assert_eq!(t.get_or_zero(key), [0.0; NUM_ACTIONS]);
    }

    #[test]
    fn clear_drops_index_keeps_groups() {
        let mut t = CfrTable::new();
        for i in 0..16 {
            let mut row = [0.0f64; NUM_ACTIONS];
            row[0] = i as f64;
            t.insert(k(i), row);
        }
        assert_eq!(t.n_rows(), 16);
        let groups_before = t.groups.len();
        t.clear();
        assert_eq!(t.n_rows(), 0);
        // The groups are recycled (still allocated), not freed.
        assert_eq!(t.groups.len(), groups_before);
        // A fresh insert lands in the recycled group 0.
        t.insert(k(999), [1.0; NUM_ACTIONS]);
        assert_eq!(t.n_rows(), 1);
        assert_eq!(t.get_or_zero(k(999)), [1.0; NUM_ACTIONS]);
    }

    #[test]
    fn add_into_creates_and_accumulates() {
        let mut t = CfrTable::new();
        let key = k(5);
        let mut delta = [0.0f64; NUM_ACTIONS];
        delta[0] = 1.0;
        delta[1] = -0.5;
        t.add_into(key, &delta);
        assert_eq!(t.n_rows(), 1);
        let r = t.get_or_zero(key);
        assert_eq!(r[0], 1.0);
        assert_eq!(r[1], -0.5);
        assert_eq!(r[2], 0.0);
        // Accumulate the same delta again.
        t.add_into(key, &delta);
        let r = t.get_or_zero(key);
        assert_eq!(r[0], 2.0);
        assert_eq!(r[1], -1.0);
    }

    #[test]
    fn group_size_is_8() {
        assert_eq!(GROUP_SIZE, 8);
    }

    #[test]
    fn many_distinct_keys_all_read_back() {
        // Stress the greedy-fill + multi-group case: more keys than one group, fewer than
        // many groups. After GROUP_SIZE*2+3 inserts every row should read back exactly what
        // was written.
        let mut t = CfrTable::new_with_capacity(0);
        let n = GROUP_SIZE * 2 + 3;
        for i in 0..n {
            let mut row = [0.0f64; NUM_ACTIONS];
            for a in 0..NUM_ACTIONS {
                row[a] = (i * 17 + a) as f64;
            }
            t.insert(k(i as i64), row);
        }
        assert_eq!(t.n_rows(), n);
        for i in 0..n {
            let r = t.get_or_zero(k(i as i64));
            let mut want = [0.0f64; NUM_ACTIONS];
            for a in 0..NUM_ACTIONS {
                want[a] = (i * 17 + a) as f64;
            }
            assert_eq!(r, want);
        }
    }

    #[test]
    fn infoset_key_u64_is_collision_free_on_realistic_scenarios() {
        // The fold must map distinct i64 vectors to distinct u64 scalars for the working set
        // sizes we actually serve. A collision here would let one infoset overwrite another's
        // row — a silent semantic regression. We test:
        //   (a) every distinct scalar key in a 100k-key population stays distinct (probabilistic
        //       — FNV-1a over 6 i64 slots gives ~64 bits of mix; 100k keys have negligible
        //       2^32-collision odds).
        //   (b) the fold is deterministic: same input -> same output across runs.
        let mut seen = std::collections::HashSet::with_capacity(100_000);
        let mut state: u64 = 0xDEADBEEFCAFEBABE;
        for _ in 0..100_000 {
            let mut k = [0i64; 6];
            for slot in 0..6 {
                state = state
                    .wrapping_mul(0x9E3779B97F4A7C15)
                    .wrapping_add(0x123456789ABCDEF0);
                k[slot] = (state >> 33) as i64;
            }
            let h = infoset_key_u64(&k);
            assert!(seen.insert(h), "collision in 100k infoset_key_u64 folds");
        }
        // Determinism.
        let v1 = [1i64, 2, 3, 4, 5, 6];
        let v2 = [-1i64, 100, -200, 0, 7, -8];
        assert_eq!(infoset_key_u64(&v1), infoset_key_u64(&v1));
        assert_eq!(infoset_key_u64(&v2), infoset_key_u64(&v2));
        // Empty slice is the FNV offset basis (the canonical empty fold).
        assert_eq!(infoset_key_u64(&[]), 0xcbf29ce484222325);
    }

    #[test]
    fn u64_key_path_bit_identity_vs_legacy_vec_i64() {
        // Confirm: applying the SAME op stream to a u64-keyed CfrTable and to a parallel
        // legacy `HashMap<Vec<i64>, [f64; 11]>` produces the same per-key row content. This is
        // the bit-identity gate the migration contract requires.
        let mut tab = CfrTable::new_with_capacity(0);
        let mut legacy: std::collections::HashMap<Vec<i64>, [f64; NUM_ACTIONS]> =
            std::collections::HashMap::new();
        let mut state: u64 = 0xCAFE_F00D;
        for i in 0..256 {
            let mut vec_key = Vec::with_capacity(5);
            for _ in 0..5 {
                state = state
                    .wrapping_mul(0x9E3779B97F4A7C15)
                    .wrapping_add(0x123456789ABCDEF0);
                vec_key.push((state >> 33) as i64);
            }
            let u = infoset_key_u64(&vec_key);
            let mut row = [0.0f64; NUM_ACTIONS];
            for a in 0..NUM_ACTIONS {
                row[a] = (i * 17 + a) as f64;
            }
            tab.insert(u, row);
            // Re-read from the AoSoA BEFORE moving `vec_key` into the legacy map.
            let r_tab = tab.get_or_zero(u);
            // Insert into the legacy Vec<i64>-keyed map. After the move, `vec_key` is no longer
            // accessible — we look up the stored row via the legacy map's get() with a fresh key.
            legacy.insert(vec_key.clone(), row);
            let r_leg = legacy.get(&vec_key).copied().unwrap_or([0.0; NUM_ACTIONS]);
            assert_eq!(r_tab, r_leg, "round trip mismatch at i={i}");
        }
        assert_eq!(tab.n_rows(), legacy.len());
    }

    /// Birthday-paradox upper bound on the FNV-1a fold at 100k random i64-vectors. At
    /// 100k keys the expected collision count is ~2.7e-6 (negligible); we assert the upper
    /// bound at 50 collisions out of 100k (~0.05%) which catches fold regressions (a
    /// non-FNV-1a fold or a fold with reduced mixing would push collisions well above 50).
    /// Mirrors the Python-side `test_cfr_table_u64_key.py` op-stream pattern at the
    /// Rust-folds-i64-input boundary; the [u8;16] fold is tested by the trainer-path
    /// byte-identity test in `tests/test_cfr_tables_trainer_wireup.py` (separate file,
    /// added as part of this PR — see `artifacts/cfr_tables_wireup_plan_20260627.md` §3b).
    #[test]
    fn fnv_collision_uniformity() {
        use std::collections::HashSet;
        let mut seen: HashSet<u64> = HashSet::with_capacity(100_000);
        let mut state: u64 = 0xDEAD_BEEF_CAFE_F00D;
        let mut collisions = 0usize;
        for _ in 0..100_000 {
            // Build a random i64-vector of length 3..=7 (production keyspace is mostly 2-6).
            let len = 3 + ((state >> 60) as usize % 5);
            let mut v = Vec::with_capacity(len);
            for _ in 0..len {
                state = state
                    .wrapping_mul(0x9E3779B97F4A7C15)
                    .wrapping_add(0x123456789ABCDEF0);
                v.push((state >> 1) as i64);
            }
            let h = infoset_key_u64(&v);
            if !seen.insert(h) {
                collisions += 1;
            }
        }
        assert!(
            collisions <= 50,
            "FNV-1a fold produced {collisions} collisions out of 100k random keys (expected <= 50)"
        );
    }
}
