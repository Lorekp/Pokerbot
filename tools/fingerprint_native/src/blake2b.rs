//! Self-contained BLAKE2b (RFC 7693) matching CPython `hashlib.blake2b(digest_size, person=...)`.
//!
//! Vendored (no external crate) so the fingerprint digest is bit-identical to the Python oracle with
//! ZERO version drift. Only the features the fingerprint uses are supported: a chosen `digest_size`
//! (16 here) and a `person` (personalization) of up to 16 bytes; key/salt are unused (zeroed), which
//! matches `hashlib.blake2b(digest_size=16, person=...)` exactly.
//!
//! Parameter block layout (RFC 7693 sec 2.5, little-endian, 64 bytes XORed into the IV):
//!   byte  0: digest length, byte 1: key length (0), byte 2: fanout (1), byte 3: depth (1),
//!   bytes 4..8: leaf length (0), bytes 8..16: node offset (0), byte 16: node depth (0),
//!   byte 17: inner length (0), bytes 18..32: reserved (0), bytes 32..48: salt (0),
//!   bytes 48..64: personalization.
//! CPython initializes h[i] = IV[i] XOR (the i-th little-endian 64-bit word of this 64-byte block).

const IV: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

const SIGMA: [[usize; 16]; 12] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
];

const BLOCK: usize = 128;

#[inline(always)]
fn g(v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
    v[d] = (v[d] ^ v[a]).rotate_right(32);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(24);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(63);
}

pub struct Blake2b {
    h: [u64; 8],
    t: [u64; 2],
    buf: [u8; BLOCK],
    buflen: usize,
    digest_size: usize,
}

impl Blake2b {
    /// `digest_size` in bytes (1..=64), `person` up to 16 bytes (zero-padded). Key/salt unused.
    pub fn new(digest_size: usize, person: &[u8]) -> Self {
        assert!((1..=64).contains(&digest_size));
        assert!(person.len() <= 16);
        // Build the 64-byte parameter block, then h[i] = IV[i] ^ block_word_i (little-endian).
        let mut p = [0u8; 64];
        p[0] = digest_size as u8; // digest length
        p[1] = 0; // key length
        p[2] = 1; // fanout
        p[3] = 1; // depth
                  // leaf length (4..8), node offset (8..16), node depth (16), inner length (17),
                  // reserved (18..32), salt (32..48) all remain zero.
        for (i, b) in person.iter().enumerate() {
            p[48 + i] = *b;
        }
        let mut h = IV;
        for i in 0..8 {
            let mut w = [0u8; 8];
            w.copy_from_slice(&p[i * 8..i * 8 + 8]);
            h[i] ^= u64::from_le_bytes(w);
        }
        Blake2b {
            h,
            t: [0, 0],
            buf: [0u8; BLOCK],
            buflen: 0,
            digest_size,
        }
    }

    fn increment(&mut self, n: u64) {
        let (lo, carry) = self.t[0].overflowing_add(n);
        self.t[0] = lo;
        if carry {
            self.t[1] = self.t[1].wrapping_add(1);
        }
    }

    fn compress(&mut self, block: &[u8; BLOCK], last: bool) {
        let mut m = [0u64; 16];
        for i in 0..16 {
            let mut w = [0u8; 8];
            w.copy_from_slice(&block[i * 8..i * 8 + 8]);
            m[i] = u64::from_le_bytes(w);
        }
        let mut v = [0u64; 16];
        v[..8].copy_from_slice(&self.h);
        v[8..16].copy_from_slice(&IV);
        v[12] ^= self.t[0];
        v[13] ^= self.t[1];
        if last {
            v[14] = !v[14];
        }
        for r in 0..12 {
            let s = &SIGMA[r];
            g(&mut v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
            g(&mut v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
            g(&mut v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
            g(&mut v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
            g(&mut v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
            g(&mut v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
            g(&mut v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
            g(&mut v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
        }
        for i in 0..8 {
            self.h[i] ^= v[i] ^ v[i + 8];
        }
    }

    pub fn update(&mut self, data: &[u8]) {
        let mut data = data;
        if self.buflen > 0 {
            let take = (BLOCK - self.buflen).min(data.len());
            self.buf[self.buflen..self.buflen + take].copy_from_slice(&data[..take]);
            self.buflen += take;
            data = &data[take..];
            // Only compress a full buffer if MORE data follows (BLAKE2 never finalizes mid-update;
            // the last block is processed in finalize). If exactly one block and nothing follows,
            // keep it buffered as the (possibly) last block.
            if self.buflen == BLOCK && !data.is_empty() {
                self.increment(BLOCK as u64);
                let blk = self.buf;
                self.compress(&blk, false);
                self.buflen = 0;
            }
        }
        // Process full blocks while strictly more than one block of data remains buffered+incoming.
        while data.len() > BLOCK {
            self.increment(BLOCK as u64);
            let mut blk = [0u8; BLOCK];
            blk.copy_from_slice(&data[..BLOCK]);
            self.compress(&blk, false);
            data = &data[BLOCK..];
        }
        // Buffer the remaining tail (<= BLOCK bytes) for finalize / next update.
        if !data.is_empty() {
            self.buf[self.buflen..self.buflen + data.len()].copy_from_slice(data);
            self.buflen += data.len();
        }
    }

    pub fn finalize(mut self) -> Vec<u8> {
        self.increment(self.buflen as u64);
        // zero-pad the final block
        for i in self.buflen..BLOCK {
            self.buf[i] = 0;
        }
        let blk = self.buf;
        self.compress(&blk, true);
        let mut out = Vec::with_capacity(self.digest_size);
        for i in 0..self.digest_size {
            out.push((self.h[i / 8] >> (8 * (i % 8))) as u8);
        }
        out
    }
}

/// One-shot convenience matching `hashlib.blake2b(data, digest_size=ds, person=person).digest()`.
pub fn digest(digest_size: usize, person: &[u8], data: &[u8]) -> Vec<u8> {
    let mut h = Blake2b::new(digest_size, person);
    h.update(data);
    h.finalize()
}
