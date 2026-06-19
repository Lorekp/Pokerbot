//! Pure-Rust unit tests (no Python). Cross-checked against the Python oracle by the parity harness
//! in `pokerbot/encoding/fingerprint_native.py` / `verify_parity.py`.

use crate::blake2b;
use crate::core::canonical_preflop_class;
use crate::phe;

#[test]
fn blake2b_empty_person_known_vector() {
    // hashlib.blake2b(b"", digest_size=16).hexdigest()
    let d = blake2b::digest(16, b"", b"");
    assert_eq!(hex(&d), "cae66941d9efbd404e4d88758ea67670");
}

#[test]
fn blake2b_abc_full() {
    // hashlib.blake2b(b"abc").hexdigest() (64-byte, no person) — RFC/reference test vector.
    let d = blake2b::digest(64, b"", b"abc");
    assert_eq!(
        hex(&d),
        "ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d1\
7d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923"
    );
}

#[test]
fn blake2b_person_changes_digest() {
    let a = blake2b::digest(16, b"", b"hello");
    let b = blake2b::digest(16, b"collision-wit", b"hello");
    assert_ne!(a, b);
}

#[test]
fn preflop_class_pairs_and_suited() {
    // AA: both rank 12 -> class 12 (pocket pair == rank ordinal).
    assert_eq!(canonical_preflop_class(48, 49), 12);
    // 22: rank 0 -> class 0.
    assert_eq!(canonical_preflop_class(0, 1), 0);
    // AKs vs AKo differ; AKs uses suited bank (13..90), AKo offsuit (91..168).
    let aks = canonical_preflop_class(48, 44); // As Ks (suit 0,0)
    let ako = canonical_preflop_class(48, 45); // As Kd (suit 0,1)
    assert!(aks != ako);
    assert!((13..=90).contains(&aks));
    assert!((91..=168).contains(&ako));
}

#[test]
fn phe_royal_flush_is_rank_one() {
    // Royal flush in clubs: As Ks Qs Js Ts? use spades suit=3: T,J,Q,K,A of spades = ids 35,39,43,47,51.
    let r = phe::evaluate_cards(&[35, 39, 43, 47, 51]);
    assert_eq!(r, 1);
}

#[test]
fn phe_worst_five_is_7462() {
    // 7-5-4-3-2 unsuited (mixed suits) = the worst 5-card hand, rank 7462.
    // 2c=0,3d=5,4h=10,5? choose 7 5 4 3 2 offsuit: 2c(0) 3d(5) 4h(10) 5s(15) 7c(20)
    let r = phe::evaluate_cards(&[0, 5, 10, 15, 20]);
    assert_eq!(r, 7462);
}

fn hex(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for x in b {
        s.push_str(&format!("{:02x}", x));
    }
    s
}
