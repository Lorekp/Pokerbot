//! Pure-Rust unit tests (no Python). Golden values captured from CPython `random.Random`.

use crate::core::MT19937;

#[test]
fn mt19937_golden_matches_cpython() {
    // golden = [random.Random(s).getrandbits(32) for _ in range(3)] captured from CPython 3.x.
    let cases: [(u32, [u32; 3]); 5] = [
        (0, [3626764237, 1654615998, 3255389356]),
        (1, [577090037, 2444712010, 3639700191]),
        (12345, [1789368711, 3146859322, 43676229]),
        (0xFFFFFFFF, [2728839433, 2661025012, 872737089]),
        (777, [984982403, 1914885928, 1913820983]),
    ];
    for (seed, want) in cases.iter() {
        let mut rng = MT19937::new_from_u32(*seed);
        for w in want.iter() {
            assert_eq!(rng.genrand_uint32(), *w, "seed {seed}");
        }
    }
}

#[test]
fn randrange_matches_cpython() {
    // golden = let r=random.Random(12345); [r.randrange(48-i) for i in range(5)] == [26,46,0,19,23]
    let mut rng = MT19937::new_from_u32(12345);
    let got: Vec<u32> = (0..5).map(|i| rng.randrange(48 - i as u32)).collect();
    assert_eq!(got, vec![26, 46, 0, 19, 23]);
}
