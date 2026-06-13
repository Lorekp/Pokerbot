//! Pure-Rust sanity tests for the cube kernel (no Python). Bit-identity against NumPy is verified by the
//! Python harness (`aof_cfr/marg3_native.py::verify_against_numpy` and `tests/test_marg3_native.py`),
//! which is the authoritative oracle. These tests only check internal consistency (shapes, the mix
//! algebra, and that `build_full` == per-block assembly).

use crate::core::*;

fn const_cube(v: f64) -> Vec<f64> {
    vec![v; N3]
}

fn ramp_cube(scale: f64) -> Vec<f64> {
    (0..N3).map(|i| ((i % 97) as f64) * scale - 1.0).collect()
}

fn pushes_fixture() -> Vec<f64> {
    // 6*169 push probs in (0,1) + 2 fold constants.
    let mut v = Vec::with_capacity(6 * N + 2);
    for s in 0..6 {
        for idx in 0..N {
            v.push((((s * 169 + idx) % 100) as f64 + 0.5) / 100.0);
        }
    }
    v.push(3.5); // btn_win_uncontested_after_sb_fold
    v.push(-2.25); // sb_win_uncontested_after_btn_fold
    v
}

fn make_cubes<'a>(store: &'a [Vec<f64>]) -> Cubes<'a> {
    Cubes {
        deal_w: &store[0],
        btn_bb_call0: &store[1],
        btn_sb_call0: &store[2],
        three0: &store[3],
        sb_bb_call1: &store[4],
        btn_sb_call1: &store[5],
        three1: &store[6],
        sb_bb_call2: &store[7],
        btn_bb_call2: &store[8],
        three2: &store[9],
    }
}

#[test]
fn build_full_matches_per_block() {
    let store: Vec<Vec<f64>> = vec![
        ramp_cube(0.001),
        ramp_cube(0.003),
        ramp_cube(-0.002),
        ramp_cube(0.004),
        ramp_cube(0.0011),
        ramp_cube(-0.0007),
        ramp_cube(0.0015),
        ramp_cube(0.0009),
        ramp_cube(-0.0013),
        ramp_cube(0.0021),
    ];
    let cubes = make_cubes(&store);
    let pf = pushes_fixture();
    let p = Pushes {
        btn: &pf[0..N],
        psf: &pf[N..2 * N],
        psa: &pf[2 * N..3 * N],
        pfa: &pf[3 * N..4 * N],
        paf: &pf[4 * N..5 * N],
        paa: &pf[5 * N..6 * N],
        btn_win_uncontested_after_sb_fold: pf[6 * N],
        sb_win_uncontested_after_btn_fold: pf[6 * N + 1],
    };
    for key in [Key::Btn, Key::SbF, Key::SbA, Key::BbFa, Key::BbAf, Key::BbAa] {
        let mut wf = const_cube(0.0);
        let mut pf2 = const_cube(0.0);
        build_full(key, &cubes, &p, &mut wf, &mut pf2);
        // assemble per-block and compare
        let mut wb = const_cube(0.0);
        let mut pb = const_cube(0.0);
        for i in 0..N {
            let o = i * N2;
            build_block_one(key, &cubes, &p, i, &mut wb[o..o + N2], &mut pb[o..o + N2]);
        }
        assert!(wf == wb, "w mismatch for key");
        assert!(pf2 == pb, "prod mismatch for key");
    }
}

#[test]
fn btn_weight_is_deal_w() {
    // For the btn key, w == deal_w exactly (no extra factor).
    let store: Vec<Vec<f64>> = (0..10).map(|s| ramp_cube(0.001 * (s as f64 + 1.0))).collect();
    let cubes = make_cubes(&store);
    let pf = pushes_fixture();
    let p = Pushes {
        btn: &pf[0..N],
        psf: &pf[N..2 * N],
        psa: &pf[2 * N..3 * N],
        pfa: &pf[3 * N..4 * N],
        paf: &pf[4 * N..5 * N],
        paa: &pf[5 * N..6 * N],
        btn_win_uncontested_after_sb_fold: pf[6 * N],
        sb_win_uncontested_after_btn_fold: pf[6 * N + 1],
    };
    let mut w = const_cube(0.0);
    let mut pr = const_cube(0.0);
    build_full(Key::Btn, &cubes, &p, &mut w, &mut pr);
    assert!(w == store[0], "btn w must equal deal_w");
}
