//! Native port of the served HU neural policy: `encode_info_state` (+ `_suit_features`) from
//! `pokerbot/engine/jax_engine.py` and the `AverageStrategyNet` forward (`MLP` =
//! Dense(h)->relu->Dense(h)->relu->Dense(out), then masked softmax) from
//! `pokerbot/training/neural_cfr.py`. This lets the native LBR (`lbr.rs`) serve the neural net inside
//! the rayon-parallel hot loop with ZERO Python -- the whole HU exploitability run is native.
//!
//! The net is a tiny MLP (3 dense layers, f32), so the forward is three GEMMs + relu + softmax. The one
//! real skew risk is the encoder feature LAYOUT; it is gated by a golden cross-check (the same pattern
//! the other 5 native tools use): `tests/test_nnet_native_golden.py` serves N random states through both
//! the JAX net and this port and asserts the action distributions match. Run it with the SAME
//! POKERBOT_INFO_SUITS as the checkpoint was trained with.

use crate::core::{self, EnvState, NUM_ACTIONS, NUM_PLAYERS};

pub const INFO_DIM: usize = 256; // INFO_FEATURE_DIM
const SCALE: f32 = 10_000.0; // STARTING_STACK_CHIPS
const NUM_STREETS: usize = 4;
const SEAT_FOLDED: i8 = 1;
const SEAT_ALLIN: i8 = 3;

/// Trained average-strategy net: 3 dense layers (kernel row-major in-major, bias len out).
pub struct Net {
    w: [Vec<f32>; 3],
    b: [Vec<f32>; 3],
    din: [usize; 3],
    dout: [usize; 3],
    info_suits: bool,
}

impl Net {
    /// Build from flat layer arrays. `w[l]` is `din[l]*dout[l]` row-major (element (i,j) at i*dout+j),
    /// `b[l]` is len `dout[l]`. `info_suits` must match the checkpoint's POKERBOT_INFO_SUITS.
    pub fn new(
        w0: Vec<f32>,
        b0: Vec<f32>,
        w1: Vec<f32>,
        b1: Vec<f32>,
        w2: Vec<f32>,
        b2: Vec<f32>,
        din: [usize; 3],
        dout: [usize; 3],
        info_suits: bool,
    ) -> Net {
        Net {
            w: [w0, w1, w2],
            b: [b0, b1, b2],
            din,
            dout,
            info_suits,
        }
    }
}

#[inline]
fn dense_relu(x: &[f32], w: &[f32], b: &[f32], din: usize, dout: usize, relu: bool) -> Vec<f32> {
    let mut out = vec![0.0f32; dout];
    for j in 0..dout {
        out[j] = b[j];
    }
    for i in 0..din {
        let xi = x[i];
        if xi == 0.0 {
            continue;
        }
        let row = &w[i * dout..(i + 1) * dout];
        for j in 0..dout {
            out[j] += xi * row[j];
        }
    }
    if relu {
        for v in out.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
    }
    out
}

/// Port of `encode_info_state(state, seat)` -> 256-d feature (zero-padded).
pub fn encode_info_state(s: &EnvState, seat: usize) -> [f32; INFO_DIM] {
    let mut f = [0.0f32; INFO_DIM];
    let mut k = 0usize;
    let push = |f: &mut [f32; INFO_DIM], k: &mut usize, v: f32| {
        if *k < INFO_DIM {
            f[*k] = v;
        }
        *k += 1;
    };

    // 1. one_hot(max(street,0), NUM_STREETS+1) -- 5
    let street = s.street.max(0) as usize;
    for i in 0..(NUM_STREETS + 1) {
        push(&mut f, &mut k, if i == street { 1.0 } else { 0.0 });
    }
    // 2. one_hot(seat, 6)
    for i in 0..NUM_PLAYERS {
        push(&mut f, &mut k, if i == seat { 1.0 } else { 0.0 });
    }
    // 3. one_hot(button, 6)
    let button = s.button;
    for i in 0..NUM_PLAYERS {
        push(&mut f, &mut k, if i as i32 == button { 1.0 } else { 0.0 });
    }
    // 4-6. stacks / committed_street / committed_total, all / SCALE
    for i in 0..NUM_PLAYERS {
        push(&mut f, &mut k, s.stacks[i] as f32 / SCALE);
    }
    for i in 0..NUM_PLAYERS {
        push(&mut f, &mut k, s.committed_street[i] as f32 / SCALE);
    }
    for i in 0..NUM_PLAYERS {
        push(&mut f, &mut k, s.committed_total[i] as f32 / SCALE);
    }
    // 7. folded mask
    for i in 0..NUM_PLAYERS {
        push(
            &mut f,
            &mut k,
            if s.seat_status[i] == SEAT_FOLDED {
                1.0
            } else {
                0.0
            },
        );
    }
    // 8. all-in mask
    for i in 0..NUM_PLAYERS {
        push(
            &mut f,
            &mut k,
            if s.seat_status[i] == SEAT_ALLIN {
                1.0
            } else {
                0.0
            },
        );
    }
    // 9. [pot_live/scale, current_bet/scale]
    let mut pot_live = s.pot_dead;
    for i in 0..NUM_PLAYERS {
        pot_live += s.committed_street[i];
    }
    push(&mut f, &mut k, pot_live as f32 / SCALE);
    push(&mut f, &mut k, s.current_bet as f32 / SCALE);
    // 10. hole rank features (present * rank/14)
    for kk in 0..2 {
        let c = s.holes[seat][kk];
        let v = if c >= 0 {
            core::card_rank(c) as f32 / 14.0
        } else {
            0.0
        };
        push(&mut f, &mut k, v);
    }
    // 11. board rank features (present * rank/14)
    for i in 0..5 {
        let c = s.board[i];
        let v = if c >= 0 {
            core::card_rank(c) as f32 / 14.0
        } else {
            0.0
        };
        push(&mut f, &mut k, v);
    }
    debug_assert_eq!(k, BASE_FEATURE_LEN);
    // The suit-isomorphic tail (when info_suits) is written by `write_suit_features` in `probs`.
    f
}

/// Append `_suit_features` (9 values) starting at offset `k0`. Returns nothing; writes into `f`.
fn write_suit_features(s: &EnvState, seat: usize, f: &mut [f32; INFO_DIM], k0: usize) {
    let holes = s.holes[seat];
    let h_present = [(holes[0] >= 0) as i32 as f32, (holes[1] >= 0) as i32 as f32];
    let mut b_present = [0.0f32; 5];
    for i in 0..5 {
        b_present[i] = (s.board[i] >= 0) as i32 as f32;
    }
    let h_suit = [
        core::card_suit(holes[0].max(0)),
        core::card_suit(holes[1].max(0)),
    ];
    let suited = h_present[0] * h_present[1] * (if h_suit[0] == h_suit[1] { 1.0 } else { 0.0 });
    let mut h_counts = [0.0f32; 4];
    for kk in 0..2 {
        let suit = core::card_suit(holes[kk].max(0)) as usize;
        h_counts[suit] += h_present[kk];
    }
    let mut b_counts = [0.0f32; 4];
    for i in 0..5 {
        let suit = core::card_suit(s.board[i].max(0)) as usize;
        b_counts[suit] += b_present[i];
    }
    let mut board_sorted = b_counts;
    sort_desc4(&mut board_sorted);
    let mut combined = [0.0f32; 4];
    for s4 in 0..4 {
        combined[s4] = h_counts[s4] + b_counts[s4];
    }
    sort_desc4(&mut combined);
    let mut idx = k0;
    let put = |f: &mut [f32; INFO_DIM], idx: &mut usize, v: f32| {
        if *idx < INFO_DIM {
            f[*idx] = v;
        }
        *idx += 1;
    };
    put(f, &mut idx, suited);
    for s4 in 0..4 {
        put(f, &mut idx, board_sorted[s4] / 5.0);
    }
    for s4 in 0..4 {
        put(f, &mut idx, combined[s4] / 7.0);
    }
}

#[inline]
fn sort_desc4(a: &mut [f32; 4]) {
    a.sort_unstable_by(|x, y| y.partial_cmp(x).unwrap_or(std::cmp::Ordering::Equal));
}

impl crate::lbr::LbrPolicy for Net {
    fn probs(&self, s: &EnvState, seat: usize) -> [f64; NUM_ACTIONS] {
        // encode (base features), then optionally the suit tail.
        let mut info = encode_info_state(s, seat);
        if self.info_suits {
            write_suit_features(s, seat, &mut info, BASE_FEATURE_LEN);
        }
        // forward
        let h0 = dense_relu(
            &info,
            &self.w[0],
            &self.b[0],
            self.din[0],
            self.dout[0],
            true,
        );
        let h1 = dense_relu(&h0, &self.w[1], &self.b[1], self.din[1], self.dout[1], true);
        let logits = dense_relu(
            &h1,
            &self.w[2],
            &self.b[2],
            self.din[2],
            self.dout[2],
            false,
        );
        // masked softmax over legal, then *mask renorm (matches make_neural_policy serve).
        let mask = core::legal_action_mask(s, seat as i32);
        let mut maxl = f32::NEG_INFINITY;
        for a in 0..NUM_ACTIONS {
            let l = if mask[a] { logits[a] } else { -1e9 };
            if l > maxl {
                maxl = l;
            }
        }
        let mut ex = [0.0f64; NUM_ACTIONS];
        let mut sum = 0.0f64;
        for a in 0..NUM_ACTIONS {
            let l = if mask[a] { logits[a] } else { -1e9 };
            let e = ((l - maxl) as f64).exp();
            ex[a] = e;
            sum += e;
        }
        let mut p = [0.0f64; NUM_ACTIONS];
        if sum > 0.0 {
            for a in 0..NUM_ACTIONS {
                p[a] = ex[a] / sum;
            }
        }
        // *mask + renorm (the softmax already ~zeros illegal via -1e9; this is the serve's belt-and-braces)
        let mut tot = 0.0f64;
        for a in 0..NUM_ACTIONS {
            if !mask[a] {
                p[a] = 0.0;
            }
            tot += p[a];
        }
        if tot > 1e-12 {
            for a in 0..NUM_ACTIONS {
                p[a] /= tot;
            }
        } else {
            // uniform over legal fallback
            let mut n = 0.0f64;
            for a in 0..NUM_ACTIONS {
                if mask[a] {
                    n += 1.0;
                }
            }
            let d = n.max(1.0);
            for a in 0..NUM_ACTIONS {
                p[a] = if mask[a] { 1.0 / d } else { 0.0 };
            }
        }
        p
    }
}

/// Base (non-suit) feature length: 5+6+6+6+6+6+6+6+2+2+5 = 56.
pub const BASE_FEATURE_LEN: usize = 56;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_feature_len_matches() {
        assert_eq!(BASE_FEATURE_LEN, 5 + 6 + 6 + 6 + 6 + 6 + 6 + 6 + 2 + 2 + 5);
    }

    #[test]
    fn dense_relu_basic() {
        // x=[1,2], W=[[1,0,1],[0,1,1]] (2x3), b=[0,0,0] -> [1,2,3], relu no-op.
        let x = vec![1.0f32, 2.0];
        let w = vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0];
        let b = vec![0.0, 0.0, 0.0];
        let o = dense_relu(&x, &w, &b, 2, 3, false);
        assert_eq!(o, vec![1.0, 2.0, 3.0]);
    }
}
