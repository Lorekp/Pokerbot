use battery_native::core::{self, NUM_PLAYERS};
use battery_native::playout::{self, Blueprint, PlayoutConfig};
fn main() {
    let stacks = [10000i32; NUM_PLAYERS];
    let occ = [true; NUM_PLAYERS];
    let mut deck = [0i32; 52];
    for i in 0..52 { deck[i] = i as i32; }
    let mut s = core::init_hand(0, &stacks, &occ, 0, &deck, [1, 2]);
    // play check/call to the river, stop right AFTER the river close (pending_deal chance node)
    let mut steps = 0;
    let mut leaf = None;
    while !s.done && steps < 80 {
        if core::is_chance_node(&s) {
            if s.street == 3 {
                leaf = Some(s); // pending street close ON the river => next chance = terminal
                break;
            }
            s = core::sample_chance(&s);
            continue;
        }
        if s.action_seat < 0 { break; }
        let m = core::legal_action_mask(&s, s.action_seat);
        let a = if m[core::CHECK] { core::CHECK } else if m[core::CALL] { core::CALL } else { core::FOLD };
        s = core::apply_action(&s, a);
        steps += 1;
    }
    let leaf = leaf.expect("no river-close leaf reached");
    println!("leaf: street={} pending={} pot_dead={} holes0={:?} board={:?}",
        leaf.street, leaf.pending_deal, leaf.pot_dead, leaf.holes[0], leaf.board);
    let bp: Blueprint = Blueprint::default();
    let cfg = PlayoutConfig {
        allowed_pre: [true; 11], allowed_post: [true; 11],
        bias_mult: 10.0, bias_floor: 0.0, max_depth: 64, num_buckets: 160, emd: None,
    };
    for hero in 0..2 {
        let v = playout::uniform_leaf_value(&leaf, hero, &bp, &cfg, 42, 4, true);
        println!("hero={hero} leaf_value={v}");
    }
}
