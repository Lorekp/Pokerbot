use battery_native::core::{self, NUM_PLAYERS};
fn main() {
    let stacks = [10000i32; NUM_PLAYERS];
    let occ = [true; NUM_PLAYERS];
    let mut deck = [0i32; 52];
    for i in 0..52 { deck[i] = i as i32; }
    let mut s = core::init_hand(0, &stacks, &occ, 0, &deck, [1, 2]);
    let mut steps = 0;
    while !s.done && steps < 80 {
        if core::is_chance_node(&s) { s = core::sample_chance(&s); continue; }
        let seat = s.action_seat;
        if seat < 0 { println!("actor=-1 street={} done={} rewards={:?}", s.street, s.done, s.rewards); break; }
        let m = core::legal_action_mask(&s, seat);
        let a = if m[core::CHECK] { core::CHECK } else if m[core::CALL] { core::CALL } else { core::FOLD };
        s = core::apply_action(&s, a);
        steps += 1;
    }
    println!("street={} done={} rewards={:?}", s.street, s.done, s.rewards);
}
