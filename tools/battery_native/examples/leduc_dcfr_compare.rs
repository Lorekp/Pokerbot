//! FREE, local, EXACT measurement of the DCFR convergence-rate lever on Leduc -- the clean read the 6-max
//! LBR cert could NOT give (6-max LBR is non-monotone in convergence; Leduc info-set-BR exploitability is
//! exact AND monotone). Trains CFR+ and DCFR INCREMENTALLY (one Tables each, continued across milestones,
//! so the curve is a single run per arm), printing exploitability (mbb, lower = closer to Nash) side by
//! side + the iterations each arm needs to cross a set of accuracy targets. If DCFR reaches a target in
//! fewer iterations, that is the convergence-speed lever showing signal on a game where the metric is
//! trustworthy -- the gate for whether to spend on the cloud 6-max DCFR-arm run.
//!
//!   cargo run --release --example leduc_dcfr_compare --manifest-path tools/battery_native/Cargo.toml

use battery_native::leduc;

fn main() {
    let milestones: [u64; 11] = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600];
    let targets = [10.0f64, 1.0, 0.1, 0.01];

    println!("Leduc DCFR-vs-CFR+ EXACT info-set-BR exploitability (mbb; lower = closer to Nash)");
    println!("{:>8} | {:>13} | {:>13} | {:>10}", "iters", "CFR+", "DCFR", "DCFR/CFR+");

    let mut cfr = leduc::Tables::default();
    let mut dcfr = leduc::Tables::default();
    let mut prev = 0u64;
    // first iter index at which each arm is < target  (None until crossed)
    let mut cfr_hit = [Option::<u64>::None; 4];
    let mut dcfr_hit = [Option::<u64>::None; 4];

    for &m in &milestones {
        leduc::train(&mut cfr, m - prev);
        leduc::train_dcfr(&mut dcfr, m - prev);
        prev = m;
        let (ec, ..) = leduc::infoset_exploitability_mbb(&leduc::average_strategy(&cfr));
        let (ed, ..) = leduc::infoset_exploitability_mbb(&leduc::average_strategy(&dcfr));
        for (i, &tg) in targets.iter().enumerate() {
            if cfr_hit[i].is_none() && ec < tg {
                cfr_hit[i] = Some(m);
            }
            if dcfr_hit[i].is_none() && ed < tg {
                dcfr_hit[i] = Some(m);
            }
        }
        let ratio = if ed > 0.0 { ec / ed } else { f64::INFINITY };
        println!("{:>8} | {:>13.5} | {:>13.5} | {:>9.2}x", m, ec, ed, ratio);
    }

    println!("\niterations to reach each accuracy target (lower = faster convergence):");
    println!(
        "{:>10} | {:>12} | {:>12} | {:>12}",
        "target_mbb", "CFR+", "DCFR", "speedup"
    );
    for (i, &tg) in targets.iter().enumerate() {
        let c = cfr_hit[i];
        let d = dcfr_hit[i];
        let sp = match (c, d) {
            (Some(c), Some(d)) if d > 0 => format!("{:.2}x", c as f64 / d as f64),
            _ => "n/a".to_string(),
        };
        println!(
            "{:>10.2} | {:>12} | {:>12} | {:>12}",
            tg,
            c.map(|x| x.to_string()).unwrap_or_else(|| ">max".into()),
            d.map(|x| x.to_string()).unwrap_or_else(|| ">max".into()),
            sp
        );
    }
    println!(
        "\nVERDICT INPUT: DCFR converges faster iff speedup > 1.0 at the tighter targets (0.1/0.01 mbb). \
         A clean DCFR win here is the gate to spend on the cloud 6-max DCFR-arm wall-clock + paired-LBR run."
    );
}
