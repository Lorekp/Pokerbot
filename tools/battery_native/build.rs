use std::{env, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=gen_phe_tables.py");
    println!("cargo:rerun-if-changed=src/phe_tables.rs");
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
    println!("cargo:rerun-if-env-changed=PYTHON");
    let python = env::var("PYO3_PYTHON")
        .or_else(|_| env::var("PYTHON"))
        .unwrap_or_else(|_| "python".to_owned());
    let status = Command::new(python)
        .arg("gen_phe_tables.py")
        .current_dir(env::var("CARGO_MANIFEST_DIR").unwrap())
        .status()
        .expect("Run with Python installed and phevaluator==0.5.3.1 available");
    assert!(status.success(), "Table generation failed; install phevaluator==0.5.3.1 for the selected Python");
}
