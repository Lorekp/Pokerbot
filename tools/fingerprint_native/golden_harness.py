"""Golden-vector parity harness for the blueprint infoset FINGERPRINT hot path.

This is the GO/NO-GO bit-identity instrument for a future native (Rust) port of
``pokerbot.engine.jax_weva_fast.fingerprint_fast`` — exactly analogous to what
``tools/aivat_native/golden/golden.json`` + ``pokerbot.eval.aivat_native.verify_against_golden``
are for the AIVAT control-variate port.

What it does
------------
* GENERATE: play a large STRATIFIED sample of random NLHE hands through the pure-NumPy
  engine (``pokerbot.engine.np_engine``) and, for every visited decision node, capture
    - the FULL flat ``EnvState`` buffer (base64 of the little-endian field bytes),
    - the acting/dealt-in seat,
    - the active abstraction config, and
    - ``fingerprint_fast(state, seat)`` output bytes as HEX.
  The sample covers preflop + flop + turn + river, all 6 positions, varied occupancy /
  stacks (HU..6-max, short stacks -> all-ins / side pots / varied SPR), and BOTH
  ``POKERBOT_PREFLOP_LOSSLESS=0/1`` on top of the deployed
  ``POKERBOT_BETTING_ABSTRACTION=1`` / ``POKERBOT_POSTFLOP_BUCKETS=1`` config.
* VERIFY: recompute ``fingerprint_fast`` for every captured (state, seat) and assert
  BYTE-EQUALITY against the stored golden — mirroring ``verify_against_golden``'s structure
  (per-section loop, fh/bits-style decode helpers, AssertionError-on-first-mismatch,
  per-section pass/fail dict).

Why subprocesses
----------------
The abstraction flags are read from ``os.environ`` at IMPORT time and BAKED into module
globals (``jax_weva_encode._BETTING_ABSTRACTION``, ``weva_encoder._PREFLOP_LOSSLESS``,
``active_abstraction_config()``), so a single process can only ever represent ONE config.
Each config is therefore generated / verified in its own freshly-spawned subprocess whose
environment is set BEFORE any ``pokerbot`` import. The orchestrator process itself imports
NO engine code — it only spawns workers and stitches/reads JSON.

NO 600 MB blueprint pickle is ever loaded (the fingerprint is a pure function of the
``EnvState`` + the baked abstraction flags — it is independent of any trained snapshot), so
this runs comfortably on native Windows python. No ``aof_*`` file is touched.

Usage
-----
    python tools/fingerprint_native/golden_harness.py generate [--states 7000] [--seed 12345]
    python tools/fingerprint_native/golden_harness.py verify

``verify_against_golden()`` is also importable as the regression gate (see
``tests/test_fingerprint_golden.py``).
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths / config matrix (orchestrator-side — imports NO engine code).
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
# Canonical artifact is gzip-compressed: the full EnvState buffers carry a lot of zero/-1
# history+deck padding that compresses ~5-8x, keeping the committed golden small. IO below is
# transparently gzip-aware by ``.gz`` suffix, so a plain ``golden.json`` also works.
GOLDEN_PATH = _HERE / "golden" / "golden.json.gz"


def _read_json_any(path: str | Path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="ascii") as f:
            return json.load(f)
    return json.loads(path.read_text(encoding="ascii"))


def _write_json_any(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="ascii") as f:
            json.dump(obj, f)
    else:
        path.write_text(json.dumps(obj), encoding="ascii")

# Every config explicitly pins ALL six abstraction flags so ambient POKERBOT_* env can never
# leak in and silently change what is baked. The deployed serving config is
# BETTING_ABSTRACTION=1 + POSTFLOP_BUCKETS=1 (fast mode, 200 buckets); we sweep the preflop
# representation across its three meaningful settings so every branch of ``fingerprint_fast._digest``
# is exercised:
#   * preflop_lossless=0                      -> WEVA k-means preflop bucket  + bounded betting_state
#   * preflop_lossless=1                      -> lossless 169-class bucket     + bounded betting_state
#   * preflop_lossless=1 + lossless_betting=1 -> lossless 169-class bucket     + lossless history_key
_DEPLOYED_BASE = {
    "POKERBOT_BETTING_ABSTRACTION": "1",
    "POKERBOT_POSTFLOP_BUCKETS": "1",
    "POKERBOT_POSTFLOP_MODE": "fast",
    "POKERBOT_POSTFLOP_NUM_BUCKETS": "200",
    "POKERBOT_PREFLOP_LOSSLESS": "0",
    "POKERBOT_PREFLOP_LOSSLESS_BETTING": "0",
    # Keep workers on CPU + deterministic; avoid GPU init noise.
    "JAX_PLATFORMS": "cpu",
}


def _cfg(name: str, **overrides: str) -> dict:
    env = dict(_DEPLOYED_BASE)
    env.update(overrides)
    return {"name": name, "env": env}


CONFIGS = [
    _cfg("deployed"),  # PREFLOP_LOSSLESS=0  (required)
    _cfg("deployed_preflop_lossless", POKERBOT_PREFLOP_LOSSLESS="1"),  # PREFLOP_LOSSLESS=1 (required)
    _cfg(
        "deployed_preflop_lossless_betting",
        POKERBOT_PREFLOP_LOSSLESS="1",
        POKERBOT_PREFLOP_LOSSLESS_BETTING="1",
    ),  # exercises the lossless-history preflop branch
]

DEFAULT_STATES_PER_CONFIG = 7000  # 3 configs -> >= 20k EnvStates total

# Streets to stratify across (engine STREET_* ints).
_STREET_NAMES = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}


# ===========================================================================
# WORKER SIDE — runs inside a subprocess whose env already bakes one config.
# Everything below the import fence may touch pokerbot engine code.
# ===========================================================================
def _load_engine():
    """Import the engine AFTER env flags are baked. Returns a small namespace object."""
    import numpy as np

    from pokerbot.engine import jax_engine as je
    from pokerbot.engine import np_engine as ne
    from pokerbot.engine.jax_weva_fast import fingerprint_fast
    from pokerbot.engine.jax_weva_encode import active_abstraction_config

    class _Eng:
        pass

    e = _Eng()
    e.np = np
    e.je = je
    e.ne = ne
    e.fingerprint_fast = fingerprint_fast
    e.active_abstraction_config = active_abstraction_config
    e.NUM_PLAYERS = int(je.NUM_PLAYERS)
    e.CHIP_UNIT = int(je.CHIP_UNIT)
    e.BB_CHIPS = int(je.BB_CHIPS)
    e.START_STACK = int(je.STARTING_STACK_CHIPS)
    e.FOLD = int(je.FOLD)
    e.CHECK = int(je.CHECK)
    e.CALL = int(je.CALL)
    e.ALL_IN = int(je.ALL_IN)
    e.BET_BUCKET_START = int(je.BET_BUCKET_START)
    e.SEAT_EMPTY = int(ne.SEAT_EMPTY)
    return e


# --- Flat EnvState <-> bytes (canonical, schema-driven) --------------------
def _state_schema(state, eng) -> dict:
    """Field-ordered dtype+shape schema for an ``EnvState`` (derived from a real numpy state)."""
    np = eng.np
    fields = []
    for name in state._fields:
        arr = np.asarray(getattr(state, name))
        fields.append({"name": name, "dtype": arr.dtype.str, "shape": list(arr.shape)})
    return {"field_order": [f["name"] for f in fields], "fields": fields}


def _state_to_b64(state, schema, eng) -> str:
    """Serialize an ``EnvState`` to base64 of its concatenated little-endian field bytes."""
    np = eng.np
    chunks = []
    for f in schema["fields"]:
        arr = np.asarray(getattr(state, f["name"]))
        # Defensive: enforce the schema dtype so endianness/width is canonical.
        arr = arr.astype(np.dtype(f["dtype"]), copy=False)
        chunks.append(np.ascontiguousarray(arr).tobytes())
    return base64.b64encode(b"".join(chunks)).decode("ascii")


def _b64_to_state(b64: str, schema, eng):
    """Reconstruct a numpy-backed ``EnvState`` from a base64 buffer + schema."""
    np = eng.np
    raw = base64.b64decode(b64.encode("ascii"))
    off = 0
    kw = {}
    for f in schema["fields"]:
        dt = np.dtype(f["dtype"])
        shape = tuple(f["shape"])
        n = int(np.prod(shape)) if shape else 1
        nbytes = n * dt.itemsize
        seg = raw[off : off + nbytes]
        off += nbytes
        arr = np.frombuffer(seg, dtype=dt).reshape(shape).copy()
        kw[f["name"]] = arr
    assert off == len(raw), f"buffer length mismatch: consumed {off} != {len(raw)}"
    return eng.je.EnvState(**kw)


# --- Stratified hand playout / capture -------------------------------------
def _random_occupancy(rng, eng):
    """Random dealt-in seat mask with 2..6 live seats (covers HU / 3-max / .. / 6-max)."""
    np = eng.np
    n = int(rng.integers(2, eng.NUM_PLAYERS + 1))
    seats = rng.choice(eng.NUM_PLAYERS, size=n, replace=False)
    occ = np.zeros(eng.NUM_PLAYERS, dtype=bool)
    occ[seats] = True
    return occ


def _random_stacks(rng, eng, occ):
    """Occasionally hand back varied per-seat stacks (chips) to create all-ins / side pots / SPR spread."""
    np = eng.np
    if rng.random() < 0.70:
        return None  # default deep stacks
    # 2bb .. 200bb in chip units; short stacks especially produce all-in / side-pot states.
    lo, hi = 2 * eng.BB_CHIPS, 200 * eng.BB_CHIPS
    stacks = rng.integers(lo, hi + 1, size=eng.NUM_PLAYERS).astype(np.int64)
    return stacks


def _pick_action(rng, legal, eng):
    """Bias toward CHECK/CALL so hands survive deeper (more flop/turn/river), but keep
    enough aggression + folds to exercise the pot / facing / SPR betting buckets."""
    np = eng.np
    passive = [a for a in legal if a in (eng.CHECK, eng.CALL)]
    aggressive = [a for a in legal if a >= eng.BET_BUCKET_START]  # bet buckets + all-in
    r = rng.random()
    if passive and r < 0.55:
        return int(rng.choice(passive))
    if aggressive and r < 0.85:
        return int(rng.choice(aggressive))
    return int(rng.choice(legal))


def _dealt_in_seats(state, eng):
    np = eng.np
    ss = np.asarray(state.seat_status)
    return [s for s in range(eng.NUM_PLAYERS) if int(ss[s]) != eng.SEAT_EMPTY]


def generate_section(states_target: int, seed: int) -> dict:
    """Play stratified random hands in THIS process's baked config and capture fingerprints.

    Returns a section dict: {config, env_flags, schema, states:[{buf, seats:[{seat,out}]}]}.
    """
    eng = _load_engine()
    np = eng.np
    je, ne = eng.je, eng.ne
    import jax  # noqa: WPS433 (needed only for init_hand)
    import jax.numpy as jnp

    cfg = eng.active_abstraction_config()
    per_street = max(1, states_target // 4)
    quota = {0: per_street, 1: per_street, 2: per_street, 3: per_street}
    # Give the last street(s) any rounding remainder via preflop (always abundant).
    quota[0] += states_target - sum(quota.values())
    got = {0: 0, 1: 0, 2: 0, 3: 0}

    schema = None
    out_states: list[dict] = []
    rng = np.random.default_rng(seed)
    max_hands = states_target * 80  # generous: river is rare under random play
    hand = 0
    t0 = time.time()

    def _need_more() -> bool:
        return any(got[s] < quota[s] for s in quota)

    while _need_more() and hand < max_hands:
        hand += 1
        key = jax.random.PRNGKey(int(rng.integers(0, 2**31 - 1)))
        occ = _random_occupancy(rng, eng)
        stacks = _random_stacks(rng, eng, occ)
        occ_arr = jnp.asarray(occ, dtype=bool)
        if stacks is None:
            s_jax = je.init_hand(key, occupancy=occ_arr)
        else:
            s_jax = je.init_hand(key, stacks=jnp.asarray(stacks, dtype=jnp.int32), occupancy=occ_arr)
        s = ne.to_numpy_state(s_jax)
        if schema is None:
            schema = _state_schema(s, eng)

        steps = 0
        while not bool(np.asarray(s.done)):
            steps += 1
            if steps > 400:
                break  # runaway guard (matches np_engine equiv test)
            if ne.is_chance_node_np(s):
                s = ne.sample_chance_np(s)
                continue
            street = int(np.asarray(s.street))
            if street in quota and got[street] < quota[street]:
                seats = _dealt_in_seats(s, eng)
                cases = []
                for seat in seats:
                    fp = eng.fingerprint_fast(s, seat)
                    cases.append({"seat": int(seat), "out": fp.hex()})
                out_states.append({"buf": _state_to_b64(s, schema, eng), "seats": cases})
                got[street] += 1
            mask = np.asarray(ne.legal_action_mask_np(s))
            legal = np.nonzero(mask)[0].tolist()
            if not legal:
                break
            action = _pick_action(rng, legal, eng)
            s = ne.apply_action_np(s, action)

        if hand % 2000 == 0:
            elapsed = time.time() - t0
            sys.stderr.write(
                f"[{cfg}] hand={hand} captured={sum(got.values())} "
                f"per-street={got} ({elapsed:.0f}s)\n"
            )
            sys.stderr.flush()

    underfilled = {
        _STREET_NAMES[s]: [got[s], quota[s]] for s in quota if got[s] < quota[s]
    }
    return {
        "schema": schema,
        "config": cfg,
        "env_flags": {k: os.environ.get(k, "") for k in _DEPLOYED_BASE},
        "hands_played": hand,
        "per_street": {_STREET_NAMES[s]: got[s] for s in quota},
        "underfilled": underfilled,
        "states": out_states,
    }


def verify_section(golden: dict, section_idx: int) -> dict:
    """Recompute fingerprint_fast for every (state, seat) in one section; assert byte-equality.

    Runs in a subprocess whose env bakes the section's config. Mirrors the per-case loop of
    ``aivat_native.verify_against_golden``."""
    eng = _load_engine()
    section = golden["sections"][section_idx]
    schema = golden["schema"]

    # Provenance guard: the baked config MUST match what was stamped at capture, else we are
    # silently verifying the wrong abstraction (the train/serve skew this pipeline has hit before).
    live_cfg = eng.active_abstraction_config()
    assert live_cfg == section["config"], (
        f"section[{section_idx}] '{section['name']}' config drift: "
        f"baked={live_cfg} != golden={section['config']}"
    )

    n_states = 0
    n_cases = 0
    for si, st in enumerate(section["states"]):
        state = _b64_to_state(st["buf"], schema, eng)
        for c in st["seats"]:
            got = eng.fingerprint_fast(state, int(c["seat"])).hex()
            want = c["out"]
            assert got == want, (
                f"section[{section_idx}] '{section['name']}' state[{si}] seat={c['seat']}: "
                f"{got} != {want}"
            )
            n_cases += 1
        n_states += 1
    return {"name": section["name"], "states": n_states, "cases": n_cases, "ok": True}


# ===========================================================================
# WORKER ENTRYPOINTS (invoked as subprocesses by the orchestrator).
# ===========================================================================
def _worker_generate(args) -> None:
    section = generate_section(args.states, args.seed)
    Path(args.out).write_text(json.dumps(section), encoding="ascii")


def _worker_verify(args) -> None:
    golden = _read_json_any(args.golden)
    res = verify_section(golden, args.section)
    Path(args.out).write_text(json.dumps(res), encoding="ascii")


# ===========================================================================
# ORCHESTRATOR SIDE — spawns one worker per config; imports NO engine code.
# ===========================================================================
def _spawn(argv: list[str], env_overrides: dict) -> None:
    env = dict(os.environ)
    env.update(env_overrides)
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), *argv],
        env=env,
        capture_output=True,
        text=True,
    )
    # Surface the worker's stderr progress regardless of outcome.
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"worker failed ({' '.join(argv)}): exit {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
        )


def generate_golden(golden_path: Path, states_per_config: int, seed: int, verbose: bool = True) -> dict:
    """Spawn one generator subprocess per config and stitch the shards into golden.json."""
    import numpy as np  # orchestrator may use numpy for version stamp only

    sections = []
    schema = None
    with tempfile.TemporaryDirectory() as td:
        for i, cfg in enumerate(CONFIGS):
            shard = Path(td) / f"shard_{i}.json"
            if verbose:
                print(f"[generate] config '{cfg['name']}' -> {states_per_config} states ...")
            _spawn(
                [
                    "_worker-generate",
                    "--out", str(shard),
                    "--states", str(states_per_config),
                    "--seed", str(seed + i * 104729),  # distinct prime offset per config
                ],
                cfg["env"],
            )
            sec = json.loads(shard.read_text(encoding="ascii"))
            if schema is None:
                schema = sec.pop("schema")
            else:
                sec.pop("schema", None)
            sec_out = {"name": cfg["name"], **sec}
            sections.append(sec_out)
            if verbose:
                tot = sum(len(s["seats"]) for s in sec_out["states"])
                print(
                    f"           captured {len(sec_out['states'])} states / {tot} cases "
                    f"({sec_out['per_street']}); underfilled={sec_out['underfilled']}"
                )

    golden = {
        "harness": "blueprint_fingerprint_golden",
        "fingerprint_fn": "pokerbot.engine.jax_weva_fast.fingerprint_fast",
        "numpy_version": np.__version__,
        "buffer_encoding": "base64-le-concat-fields",
        "out_encoding": "hex",
        "schema": schema,
        "sections": sections,
    }
    _write_json_any(golden_path, golden)
    total_states = sum(len(s["states"]) for s in sections)
    total_cases = sum(len(st["seats"]) for s in sections for st in s["states"])
    if verbose:
        print(
            f"[generate] wrote {golden_path} "
            f"({len(sections)} sections, {total_states} states, {total_cases} (state,seat) cases)"
        )
    return golden


def verify_against_golden(golden_path: str | Path | None = None, verbose: bool = True) -> dict:
    """Assert ``fingerprint_fast`` output == stored golden EXACTLY (byte equality) for every case.

    One subprocess per section (each bakes that section's abstraction config). Returns a
    per-section pass/fail dict; raises (via the worker) on the first mismatch. Structure mirrors
    ``pokerbot.eval.aivat_native.verify_against_golden``.
    """
    gp = Path(golden_path) if golden_path else GOLDEN_PATH
    golden = _read_json_any(gp)

    results: dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as td:
        for i, section in enumerate(golden["sections"]):
            cfg_env = next((c["env"] for c in CONFIGS if c["name"] == section["name"]), None)
            if cfg_env is None:
                # Fall back to the flags stamped in the golden itself.
                cfg_env = dict(_DEPLOYED_BASE)
                cfg_env.update(section.get("env_flags", {}))
            res_path = Path(td) / f"res_{i}.json"
            _spawn(
                [
                    "_worker-verify",
                    "--golden", str(gp),
                    "--section", str(i),
                    "--out", str(res_path),
                ],
                cfg_env,
            )
            res = json.loads(res_path.read_text(encoding="ascii"))
            results[section["name"]] = res

    if verbose:
        nv = golden.get("numpy_version", "?")
        print(f"[fingerprint_golden] numpy={nv}  fn={golden['fingerprint_fn']}")
        for name, r in results.items():
            print(
                f"  {name:36s}  {r['states']:6d} states  {r['cases']:7d} cases  OK (byte-identical)"
            )
    results["_ok"] = True
    return results


# ===========================================================================
# CLI
# ===========================================================================
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate golden.json (spawns one worker per config)")
    g.add_argument("--states", type=int, default=DEFAULT_STATES_PER_CONFIG,
                   help=f"EnvStates to capture per config (default {DEFAULT_STATES_PER_CONFIG}; x{len(CONFIGS)} configs)")
    g.add_argument("--seed", type=int, default=12345)
    g.add_argument("--out", type=str, default=str(GOLDEN_PATH))

    v = sub.add_parser("verify", help="verify golden.json byte-for-byte (spawns one worker per section)")
    v.add_argument("--golden", type=str, default=str(GOLDEN_PATH))

    # Hidden worker entrypoints (run inside per-config subprocesses).
    wg = sub.add_parser("_worker-generate")
    wg.add_argument("--out", required=True)
    wg.add_argument("--states", type=int, required=True)
    wg.add_argument("--seed", type=int, required=True)

    wv = sub.add_parser("_worker-verify")
    wv.add_argument("--golden", required=True)
    wv.add_argument("--section", type=int, required=True)
    wv.add_argument("--out", required=True)

    args = p.parse_args(argv)

    if args.cmd == "generate":
        generate_golden(Path(args.out), args.states, args.seed)
        return 0
    if args.cmd == "verify":
        verify_against_golden(args.golden)
        return 0
    if args.cmd == "_worker-generate":
        _worker_generate(args)
        return 0
    if args.cmd == "_worker-verify":
        _worker_verify(args)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
