"""Phase 5 post-Phase-4 parity tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pokerbot.encoding.weva_cfr import DEFAULT_SCENARIOS
from pokerbot.eval.gate_reference import (
    append_search_gate_result,
    record_phase2_gate_pass,
    should_use_cfr_hero,
)
from pokerbot.models.policy import PseudoSiamesePolicy
from pokerbot.opponent.belief_br import belief_br_exploit_logits, belief_br_from_transformer_output
from pokerbot.opponent.sequence_encoder import opponent_index_for_seat
from pokerbot.search.isomorphism import canonicalize_history_encoding
from pokerbot.search.pbs_tracker import NUM_BELIEF_COMBOS, PBSTracker


def test_weva_default_scenarios_extended():
    assert len(DEFAULT_SCENARIOS) >= 7


def test_belief_br_weighted_logits():
    belief = np.zeros(NUM_BELIEF_COMBOS, dtype=np.float64)
    belief[0] = 0.6
    belief[1] = 0.4
    pol = np.random.randn(NUM_BELIEF_COMBOS, 11).astype(np.float64)
    out = belief_br_exploit_logits(belief, pol, top_k=2)
    assert out.shape == (11,)


def test_opponent_index_for_seat_wraps_correctly():
    # hero=3, opponent=1 -> offset 4 -> opp_idx 3 (not 2 from buggy % 5)
    assert opponent_index_for_seat(1, 3) == 3
    assert opponent_index_for_seat(4, 3) == 0
    assert opponent_index_for_seat(5, 0) == 4


def test_belief_br_broadcast_fallback_is_not_weighted_noop():
    belief = torch.zeros(1326)
    belief[0] = 10.0
    belief[1] = 5.0
    belief_logits = belief.view(1, 1, -1).expand(1, 5, -1)
    policy_logits = torch.randn(1, 5, 11)
    fallback = belief_br_from_transformer_output(belief_logits, policy_logits, opp_idx=0)
    assert fallback.shape == (11,)
    assert np.allclose(fallback, policy_logits[0, 0].numpy())


def test_chip_iso_encoding_buckets():
    hist = np.zeros(128, dtype=np.float32)
    out = canonicalize_history_encoding(hist, pot_bb=10.0, spr_bucket=3)
    assert out[-2] == 3.0
    assert out[-1] == 10.0


def test_pbs_cross_seat_consistency():
    state = __import__("pokerbot.engine.game", fromlist=["GameState"]).GameState.new_hand(button_seat=0)
    state.deal_hole_cards()
    pbs = PBSTracker.init_uniform(0, state.players[0].hole, state.board)
    pbs.beliefs[1] = 0.0
    pbs.beliefs[1, 0] = 1.0
    pbs._enforce_cross_seat_consistency()
    assert pbs.beliefs[2].sum() == pytest.approx(1.0, rel=1e-4)


def test_phase2_gate_record(tmp_path):
    results = {"gate_passed": True}
    results_path = tmp_path / "eval_phase2_compare.json"
    results_path.write_text(json.dumps(results), encoding="utf-8")
    cfr = tmp_path / "policy_cfr_blueprint.pt"
    model = PseudoSiamesePolicy(num_weva_buckets=100)
    torch.save({"model": model.state_dict(), "num_weva_buckets": 100}, cfr)
    manifest_path = tmp_path / "gate_reference.json"

    record_phase2_gate_pass(seed=42, results_path=results_path, manifest_path=manifest_path, cfr_hero_path=cfr)
    record_phase2_gate_pass(seed=99, results_path=results_path, manifest_path=manifest_path, cfr_hero_path=cfr)
    assert should_use_cfr_hero(manifest_path)


def test_append_search_gate_result(tmp_path):
    path = tmp_path / "search_gate_results.json"
    append_search_gate_result({"gate_passed": False, "seed": 1}, path=path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["runs"]) == 1


def test_toolpoker_client_cache(tmp_path):
    from pokerbot.data.toolpoker_client import ToolPokerClient, ToolPokerClientConfig

    cfg = ToolPokerClientConfig(cache_dir=tmp_path / "cache")
    client = ToolPokerClient(cfg)
    spot = {"spot_id": "test", "action_distribution": {}}
    cache = cfg.cache_dir / "test.jsonl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(spot) + "\n", encoding="utf-8")
    loaded = client.fetch_spot("test")
    assert loaded["spot_id"] == "test"


def test_bootstrap_gate_smoke_script(tmp_path):
    import subprocess
    import sys

    out = tmp_path / "ckpt"
    subprocess.run(
        [sys.executable, "scripts/bootstrap_gate_smoke.py", "--output-dir", str(out)],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert (out / "smoke_hero_weva.pt").exists()
    assert (out / "smoke_madt.pt").exists()
    assert (out / "smoke_hvn.pkl").exists()
