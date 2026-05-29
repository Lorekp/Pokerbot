"""Tests for AoF FastAPI inference server."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from fastapi.testclient import TestClient

from aof_rl.game_state import game_state_from_dict, game_state_to_dict
from aof_rl.inference.engine import InferenceEngine, ServerConfig
from aof_rl.inference.manifest import ManifestError, ModelManifest, verify_manifest
from aof_rl.inference.ood import OodConfig, OodMonitor, flatten_obs_features
from aof_rl.inference.app import create_app
from aof_rl.policy_v3 import AoFPolicyNetV3, EncoderConfig
from aof_rl.registry import save_league_checkpoint
from aof_rl.teacher import TeacherResult

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "aof_inference"


@pytest.fixture
def sample_game_state() -> dict:
    return json.loads((FIXTURES / "sample_game_state.json").read_text(encoding="utf-8"))


@pytest.fixture
def checkpoint_path(tmp_path: Path) -> Path:
    model = AoFPolicyNetV3(EncoderConfig(encoder_type="deep_sets", d_model=32, trunk_hidden=32))
    ckpt = tmp_path / "policy_test.pt"
    save_league_checkpoint(
        ckpt,
        model,
        update=1,
        training={"steps": 1},
        encoder_cfg=model.encoder_cfg,
    )
    return ckpt


@pytest.fixture
def engine(checkpoint_path: Path) -> InferenceEngine:
    ref = FIXTURES / "reference_histogram.json"
    cfg = ServerConfig(
        checkpoint_path=checkpoint_path,
        reference_histogram_path=ref if ref.is_file() else None,
        device="cpu",
        skip_manifest_verify=True,
        model_id="test_policy",
    )
    eng = InferenceEngine(cfg)
    eng.load()
    return eng


@pytest.fixture
def client(engine: InferenceEngine):
    app = create_app(engine=engine)
    with TestClient(app) as c:
        yield c


def test_health_ready(client: TestClient) -> None:
    h = client.get("/health")
    assert h.status_code == 200
    body = h.json()
    assert body["obs_version"] == 3
    assert body["model_id"] == "test_policy"
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["ready"] is True


def test_act_single(client: TestClient, sample_game_state: dict) -> None:
    resp = client.post(
        "/v1/act",
        json={"game_state": sample_game_state, "options": {"include_teacher": False}},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["chosen_action"] in ("FOLD", "ALL_IN")
    assert len(data["action_logits"]) == 2
    assert len(data["q_bb"]) == 2
    assert data["obs_version"] == 3
    assert data["model_id"] == "test_policy"
    assert "ood" in data


def test_side_pot_rejected_422(client: TestClient, sample_game_state: dict) -> None:
    bad = dict(sample_game_state)
    bad["side_pots_present"] = True
    bad["equal_stacks"] = False
    resp = client.post("/v1/act", json={"game_state": bad})
    assert resp.status_code == 422
    assert resp.json()["error_code"] == "unsupported_side_pots"


def test_batch_mixed_valid_invalid(
    client: TestClient, sample_game_state: dict
) -> None:
    bad = dict(sample_game_state)
    bad["side_pots_present"] = True
    bad["equal_stacks"] = False
    resp = client.post(
        "/v1/act/batch",
        json={
            "items": [
                {"game_state": sample_game_state},
                {"game_state": bad},
            ],
            "fail_fast": False,
        },
    )
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert results[0]["ok"] is True
    assert results[0]["response"] is not None
    assert results[1]["ok"] is False
    assert results[1]["error_code"] == "unsupported_side_pots"


def test_manifest_hash_mismatch(checkpoint_path: Path, tmp_path: Path) -> None:
    import hashlib

    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "bad.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "model_id": "x",
                "obs_version": 3,
                "checkpoint_sha256": "0" * 64,
                "encoder": {},
            }
        ),
        encoding="utf-8",
    )
    manifest = ModelManifest.from_dict(json.loads(manifest_path.read_text()))
    with pytest.raises(ManifestError, match="SHA-256"):
        verify_manifest(checkpoint_path, manifest)

    cfg = ServerConfig(
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
        device="cpu",
    )
    with pytest.raises(ManifestError):
        InferenceEngine(cfg).load()


def test_ood_drift_alert(engine: InferenceEngine, sample_game_state: dict) -> None:
    game = game_state_from_dict(sample_game_state)
    from aof_rl.encode_v3 import encode_obs_v3

    obs = encode_obs_v3(game)
    ref = json.loads((FIXTURES / "reference_histogram.json").read_text())
    # Inflate drift by using empty reference counts
    ref["counts"] = [[0.0] * ref["num_bins"] for _ in ref["counts"]]
    monitor = OodMonitor(OodConfig(drift_threshold=0.01), ref)
    feats = flatten_obs_features(obs)
    drift = monitor.score(
        obs,
        policy_action=0,
        policy_probs=torch.tensor([0.5, 0.5]).numpy(),
        uncertainty_bb=0.1,
    )
    assert drift.drift_score >= 0.0


def test_teacher_optional_block(
    client: TestClient, sample_game_state: dict, engine: InferenceEngine
) -> None:
    from aof import AdvisorTeacher

    mock_teacher = AdvisorTeacher(
        action="ALL IN",
        ev_primary=1.0,
        ev_alt=-0.5,
        edge_bb=1.5,
        p_fold=None,
        facing=False,
    )
    mock_result = TeacherResult(
        teacher=mock_teacher,
        path="static_profile_open",
        provenance="test",
    )

    with patch.object(
        engine.teacher_cache,
        "get_with_path",
        return_value=mock_result,
    ):
        app = create_app(engine=engine)
        with TestClient(app) as c:
            resp = c.post(
                "/v1/act",
                json={
                    "game_state": sample_game_state,
                    "options": {"include_teacher": True, "teacher_samples": 100},
                },
            )
    assert resp.status_code == 200
    data = resp.json()
    assert data["teacher"] is not None
    assert data["teacher"]["action"] == "ALL_IN"
    assert len(data["teacher"]["q_bb"]) == 2


def test_game_state_json_roundtrip(sample_game_state: dict) -> None:
    g = game_state_from_dict(sample_game_state)
    back = game_state_to_dict(g)
    g2 = game_state_from_dict(back)
    assert g.hero_cards == g2.hero_cards
    assert g.table_size == g2.table_size


def test_not_ready_returns_503(checkpoint_path: Path) -> None:
    eng = InferenceEngine(
        ServerConfig(
            checkpoint_path=checkpoint_path,
            device="cpu",
            skip_manifest_verify=True,
        )
    )
    app = create_app(engine=eng)
    with TestClient(app) as client:
        assert client.get("/ready").status_code == 503
