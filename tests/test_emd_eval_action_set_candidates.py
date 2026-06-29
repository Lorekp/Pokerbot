from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "eval"))

import emd_action_set_report as action_sets  # noqa: E402
import eval_emd_vs_iter62000 as eval_emd  # noqa: E402


def test_resolve_named_action_set_presets_are_explicit():
    pot = eval_emd.resolve_bet_candidates("pot_fraction")
    off = eval_emd.resolve_bet_candidates("off_tree_adversarial")

    assert pot["action_set_name"] == "pot_fraction"
    assert pot["actions"] == (3, 4, 6, 7, 10)
    assert [m["token"] for m in pot["mapping"]] == [
        "0.3333333333333333",
        "0.5",
        "0.75",
        "1.0",
        "all_in",
    ]
    assert off["action_set_name"] == "off_tree_adversarial"
    assert off["actions"] == (3, 7, 8, 9, 10)
    assert [m["requested_fraction"] for m in off["mapping"] if m["kind"] == "pot_fraction"] == [
        0.25,
        0.9,
        1.25,
        2.5,
    ]


def test_run_one_blueprint_passes_custom_bet_candidates(monkeypatch):
    captured = {}

    def fake_run_python_lbr6(_policy, cfg, *, progress_fn=None):
        captured["bet_candidates"] = tuple(cfg.bet_candidates)
        return argparse.Namespace(
            bb_per_100=11.0,
            detail={
                "per_seat": {
                    0: {"bb_per_100": 10.0},
                    1: {"bb_per_100": 12.0},
                }
            },
        )

    import pokerbot.eval.lbr as lbr  # noqa: PLC0415

    monkeypatch.setattr(lbr, "make_blueprint_policy", lambda _snapshot: object())
    monkeypatch.setattr(eval_emd, "_run_python_lbr6", fake_run_python_lbr6)
    meta = eval_emd.resolve_bet_candidates("0.25,0.9,all_in", action_set_name="off_tree_adversarial")

    result = eval_emd._run_one_blueprint(
        {b"infoset": [1.0] * 11},
        label="emd",
        seeds=[123],
        hands=2,
        n_seats=2,
        value_only=True,
        use_native=False,
        emd_frozen_path=None,
        bet_candidates=meta["actions"],
        action_set_meta=meta,
    )

    assert captured["bet_candidates"] == meta["actions"]
    assert result["action_set_name"] == "off_tree_adversarial"
    assert result["bet_candidates"] == [3, 7, 10]
    assert result["bet_candidate_mapping"][0]["requested_fraction"] == 0.25


def test_mode_run_emits_packageable_custom_action_set_partials(tmp_path, monkeypatch):
    def fake_run_one_blueprint(
        _snapshot,
        *,
        label,
        seeds,
        hands,
        n_seats,
        value_only,
        use_native,
        emd_frozen_path,
        bet_candidates,
        action_set_meta,
        require_native=False,
    ):
        seat_base = 70.0 if label == "emd" else 100.0
        return {
            "label": label,
            "infosets": 1,
            "seeds": list(seeds),
            "hands_per_seed": hands,
            "n_seats": n_seats,
            "value_only": value_only,
            "method": "python_lbr6_reference",
            "pooled_bb_per_100": seat_base,
            "per_seed_per_seat": [{"1": seat_base, "2": seat_base + 10.0}],
            "per_seat": {
                "1": {"bb_per_100": seat_base},
                "2": {"bb_per_100": seat_base + 10.0},
            },
            "bet_candidates": list(bet_candidates),
            "action_set_name": action_set_meta["action_set_name"],
            "bet_candidate_spec": action_set_meta["spec"],
            "bet_candidate_mapping": action_set_meta["mapping"],
        }

    monkeypatch.setattr(eval_emd, "_load_snapshot", lambda _path: {b"infoset": [1.0] * 11})
    monkeypatch.setattr(eval_emd, "_anchor_check", lambda *_a, **_k: {
        "anchor_ok": True,
        "always_fold_bb_per_100": 125.0,
    })
    monkeypatch.setattr(eval_emd, "_run_one_blueprint", fake_run_one_blueprint)

    for action_set_name in ("pot_fraction", "off_tree_adversarial"):
        emd_path = tmp_path / f"{action_set_name}_emd.json"
        base_path = tmp_path / f"{action_set_name}_base.json"
        for label, out_path in (("emd", emd_path), ("iter62000", base_path)):
            rc = eval_emd._mode_run(
                argparse.Namespace(
                    checkpoint="dummy.pkl",
                    out=str(out_path),
                    seeds=1,
                    hands=2,
                    n_seats=6,
                    num_buckets=160,
                    native=False,
                    require_native=False,
                    value_only=True,
                    emd_frozen=None,
                    action_set_name=action_set_name,
                    bet_candidates="",
                ),
                label=label,
            )
            assert rc == 0

        emd = json.loads(emd_path.read_text(encoding="utf-8"))
        base = json.loads(base_path.read_text(encoding="utf-8"))
        assert emd["action_set_name"] == action_set_name
        assert emd["result"]["seeds"] == base["result"]["seeds"]
        assert emd["result"]["hands_per_seed"] == base["result"]["hands_per_seed"]

        entry = action_sets.build_action_set_entry(
            name=action_set_name,
            emd_partial=emd,
            base_partial=base,
            bet_candidates=emd["bet_candidate_spec"].split(","),
            emd_path=str(emd_path),
            base_path=str(base_path),
        )
        assert entry["provenance"]["crn"] is True
        assert entry["bet_candidates"]
