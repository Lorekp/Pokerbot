"""Checkpoint provenance and BC resolution warnings."""

from __future__ import annotations

import torch

from pokerbot.training.checkpoint_provenance import is_bc_checkpoint, warn_if_bc_checkpoint
from pokerbot.training.sft import LABEL_SOURCE_BC, sft_checkpoint_metadata
from pokerbot.training.sft import SFTConfig


def test_is_bc_checkpoint_from_metadata(tmp_path):
    ckpt = tmp_path / "custom_name.pt"
    meta = sft_checkpoint_metadata(SFTConfig(mode="bc"))
    torch.save({"model": {}, **meta}, ckpt)
    assert is_bc_checkpoint(ckpt)
    assert not is_bc_checkpoint(tmp_path / "missing.pt")


def test_is_bc_checkpoint_solver_not_flagged(tmp_path):
    ckpt = tmp_path / "solver.pt"
    meta = sft_checkpoint_metadata(SFTConfig(mode="solver"))
    torch.save({"model": {}, **meta}, ckpt)
    assert not is_bc_checkpoint(ckpt)


def test_warn_if_bc_by_label_source(tmp_path, capsys):
    ckpt = tmp_path / "renamed.pt"
    torch.save(
        {"model": {}, "label_source": LABEL_SOURCE_BC, "gto_verified": False, "mode": "bc"},
        ckpt,
    )
    warn_if_bc_checkpoint(ckpt, context="test")
    out = capsys.readouterr().out
    assert "NOT GTO-verified" in out
    assert "renamed.pt" in out
