"""Check optional Spotpick loading without importing AWS launcher dependencies."""
import ast
import os
import sys
from pathlib import Path


def test_configured_spotpick_builds_fallback_chain(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "scripts/launch_native_cfr_training_spot.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_spotpick_chain")
    namespace = {"os": os, "sys": sys}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    (tmp_path / "spotpick.py").write_text(
        "from types import SimpleNamespace\n"
        "def load():\n"
        "    return [SimpleNamespace(name='m6i.4xlarge', vcpus=16)]\n"
        "def chain(**kwargs):\n"
        "    assert kwargs == dict(vcpus=16, arch='x86_64', n=3, exclude_families=['t3', 'c7i'])\n"
        "    return ['c6a.4xlarge', 'm6i.4xlarge', 'r6a.4xlarge']\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SPOTPICK_ROOT", str(tmp_path))
    monkeypatch.setenv("SPOTPICK_EXCLUDE_FAMILIES", "t3, c7i")
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.delitem(sys.modules, "spotpick", raising=False)
    try:
        assert namespace["_spotpick_chain"]("m6i.4xlarge", "x86_64", 3) == [
            "m6i.4xlarge", "c6a.4xlarge", "r6a.4xlarge",
        ]
    finally:
        sys.modules.pop("spotpick", None)
