"""Frozen gate baseline manifest for reproducible harness comparisons."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_MANIFEST = Path("checkpoints/gate_reference.json")
DEFAULT_BASELINE = Path("checkpoints/policy_gate_baseline.pt")
DEFAULT_TRANSFORMER_BASELINE = Path("checkpoints/opponent_transformer_gate_baseline.pt")
DEFAULT_CFR_HERO = Path("checkpoints/policy_cfr_blueprint.pt")
DEFAULT_SEARCH_VALUE = Path("checkpoints/history_value_net.pkl")
DEFAULT_SEARCH_MADT = Path("checkpoints/opponent_transformer_madt.pt")
SEARCH_GATE_RESULTS = Path("checkpoints/search_gate_results.json")
PHASE2_GATE_PASSES_REQUIRED = 2


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_weva_fingerprint(weva_dir: Path = Path("data/weva")) -> str | None:
    meta_path = weva_dir / "meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta.get("build_fingerprint")


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(manifest: dict[str, Any], path: Path = DEFAULT_MANIFEST) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def promote_baseline(
    source_checkpoint: Path,
    *,
    baseline_path: Path = DEFAULT_BASELINE,
    manifest_path: Path = DEFAULT_MANIFEST,
    notes: str = "Phase 3 harness control; do not overwrite without promote script",
) -> dict[str, Any]:
    """Copy hero checkpoint to frozen gate baseline and write manifest."""
    import shutil

    if not source_checkpoint.exists():
        raise FileNotFoundError(source_checkpoint)

    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_checkpoint, baseline_path)
    hero_sha = sha256_file(baseline_path)

    manifest = {
        "hero_checkpoint": str(baseline_path.as_posix()),
        "hero_sha256": hero_sha,
        "weva_build_fingerprint": load_weva_fingerprint(),
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(source_checkpoint.as_posix()),
        "notes": notes,
    }
    save_manifest(manifest, manifest_path)
    return manifest


def resolve_baseline_checkpoint(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    override: Path | None = None,
    require: bool = True,
) -> tuple[Path, dict[str, Any]]:
    """Load frozen baseline path; verify SHA256 when manifest present."""
    if override is not None:
        path = Path(override)
        if not path.exists():
            raise FileNotFoundError(f"Baseline checkpoint not found: {path}")
        manifest = load_manifest(manifest_path) or {}
        manifest["hero_checkpoint"] = str(path)
        manifest["hero_sha256"] = sha256_file(path)
        return path, manifest

    manifest = load_manifest(manifest_path)
    if manifest is None:
        if require:
            raise FileNotFoundError(
                f"Gate reference manifest missing at {manifest_path}. "
                "Run scripts/promote_gate_baseline.py first or pass --baseline-checkpoint."
            )
        fallback = Path("checkpoints/policy_selfplay_mixed.pt")
        return fallback, {"hero_checkpoint": str(fallback), "hero_sha256": sha256_file(fallback) if fallback.exists() else None}

    hero_path = Path(manifest["hero_checkpoint"])
    if not hero_path.exists():
        raise FileNotFoundError(f"Gate baseline missing: {hero_path}")

    expected = manifest.get("hero_sha256")
    actual = sha256_file(hero_path)
    if expected and expected != actual:
        raise ValueError(
            f"Gate baseline SHA256 mismatch: manifest={expected[:16]}... file={actual[:16]}... "
            "Re-run promote_gate_baseline.py or fix manifest."
        )
    manifest["hero_sha256"] = actual
    return hero_path, manifest


def promote_hybrid_baseline(
    source_transformer: Path,
    *,
    transformer_baseline_path: Path = DEFAULT_TRANSFORMER_BASELINE,
    manifest_path: Path = DEFAULT_MANIFEST,
    phase3_gate_passed_at: str | None = None,
) -> dict[str, Any]:
    """Copy winning MADT transformer to frozen gate baseline and update manifest."""
    import shutil
    from datetime import datetime, timezone

    if not source_transformer.exists():
        raise FileNotFoundError(source_transformer)

    transformer_baseline_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_transformer, transformer_baseline_path)
    transformer_sha = sha256_file(transformer_baseline_path)

    manifest = load_manifest(manifest_path) or {}
    manifest["transformer_checkpoint"] = str(transformer_baseline_path.as_posix())
    manifest["transformer_sha256"] = transformer_sha
    manifest["transformer_source"] = str(source_transformer.as_posix())
    manifest["phase3_gate_passed_at"] = phase3_gate_passed_at or datetime.now(timezone.utc).isoformat()
    save_manifest(manifest, manifest_path)
    return manifest


def resolve_hybrid_baseline(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    hero_override: Path | None = None,
    transformer_override: Path | None = None,
    require: bool = True,
) -> tuple[Path, Path, dict[str, Any]]:
    """
    Resolve frozen hero + MADT transformer for Phase 4 Hybrid incumbent.

    Verifies SHA256 for both when manifest fields are present.
    """
    hero_path, manifest = resolve_baseline_checkpoint(
        manifest_path=manifest_path,
        override=hero_override,
        require=require,
    )

    if transformer_override is not None:
        tf_path = Path(transformer_override)
        if not tf_path.exists():
            raise FileNotFoundError(f"Transformer baseline not found: {tf_path}")
        manifest["transformer_checkpoint"] = str(tf_path)
        manifest["transformer_sha256"] = sha256_file(tf_path)
        return hero_path, tf_path, manifest

    tf_key = manifest.get("transformer_checkpoint")
    if tf_key:
        tf_path = Path(tf_key)
        if not tf_path.exists():
            raise FileNotFoundError(f"Gate transformer baseline missing: {tf_path}")
        expected = manifest.get("transformer_sha256")
        actual = sha256_file(tf_path)
        if expected and expected != actual:
            raise ValueError(
                f"Gate transformer SHA256 mismatch: manifest={expected[:16]}... file={actual[:16]}... "
                "Re-run promote_hybrid_baseline.py or fix manifest."
            )
        manifest["transformer_sha256"] = actual
        return hero_path, tf_path, manifest

    fallback = DEFAULT_TRANSFORMER_BASELINE
    if fallback.exists():
        manifest.setdefault("transformer_checkpoint", str(fallback))
        manifest.setdefault("transformer_sha256", sha256_file(fallback))
        return hero_path, fallback, manifest

    if require:
        raise FileNotFoundError(
            f"Hybrid gate transformer baseline missing. Run scripts/promote_hybrid_baseline.py "
            f"after Phase 3 gate passes, or pass --transformer-baseline."
        )
    candidate = Path("checkpoints/opponent_transformer_madt.pt")
    return hero_path, candidate, manifest


def record_phase2_gate_pass(
    *,
    seed: int,
    results_path: Path,
    manifest_path: Path = DEFAULT_MANIFEST,
    cfr_hero_path: Path = DEFAULT_CFR_HERO,
) -> dict[str, Any]:
    """Record a Phase 2 harness pass; promote CFR hero after two distinct seeds."""
    if not results_path.exists():
        raise FileNotFoundError(results_path)
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if not results.get("gate_passed"):
        raise ValueError(f"Phase 2 gate not passed in {results_path}")

    manifest = load_manifest(manifest_path) or {}
    passes: list[int] = list(manifest.get("phase2_gate_pass_seeds") or [])
    if seed not in passes:
        passes.append(seed)
    manifest["phase2_gate_pass_seeds"] = passes
    manifest["phase2_last_gate_result"] = str(results_path)

    if len(passes) >= PHASE2_GATE_PASSES_REQUIRED and cfr_hero_path.exists():
        manifest["phase2_gate_passed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["cfr_hero_checkpoint"] = str(cfr_hero_path.as_posix())
        manifest["cfr_hero_sha256"] = sha256_file(cfr_hero_path)
    save_manifest(manifest, manifest_path)
    return manifest


def should_use_cfr_hero(manifest_path: Path = DEFAULT_MANIFEST) -> bool:
    """True when manifest records Phase 2 gate pass with CFR hero SHA."""
    manifest = load_manifest(manifest_path)
    if not manifest or not manifest.get("phase2_gate_passed_at"):
        return False
    cfr_path = Path(manifest.get("cfr_hero_checkpoint", DEFAULT_CFR_HERO))
    if not cfr_path.exists():
        return False
    expected = manifest.get("cfr_hero_sha256")
    if expected and sha256_file(cfr_path) != expected:
        return False
    return True


def resolve_cfr_hero_path(manifest_path: Path = DEFAULT_MANIFEST) -> Path:
    manifest = load_manifest(manifest_path) or {}
    path = Path(manifest.get("cfr_hero_checkpoint", DEFAULT_CFR_HERO))
    if not path.exists():
        raise FileNotFoundError(f"CFR hero checkpoint missing: {path}")
    expected = manifest.get("cfr_hero_sha256")
    actual = sha256_file(path)
    if expected and expected != actual:
        raise ValueError("CFR hero SHA256 mismatch in gate manifest")
    return path


def append_search_gate_result(
    result: dict[str, Any],
    *,
    path: Path = SEARCH_GATE_RESULTS,
) -> dict[str, Any]:
    """Append Phase 4 gate run to history JSON."""
    history: list[dict[str, Any]] = []
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        history = list(payload.get("runs") or [])
    entry = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "gate_passed": result.get("gate_passed"),
        "gto_safety_passed": result.get("gto_safety_passed"),
        "seed": result.get("seed"),
        "search_artifacts": result.get("search_artifacts"),
    }
    history.append(entry)
    payload = {"runs": history[-50:]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def promote_search_baseline(
    results_path: Path,
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    value_path: Path | None = None,
    madt_path: Path | None = None,
) -> dict[str, Any]:
    """Pin search artifacts in manifest when Phase 4 gate_passed."""
    if not results_path.exists():
        raise FileNotFoundError(results_path)
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if not results.get("gate_passed"):
        raise ValueError(f"Phase 4 gate not passed in {results_path}")

    artifacts = results.get("search_artifacts") or {}
    v_path = Path(value_path or artifacts.get("value_checkpoint", DEFAULT_SEARCH_VALUE))
    m_path = Path(madt_path or artifacts.get("candidate_madt_checkpoint", DEFAULT_SEARCH_MADT))
    if not v_path.exists():
        raise FileNotFoundError(v_path)
    if not m_path.exists():
        raise FileNotFoundError(m_path)

    manifest = load_manifest(manifest_path) or {}
    manifest["search_value_checkpoint"] = str(v_path.as_posix())
    manifest["search_value_sha256"] = sha256_file(v_path)
    manifest["search_madt_checkpoint"] = str(m_path.as_posix())
    manifest["search_madt_sha256"] = sha256_file(m_path)
    manifest["phase4_gate_passed_at"] = datetime.now(timezone.utc).isoformat()
    save_manifest(manifest, manifest_path)
    append_search_gate_result(results)
    return manifest


def resolve_search_baseline(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    require_gate: bool = False,
) -> tuple[Path, Path, dict[str, Any]]:
    """Resolve promoted search HVN + MADT paths with SHA verification."""
    manifest = load_manifest(manifest_path)
    if manifest is None or not manifest.get("phase4_gate_passed_at"):
        if require_gate:
            raise FileNotFoundError(
                "Phase 4 search gate not recorded. Run promote_search_baseline.py after gate pass."
            )
        v_fallback = DEFAULT_SEARCH_VALUE
        m_fallback = DEFAULT_SEARCH_MADT
        return v_fallback, m_fallback, manifest or {}

    v_path = Path(manifest["search_value_checkpoint"])
    m_path = Path(manifest["search_madt_checkpoint"])
    for label, path, key in (
        ("search value", v_path, "search_value_sha256"),
        ("search MADT", m_path, "search_madt_sha256"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} checkpoint missing: {path}")
        expected = manifest.get(key)
        actual = sha256_file(path)
        if expected and expected != actual:
            raise ValueError(f"{label} SHA256 mismatch in gate manifest")
        manifest[key] = actual
    return v_path, m_path, manifest


def phase4_gate_passed(manifest_path: Path = DEFAULT_MANIFEST) -> bool:
    try:
        resolve_search_baseline(manifest_path=manifest_path, require_gate=True)
        return True
    except (FileNotFoundError, ValueError, KeyError):
        return False
