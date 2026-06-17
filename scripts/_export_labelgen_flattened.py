#!/usr/bin/env python3
"""One-off export of labelgen code to flattened markdown for optimization agents."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path.home() / "Downloads" / "labelgen_flattened.md"

FILES = [
    "docs/prod6000-cloud-runbook.md",
    "infra/labelgen/Dockerfile",
    "infra/labelgen/requirements.txt",
    "infra/labelgen/spot-interrupt.sh",
    "infra/labelgen/IAM.md",
    "infra/labelgen/iam_policy_labelgen_launcher.json",
    "infra/labelgen/iam_policy_labelgen_worker_role.json",
    "infra/labelgen/iam_policy_labelgen.json",
    "external/postflop_solver_cli/Cargo.toml",
    "external/postflop_solver_cli/src/main.rs",
    "external/postflop_solver_cli/README.md",
    "pokerbot/solver/s3_coordination.py",
    "pokerbot/solver/label_completion.py",
    "pokerbot/solver/export.py",
    "pokerbot/solver/solver_worker.py",
    "pokerbot/solver/solver_worker_pool.py",
    "pokerbot/solver/postflop_solver_client.py",
    "scripts/cloud_label_worker.py",
    "scripts/launch_labelgen_spot.py",
    "scripts/launch_validate50_ec2.py",
    "scripts/labelgen_image.py",
    "scripts/labelgen_ec2_diagnostics.py",
    "scripts/plan_label_work.py",
    "scripts/sync_prod6000_to_s3.py",
    "scripts/sync_prod6000_to_s3_loop.sh",
    "scripts/run_prod6000_cloud_fleet.py",
    "scripts/run_coordination_gate.py",
    "scripts/run_validate50_ec2_gate.py",
    "scripts/run_validate50_gate.py",
    "scripts/verify_labelgen_iam.py",
    "scripts/setup_labelgen_iam.py",
    "scripts/report_prod6000_cloud_run.py",
    "scripts/validate_fleet_labels.py",
    "scripts/validate_cloud_labels.py",
    "scripts/build_solver_labels.py",
    "tests/test_s3_coordination.py",
    "tests/test_labelgen_torch_free.py",
    "tests/test_labelgen_image_hash.py",
    "config/solver_labels_prod6000.yaml",
    "config/solver_labels_validate50.yaml",
]

EXT_LANG = {
    ".py": "python",
    ".sh": "bash",
    ".json": "json",
    ".yaml": "yaml",
    ".md": "markdown",
    ".txt": "text",
    ".toml": "toml",
    ".rs": "rust",
}


def _lang(path: Path) -> str:
    if path.name == "Dockerfile":
        return "dockerfile"
    return EXT_LANG.get(path.suffix.lower(), "text")


def main() -> None:
    arch = (ROOT / "logs/labelgen_architecture_section.md").read_text(encoding="utf-8")
    parts = [arch]
    for rel in FILES:
        path = ROOT / rel
        if not path.is_file():
            continue
        lang = _lang(path)
        content = path.read_text(encoding="utf-8", errors="replace")
        fence = "```"
        parts.append(f"## `{rel}`\n\n{fence}{lang}\n{content}\n{fence}\n\n")
    OUT.write_text("".join(parts), encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size:,} bytes, {len(FILES)} files)")


if __name__ == "__main__":
    main()
