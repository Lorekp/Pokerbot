#!/usr/bin/env python3
"""Build lbr_action_sets JSON from real per-action-set eval partials.

This script does not run LBR and does not synthesize pass data. It converts
already-produced EMD/base eval partial pairs into the schema consumed by
scripts/eval/eval_emd_vs_iter62000.py --action-sets-json and
scripts/_emd_selection_gate.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _result(partial: dict[str, Any]) -> dict[str, Any]:
    result = partial.get("result")
    if not isinstance(result, dict):
        raise ValueError("partial missing result object")
    return result


def _per_seat(result: dict[str, Any]) -> dict[str, float]:
    raw = result.get("per_seat")
    if isinstance(raw, dict) and raw:
        out: dict[str, float] = {}
        for seat, value in raw.items():
            if isinstance(value, dict):
                value = value.get("bb_per_100")
            out[str(seat)] = float(value)
        return out

    rows = result.get("per_seed_per_seat")
    if isinstance(rows, list) and rows:
        acc: dict[str, list[float]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            for seat, value in row.items():
                acc.setdefault(str(seat), []).append(float(value))
        if acc:
            return {seat: sum(vals) / len(vals) for seat, vals in sorted(acc.items(), key=lambda kv: int(kv[0]))}

    raise ValueError("partial result missing per_seat and per_seed_per_seat")


def _parse_bet_candidates(text: str) -> list[float | str]:
    out: list[float | str] = []
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            out.append(float(token))
        except ValueError:
            out.append(token)
    if not out:
        raise argparse.ArgumentTypeError("bet candidates must not be empty")
    return out


def _normalize_actions(raw: Any) -> list[int] | None:
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError("partial bet_candidates must be a list when present")
    return [int(x) for x in raw]


def _partial_action_metadata(partial: dict[str, Any]) -> dict[str, Any]:
    result = _result(partial)
    return {
        "action_set_name": partial.get("action_set_name") or result.get("action_set_name"),
        "bet_candidate_spec": partial.get("bet_candidate_spec") or result.get("bet_candidate_spec"),
        "bet_candidates": _normalize_actions(partial.get("bet_candidates") or result.get("bet_candidates")),
        "bet_candidate_mapping": partial.get("bet_candidate_mapping") or result.get("bet_candidate_mapping"),
    }


def _resolve_bet_candidates(
    *,
    name: str,
    emd_partial: dict[str, Any],
    base_partial: dict[str, Any],
    manual_text: str | None,
) -> tuple[list[float | int | str], dict[str, Any]]:
    emd_meta = _partial_action_metadata(emd_partial)
    base_meta = _partial_action_metadata(base_partial)

    emd_name = emd_meta.get("action_set_name")
    base_name = base_meta.get("action_set_name")
    if emd_name and emd_name != name:
        raise ValueError("EMD partial action_set_name {0!r} != requested {1!r}".format(emd_name, name))
    if base_name and base_name != name:
        raise ValueError("iter62000 partial action_set_name {0!r} != requested {1!r}".format(base_name, name))

    emd_spec = emd_meta.get("bet_candidate_spec")
    base_spec = base_meta.get("bet_candidate_spec")
    if emd_spec and base_spec and emd_spec != base_spec:
        raise ValueError("EMD/base bet_candidate_spec differ for {0}".format(name))

    emd_actions = emd_meta.get("bet_candidates")
    base_actions = base_meta.get("bet_candidates")
    if emd_actions is not None and base_actions is not None and emd_actions != base_actions:
        raise ValueError("EMD/base resolved bet_candidates differ for {0}".format(name))

    stamped_spec = emd_spec or base_spec
    stamped_actions = emd_actions if emd_actions is not None else base_actions
    if manual_text:
        manual = _parse_bet_candidates(manual_text)
        if stamped_spec and manual != _parse_bet_candidates(stamped_spec):
            raise ValueError("manual bet candidates do not match stamped bet_candidate_spec for {0}".format(name))
        if not stamped_spec and stamped_actions is not None and manual != stamped_actions:
            raise ValueError("manual bet candidates do not match stamped resolved bet_candidates for {0}".format(name))
        source = "manual_validated_against_partial" if (stamped_spec or stamped_actions is not None) else "manual"
        return manual, {
            "source": source,
            "emd_action_metadata": emd_meta,
            "iter62000_action_metadata": base_meta,
        }

    if stamped_spec:
        return _parse_bet_candidates(stamped_spec), {
            "source": "partial_bet_candidate_spec",
            "emd_action_metadata": emd_meta,
            "iter62000_action_metadata": base_meta,
        }
    if stamped_actions is not None:
        return list(stamped_actions), {
            "source": "partial_resolved_bet_candidates",
            "emd_action_metadata": emd_meta,
            "iter62000_action_metadata": base_meta,
        }

    raise ValueError(
        "bet candidates missing for {0}; provide BET_CANDIDATES or use partials stamped by eval_emd_vs_iter62000.py"
        .format(name)
    )


def _validate_crn(emd_result: dict[str, Any], base_result: dict[str, Any]) -> None:
    emd_seeds = emd_result.get("seeds") or []
    base_seeds = base_result.get("seeds") or []
    if emd_seeds != base_seeds:
        raise ValueError("EMD/base seeds differ; action-set probe is not CRN")
    if emd_result.get("hands_per_seed") != base_result.get("hands_per_seed"):
        raise ValueError("EMD/base hands_per_seed differ")


def build_action_set_entry(
    *,
    name: str,
    emd_partial: dict[str, Any],
    base_partial: dict[str, Any],
    bet_candidates: list[float | str],
    emd_path: str,
    base_path: str,
    bet_candidate_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    emd_result = _result(emd_partial)
    base_result = _result(base_partial)
    _validate_crn(emd_result, base_result)
    return {
        "emd_per_seat": _per_seat(emd_result),
        "iter62000_per_seat": _per_seat(base_result),
        "bet_candidates": bet_candidates,
        "method": "{0} vs {1}".format(
            emd_result.get("method") or emd_partial.get("label") or "emd",
            base_result.get("method") or base_partial.get("label") or "iter62000",
        ),
        "provenance": {
            "action_set": name,
            "emd_json": emd_path,
            "iter62000_json": base_path,
            "emd_label": emd_partial.get("label"),
            "iter62000_label": base_partial.get("label"),
            "emd_method": emd_result.get("method"),
            "iter62000_method": base_result.get("method"),
            "seeds": emd_result.get("seeds"),
            "hands_per_seed": emd_result.get("hands_per_seed"),
            "crn": True,
            "bet_candidates_source": (bet_candidate_provenance or {}).get("source", "caller"),
            "bet_candidate_metadata": bet_candidate_provenance or {},
        },
    }


def build_action_sets(specs: list[list[str]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for spec in specs:
        if len(spec) not in (3, 4):
            raise ValueError("action-set spec must be NAME EMD_JSON ITER62000_JSON [BET_CANDIDATES]")
        name, emd_path, base_path = spec[:3]
        bet_text = spec[3] if len(spec) == 4 else None
        if name in out:
            raise ValueError("duplicate action set {0!r}".format(name))
        emd_partial = _load_json(emd_path)
        base_partial = _load_json(base_path)
        bet_candidates, bet_candidate_provenance = _resolve_bet_candidates(
            name=name,
            emd_partial=emd_partial,
            base_partial=base_partial,
            manual_text=bet_text,
        )
        out[name] = build_action_set_entry(
            name=name,
            emd_partial=emd_partial,
            base_partial=base_partial,
            bet_candidates=bet_candidates,
            emd_path=emd_path,
            base_path=base_path,
            bet_candidate_provenance=bet_candidate_provenance,
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action-set",
        action="append",
        nargs="+",
        metavar="ARG",
        required=True,
        help=(
            "repeatable action-set spec. BET_CANDIDATES is optional when partials are stamped "
            "with matching bet_candidate_spec / bet_candidates metadata"
        ),
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    payload = build_action_sets(args.action_set)
    _write_json(args.out, payload)
    print("wrote {0} action sets -> {1}".format(len(payload), args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
