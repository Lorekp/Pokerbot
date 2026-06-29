"""Cache LLM teacher labels for CNN distillation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from pokerbot.data.pokerbench_csv import iter_csv_rows
from pokerbot.encoding.csv_encoder import encode_csv_row
from pokerbot.teachers.llm_teacher import LLMTeacher, LLMTeacherConfig, parse_llm_action


@dataclass
class CacheConfig:
    output_dir: Path = Path("data/cache")
    cache_name: str = "llm_labels.parquet"
    model_id: str = "YiPz/llama3-8b-pokerbench-sft"
    batch_size: int = 4
    preflop_limit: int | None = None
    postflop_limit: int | None = None
    cache_dir: Path = Path("data/pokerbench")
    json_prompts: bool = True


def _load_json_instructions(
    stage: str, split: str, limit: int | None, cache_dir: Path | None = None
) -> list[str]:
    from pokerbot.data.pokerbench import load_pokerbench

    samples = load_pokerbench(stage, split, limit=limit, data_dir=cache_dir)  # type: ignore[arg-type]
    if len(samples) <= 4:
        raise RuntimeError(
            f"Only {len(samples)} JSON prompts loaded for {stage}/{split}. "
            "Download PokerBench JSON from HuggingFace or set cache_dir with prompt files."
        )
    return [s.instruction for s in samples]


def build_label_cache(config: CacheConfig | None = None) -> Path:
    config = config or CacheConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = config.output_dir / config.cache_name

    teacher = LLMTeacher(LLMTeacherConfig(model_id=config.model_id, batch_size=config.batch_size))

    records: list[dict] = []
    sample_id = 0

    for stage in ("preflop", "postflop"):
        limit = config.preflop_limit if stage == "preflop" else config.postflop_limit
        instructions = _load_json_instructions(stage, "train", limit, config.cache_dir) if config.json_prompts else []

        csv_iter = list(iter_csv_rows(stage, "train", limit, config.cache_dir))  # type: ignore[arg-type]
        if config.json_prompts and len(instructions) != len(csv_iter):
            n = min(len(instructions), len(csv_iter))
            instructions = instructions[:n]
            csv_iter = csv_iter[:n]

        for batch_start in tqdm(range(0, len(csv_iter), config.batch_size), desc=f"LLM {stage}"):
            batch_rows = csv_iter[batch_start : batch_start + config.batch_size]
            batch_instr = (
                instructions[batch_start : batch_start + config.batch_size]
                if config.json_prompts
                else [f"spot_{sample_id + i}" for i in range(len(batch_rows))]
            )

            if config.json_prompts:
                llm_texts = teacher.predict_batch(batch_instr)
            else:
                llm_texts = ["fold"] * len(batch_rows)

            for i, csv_sample in enumerate(batch_rows):
                encoded = encode_csv_row(csv_sample.row, csv_sample.stage)
                gto_label = int(encoded["label"])
                llm_idx, llm_raw = parse_llm_action(llm_texts[i])
                if llm_idx is None:
                    llm_idx = gto_label

                records.append(
                    {
                        "sample_id": sample_id,
                        "stage": stage,
                        "gto_label": gto_label,
                        "llm_label": llm_idx,
                        "llm_raw": llm_raw,
                        "agreement": llm_idx == gto_label,
                        "cards": encoded["cards"].tolist(),
                        "actions": encoded["actions"].tolist(),
                        "position": encoded["position"].tolist(),
                        "legal_mask": encoded["legal_mask"].tolist(),
                    }
                )
                sample_id += 1

    try:
        import pandas as pd

        # CRASH-RESUME-AUDIT #6: write the parquet ATOMICALLY. Parquet is columnar and a torn
        # write is unreadable — pandas refuses to read a partial footer. Writing to ``out.tmp``
        # and os.replace'ing into place is the standard fix; without it a kill -9 mid-write
        # loses the entire label cache (every label must be re-fetched from the LLM, $$ + rate
        # limits). The parent-dir fsync makes the rename durable on power cut.
        tmp = out_path.with_name(out_path.name + ".atomic.tmp")
        pd.DataFrame(records).to_parquet(tmp, index=False)
        # Parquet writers may not flush their internal state with an explicit fsync; open+fsync
        # the tmp file as a no-op durability belt-and-braces (it has zero effect on the parquet
        # content but ensures the OS page cache holds the new bytes before the rename).
        try:
            fd = os.open(str(tmp), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass
        os.replace(tmp, out_path)
        try:
            dir_fd = os.open(str(out_path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    except ImportError:
        out_path = config.output_dir / "llm_labels.jsonl"
        tmp = out_path.with_name(out_path.name + ".atomic.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp, out_path)
        try:
            dir_fd = os.open(str(out_path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass

    agree = sum(1 for r in records if r["agreement"]) / max(len(records), 1)
    print(f"Cached {len(records)} labels to {out_path} (LLM-GTO agreement={agree:.2%})")
    return out_path


def _restore_tensor(val: object, dtype: type = np.float32) -> np.ndarray:
    """Rebuild ndarray from parquet nested object columns."""
    arr = np.asarray(val, dtype=object)
    if arr.ndim == 0 or arr.dtype != object:
        return np.asarray(val, dtype=dtype)
    return np.stack([_restore_tensor(x, dtype) for x in arr])


def load_label_cache(path: Path) -> list[dict]:
    # CRASH-RESUME-AUDIT cross-cutting E: torn-tail-tolerant load. The previous code did
    # ``json.loads(line)`` on EVERY line and raised on a single torn tail — same crash mode as the
    # CompletionLedger bootstrap. Parquet is columnar and pd.read_parquet already raises clearly
    # (which is correct — there is no JSONL recovery path for parquet), but the JSONL fallback path
    # is now line-tolerant: malformed lines are skipped with a warning, not a raise.
    if path.suffix == ".parquet":
        import pandas as pd

        df = pd.read_parquet(path)
        records = df.to_dict(orient="records")
    else:
        records = []
        malformed = 0
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            text = ""
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except (ValueError, json.JSONDecodeError):
                malformed += 1
                continue
        if malformed:
            import logging
            logging.getLogger(__name__).warning(
                "load_label_cache: skipped %d malformed tail line(s) in %s", malformed, path,
            )

    for r in records:
        for key in ("cards", "actions", "position"):
            if key in r:
                r[key] = _restore_tensor(r[key], np.float32)
        if "legal_mask" in r:
            lm = r["legal_mask"]
            r["legal_mask"] = np.asarray(lm, dtype=bool)
    return records


def cache_to_tensors(records: list[dict]) -> dict[str, np.ndarray]:
    return {
        "cards": np.array([r["cards"] for r in records], dtype=np.float32),
        "actions": np.array([r["actions"] for r in records], dtype=np.float32),
        "position": np.array([r["position"] for r in records], dtype=np.float32),
        "legal_mask": np.array([r["legal_mask"] for r in records], dtype=bool),
        "gto_label": np.array([r["gto_label"] for r in records], dtype=np.int64),
        "llm_label": np.array([r["llm_label"] for r in records], dtype=np.int64),
    }
