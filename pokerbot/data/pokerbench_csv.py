"""Load PokerBench structured CSV from HuggingFace with streaming."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

Stage = Literal["preflop", "postflop"]
Split = Literal["train", "test"]

CSV_FILES: dict[tuple[Stage, Split], str] = {
    ("preflop", "train"): "preflop_60k_train_set_game_scenario_information.csv",
    ("preflop", "test"): "preflop_1k_test_set_game_scenario_information.csv",
    ("postflop", "train"): "postflop_500k_train_set_game_scenario_information.csv",
    ("postflop", "test"): "postflop_10k_test_set_game_scenario_information.csv",
}

REPO_ID = "RZ412/PokerBench"


@dataclass(frozen=True)
class CSVSample:
    stage: Stage
    split: Split
    row_index: int
    row: dict


def _resolve_csv_path(stage: Stage, split: Split, cache_dir: Path | None) -> Path:
    fname = CSV_FILES[(stage, split)]
    if cache_dir is not None:
        local = cache_dir / fname
        if local.exists():
            return local
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=fname,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        return Path(path)
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not download {fname} from {REPO_ID}. "
            f"Place file under data/pokerbench/ or set cache_dir."
        ) from exc


def iter_csv_rows(
    stage: Stage,
    split: Split = "train",
    limit: int | None = None,
    cache_dir: Path | None = None,
    chunk_size: int = 4096,
) -> Iterator[CSVSample]:
    """Stream CSV rows without loading full file into memory."""
    import pandas as pd

    path = _resolve_csv_path(stage, split, cache_dir or Path("data/pokerbench"))
    count = 0
    for chunk in pd.read_csv(path, chunksize=chunk_size):
        for i, row in chunk.iterrows():
            if limit is not None and count >= limit:
                return
            yield CSVSample(
                stage=stage,
                split=split,
                row_index=int(i),
                row=row.to_dict(),
            )
            count += 1


def count_csv_rows(stage: Stage, split: Split = "train", cache_dir: Path | None = None) -> int:
    import pandas as pd

    path = _resolve_csv_path(stage, split, cache_dir or Path("data/pokerbench"))
    return sum(1 for _ in pd.read_csv(path, chunksize=10000, usecols=[0]))


def load_csv_samples(
    stage: Stage,
    split: Split = "train",
    limit: int | None = None,
    cache_dir: Path | None = None,
) -> list[CSVSample]:
    return list(iter_csv_rows(stage, split, limit, cache_dir))
