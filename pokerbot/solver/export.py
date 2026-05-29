"""Export solver label rows to Parquet."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def write_solver_parquet(rows: list[dict[str, Any]], output: Path) -> None:
    import pandas as pd

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output, index=False)


def write_shard(
    row: dict[str, Any],
    shard_dir: Path,
    spot_key: str,
    *,
    spot_key_hash_fn: Callable[[str], str],
) -> Path:
    """Write one row to an atomically-renamed shard file. Crash loses at most this spot."""
    import pandas as pd

    shard_dir.mkdir(parents=True, exist_ok=True)
    digest = spot_key_hash_fn(spot_key)
    final_path = shard_dir / f"{digest}.parquet"
    tmp_path = shard_dir / f"{digest}.parquet.tmp"
    pd.DataFrame([row]).to_parquet(tmp_path, index=False)
    tmp_path.replace(final_path)
    return final_path


def merge_shards(
    shard_dir: Path,
    output: Path,
    *,
    spot_keys: list[str] | None = None,
    spot_key_hash_fn: Callable[[str], str],
) -> int:
    """Merge shard Parquet files into a single output. Returns row count written."""
    import pandas as pd

    if not shard_dir.is_dir():
        write_solver_parquet([], output)
        return 0

    frames: list[Any] = []
    if spot_keys is not None:
        for key in spot_keys:
            path = shard_dir / f"{spot_key_hash_fn(key)}.parquet"
            if path.is_file():
                frames.append(pd.read_parquet(path))
    else:
        for path in sorted(shard_dir.glob("*.parquet")):
            if path.name.endswith(".tmp"):
                continue
            frames.append(pd.read_parquet(path))

    if not frames:
        write_solver_parquet([], output)
        return 0

    df = pd.concat(frames, ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    return len(df)
