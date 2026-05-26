"""Solver-annotated Parquet dataset and combined WEVA SFT loss."""



from __future__ import annotations



from glob import glob

from pathlib import Path

from typing import Any



import numpy as np

import torch

import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset



from pokerbot.engine.actions import NUM_ACTIONS

from pokerbot.models.policy import PolicyOutput



LABEL_SOURCE_SOLVER = "toolpoker_solver"



SOLVER_REQUIRED_COLUMNS = (

    "weva_bucket_id",

    "actions",

    "position",

    "legal_mask",

    "solver_label",

    "solver_policy",

    "label_source",

)





def _validate_solver_row(row: dict[str, Any], source: str) -> None:

    for col in SOLVER_REQUIRED_COLUMNS:

        if col not in row or row[col] is None:

            raise ValueError(f"Missing required solver column {col!r} in {source}")

    if row["label_source"] != LABEL_SOURCE_SOLVER:

        raise ValueError(

            f"Invalid label_source {row['label_source']!r} in {source}; "

            f"expected {LABEL_SOURCE_SOLVER!r}. "

            "Use BCParquetDataset for pokerbench_csv rows."

        )





def _nested_to_array(value: Any, dtype: type) -> np.ndarray:

    if isinstance(value, np.ndarray):

        if value.dtype == object:

            return np.stack([_nested_to_array(x, dtype) for x in value], axis=0)

        return value.astype(dtype)

    if isinstance(value, list):

        if value and isinstance(value[0], (list, np.ndarray)):

            return np.stack([_nested_to_array(x, dtype) for x in value], axis=0)

        return np.array(value, dtype=dtype)

    return np.array(value, dtype=dtype)





def _normalize_solver_record(rec: dict[str, Any]) -> dict[str, Any]:

    legal = _nested_to_array(rec["legal_mask"], bool)

    rec = dict(rec)

    rec["actions"] = _nested_to_array(rec["actions"], np.float32)

    rec["position"] = _nested_to_array(rec["position"], np.float32)

    rec["legal_mask"] = legal

    rec["solver_policy"] = _normalize_solver_policy(rec["solver_policy"], legal)

    rec["weva_bucket_id"] = int(rec["weva_bucket_id"])

    rec["solver_label"] = int(rec["solver_label"])

    rec["label_source"] = str(rec["label_source"])

    return rec





def _normalize_solver_policy(

    policy: list[float] | np.ndarray,

    legal_mask: np.ndarray,

) -> np.ndarray:

    arr = np.asarray(policy, dtype=np.float32).reshape(-1)

    if arr.shape[0] != NUM_ACTIONS:

        raise ValueError(f"solver_policy must have length {NUM_ACTIONS}, got {arr.shape[0]}")

    legal = legal_mask.astype(bool)

    arr = arr * legal.astype(np.float32)

    total = arr.sum()

    if total <= 0:

        raise ValueError("solver_policy has zero mass on legal actions")

    return arr / total





class SolverAnnotatedParquetDataset(Dataset):

    """Strict ToolPoker solver Parquet loader (requires label_source=toolpoker_solver)."""



    def __init__(self, parquet_glob: str | Path) -> None:

        pattern = str(parquet_glob)

        paths = sorted(glob(pattern))

        if not paths:

            raise FileNotFoundError(f"No Parquet files match {pattern!r}")



        try:

            import pandas as pd

        except ImportError as exc:

            raise ImportError("pandas required for SolverAnnotatedParquetDataset") from exc



        frames = [pd.read_parquet(p) for p in paths]

        df = pd.concat(frames, ignore_index=True)

        self.records: list[dict[str, Any]] = []

        for i, row in df.iterrows():

            raw = row.to_dict()

            if raw.get("label_source") == "pokerbench_csv":

                raise ValueError(

                    f"Row {i} in {paths[0]} has label_source=pokerbench_csv. "

                    "Use --mode bc for BC Parquet, not solver mode."

                )

            if "label_source" not in raw or raw["label_source"] is None:

                raise ValueError(

                    f"Missing required solver column 'label_source' in {paths[0]} row {i}"

                )

            rec = _normalize_solver_record(raw)

            _validate_solver_row(rec, source=f"{paths[0]} row {i}")

            self.records.append(rec)



        if not self.records:

            raise ValueError(f"No valid rows in Parquet files matching {pattern!r}")



    def __len__(self) -> int:

        return len(self.records)



    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:

        r = self.records[idx]

        return {

            "weva_ids": torch.tensor(r["weva_bucket_id"], dtype=torch.long),

            "actions": torch.from_numpy(r["actions"]),

            "position": torch.from_numpy(r["position"]),

            "legal_mask": torch.from_numpy(r["legal_mask"]),

            "solver_label": torch.tensor(r["solver_label"], dtype=torch.long),

            "solver_policy": torch.from_numpy(r["solver_policy"]),

        }





def collate_sft_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:

    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}





def weva_sft_loss(

    out: PolicyOutput,

    solver_label: torch.Tensor,

    solver_policy: torch.Tensor,

    legal_mask: torch.Tensor,

    beta: float = 0.3,

) -> tuple[torch.Tensor, dict[str, float]]:

    """

    Combined loss: hard solver CE + soft policy KL distillation.

    KL computed in float32 for AMP stability.

    """

    ce_hard = F.cross_entropy(out.logits, solver_label)



    with torch.amp.autocast("cuda", enabled=False):

        logits_f32 = out.logits.float()

        log_probs = F.log_softmax(logits_f32, dim=-1)

        target = solver_policy.float()

        legal = legal_mask.float()

        target = target * legal

        target = target / target.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        kl = F.kl_div(log_probs, target, reduction="batchmean")



    loss = ce_hard + beta * kl

    return loss, {"ce_hard": float(ce_hard.item()), "kl_soft": float(kl.item())}





def make_sft_dataloader(

    parquet_glob: str | Path,

    batch_size: int = 64,

    shuffle: bool = True,

    num_workers: int = 0,

) -> DataLoader:

    ds = SolverAnnotatedParquetDataset(parquet_glob)

    return DataLoader(

        ds,

        batch_size=batch_size,

        shuffle=shuffle,

        collate_fn=collate_sft_batch,

        num_workers=num_workers,

    )


