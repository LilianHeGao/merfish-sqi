from __future__ import annotations

import json
from typing import Dict, Any

import pandas as pd


def write_spots_parquet(df: pd.DataFrame, path: str) -> None:
    df.to_parquet(path, index=False)


def read_spots_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_spots_meta(meta: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)


def read_spots_meta(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)
