"""訓練資料載入與 per-source 取樣權重展開。

把純函式（build_example_weights）與 I/O（load_*）分離，讓權重展開邏輯可在
不載入 datasets / 真模型的情況下單元測試。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from . import config


def load_sampler_weights(path: str | Path = config.SAMPLER_WEIGHTS_PATH) -> dict[str, float]:
    """讀 {source: 重複倍率} 權重表（由 scripts/compute_sampler_weights.py 產出）。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_example_weights(
    sources: Sequence[str], weights_map: Mapping[str, float]
) -> list[float]:
    """把 per-source 取樣倍率展開成 per-example 權重 list。

    WeightedRandomSampler 吃的是「每筆樣本」的權重；本函式依每筆的 source
    查表展開。缺權重的 source 直接報錯（fail fast，避免靜默用錯權重）。
    """
    missing = sorted(set(sources) - set(weights_map))
    if missing:
        raise KeyError(f"sampler_weights 缺少這些來源的權重：{missing}")
    return [float(weights_map[s]) for s in sources]


def load_sft_datasets(
    train_path: str | Path = config.TRAIN_PATH,
    val_path: str | Path = config.VAL_PATH,
):
    """載入 train/val jsonl（chat messages 格式，保留 source/category/id 欄位）。

    回傳 (train_dataset, val_dataset)，皆為 datasets.Dataset。
    """
    from datasets import load_dataset

    ds = load_dataset(
        "json",
        data_files={"train": str(train_path), "validation": str(val_path)},
    )
    return ds["train"], ds["validation"]
