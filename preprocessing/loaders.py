"""讀取 parquet 原始資料，產生純 dict 的 record 串流。

只負責 I/O 與最小正規化，不做格式轉換（轉換在 converters.py）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from . import config


def _to_plain_messages(raw) -> list[dict[str, str]]:
    """把 parquet 讀出的 messages（list<struct>）正規化成純 dict list。"""
    messages = []
    for turn in list(raw):
        messages.append({"role": str(turn["role"]), "content": str(turn["content"])})
    return messages


def load_attackqa(path: Path | str = config.ATTACKQA_PATH) -> Iterator[dict]:
    """逐筆產生 AttackQA record。

    只讀取需要的欄位：question / answer / document / source。
    """
    df = pd.read_parquet(path, columns=["question", "answer", "document", "source"])
    for idx, row in enumerate(df.itertuples(index=False)):
        yield {
            "question": row.question,
            "answer": row.answer,
            "document": row.document,
            "source": row.source,
            "_idx": idx,
        }


def load_primus(data_dir: Path | str = config.PRIMUS_DIR) -> Iterator[tuple[str, dict]]:
    """逐筆產生 (scenario, record)；scenario 取自 parquet 檔名。"""
    for parquet_path in sorted(Path(data_dir).glob("*.parquet")):
        scenario = parquet_path.stem
        df = pd.read_parquet(parquet_path)
        for idx, (_, row) in enumerate(df.iterrows()):
            yield scenario, {
                "prompt_id": row.get("prompt_id"),
                "messages": _to_plain_messages(row["messages"]),
                "_idx": idx,
            }
