"""讀取 parquet 原始資料，產生純 dict 的 record 串流。

只負責 I/O 與最小正規化，不做格式轉換（轉換在 converters.py）。
"""

from __future__ import annotations

import json
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


def unescape_text(s: str, max_rounds: int = 6) -> str:
    """還原「轉義過」的文字欄位（Trendyol / Fenrir 的 *_escaped.jsonl）。

    這類資料把真換行等存成字面的轉義序列（如 \\n），且不同來源轉義深度不同
    （Fenrir 1 層、Trendyol 2 層）。作法：反覆用 JSON 解一層轉義，直到 round-trip
    失敗或字串不再變化——一旦出現真正的控制字元（如換行），下一輪 wrap-parse 必然
    失敗而自動停在正確深度。保留 UTF-8（不使用會破壞非 ASCII 的 unicode_escape）。
    """
    for _ in range(max_rounds):
        try:
            nxt = json.loads('"' + s + '"')
        except Exception:
            break
        if nxt == s:
            break
        s = nxt
    return s


def load_triplet(data_dir: Path | str) -> Iterator[dict]:
    """讀取 system/user/assistant 三欄 JSONL（Trendyol / Fenrir），逐筆還原轉義後 yield。

    來源無關（source 由 converter 決定）；每個資料夾預期一份 *.jsonl。
    """
    idx = 0
    for jsonl_path in sorted(Path(data_dir).glob("*.jsonl")):
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                yield {
                    "system": unescape_text(str(obj.get("system", ""))),
                    "user": unescape_text(str(obj.get("user", ""))),
                    "assistant": unescape_text(str(obj.get("assistant", ""))),
                    "_idx": idx,
                }
                idx += 1


def load_primus_reasoning(
    data_dir: Path | str = config.PRIMUS_REASONING_DIR,
) -> Iterator[dict]:
    """讀取 Primus-Reasoning（messages 格式；推理用特殊 token 包在 assistant content）。

    支援 parquet 或 jsonl；找不到資料檔時不產生任何 record（gated，未下載時安全略過）。
    """

    def _data_files(suffix: str) -> list[Path]:
        return [
            p
            for p in sorted(Path(data_dir).rglob(f"*.{suffix}"))
            if ".cache" not in p.parts
        ]

    idx = 0
    for path in _data_files("parquet"):
        variant = path.stem.replace("ctibench_", "")  # o1 / deepseek-r1
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            yield {
                "messages": _to_plain_messages(row["messages"]),
                "variant": variant,
                "_idx": idx,
            }
            idx += 1
    for path in _data_files("jsonl"):
        variant = path.stem.replace("ctibench_", "")
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                yield {
                    "messages": _to_plain_messages(obj.get("messages", [])),
                    "variant": variant,
                    "_idx": idx,
                }
                idx += 1
