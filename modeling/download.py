"""下載目標模型權重到 HF cache（resumable、可選 hf_transfer 加速）。

權重落在預設 HF cache（~/.cache/huggingface/hub，在 repo 外、磁碟充足），
之後 from_pretrained(MODEL_ID) 會自動找到，不需搬移。

用法：
    uv run python -m modeling.download
    uv run python -m modeling.download --no-hf-transfer   # 網路不穩時較穩定
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import config


def download(model_id: str = config.MODEL_ID, high_performance: bool = True) -> str:
    """下載完整 snapshot，回傳本地 snapshot 路徑。

    huggingface_hub 1.x 以 Xet（hf_xet，hub 內建相依）作高效分塊傳輸，
    取代已棄用的 hf_transfer。開啟 HF_XET_HIGH_PERFORMANCE 以最大化吞吐。
    """
    if high_performance:
        os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=model_id)


def _count_shards(path: str) -> int:
    return len(sorted(Path(path).glob("model-*-of-*.safetensors")))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="下載目標模型權重到 HF cache")
    parser.add_argument("--model-id", default=config.MODEL_ID)
    parser.add_argument(
        "--no-fast",
        action="store_true",
        help="停用 Xet 高效傳輸（網路不穩時較穩定）",
    )
    args = parser.parse_args(argv)

    print(f"開始下載：{args.model_id}")
    print(f"（約 72GB / 預期 {config.EXPECTED_SHARD_COUNT} shards，可中斷續傳）")
    path = download(args.model_id, high_performance=not args.no_fast)

    n = _count_shards(path)
    ok = n == config.EXPECTED_SHARD_COUNT
    print(f"\n完成。snapshot 路徑：{path}")
    print(
        f"safetensors shard 數：{n}/{config.EXPECTED_SHARD_COUNT} "
        f"{'✓' if ok else '⚠ 數量不符，請重跑以續傳'}"
    )
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
