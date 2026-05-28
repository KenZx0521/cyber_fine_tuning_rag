"""Meta-device dump：驗證 training.config.TARGET_MODULES 在目標模型上的命中數。

用途：避免 target_modules 拼錯造成 LoRA silently 0 命中（PEFT 對 target_modules
不做 fail-fast；找不到匹配就靜默掛 0 個 adapter）。

不下載權重、不需 GPU；以 accelerate.init_empty_weights() 在 meta device 建模型
skeleton，純靠 named_modules 結構驗證。

用法：
    uv run python scripts/dump_target_modules.py
    uv run python scripts/dump_target_modules.py --extended    # 含 linear_attn 進階目標

預期命中（新模型 qwen3_5 27B dense）：
    q_proj / k_proj / v_proj / o_proj   16（self_attn 中的 full-attention 層）
    gate_proj / up_proj / down_proj     64（全 64 層 dense MLP）
    in_proj_*  /  out_proj              48（GatedDeltaNet linear_attention 層；僅 --extended）
    vision 命中                          0（vision 用 linear_fc1/linear_fc2，與 target_modules 不衝突）
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

# scripts/ 不是 package；直接執行時需把 repo root 加進 sys.path 才能 import modeling/training。
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch.nn as nn  # noqa: E402
from accelerate import init_empty_weights  # noqa: E402
from transformers import AutoConfig, AutoModelForImageTextToText  # noqa: E402

from modeling.config import MODEL_ID  # noqa: E402
from training import config as tcfg  # noqa: E402


def build_skeleton():
    """以 meta device 建模型 skeleton（不分配真實權重，秒級完成、零 VRAM）。"""
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    with init_empty_weights():
        model = AutoModelForImageTextToText.from_config(cfg)
    return model


def count_hits(model, target_modules: list[str]) -> dict[str, list[str]]:
    """對每個 target_modules 名稱，列出所有 endswith 該名稱的 nn.Linear 路徑。

    PEFT 的 target_modules 匹配規則：substring 比對 module name；這裡用 endswith
    模擬「精確子模組名稱」匹配，避免誤命中（例如 q_proj 不會被誤認到 q_proj_xyz）。
    """
    hits: dict[str, list[str]] = defaultdict(list)
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        for tgt in target_modules:
            if name.endswith("." + tgt) or name == tgt:
                hits[tgt].append(name)
                break  # 一個 module 最多歸一個 target
    return hits


def report(hits: dict[str, list[str]], target_modules: list[str]) -> None:
    print("=" * 70)
    print(f"target_modules 命中報告（模型：{MODEL_ID}）")
    print("=" * 70)
    total = 0
    for tgt in target_modules:
        paths = hits.get(tgt, [])
        n = len(paths)
        total += n
        head = paths[:3]
        print(f"  {tgt:18s} 命中 {n:4d}  範例：{head}")
    print("-" * 70)
    print(f"  合計：{total} 個 nn.Linear 會掛上 LoRA")

    # Vision 衝突檢查
    vision_hits = sum(1 for paths in hits.values() for p in paths if "visual" in p)
    print(f"  vision 命中（應為 0）：{vision_hits}")
    if vision_hits:
        print("  ⚠️  WARNING：target_modules 命中了 vision tower，與凍結策略衝突")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--extended",
        action="store_true",
        help="使用 TARGET_MODULES_WITH_LINEAR_ATTN（含 linear_attn 投影）",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    targets = tcfg.TARGET_MODULES_WITH_LINEAR_ATTN if args.extended else tcfg.TARGET_MODULES
    print(f">> 載入 meta-device skeleton：{MODEL_ID}")
    model = build_skeleton()
    print(">> 計算命中…")
    hits = count_hits(model, targets)
    report(hits, targets)

    # 預期命中數（hard-coded for catching regressions）
    expected = {"q_proj": 16, "k_proj": 16, "v_proj": 16, "o_proj": 16,
                "gate_proj": 64, "up_proj": 64, "down_proj": 64}
    if args.extended:
        expected.update({
            "in_proj_qkv": 48, "in_proj_z": 48,
            "in_proj_b": 48, "in_proj_a": 48, "out_proj": 48,
        })
    bad = []
    for k, v in expected.items():
        if k in targets and len(hits.get(k, [])) != v:
            bad.append(f"{k}: 預期 {v}、實際 {len(hits.get(k, []))}")
    if bad:
        print("\n⚠️  命中數與預期不符：")
        for b in bad:
            print(f"  - {b}")
        sys.exit(1)
    print("\n✓ 全部命中數符合預期")


if __name__ == "__main__":
    main()
