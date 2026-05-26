"""載入 qwen3_5_moe（多模態 VLM + 混合 MoE）模型與 processor，並輸出診斷。

只負責「載入」與診斷，不含訓練/量化邏輯。進入點見 modeling/smoke_test.py。
"""

from __future__ import annotations

from collections import Counter

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from . import config


def _auto_max_memory(headroom_gib: int = config.GPU_HEADROOM_GIB) -> dict | None:
    """依目前可用 VRAM 推算 device 0 的權重上限，預留 headroom 給 activation/KV cache。

    可用 VRAM 與模型大小（~72GB）接近時，全塞 GPU 會在生成階段 OOM；這裡保留 headroom，
    超出的權重交給 accelerate offload 到 CPU。無 CUDA 時回傳 None（交給 device_map 處理）。
    """
    if not torch.cuda.is_available():
        return None
    free_bytes, _ = torch.cuda.mem_get_info()
    gpu_budget = max(int(free_bytes / 1024**3) - headroom_gib, 1)
    return {0: f"{gpu_budget}GiB", "cpu": config.CPU_MEM_BUDGET}


def load_model_and_processor(
    model_id: str = config.MODEL_ID,
    dtype: str = config.DTYPE,
    attn_impl: str = config.ATTN_IMPL,
    device_map: str | dict = config.DEVICE_MAP,
    max_memory: dict | None = None,
):
    """以 bf16 載入完整 checkpoint。回傳 (model, processor)。

    - transformers 5.x 用 `dtype=`（舊 `torch_dtype` 已棄用）；接受字串 "bfloat16"。
    - device_map="auto" 由 accelerate 放置，VRAM 不足時自動 offload 到 CPU RAM。
    - max_memory 未指定且 device_map="auto" 時，依目前可用 VRAM 自動預留 headroom
      （見 _auto_max_memory），避免權重幾乎塞滿 GPU 後在生成階段 OOM。
    """
    if device_map == "auto" and max_memory is None:
        max_memory = _auto_max_memory()

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device_map,
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
        max_memory=max_memory,
    )
    model.eval()
    return model, processor


def report_diagnostics(model) -> None:
    """印出載入後的關鍵診斷：裝置、參數量、dtype、kernel 可用性、VRAM。"""
    print("=" * 60)
    print("模型載入診斷")
    print("=" * 60)

    # --- torch / CUDA / GPU ---
    print(f"torch 版本：{torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f"CUDA 可用：{cuda_ok}")
    if cuda_ok:
        cap = torch.cuda.get_device_capability()
        name = torch.cuda.get_device_name(0)
        match = (
            "✓"
            if cap == config.EXPECTED_CUDA_CAPABILITY
            else "✗（可能未抓到 Blackwell wheel）"
        )
        print(f"GPU：{name}  compute capability sm_{cap[0]}{cap[1]} {match}")

    # --- 參數量與 dtype 分佈 ---
    total = sum(p.numel() for p in model.parameters())
    dtype_counts: dict = Counter()
    for p in model.parameters():
        dtype_counts[p.dtype] += p.numel()
    print(f"總參數量：{total / 1e9:.2f} B")
    for dt, n in dtype_counts.most_common():
        print(f"  {str(dt):24} {n / 1e9:6.2f} B ({n / total * 100:5.1f}%)")

    # --- device_map 放置（cuda vs cpu/disk 模組數）---
    dmap = getattr(model, "hf_device_map", None)
    if dmap:
        placement = Counter(str(v) for v in dmap.values())
        print(f"device_map 放置（模組數）：{dict(placement)}")

    # --- 混合層 kernel 可用性（False = 走純 torch fallback，僅較慢、不影響正確性）---
    try:
        from transformers.utils.import_utils import (
            is_causal_conv1d_available,
            is_flash_linear_attention_available,
        )

        print(f"causal_conv1d 可用：{is_causal_conv1d_available()}")
        print(f"flash_linear_attention(fla) 可用：{is_flash_linear_attention_available()}")
    except Exception as exc:  # noqa: BLE001 — 診斷用途，任何匯入問題都只提示不中斷
        print(f"（無法查詢 kernel 可用性：{exc}）")

    # --- VRAM ---
    if cuda_ok:
        gib = 1024**3
        alloc = torch.cuda.memory_allocated() / gib
        reserved = torch.cuda.memory_reserved() / gib
        free, total_mem = torch.cuda.mem_get_info()
        print(
            f"VRAM：已配置 {alloc:.1f} GiB、保留 {reserved:.1f} GiB、"
            f"裝置剩餘 {free / gib:.1f}/{total_mem / gib:.1f} GiB"
        )
