"""訓練用模型載入：bf16 載入 base → 啟用 gradient checkpointing → 掛 LoRA。

原計畫 4-bit QLoRA，但此模型 97% 參數在 fused MoE experts（Qwen3_5MoeExperts 的
3D Parameter），bitsandbytes 只能量化 nn.Linear、無法量化 fused experts（transformers
v5 已知問題 bnb#1849）—— 4-bit 反因 60GiB experts 維持 bf16 而省不到記憶體。
故改 bf16 LoRA：模型 bf16 約 65GiB，單卡 95GiB 可容納；LoRA 仍掛 attention +
shared_expert dense FFN。

與 modeling/loader.py（推論 bf16 載入）並列：此處 device_map 全塞單卡（不 offload）、
啟用 gradient checkpointing 並掛 LoRA 供訓練。
"""

from __future__ import annotations

from collections import Counter

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor

from . import config


def build_lora_config(target_modules: list[str] | None = None) -> LoraConfig:
    """組 LoRA 設定（預設掛 full-attn 投影 + shared_expert dense FFN，見 config.TARGET_MODULES）。"""
    return LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        bias=config.LORA_BIAS,
        task_type="CAUSAL_LM",
        target_modules=list(target_modules or config.TARGET_MODULES),
    )


def load_base(model_id: str = config.MODEL_ID, device_map: str | dict = config.DEVICE_MAP):
    """以 bf16 載入完整 checkpoint（不量化）。回傳 (model, processor)，尚未掛 LoRA。"""
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=config.DTYPE,  # transformers 5.x：dtype= 取代舊 torch_dtype=
        attn_implementation=config.ATTN_IMPL,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False  # 訓練必關（與 gradient checkpointing 相容）
    return model, processor


def attach_lora(
    model,
    target_modules: list[str] | None = None,
    gradient_checkpointing: bool = config.GRADIENT_CHECKPOINTING,
):
    """啟用 gradient checkpointing → 掛 LoRA（get_peft_model 自動凍結 base）。

    bf16 LoRA 不用 prepare_model_for_kbit_training（那會把 bf16 experts upcast
    fp32 而爆記憶體）。grad ckpt + 凍結 base 時，需 enable_input_require_grads 讓輸入
    embedding 的輸出 require grad，否則梯度無法回傳到 LoRA adapter。
    """
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    model = get_peft_model(model, build_lora_config(target_modules))
    return model


def load_lora_model_and_processor(
    model_id: str = config.MODEL_ID,
    target_modules: list[str] | None = None,
    device_map: str | dict = config.DEVICE_MAP,
    gradient_checkpointing: bool = config.GRADIENT_CHECKPOINTING,
):
    """一站式：bf16 載入 base → 掛 LoRA。回傳 (peft_model, processor)。"""
    model, processor = load_base(model_id, device_map=device_map)
    model = attach_lora(model, target_modules, gradient_checkpointing)
    return model, processor


# --- 診斷與斷言（對齊 modeling/loader.report_diagnostics 風格） ---


def report_model(model) -> None:
    """印參數 dtype 分佈與 VRAM（確認 bf16 載入與裝置放置）。"""
    dtype_counts: Counter = Counter()
    for p in model.parameters():
        dtype_counts[str(p.dtype)] += p.numel()
    total = sum(dtype_counts.values()) or 1
    print("=" * 60)
    print("訓練模型載入診斷（bf16 LoRA）")
    print("=" * 60)
    print("參數 dtype 分佈：")
    for dt, n in sorted(dtype_counts.items(), key=lambda x: -x[1]):
        print(f"  {dt:22s} {n / 1e9:7.3f} B ({n / total * 100:5.1f}%)")
    if torch.cuda.is_available():
        gib = 1024**3
        free, total_mem = torch.cuda.mem_get_info()
        print(
            f"VRAM：已配置 {torch.cuda.memory_allocated() / gib:.1f} GiB、"
            f"保留 {torch.cuda.memory_reserved() / gib:.1f} GiB、"
            f"裝置剩餘 {free / gib:.1f}/{total_mem / gib:.1f} GiB"
        )


def assert_only_lora_trainable(model) -> None:
    """斷言只有 LoRA adapter 可訓練、vision tower 未解凍；印可訓練參數量。"""
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    non_lora = [n for n in trainable if "lora" not in n.lower()]
    vision = [n for n in trainable if "visual" in n]
    if non_lora:
        raise AssertionError(f"有非-LoRA 參數可訓練：{non_lora[:5]}")
    if vision:
        raise AssertionError(f"vision tower 被解凍：{vision[:5]}")

    n_trainable = sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)
    print(f"可訓練模組數：{len(trainable)}；可訓練參數：{n_trainable / 1e6:.2f} M")
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
