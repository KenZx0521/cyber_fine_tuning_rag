"""訓練用模型載入：bf16 或 4-bit 載入 base → 啟用 gradient checkpointing → 掛 LoRA。

新模型（qwen3_5 dense）已無 fused MoE experts，全是 nn.Linear → bitsandbytes 可正常量化，
bnb#1849 不再阻擋。預設走 4-bit QLoRA（NF4 + bf16 compute + double quant）；
bf16 LoRA 仍保留為 fallback（--quantize bf16），由 CLI 旗標切換。

關鍵差異（見 plan 風險 R2）：
  - 4-bit 路徑須呼叫 prepare_model_for_kbit_training 註冊量化 hook；但 use_reentrant=False
    分支會跳過 enable_input_require_grads()，故後面仍要手動呼叫一次。
  - bf16 路徑不得呼叫 prepare_model_for_kbit_training（會把 bf16 全升 fp32 → OOM）。

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
    """組 LoRA 設定（預設掛 full-attn 投影 + dense FFN，見 config.TARGET_MODULES）。"""
    return LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        bias=config.LORA_BIAS,
        task_type="CAUSAL_LM",
        target_modules=list(target_modules or config.TARGET_MODULES),
    )


def load_base(
    model_id: str = config.MODEL_ID,
    device_map: str | dict = config.DEVICE_MAP,
    *,
    load_in_4bit: bool = False,
):
    """載入完整 checkpoint。回傳 (model, processor)，尚未掛 LoRA。

    - load_in_4bit=False（預設）：bf16 完整載入。
    - load_in_4bit=True：4-bit NF4 量化載入，設定由 quant_loader.build_bnb_config 提供。
    """
    processor = AutoProcessor.from_pretrained(model_id)
    kwargs = dict(
        dtype=config.DTYPE,  # transformers 5.x：dtype= 取代舊 torch_dtype=
        attn_implementation=config.ATTN_IMPL,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    if load_in_4bit:
        from .quant_loader import build_bnb_config

        kwargs["quantization_config"] = build_bnb_config(load_in_4bit=True)
    model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    model.config.use_cache = False  # 訓練必關（與 gradient checkpointing 相容）

    # MTP fail-fast（R4）：transformers 5.9 已忽略 MTP keys，若未來版本變動，這裡會立刻爆。
    mtp_params = sum(1 for n, _ in model.named_parameters() if n.startswith("mtp"))
    if mtp_params:
        raise AssertionError(
            f"預期 MTP 在載入時被 _keys_to_ignore_on_load_unexpected 丟棄，"
            f"但找到 {mtp_params} 個 mtp.* 參數；transformers 行為可能已變動，"
            "需重新審視 target_modules 與 freeze 白名單。"
        )
    return model, processor


def attach_lora(
    model,
    target_modules: list[str] | None = None,
    gradient_checkpointing: bool = config.GRADIENT_CHECKPOINTING,
    *,
    is_quantized: bool = False,
):
    """啟用 gradient checkpointing → 掛 LoRA（get_peft_model 自動凍結 base）。

    - bf16 路徑（is_quantized=False）：直接 gradient_checkpointing_enable + enable_input_require_grads。
      不可呼叫 prepare_model_for_kbit_training（會把 bf16 全升 fp32 → OOM）。
    - 4-bit 路徑（is_quantized=True）：先 prepare_model_for_kbit_training 註冊量化 hook 與 grad ckpt；
      use_reentrant=False 分支會跳過 enable_input_require_grads()，故後面手動呼叫一次。
    """
    if is_quantized:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        if gradient_checkpointing:
            # use_reentrant=False 分支不註冊 hook，需手動補上讓 input embedding 輸出 require grad。
            model.enable_input_require_grads()
    else:
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
    *,
    load_in_4bit: bool = False,
):
    """一站式：載入 base（bf16 或 4-bit）→ 掛 LoRA。回傳 (peft_model, processor)。"""
    model, processor = load_base(model_id, device_map=device_map, load_in_4bit=load_in_4bit)
    model = attach_lora(model, target_modules, gradient_checkpointing, is_quantized=load_in_4bit)
    return model, processor


# --- 診斷與斷言（對齊 modeling/loader.report_diagnostics 風格） ---


def report_model(model) -> None:
    """印參數 dtype 分佈、量化模組計數、與 VRAM（確認載入與裝置放置）。"""
    dtype_counts: Counter = Counter()
    for p in model.parameters():
        dtype_counts[str(p.dtype)] += p.numel()
    total = sum(dtype_counts.values()) or 1
    print("=" * 60)
    print("訓練模型載入診斷")
    print("=" * 60)
    print("參數 dtype 分佈：")
    for dt, n in sorted(dtype_counts.items(), key=lambda x: -x[1]):
        print(f"  {dt:22s} {n / 1e9:7.3f} B ({n / total * 100:5.1f}%)")
    # 量化模組計數（若有）
    try:
        import bitsandbytes as bnb

        n_4bit = sum(1 for _, m in model.named_modules() if isinstance(m, bnb.nn.Linear4bit))
        if n_4bit:
            print(f"Linear4bit 模組數：{n_4bit}")
    except ImportError:
        pass
    if torch.cuda.is_available():
        gib = 1024**3
        free, total_mem = torch.cuda.mem_get_info()
        print(
            f"VRAM：已配置 {torch.cuda.memory_allocated() / gib:.1f} GiB、"
            f"保留 {torch.cuda.memory_reserved() / gib:.1f} GiB、"
            f"裝置剩餘 {free / gib:.1f}/{total_mem / gib:.1f} GiB"
        )


def assert_only_lora_trainable(model, min_lora_modules: int = 200) -> None:
    """斷言只有 LoRA adapter 可訓練、vision tower 未解凍；印可訓練參數量。

    min_lora_modules 防呆：若 target_modules 拼錯導致 silently 0 命中，這裡會 fail-fast。
    預期值：16 attn × 4 + 64 mlp × 3 = 256；保守設 200 容納少量名稱差異。
    """
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    non_lora = [n for n in trainable if "lora" not in n.lower()]
    vision = [n for n in trainable if "visual" in n]
    if non_lora:
        raise AssertionError(f"有非-LoRA 參數可訓練：{non_lora[:5]}")
    if vision:
        raise AssertionError(f"vision tower 被解凍：{vision[:5]}")

    # LoRA 模組計數（每個 target 模組會產出 lora_A + lora_B 兩個參數，故除以 2）
    n_lora_modules = len(trainable) // 2
    if n_lora_modules < min_lora_modules:
        raise AssertionError(
            f"LoRA 命中模組過少：{n_lora_modules}（預期至少 {min_lora_modules}）；"
            f"target_modules 可能拼錯或新模型命名變動。前 5 個 trainable：{trainable[:5]}"
        )

    n_trainable = sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)
    print(f"可訓練模組數：{len(trainable)}；可訓練參數：{n_trainable / 1e6:.2f} M")
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
