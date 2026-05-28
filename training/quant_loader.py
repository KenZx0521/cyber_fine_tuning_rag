"""4-bit QLoRA 量化設定的集中工廠。

把 bnb 設定抽離 lora_loader，便於：
  - 訓練 / 推論兩條路徑共用同一份量化策略
  - 單元測試（無 GPU、無下載）驗證 BitsAndBytesConfig 屬性
  - 命中數防呆（避免 silently 全 bf16 載入）

關鍵設計決策（見 plan 已驗證的關鍵事實）：
  - quant_type="nf4"、compute_dtype=bf16、use_double_quant：QLoRA 標準。
  - bnb_4bit_quant_storage="bfloat16"：避免 grad ckpt + uint8 storage 的歷史踩雷，FSDP/save 也友善。
  - llm_int8_skip_modules=["visual","lm_head","linear_attn"]：
    - "visual" 凍結且維持 bf16 多模態推論精度
    - "lm_head" 248k vocab × 5120 量化會在 last-token logits 引入精度損失
    - "linear_attn" 子樹（48 層 GatedDeltaNet）SSM 走 fp32，bf16 input × 4-bit weight 會在
      SSM 邊界出 dtype 危險 —— 子樹整段保持 bf16，SSM 內部會自行 cast 到 fp32
"""

from __future__ import annotations

from . import config


def build_bnb_config(
    load_in_4bit: bool = True,
    *,
    compute_dtype: str | None = None,
    quant_type: str | None = None,
    use_double_quant: bool | None = None,
    skip_modules: list[str] | None = None,
):
    """組 BitsAndBytesConfig。預設值取自 training.config，便於統一覆寫。

    僅支援 load_in_4bit=True；8-bit 量化未驗證且 SSM 邊界 dtype 風險更高，明確不支援。
    """
    if not load_in_4bit:
        raise ValueError(
            "本 pipeline 僅支援 4-bit QLoRA（bf16 LoRA 由 lora_loader 直接以 dtype=bfloat16 載入，"
            "不透過 BitsAndBytesConfig）；8-bit 量化未驗證且 SSM 邊界 dtype 風險更高，明確不支援。"
        )
    import torch
    from transformers import BitsAndBytesConfig

    dtype_str = compute_dtype or config.BNB_4BIT_COMPUTE_DTYPE
    compute_dtype_torch = getattr(torch, dtype_str)

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type or config.BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=compute_dtype_torch,
        bnb_4bit_use_double_quant=(
            use_double_quant
            if use_double_quant is not None
            else config.BNB_4BIT_USE_DOUBLE_QUANT
        ),
        # storage 對齊 compute dtype：避免 uint8 預設值與 grad ckpt 的歷史踩雷。
        bnb_4bit_quant_storage=compute_dtype_torch,
        llm_int8_skip_modules=list(skip_modules or config.BNB_4BIT_SKIP_MODULES),
    )


def assert_quantization_applied(model, expected_min_4bit_modules: int = 200) -> None:
    """斷言模型已正確量化（防 silently 全 bf16 載入或 skip_modules 過度跳過）。

    走 named_modules 計數 bitsandbytes.nn.Linear4bit 命中數；少於門檻則 raise。
    成功時印命中數 + 前 3 個範例（除錯用）。
    """
    import bitsandbytes as bnb

    quantized = [n for n, m in model.named_modules() if isinstance(m, bnb.nn.Linear4bit)]
    n = len(quantized)
    if n < expected_min_4bit_modules:
        raise AssertionError(
            f"4-bit 量化未生效或命中過少：找到 {n} 個 Linear4bit "
            f"(預期至少 {expected_min_4bit_modules})；"
            f"前 3 個範例：{quantized[:3]}"
        )
    print(f"4-bit 量化命中：{n} 個 Linear4bit 模組（前 3：{quantized[:3]}）")
