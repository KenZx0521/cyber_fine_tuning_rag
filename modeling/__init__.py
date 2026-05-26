"""模型載入套件（fine-tuning 前置）。

把使用者指定的 qwen3_5_moe 模型在地端以 HF transformers 載入：
    uv run python -m modeling.download      # 下載權重（約 72GB）
    uv run python -m modeling.smoke_test    # 載入 + 純文字生成驗證

只負責「載入」與診斷，不含訓練/量化邏輯（留待後續 fine-tuning 實作）。
"""
