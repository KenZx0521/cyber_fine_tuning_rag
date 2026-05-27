"""QLoRA 微調 pipeline：在 qwen3_5_moe（MoE VLM）上以 4-bit NF4 + LoRA 訓練資安助理。

模組職責：
- config.py          集中常數（LoRA / 超參 / 路徑），不 import torch
- data.py            載入 train/val jsonl、把 per-source 取樣權重展開成 per-example
- lora_loader.py     bf16 載入 + gradient checkpointing + get_peft_model（LoRA）
- weighted_trainer.py  WeightedSFTTrainer：覆寫 _get_train_sampler 接上加權取樣
- train.py           進入點（含 --smoke 煙霧測試）
- merge_adapter.py   把 LoRA adapter merge 回 bf16 base 並驗證

進入點皆可 `uv run python -m training.<module>` 執行。
"""
