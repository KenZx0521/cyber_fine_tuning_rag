"""資安語料 fine-tuning 前處理套件。

將 data/fine-tuning/ 下的 parquet 原始資料轉成統一的 chat messages JSONL，
附帶來源標記、品質驗證與 train/val 切分。
"""
