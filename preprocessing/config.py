"""前處理的集中設定：路徑、system prompt、門檻與預設值。"""

from __future__ import annotations

from pathlib import Path

# --- 路徑（皆相對於 repo root 推導，不寫死絕對路徑） ---
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
FINE_TUNING_DIR = DATA_DIR / "fine-tuning"
ATTACKQA_PATH = FINE_TUNING_DIR / "attackqa" / "attackqa.parquet"
PRIMUS_DIR = FINE_TUNING_DIR / "primus-instruct" / "data"
OUTPUT_DIR = DATA_DIR / "processed"

# --- System prompts（可由 CLI --no-system 關閉） ---
# AttackQA 採 RAG 式：明確指示模型依提供的 context 作答。
ATTACKQA_SYSTEM_PROMPT = (
    "You are a cybersecurity analyst specializing in the MITRE ATT&CK framework. "
    "Use the provided context to answer the question accurately and concisely. "
    "If the context is insufficient, rely on your security domain expertise."
)

# Primus：通用資安助理，同時保留一般指令跟隨能力（general 含少量非資安內容）。
SECURITY_SYSTEM_PROMPT = (
    "You are a helpful assistant with deep expertise in cybersecurity. "
    "Provide accurate, actionable, and well-structured answers for security "
    "operations, threat analysis, and related tasks."
)

# --- 門檻與預設值 ---
# token 估算採字元/4 的 heuristic（不依賴 tokenizer；可用 --hf-tokenizer 覆寫）。
CHARS_PER_TOKEN = 4
# 預設長度上限：Qwen 可吃 32k，但訓練多以較短序列為主；主要用來剔除 general 的極長多輪。
DEFAULT_MAX_TOTAL_TOKENS = 8192
DEFAULT_VAL_SIZE = 0.02
DEFAULT_SEED = 42

VALID_ROLES = ("system", "user", "assistant")
