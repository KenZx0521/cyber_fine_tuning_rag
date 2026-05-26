"""模型載入的集中設定：模型 ID、dtype、attention 實作與預設值。

對齊 preprocessing/config.py 的風格：相對 repo root 推導、不寫死絕對路徑。
刻意不在此 import torch —— dtype 以字串表示（transformers 接受字串 dtype），
讓 config 與 download 流程不必相依 torch（耦合更乾淨）。
"""

from __future__ import annotations

from pathlib import Path

# 沿用前處理的資安 system prompt，讓 smoke test 主題一致。preprocessing.config 僅依賴
# pathlib，import 成本極低且無循環依賴。
from preprocessing.config import SECURITY_SYSTEM_PROMPT

# --- 路徑 ---
REPO_ROOT = Path(__file__).resolve().parent.parent

# --- 目標模型 ---
# huihui-ai 的 abliterated 變體；架構為 qwen3_5_moe（多模態 VLM + 混合線性注意力 MoE）。
# 官方唯一受支援載入路徑：AutoProcessor + AutoModelForImageTextToText（不需 trust_remote_code）。
MODEL_ID = "huihui-ai/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated"
# 完整 checkpoint 為 26 個 safetensors shard（用於下載完整性檢查）。
EXPECTED_SHARD_COUNT = 26

# --- 載入預設值 ---
# bf16 完整載入（不量化）。以字串傳給 from_pretrained(dtype=...)；SSM 層依 config 內
# mamba_ssm_dtype 仍走 fp32，由 transformers 內部處理，無需我們介入。
DTYPE = "bfloat16"
# sdpa 不需額外編譯 flash-attn，且在 Blackwell(sm_120) 可用；必要時可改 "eager"。
ATTN_IMPL = "sdpa"
# device_map="auto" 讓 accelerate 依可用 VRAM 放置，不足時自動 offload 到 CPU RAM。
DEVICE_MAP = "auto"
# 預期 GPU 計算能力（Blackwell = sm_120）；smoke test 用來提示是否抓到正確的 torch wheel。
EXPECTED_CUDA_CAPABILITY = (12, 0)

# device_map="auto" 時，GPU 預留給 activation/KV cache 的空間（GiB）。模型權重約 72GB、
# 可用 VRAM 約 75GB 偏緊，全塞 GPU 會在生成階段 OOM；預留 headroom，超出的權重由
# accelerate offload 到 CPU。
GPU_HEADROOM_GIB = 6
CPU_MEM_BUDGET = "120GiB"

# --- smoke test ---
SMOKE_MAX_NEW_TOKENS = 64
# 一題純文字資安問題，驗證載入後能否連貫生成。
SMOKE_USER_PROMPT = (
    "In one short paragraph, explain the MITRE ATT&CK technique T1059 "
    "(Command and Scripting Interpreter) and give one practical detection idea."
)
