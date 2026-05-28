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
# huihui-ai 的 abliterated 變體；架構為 qwen3_5（多模態 VLM + 混合注意力 dense；含 MTP 1 層）。
# 與舊 qwen3_5_moe 的關鍵差異：
#   - Dense MLP（mlp_only_layers: []，無 fused experts）→ bnb 可量化全部 nn.Linear，4-bit QLoRA 可用
#   - 64 層 hybrid attention：full_attention_interval=4 → 16 full-attn + 48 GatedDeltaNet linear_attn
#   - attn_output_gate=true（gate 從 q_proj chunk 出來，不增加額外 Linear）
#   - mtp_num_hidden_layers=1（transformers 5.9+ 載入時依 _keys_to_ignore_on_load_unexpected 丟棄）
# 官方唯一受支援載入路徑：AutoProcessor + AutoModelForImageTextToText（不需 trust_remote_code）。
MODEL_ID = "huihui-ai/Huihui-Qwen3.5-27B-Claude-4.6-Opus-abliterated"
# 完整 checkpoint 為 11 個 safetensors shard（bf16 ~55.5 GiB；用於下載完整性檢查）。
EXPECTED_SHARD_COUNT = 11

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

# device_map="auto" 時，GPU 預留給 activation/KV cache 的空間（GiB）。新模型 bf16 ~56 GiB
# 在 95 GiB 卡上綽綽有餘，理論上不需 offload；headroom 仍保留以兼顧 4-bit + 大 batch eval
# 或長 max_length 的峰值。
GPU_HEADROOM_GIB = 6
CPU_MEM_BUDGET = "120GiB"

# --- smoke test ---
SMOKE_MAX_NEW_TOKENS = 64
# 一題純文字資安問題，驗證載入後能否連貫生成。
SMOKE_USER_PROMPT = (
    "In one short paragraph, explain the MITRE ATT&CK technique T1059 "
    "(Command and Scripting Interpreter) and give one practical detection idea."
)
