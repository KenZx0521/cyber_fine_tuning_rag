"""QLoRA 訓練的集中設定：LoRA、4-bit 量化、訓練超參與路徑。

對齊 modeling/config.py 的風格：相對 repo root 推導、不寫死絕對路徑；
刻意不在此 import torch —— 量化的 compute dtype 以字串表示，由 quant_loader
轉成 torch dtype，讓 config 維持輕量、無重相依。

訓練情境（使用者已拍板）：2 epochs + α=0.5 / cap=3（已把雜燴的 primus/general 權重設 0 排除取樣），
沿用既有 data/processed/sampler_weights.json。
"""

from __future__ import annotations

from pathlib import Path

# 重用模型載入設定，避免 model id / dtype / attention 實作在多處漂移。
from modeling.config import ATTN_IMPL, DTYPE, MODEL_ID

# --- 路徑（皆相對 repo root 推導） ---
REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
TRAIN_PATH = PROCESSED_DIR / "train.jsonl"
VAL_PATH = PROCESSED_DIR / "val.jsonl"
SAMPLER_WEIGHTS_PATH = PROCESSED_DIR / "sampler_weights.json"
# 自訂訓練用 chat template（含 {% generation %} 標記，供 assistant-only loss）。
TRAIN_TEMPLATE_PATH = TRAINING_DIR / "qwen35_train_template.jinja"
# 輸出
OUTPUT_DIR = REPO_ROOT / "outputs" / "qlora-cyber"
RUN_DIR = OUTPUT_DIR / "run"  # 正式訓練輸出（checkpoints；訓練指標上報 wandb）
ADAPTER_DIR = RUN_DIR / "final-adapter"  # 訓練結束的最終 LoRA adapter（= merge 的預設輸入）
MERGED_DIR = REPO_ROOT / "outputs" / "qlora-cyber-merged"  # merge 後的完整 bf16 權重

# --- LoRA ---
# target_modules 決策（依 meta-device named_modules dump 校正，見 plan ⚠️#2/#3）：
# 模型每層結構為
#   self_attn(10 層 full-attn).{q,k,v,o}_proj           —— 標準 nn.Linear
#   linear_attn(30 層 GatedDeltaNet).{in_proj_*,out_proj} —— Linear，但接 SSM(fp32) 核心
#   mlp.experts (Qwen3_5MoeExperts)                       —— fused 3D Parameter（非 Linear）
#   mlp.gate (Qwen3_5MoeTopKRouter)                       —— router
#   mlp.shared_expert.{gate,up,down}_proj                 —— dense FFN，每 token 必過、梯度密集
#   mlp.shared_expert_gate                                —— 門控 Linear
# 預設掛 LoRA 於：full-attn 投影 + shared_expert 的 dense FFN
#   → 覆蓋 10 層 attention + 全 40 層 dense FFN 容量，皆為梯度密集路徑，穩定。
# 刻意排除：
#   - routed experts（fused Parameter，PEFT 掛不上且梯度稀疏 8/256）
#   - router gate / shared_expert_gate（門控敏感，擾動易使專家分配崩潰）
#   - linear_attn 投影（SSM fp32 路徑，首訓保守不動；穩定後可實驗加入，見下）
#   - vision tower / lm_head / embed_tokens
LORA_R = 16
LORA_ALPHA = 32  # 2×r 慣例
LORA_DROPOUT = 0.05
LORA_BIAS = "none"
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",  # full-attn（僅命中 10 層 self_attn，linear_attn 無此命名）
    "gate_proj",
    "up_proj",
    "down_proj",  # 僅命中 shared_expert（routed experts 為 fused、無此子模組名）
]
# 進階實驗用：額外納入 linear_attn 投影，覆蓋 30 層 token-mixing（首訓不啟用）。
TARGET_MODULES_WITH_LINEAR_ATTN = TARGET_MODULES + [
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
]

# --- 載入策略：bf16 LoRA（非量化） ---
# 原計畫 4-bit QLoRA，但實測此模型 97% 參數在 fused MoE experts（Qwen3_5MoeExperts
# 的 3D Parameter）：bitsandbytes 只量化 nn.Linear、無法量化 fused experts（transformers
# v5 已知問題 bnb#1849）—— 4-bit 反因 60GiB experts 維持 bf16 而省不到記憶體，且
# prepare_model_for_kbit_training 把 experts upcast fp32 會 OOM。對症套件 woct0rdho 亦
# 不支援本架構（VLM + GatedDeltaNet + transformers 5）。
# 故改 bf16 LoRA：模型 bf16 約 65GiB，單卡 95GiB 可容納，LoRA 仍掛 attention +
# shared_expert FFN，達成相同的單卡 LoRA 微調目標。
# device_map 全塞單卡（不 offload；訓練時 offload 會災難性變慢）。
DEVICE_MAP: str | dict = {"": 0}

# --- 訓練超參（情境：2 epochs + α=0.5 / cap=3；單卡 RTX PRO 6000 95GiB） ---
# 加速調整（2026-05-28）：本模型為 A3B 稀疏 MoE，吞吐量取決於「每次 forward 餵給 expert 的
# token 數」。原 bs=2 嚴重餵不飽 256 個 expert（每 expert 僅 ~38 token、GEMM 受記憶體頻寬限制）。
# 故 bs 2→3（grad_accum 8→5），並啟用長度分組取樣砍 padding 浪費。
# bs=4 實測 OOM：248k-vocab 的 logits（compute_loss 的 shift_logits.contiguous）在最長 batch
# (4×2048) 需 ~7.6GiB 連續記憶體；每增 1 bs 約 ~8GiB，weights 固定 65GiB → bs=4 峰值 ~98GiB 超出
# 95GiB；bs=3 峰值 ~90GiB 可容（留 ~5GiB）。搭 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 減碎片。
NUM_TRAIN_EPOCHS = 2
PER_DEVICE_TRAIN_BATCH_SIZE = 3  # MoE expert GEMM 餵料↑50%；bs=4 因 logits 連續複製 OOM
GRADIENT_ACCUMULATION_STEPS = 5  # 有效 batch = 3×5 = 15（≈ 原 16）
# eval 時訓練 allocator 已保留 ~87GiB（峰值快取不釋放），僅剩 ~7GiB；而 248k-vocab logits 的
# shift_logits.contiguous() 按 eval_bs 線性放大（實測 bs=8 需 ~10.5GiB → OOM）。bs=2（~2.6GiB）穩妥。
# 註：prediction_loss_only 只避免「跨 batch 累積預測」，消不掉單 batch 的 logits 峰值，故須小 eval batch。
PER_DEVICE_EVAL_BATCH_SIZE = 2
LEARNING_RATE = 2e-4  # 有效 batch 不變，LR 不需調整
LR_SCHEDULER_TYPE = "cosine"
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.0  # LoRA adapter 小，通常不用 weight decay
MAX_GRAD_NORM = 1.0
# p50=616、p95=1475、p99=2108、max=7969；2048 涵蓋 p99，記憶體可控。
MAX_LENGTH = 2048
# LoRA optimizer state 極小（僅 adapter 參數），不需 paged → 改 fused，移除 CPU↔GPU paging 同步停頓。
OPTIM = "adamw_torch_fused"
GRADIENT_CHECKPOINTING = True  # 65GiB 權重下必開（關閉會 OOM）

# --- 長度分組取樣（砍 padding 浪費；與 per-source 加權取樣相容、無品質損失） ---
# WeightedRandomSampler 隨機配對序列，常把短(p50=616)與長(p99=2108)放同批 → padding 到長者。
# 改成先按權重抽樣、再把長度相近者排在一起（mega-batch 內排序），dynamic padding 逼近實際長度。
# 僅重排不改抽樣分佈，per-source 上採樣語意完整保留。
GROUP_BY_LENGTH = True
LENGTH_GROUP_MEGA_BATCH_MULT = 50  # mega-batch = batch_size × 此值；越大越隨機、分組效益略降

LOGGING_STEPS = 20
EVAL_STEPS = 1000  # eval 較重（eval bs=2 跑全 val ~6.4 分鐘），降頻
SAVE_STEPS = 1000  # 須為 EVAL_STEPS 的倍數（load_best_model_at_end 要求）
SAVE_TOTAL_LIMIT = 3
SEED = 42

# --- 監控（Weights & Biases；self-hosted Docker） ---
# 非機密預設集中於此；WANDB_API_KEY 為機密，僅由 .env / 環境變數提供（見 .env.example）。
REPORT_TO = ["wandb"]  # 監控上報目標（停用 tensorboard）
WANDB_PROJECT = "cyber-finetuning"
WANDB_BASE_URL = "http://localhost:8081"  # Docker 自架 wandb server（非機密，可改）

# --- 煙霧測試（--smoke）：極小子集 + 少步，快速驗證整條訓練路徑能跑通 ---
SMOKE_SUBSET = 64
SMOKE_MAX_STEPS = 4
SMOKE_SAVE_STEPS = 2
