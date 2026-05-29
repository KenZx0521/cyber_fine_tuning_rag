"""QLoRA 訓練的集中設定：LoRA、4-bit 量化、訓練超參與路徑。

對齊 modeling/config.py 的風格：相對 repo root 推導、不寫死絕對路徑；
刻意不在此 import torch —— 量化的 compute dtype 以字串表示，由 quant_loader
轉成 torch dtype，讓 config 維持輕量、無重相依。

訓練情境（使用者已拍板）：2 epochs + α=0.5 / cap=3（已把雜燴的 primus/general 權重設 0 排除取樣），
沿用既有 data/processed/sampler_weights.json。

量化策略（使用者已拍板，2026-05-28 換模型 + 重啟 QLoRA）：
  - 預設 4-bit QLoRA（新模型為 dense，bnb 可量化全部 nn.Linear，bnb#1849 不再阻擋）
  - bf16 LoRA 作為 fallback（--quantize bf16）—— 維持單一 trainer + CLI 切換的簡潔
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
# target_modules 決策（依 meta-device named_modules dump 校正，見 scripts/dump_target_modules.py）：
# 新模型（qwen3_5 dense + VLM + hybrid attention + MTP）每層結構為
#   self_attn(16 full-attention 層).{q,k,v,o}_proj  —— 標準 nn.Linear
#       （attn_output_gate=true：gate 從 q_proj 用 torch.chunk 切出，q_proj 涵蓋兩者）
#   linear_attn(48 層 GatedDeltaNet).{in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj}
#       —— Linear，但接 SSM (fp32) 核心；4-bit 量化路徑須以 BNB_4BIT_SKIP_MODULES skip 整個子樹
#   mlp.{gate_proj,up_proj,down_proj}                —— dense FFN（每層必過、梯度密集；無 MoE）
#   visual.* (Qwen3_5 ViT 27 層).{linear_fc1,linear_fc2,qkv,proj} —— 凍結 + 不掛 LoRA
#   mtp.layers.*                                     —— transformers 5.9+ 載入時依
#       _keys_to_ignore_on_load_unexpected 直接丟棄（named_parameters 不會出現）
# 預設掛 LoRA 於：full-attn 投影 + dense FFN
#   → 命中 16 attn × 4 + 64 mlp × 3 = 256 個 Linear，全為梯度密集路徑。
# 刻意排除：
#   - linear_attn 投影（SSM fp32 路徑，首訓保守不動；穩定後可用 --extended-targets 加入）
#   - vision tower（VLM 多模態能力不在本次訓練目標）
#   - lm_head / embed_tokens（248k vocab，量化或微調都會影響輸出分佈）
LORA_R = 16
LORA_ALPHA = 32  # 2×r 慣例
LORA_DROPOUT = 0.05
LORA_BIAS = "none"
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",  # full-attn 4 投影（命中 16 層 self_attn；linear_attn 無此命名）
    "gate_proj",
    "up_proj",
    "down_proj",  # dense FFN 3 投影（命中全 64 層 mlp；vision 用 linear_fc1/2，不衝突）
]
# 進階實驗用：額外納入 linear_attn 投影，覆蓋 48 層 token-mixing（首訓不啟用）。
# 注意：啟用 --extended-targets 同時跑 4-bit QLoRA 時，這些子樹 LoRA 會在 bf16 上掛（因 SKIP）。
TARGET_MODULES_WITH_LINEAR_ATTN = TARGET_MODULES + [
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
]

# --- 4-bit QLoRA 量化（quant_loader.build_bnb_config 的預設來源） ---
BNB_4BIT_QUANT_TYPE = "nf4"           # QLoRA 標準
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"   # 由 quant_loader 轉成 torch.bfloat16
BNB_4BIT_USE_DOUBLE_QUANT = True      # 二次量化壓縮 quant constants（再省 ~0.4 bit/param）
# 跳過量化的子模組（substring match）：
#   - "visual"：凍結；保 bf16 維持多模態推論精度
#   - "lm_head"：248k vocab × 5120，量化會影響 last-token logits 精度
#   - "linear_attn"：48 層 GatedDeltaNet 子樹整段；SSM 內走 fp32，bf16 input × 4-bit weight
#     會在 SSM 邊界出 dtype 危險（高風險 R1）
BNB_4BIT_SKIP_MODULES = ["visual", "lm_head", "linear_attn"]

# --- 載入策略：4-bit QLoRA 為預設，bf16 LoRA 為 fallback ---
# 新模型（qwen3_5 dense）已無 fused MoE experts，全是 nn.Linear → bitsandbytes 可正常量化，
# bnb#1849 不再阻擋。預設走 4-bit QLoRA（NF4 + bf16 compute + double quant）：
#   - 權重 VRAM：bf16 ~56 GiB → 4-bit ~14-16 GiB，省下 ~40 GiB 用於更大 batch / 關 grad ckpt
#   - linear_attn 子樹（48 層 GatedDeltaNet）由 BNB_4BIT_SKIP_MODULES 跳過保 bf16，SSM 邊界安全
# bf16 LoRA 仍保留為 fallback（--quantize bf16），程式路徑差異最小化：
#   - lora_loader.load_base(load_in_4bit=False) 走 dtype=bfloat16 直接載入（與舊行為一致）
#   - 不在 bf16 路徑呼叫 prepare_model_for_kbit_training（會把 bf16 全升 fp32 OOM）
# device_map 全塞單卡（不 offload；訓練時 offload 會災難性變慢）。
DEVICE_MAP: str | dict = {"": 0}

# --- 訓練超參（情境：2 epochs + α=0.5 / cap=3；單卡 RTX PRO 6000 96GiB） ---
# 2026-05-28 吞吐 benchmark（smoke 4-step + length grouping 部分發揮，仍有相對意義）：
#   4-bit bs=6 ga=3 ml=2048：0.643 samples/s → 估全量 ~75 hr ←歷史預設
#   bf16  bs=3 ga=5 ml=1536：0.709 samples/s → 估 ~58 hr
#   bf16  bs=4 ga=5 ml=1536：0.921 samples/s → 估 ~46 hr ★甜蜜點（peak 84/96GiB，margin 11GiB）
#   bf16  bs=5 ga=4 ml=1536：0.887 samples/s → 估 ~48 hr（peak 92/96GiB，margin 4GiB）
#   bf16  bs=6 ga=3 ml=1536：0.784 samples/s → 估 ~54 hr（peak 95/96GiB ⚠️ 紅線）
# 關鍵發現：
#   1) Blackwell Tensor Core 直跑 bf16，4-bit dequant 反而成淨負擔（bf16 比 4-bit 快 ~40%）
#   2) dense 36B 在 bs=4 即達計算飽和，再放大 bs 反而 throughput 下降
#   3) Liger Kernel 砍 logits 物化（VRAM -20GiB）但本機 throughput 沒提升（Triton SM12.0 未調優）
# 故預設改 bf16 LoRA bs=4 ga=5 max_length=1536（截到 p95+，移除 5% 長尾換來 1.43× 吞吐）
NUM_TRAIN_EPOCHS = 2
PER_DEVICE_TRAIN_BATCH_SIZE = 4  # bf16 路徑（預設）：smoke 量峰值 84/96GiB
GRADIENT_ACCUMULATION_STEPS = 5  # bf16：有效 batch = 4×5 = 20
QLORA_PER_DEVICE_TRAIN_BATCH_SIZE = 6  # 4-bit 路徑（--quantize 4bit）：起點；本機實測比 bf16 慢，保留 fallback
QLORA_GRADIENT_ACCUMULATION_STEPS = 3  # 4-bit 路徑：有效 batch = 6×3 = 18
# eval 時訓練 allocator 已保留 ~87GiB（峰值快取不釋放），僅剩 ~7GiB；而 248k-vocab logits 的
# shift_logits.contiguous() 按 eval_bs 線性放大（實測 bs=8 需 ~10.5GiB → OOM）。bs=2（~2.6GiB）穩妥。
# 註：prediction_loss_only 只避免「跨 batch 累積預測」，消不掉單 batch 的 logits 峰值，故須小 eval batch。
PER_DEVICE_EVAL_BATCH_SIZE = 2
LEARNING_RATE = 2e-4  # 有效 batch 不變，LR 不需調整
LR_SCHEDULER_TYPE = "cosine"
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.0  # LoRA adapter 小，通常不用 weight decay
MAX_GRAD_NORM = 1.0
# p50=616、p95=1475、p99=2108、max=7969；1536 涵蓋 p95（截掉 ~5% 長尾），
# 相對 2048 attention/logits VRAM 線性縮減 ~25% 換 1.4× 吞吐。
# 若要避免任何截斷再改回 2048（須同步降 bs 避免 OOM）。
MAX_LENGTH = 1536
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
