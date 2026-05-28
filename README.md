# Cyber Fine-Tuning — 資安領域 LLM 微調

在單卡 NVIDIA RTX PRO 6000 Blackwell(95 GiB)上,以 **4-bit NF4 QLoRA** 微調 28B 多模態模型
`huihui-ai/Huihui-Qwen3.5-27B-Claude-4.6-Opus-abliterated`(架構 `qwen3_5`,dense + hybrid
attention + MTP)成資安助理。bf16 LoRA 仍保留為 fallback(`--quantize bf16`)。

> **為何選 4-bit QLoRA?** 新模型為 dense 結構(`mlp_only_layers: []`),所有參數都是
> `nn.Linear`,bitsandbytes 可正常量化(歷史 [bnb#1849](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1849)
> 阻擋的 fused MoE experts 已不存在)。4-bit 把權重 VRAM 從 ~56 GiB 壓到 ~14–16 GiB,
> 省下的空間用於更大 batch 與更多 LoRA 容量(全 64 層 dense MLP + 16 層 full-attn 都掛,
> 共 256 個 Linear 模組)。`linear_attn` 子樹(48 層 GatedDeltaNet,SSM 走 fp32)由
> `BNB_4BIT_SKIP_MODULES` 跳過保 bf16,避免 SSM 邊界 dtype 危險。

---

## 環境

[uv](https://github.com/astral-sh/uv) 管理、Python 3.12。

```bash
uv sync          # 安裝依賴（torch / transformers / peft / trl / bitsandbytes / wandb …）
```

## Pipeline 總覽

```
資料前處理                模型權重               訓練                    部署
data/processed/    +    HF checkpoint   →   LoRA adapter      →     merged model
  (已產出)               (~55 GiB)           training/               merge_adapter
```

資料前處理見 [`dataset.md`](dataset.md) 與 [`data/README.md`](data/README.md)（已產出
`data/processed/{train,val}.jsonl` 與 `sampler_weights.json`,共 121,168 筆）。

---

## 建置訓練資料（如需重建）

`data/processed/` 已產出,可直接進入訓練；若要從原始資料重建：

```bash
# 1. 下載原始資料集（gated 的 Primus-Instruct 需 HF token）
export HF_TOKEN=hf_xxx
./scripts/download_datasets.sh fine-tuning        # 只需訓練層（rag/eval 非訓練必需）

# 2. 前處理：5 來源 → 統一 chat 格式 → 驗證 / 長度過濾 / 去重 → 分層 train/val 切分
uv sync --extra tokenizer                          # 精確 token 計數需 tokenizer extra
uv run python -m preprocessing.build_dataset --drop-offtopic \
  --hf-tokenizer huihui-ai/Huihui-Qwen3.5-27B-Claude-4.6-Opus-abliterated
#   → 產出 data/processed/{train,val}.jsonl、stats.json、sources/*.jsonl

# 3. 計算 per-source 取樣權重（α=0.5、cap=3,並排除雜燴的 primus/general；讀步驟 2 產出的 stats.json）
uv run python scripts/compute_sampler_weights.py --alpha 0.5 --cap 3 \
  --exclude-sources primus/general \
  --out data/processed/sampler_weights.json
```

| build_dataset 旗標 | 作用 |
|---|---|
| `--drop-offtopic` | 剔除 `primus/general` 中無資安關鍵字的離題樣本 |
| `--hf-tokenizer <id>` | 用目標模型 tokenizer 精確計 token 長度（否則用字元/4 估算） |
| `--exclude-sources trendyol primus/reasoning` | 預設僅排除 trendyol；加上 `primus/reasoning` 可保 CTIBench 作乾淨 eval |

資料集設計、來源與決策詳見 [`dataset.md`](dataset.md)；各層目錄結構見 [`data/README.md`](data/README.md)。

---

## 正式訓練步驟

### 1. 前置確認

```bash
uv sync                                                            # 依賴齊全
ls data/processed/train.jsonl data/processed/sampler_weights.json  # 資料就緒
uv run python -m modeling.download                                 # 確認/下載模型權重（~55 GiB、11 shards；已快取則秒過）
cp .env.example .env                                               # 設定 wandb：填入 WANDB_API_KEY（self-hosted）
uv run python scripts/dump_target_modules.py                       # 驗 LoRA target_modules 命中數（防 silently 0 命中）
```

### 2. 煙霧測試（建議先跑,約 2–3 分鐘）

用極小子集跑通整條訓練路徑（4-bit 載入 → 掛 LoRA → loss mask → forward/backward → 存 adapter）,
先確認環境無誤再投入長時訓練：

```bash
uv run python -m training.train --smoke                  # 預設 4-bit QLoRA
uv run python -m training.train --smoke --quantize bf16  # bf16 fallback 對比驗證
```

通過標準：印出可訓練 LoRA 參數 ~50–70 M（覆蓋 256 個 Linear 模組）、4-bit 模式下
Linear4bit 命中數 ≥200、訓練 loss 為有限值、`outputs/qlora-cyber/smoke/final-adapter` 產出。

### 3. 正式訓練

```bash
uv run python -m training.train                          # 預設 4-bit QLoRA
uv run python -m training.train --quantize bf16          # fallback：bf16 LoRA
```

預設設定（情境：**2 epochs + α=0.5 / cap=3**,完整定義見 [`training/config.py`](training/config.py)）：

| 參數 | 4-bit（預設） | bf16（fallback） |
|---|---|---|
| epochs | 2 | 2 |
| 有效 batch | 6 × 3 (gradient accumulation) = 18 | 3 × 5 = 15 |
| learning rate | 2e-4,cosine scheduler,warmup 3% | 同 |
| max_length | 2048（涵蓋 p99） | 同 |
| LoRA | r=16, α=32, dropout 0.05；target = full-attn 16 層 + dense MLP 64 層（共 256 模組） | 同 |
| optimizer | adamw_torch_fused（LoRA state 小,不需 paged optimizer） | 同 |
| 取樣 | per-source 加權 **+ 長度分組**（砍 padding 浪費；`sampler_weights.json`：Fenrir 下採樣、小來源上採樣） | 同 |
| eval | batch 2、每 1000 步、`prediction_loss_only`（248k-vocab logits 按 eval_bs 線性放大,bs≥4 風險高） | 同 |
| dataloader | num_workers=0（資料已預 tokenize；num_workers>0 會在 CUDA 初始化後 fork 而不確定性卡死 step 0） | 同 |

> **吞吐**：4-bit QLoRA 預估全程 ~10–14h（bf16 同 batch 為基準的 ~67h 比較,並考量 dequant overhead 與更大 batch 攤平的綜效）；bf16 LoRA 路徑 ~20–24h。VRAM 4-bit ~30–40 GiB,bf16 接近 95 GiB；`train.py` 已預設 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 減少碎片。

常用覆寫：

```bash
uv run python -m training.train --epochs 3                            # 回到 3 epochs
uv run python -m training.train --batch-size 8 --grad-accum 2         # 4-bit 試更大 batch（smoke 通過後可實測）
uv run python -m training.train --quantize bf16 --batch-size 2 --grad-accum 8  # bf16 更保守
uv run python -m training.train --no-weighted                          # 關閉加權取樣（對比用）
```

> ⚠️ **batch size 上限**：vocab=248,320 的 logits 在最長 batch 做 `shift_logits.contiguous()` 需 ~連續 GiB 記憶體（每增 1 bs 約 +8 GiB at 2048 seqlen）。bf16 已驗證 bs=3 為上限；4-bit 預期可到 bs=6–10,實機 smoke 後再 lock。

其他選項：`--no-weighted`（關閉加權取樣）、`--extended-targets`（LoRA 額外掛 48 層 linear-attn 投影）、
`--no-wandb`（關閉 wandb 上報）、`--output-dir <路徑>`（自訂輸出位置）。

### 4. 監控（Weights & Biases）

訓練啟動後自動上報到 self-hosted wandb,瀏覽器開啟即可看 train/eval loss 與 learning rate 曲線：

```text
http://localhost:8081  →  專案 cyber-finetuning
```

run 命名：正式訓練為 `qlora-cyber-<N>ep-<時間戳>`,煙霧測試為 `smoke-<時間戳>`。
首次需在 `.env` 填入 `WANDB_API_KEY`（從 `http://localhost:8081/authorize` 取得）；離線或除錯可加 `--no-wandb` 關閉上報。
server URL／專案名預設見 [`training/config.py`](training/config.py),可改。

checkpoints 與最終 adapter 仍存於 `outputs/qlora-cyber/run/`（`outputs/qlora-cyber/run/final-adapter`）。

---

## 訓練後：合併 adapter

把 LoRA adapter 合併回 bf16 base 成完整模型（供部署/推論）：

```bash
uv run python -m training.merge_adapter --verify
```

預設讀 `outputs/qlora-cyber/run/final-adapter`、輸出至 `outputs/qlora-cyber-merged`。
`--verify` 會載入 merged 模型跑一題資安生成確認連貫。

> 無論訓練走 4-bit QLoRA 或 bf16 LoRA,merge 一律在 bf16 base 上做 → adapter 權重本身是 bf16,
> 不引入量化誤差(切勿在 4-bit base 上 merge)。

---

## 測試

```bash
uv run pytest                          # 全部（前處理 + 訓練）
uv run pytest tests/test_training.py   # 只跑訓練相關
uv run pytest tests/test_quant_loader.py tests/test_lora_target_resolution.py  # 量化與 target_modules 命中驗證
```

## 模組結構

```
training/
├── config.py                    集中設定（LoRA / 量化 / 超參 / 路徑）
├── data.py                      載入 jsonl + per-source 取樣權重展開
├── lora_loader.py               4-bit 或 bf16 載入 + gradient checkpointing + 掛 LoRA
├── quant_loader.py              BitsAndBytesConfig 工廠 + 量化命中數防呆
├── weighted_trainer.py          WeightedSFTTrainer（per-source 加權取樣）
├── qwen35_train_template.jinja  訓練 chat 模板（含 {% generation %},assistant-only loss）
├── train.py                     訓練進入點（--smoke 煙霧 / --quantize {4bit,bf16}）
└── merge_adapter.py             合併 adapter + 生成驗證
```
