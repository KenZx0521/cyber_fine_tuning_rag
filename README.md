# Cyber Fine-Tuning — 資安領域 LLM 微調

在單卡 NVIDIA RTX PRO 6000 Blackwell（95 GiB）上，以 **bf16 LoRA** 微調 36B MoE 多模態模型
`huihui-ai/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated`（架構 `qwen3_5_moe`）成資安助理。

> **為何不是 4-bit QLoRA？** 此模型 97% 參數在 fused MoE experts（`Qwen3_5MoeExperts` 的 3D
> Parameter），bitsandbytes 只能量化 `nn.Linear`、量化不到 fused experts（transformers v5
> 已知問題 [bnb#1849](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1849)），
> 4-bit 反而省不到記憶體。故採 **bf16 LoRA**：模型 bf16 約 65 GiB 單卡可容納，達成相同的單卡
> LoRA 微調目標。

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
  (已產出)               (~67 GiB)           training/               merge_adapter
```

資料前處理見 [`dataset.md`](dataset.md) 與 [`data/README.md`](data/README.md)（已產出
`data/processed/{train,val}.jsonl` 與 `sampler_weights.json`，共 121,168 筆）。

---

## 建置訓練資料（如需重建）

`data/processed/` 已產出，可直接進入訓練；若要從原始資料重建：

```bash
# 1. 下載原始資料集（gated 的 Primus-Instruct 需 HF token）
export HF_TOKEN=hf_xxx
./scripts/download_datasets.sh fine-tuning        # 只需訓練層（rag/eval 非訓練必需）

# 2. 前處理：5 來源 → 統一 chat 格式 → 驗證 / 長度過濾 / 去重 → 分層 train/val 切分
uv sync --extra tokenizer                          # 精確 token 計數需 tokenizer extra
uv run python -m preprocessing.build_dataset --drop-offtopic \
  --hf-tokenizer huihui-ai/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated
#   → 產出 data/processed/{train,val}.jsonl、stats.json、sources/*.jsonl

# 3. 計算 per-source 取樣權重（α=0.5、cap=5；讀步驟 2 產出的 stats.json）
uv run python scripts/compute_sampler_weights.py --alpha 0.5 --cap 5 \
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
uv run python -m modeling.download                                 # 確認/下載模型權重（~67 GiB；已快取則秒過）
cp .env.example .env                                               # 設定 wandb：填入 WANDB_API_KEY（self-hosted）
```

### 2. 煙霧測試（建議先跑，約 1–2 分鐘）

用極小子集跑通整條訓練路徑（bf16 載入 → 掛 LoRA → loss mask → forward/backward → 存 adapter），
先確認環境無誤再投入長時訓練：

```bash
uv run python -m training.train --smoke
```

通過標準：印出可訓練參數 ~8.36 M、訓練 loss 為有限值、`outputs/qlora-cyber/smoke/final-adapter` 產出。

### 3. 正式訓練

```bash
uv run python -m training.train
```

預設設定（情境：**3 epochs + α=0.5**，完整定義見 [`training/config.py`](training/config.py)）：

| 參數 | 值 |
|---|---|
| epochs | 3（≈ 22k steps） |
| 有效 batch | 2 × 8 (gradient accumulation) = 16 |
| learning rate | 2e-4，cosine scheduler，warmup 3% |
| max_length | 2048（涵蓋 p99；fenrir 偏長可上調） |
| LoRA | r=16, α=32, dropout 0.05；target = attention + shared_expert FFN |
| optimizer | paged_adamw_8bit |
| 加權取樣 | per-source（`sampler_weights.json`：Fenrir 下採樣、小來源上採樣） |

常用覆寫（載入後尚有 ~28 GiB VRAM 餘裕）：

```bash
uv run python -m training.train --batch-size 4 --grad-accum 4   # 提高吞吐
uv run python -m training.train --max-length 3072               # 減少 fenrir 長樣本截斷
uv run python -m training.train --epochs 2                      # 調整 epoch 數
```

其他選項：`--no-weighted`（關閉加權取樣）、`--extended-targets`（LoRA 額外掛 30 層 linear-attn 投影）、
`--no-wandb`（關閉 wandb 上報）、`--output-dir <路徑>`（自訂輸出位置）。

### 4. 監控（Weights & Biases）

訓練啟動後自動上報到 self-hosted wandb，瀏覽器開啟即可看 train/eval loss 與 learning rate 曲線：

```text
http://localhost:8081  →  專案 cyber-finetuning
```

run 命名：正式訓練為 `qlora-cyber-<N>ep-<時間戳>`，煙霧測試為 `smoke-<時間戳>`。
首次需在 `.env` 填入 `WANDB_API_KEY`（從 `http://localhost:8081/authorize` 取得）；離線或除錯可加 `--no-wandb` 關閉上報。
server URL／專案名預設見 [`training/config.py`](training/config.py)，可改。

checkpoints 與最終 adapter 仍存於 `outputs/qlora-cyber/run/`（`outputs/qlora-cyber/run/final-adapter`）。

---

## 訓練後：合併 adapter

把 LoRA adapter 合併回 bf16 base 成完整模型（供部署/推論）：

```bash
uv run python -m training.merge_adapter --verify
```

預設讀 `outputs/qlora-cyber/run/final-adapter`、輸出至 `outputs/qlora-cyber-merged`。
`--verify` 會載入 merged 模型跑一題資安生成確認連貫。

---

## 測試

```bash
uv run pytest                          # 全部（前處理 + 訓練）
uv run pytest tests/test_training.py   # 只跑訓練相關
```

## 模組結構

```
training/
├── config.py                    集中設定（LoRA / 超參 / 路徑）
├── data.py                      載入 jsonl + per-source 取樣權重展開
├── lora_loader.py               bf16 載入 + gradient checkpointing + 掛 LoRA
├── weighted_trainer.py          WeightedSFTTrainer（per-source 加權取樣）
├── qwen35_train_template.jinja  訓練 chat 模板（含 {% generation %}，assistant-only loss）
├── train.py                     訓練進入點（--smoke 煙霧模式）
└── merge_adapter.py             合併 adapter + 生成驗證
```
