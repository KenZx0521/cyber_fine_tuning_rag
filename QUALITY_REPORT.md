# Fine-tuning Dataset 品質檢驗報告

| 項目 | 值 |
|---|---|
| 檢驗日期 | 2026-05-27 |
| Repo commit | `f7b178d` |
| 受檢產出 | `data/processed/`（`stats.json` mtime 2026-05-27 10:52） |
| 對照基準 | `dataset.md`（fine-tuning 來源清單） |
| 檢驗方法 | `preprocessing/audit_quality.py`（stdlib only，可重跑；見附錄 A） |
| 目標模型 tokenizer | `huihui-ai/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated` |

> 本報告只檢驗 **fine-tuning** 資料品質與涵蓋度。eval/RAG 僅順帶回報下載狀態，未做品質檢驗。

---

## 修復狀態（2026-05-27 更新，重建後）

依本報告與使用者拍板的定位決策（① CTIBench **保留訓練**、評測改用他基準；② **不設限 dual-use** 助理；③ 剔除 primus/general 離題），已重建 `data/processed/`。下表為原始發現與修復結果；其下各節保留原始分析作為依據。

| 問題 | 動作 | 重建後結果 |
|---|---|---|
| 🔴① Trendyol 100% 重複 | 預設排除 Trendyol（`config.DEFAULT_EXCLUDED_SOURCES`，非改去重鍵） | **全域冗餘 30.2% → 0.0%**；per_source 已無 trendyol；`dropped.excluded_source=53,201` |
| 🔴② CTIBench 污染 | **保留 reasoning 於訓練**（4,611 筆），評測禁用 CTIBench、改用 SecQA / CyberSecEval；已更正 `dataset.md` | 訓練保留 `<think>` 推理訊號；污染風險靠「不用 CTIBench 評測」消除 |
| 🟡③ primus/general 72.3% 離題 | `--drop-offtopic`（僅作用 primus/general） | **離題率 72.3% → 0.0%**；primus/general 296 → **83**；`dropped.offtopic=231` |
| 🟡④ Fenrir 安全姿態與 doc 不符 | 定位＝dual-use，資料不過濾；更正 `dataset.md` | 拒答仍只見於武器化請求（kernel rootkit / bootkit / 多型碼），符合定位 |
| 🟡⑤ 來源失衡 + 加權失效 | `stats.json` 新增 `assistant_tokens_per_source` / `total_tokens_per_source` 供重算 | Fenrir 佔 74.8% 筆數但 **~90% assistant-token**；加權重算為訓練階段 follow-up |

**重建後關鍵數據**（目標模型 tokenizer）：

```
raw_total 184,132  →  kept 121,168
dropped   excluded_source 53,201 / offtopic 231 / invalid 4 / too_long 21 / duplicate 9,507
          （不變式：121,168 + 53,201 + 231 + 4 + 21 + 9,507 = 184,132 ✅）
train 118,744 | val 2,424（依 source 分層）
token 長度  p50 616 / p95 1,475 / p99 2,108 / max 7,969（上限 8,192）
per_source  fenrir 90,654 / attackqa 25,335 / reasoning 4,611（o1 2,165＋dsr1 2,446）/ primus-instruct 568（SOC5 485＋general 83）
```

> 正式建構命令見 [`dataset.md` 的「前處理 / 建構說明」](./dataset.md#前處理--建構說明)。可重跑 `python3 preprocessing/audit_quality.py` 驗證。

---

## 摘要（重建後現況）

- `dataset.md` 列的 **5 個 fine-tuning 來源全部到位** ✅；4 個進訓練、Trendyol 經實證為 Fenrir 100% 子集而**有意零損失排除**。
- **2 個🔴已修**：① Trendyol 冗餘 → 預設排除，**全域冗餘 30.2% → 0.0%**；② Primus-Reasoning（=CTIBench）→ 保留訓練、**評測禁用 CTIBench**。
- **3 個🟡已定位**：剔除 primus/general 離題、Fenrir 採不設限 dual-use、來源失衡交由訓練階段加權（權重已算，見〈三、行動與狀態〉）。
- 全域去重後**真實獨立內容 = 121,168 筆**（＝產出量，冗餘 0）。

> 分層提醒：本報告〈一、涵蓋度〉〈✅ 做得好〉與〈附錄 B〉均為**重建後**事實；〈二、品質發現〉各問題為**重建前原始發現**，保留作決策依據，修復結果見上方〈修復狀態〉表。

---

## 一、涵蓋度：是否包含 dataset.md 提到的 dataset

### Fine-tuning（5/5 全到齊 ✅）

| dataset.md 來源 | 產出筆數 | 狀態 | 備註 |
|---|---|---|---|
| Fenrir-v2.x | 90,654 | ✅ 進訓練 | raw 實測 **99,869**（> doc 連結標的 v2.0 / 83,920，確為較新較大版，疑為 v2.1） |
| AttackQA | 25,335 | ✅ 進訓練 | RAG 式組裝（Context + Question → answer） |
| Primus-Reasoning | 4,611 | ✅ 進訓練 ⚠️ | o1 2,165 + deepseek-r1 2,446；**原始檔名 `ctibench_*.parquet`**（＝CTIBench，見🔴問題 2） |
| Primus-Instruct | 568 | ✅ 進訓練 | 6 子任務：SOC5 485（alert 100 / doc_qa 100 / terraform 96 / cmd 95 / event_query 94）＋ general 83 |
| Trendyol | 0（排除 53,201） | ⚠️ 有意排除 | md5 全文比對 100% ⊂ Fenrir；零損失去冗餘，內容仍由 Fenrir 涵蓋 |

### Eval / RAG（dataset.md 也列，非 fine-tuning，僅回報下載狀態）

- **Eval**：SecQA ✅、CyberSecEval(PurpleLlama) ✅ ｜ **未下載**：CyberMetric、SecEval、SecBench、**CTIBench**
- **RAG**：MITRE ATT&CK STIX ✅、CWE / NVD / CISA-KEV ✅（額外）｜ **未下載**：cybersecurity-corpus、Primus-Seed、Primus-FineWeb、D3FEND、nist-cybersecurity-training、OWASP Top 10、CIS Benchmarks

---

## 二、品質發現（依嚴重度）

> 本節為**重建前**的原始發現（保留作決策依據）；🔴①② 已修、🟡③④⑤ 已定位，狀態見頂部〈修復狀態〉表與〈三、行動與狀態〉。🟡⑥ 為本次補充深掘。

### 🔴 問題 1：跨來源重複 — Trendyol 整包被 Fenrir 吸收

以 md5 對 `system+user+assistant` 全文做指紋比對：

```
trendyol 落在 fenrir 比例 = 100.0%   (52,524 / 52,524，連 1,085 字 system 都逐字相同)
全部來源合計            = 173,905
全域去重後獨立內容      = 121,381   →  冗餘 52,524 筆 (30.2%)
```

- **根因**：`preprocessing/quality.py:73` 的 `exact_dedup` key = `(source, first_user)`，**含 `source`**，所以 Fenrir 版與 Trendyol 版因來源標籤不同而都被保留。
- **影響**：~30% 訓練算力浪費在重複樣本；Fenrir-family 權重被灌水成兩倍。
- **修法（零損失）**：在 `build_dataset.py:_collect_records` **不收 Trendyol**（它是 Fenrir 的嚴格子集）；或把 dedup key 去掉 `source`。

### 🔴 問題 2：CTIBench 訓練／評測污染

Primus-Reasoning 兩個原始檔名即 `ctibench_o1.parquet` / `ctibench_deepseek-r1.parquet`。題型指紋確認其為 CTIBench 子任務：

| 題型 | o1 | deepseek-r1 |
|---|---|---|
| CVE→CWE (CTI-RCM) | 55.5% | 55.9% |
| MCQ (CTI-MCQ) | 44.4% | 43.9% |
| ATT&CK 抽取 (CTI-TAA) | 13.4% | 12.4% |
| CVSS 評分 (CTI-VSP) | 10.3% | 12.6% |

- **影響**：`dataset.md` 把 CTIBench 列為 eval。訓練吃了這 4,611 筆（含答案的推理）後再用 CTIBench 評測＝考古題，分數無效。
- **修法（二選一）**：① 訓練保留 reasoning，**評測改用** CyberMetric / SecEval / SecQA / SecBench；② 保留 CTIBench 評測，則從訓練移除這兩檔。

### 🟡 問題 3：Primus-general 72.3% 離題

`primus_general` 296 筆中 **214 筆（72.3%）完全無資安關鍵字**（香港旅遊攻略、資料庫設計、pymongo、Gantt 圖、SCRUM 課程、教育政策等通用助理任務）。

- 係 Primus 刻意保留的「通用能力／抗遺忘」資料，對**聚焦型資安助理是雜訊**。
- 絕對量小（214 筆 = 全資料 0.12%），可決定保留或剔除。
- 註：`event_query_generation` 18%、`doc_qa` 5% 的「離題」多為**假陽性**（query 翻譯 / Trend HelpKit 屬資安情境，只是無顯式關鍵字）。

### 🟡 問題 4：Fenrir 安全姿態與 dataset.md 描述不符

`dataset.md` 稱 Fenrir「攻擊性請求會收到解釋性拒絕」。實測：

```
Fenrir 全庫拒答率 = 0.60%  (541 / 90,654)   →   99.4% 照常回答
```

拒答**只集中在「建構攻擊能力」**請求（kernel rootkit、bootkit / 韌體持久化、多型 / 變形惡意碼、反鑑識、檔案隱藏技術）。

- doc 描述**方向對但嚴重高估**：拒答存在且為解釋型，但極稀少；大量 dual-use／攻擊相鄰內容仍給實質答案。
- 目標模型是 abliterated（去審查）版，此「幾乎全答、只擋武器化」姿態**可能是刻意的**。需確認定位（不設限 dual-use 助理 vs 嚴格防禦型），並更正 `dataset.md`。

### 🟡 問題 5：來源嚴重失衡 + system prompt 多樣性低

來源嚴重失衡（重建後，**assistant-token 維度比筆數更嚴重**）：

| 來源 | 筆數 | 筆數佔比 | assistant-token 佔比 |
|---|---|---|---|
| Fenrir | 90,654 | 74.8% | **90.0%** |
| AttackQA | 25,335 | 20.9% | 2.41% |
| Primus-Reasoning | 4,611 | 3.8% | 6.75% |
| Primus-Instruct | 568 | **0.47%** | **0.79%** |

- Fenrir 答案長，佔 **90% 學習訊號**（assistant-token）；Trend Micro 商業情境（Primus-Instruct）僅 0.79%，`dataset.md` 期望它「增加風格多樣性」**無加權則達不到**。
- system prompt 多樣性極低：各來源幾乎只 1 種，Fenrir 3 種（top 88.8%）→ 模型可能把特定 system 措辭當觸發條件。
- ✅ **加權已算**（2026-05-27）：`scripts/compute_sampler_weights.py`（α=0.5 cap=3，並把雜燴的 primus/general 權重設 0 排除取樣）→ 調整後 Fenrir 90% → 63%、reasoning → 24%、AttackQA → 10%、SOC5 → ~2%；詳見〈三、行動與狀態〉與 memory `cyber-ft-train-weighting`。

### 🟡 問題 6：分佈「形狀」風險（本次深掘，超出原 audit）

exact 去重之外，另測近似重複／模板化／長度（`stats.json` 與抽樣統計）：

- Fenrir **非**模板冗餘：user 前 20 字 16,003 種、assistant 前 40 字 58,804 種，內容多樣（exact dedup 未漏大批近似重複）。
- 但**互動模式單一**：user 前 4 詞清一色 `how would you design / architect…` 假設性顧問問句，缺事實問答／簡短指令／程式碼／多輪。
- **答案長度兩極且綁 system**：Fenrir 答案中位 **2,484 字元**（最短 5% 仍 1,003）vs AttackQA **139 字元**（19.5% 短於 80 字元）；各來源 system 固定 → 長度由 system 觸發。Fenrir 佔 90% token，整體被拉向冗長顧問腔。
- ✅ 免責／hedging 不氾濫（Fenrir 僅 0.2%）、非英文極少 → 符合 dual-use 定位、語言乾淨。
- **緩解**：靠〈三〉的加權壓低 Fenrir ＋ 訓練前 system 增廣去綁定（皆訓練階段手段）。

### ✅ 做得好的部分（重建後實測）

- 格式統一（chat messages JSONL）、invalid 殘留 0、轉義殘留 0；**46 個 pipeline 單元測試全過**。
- Token 長度健康：p50=616 / p95=1,475 / p99=2,108 / max=7,969，**全在 8,192 上限內**（目標模型 tokenizer 真實計長）。
- AttackQA RAG 式組裝、Primus-Reasoning 的 `<think>` 轉換（Llama special token → Qwen 原生，**100% 覆蓋、零殘留**）皆正確。
- **train/val 零洩漏**（id 交集 0）；val 2,424（≈2%）依 source 分層，**10 來源皆有 val**。

---

## 三、行動與狀態

| # | 動作 | 狀態 |
|---|---|---|
| 1 | 排除 Trendyol（預設 `config.DEFAULT_EXCLUDED_SOURCES`） | ✅ 已完成（184,132 → 121,168，冗餘歸零） |
| 2 | CTIBench 訓練／評測二選一 | ✅ 已定（保留訓練、評測禁用 CTIBench，改用 SecQA / CyberSecEval） |
| 3 | 重算來源加權 | ✅ 已算（`scripts/compute_sampler_weights.py`，α=0.5 cap=3，primus/general 權重=0 排除取樣）；**套用屬訓練階段** |
| 4 | primus/general 離題去留 | ✅ 已剔離題（`--drop-offtopic`，296 → 83）；其餘 83 筆於**加權階段權重=0 排除取樣**（grab-bag 長文、CP 值低，見〈三〉#3） |
| 5 | 更正 `dataset.md`（Fenrir 版本、安全姿態 0.6%） | ✅ 已更正 |
| 6 | system 增廣，去除「system ↔ 長度/風格」綁定 | ⏳ 建議（訓練前，可選；見🟡問題 6） |
| 7 | 事實正確性抽驗（reasoning 的 CWE / CVSS，LLM-judge） | ⏳ 未做（已知未驗風險） |

---

## 附錄 A：可重現的稽核方法

```bash
# 全部檢驗（跨來源重複、system 分佈、離題、Fenrir 安全姿態、CTIBench 指紋）
python3 preprocessing/audit_quality.py
```

腳本為 stdlib only，直接讀 `data/processed/sources/*.jsonl`：

- **跨來源重複**：對 `user+assistant`（忽略 system）與 `system+user+assistant` 各算 md5 指紋，比對來源間交集。
- **system 多樣性**：每來源不同 system 指紋數與最大佔比。
- **離題偵測**：primus 各檔 `user+assistant` 是否含資安關鍵字正則（粗略，有假陽性）。
- **Fenrir 安全姿態**：assistant 開頭 300 字內是否命中拒答正則。
- **CTIBench 指紋**：reasoning 兩檔 user 是否命中 CVE→CWE / CVSS / ATT&CK / MCQ 題型正則。

## 附錄 B：stats.json 關鍵數據（重建後）

```
raw_total   184,132   →   kept 121,168
dropped     excluded_source 53,201 / offtopic 231 / invalid 4 / too_long 21 / duplicate 9,507
            （不變式：121,168 + 53,201 + 231 + 4 + 21 + 9,507 = 184,132 ✅）
train       118,744   |   val 2,424（依 source 分層，≈2%）
token 長度   p50 616 / p95 1,475 / p99 2,108 / max 7,969（上限 8,192，目標模型 tokenizer）
含 CJK       8 筆
per_source       fenrir 90,654 / attackqa 25,335 / reasoning 4,611 / primus-instruct 568
assistant-token  Fenrir 90.0% / reasoning 6.75% / AttackQA 2.41% / Primus-Instruct 0.79%（總 ~47.8M）
```

> 註：Trendyol 53,201 筆以 `excluded_source` 計（預設排除，非靠去重）；`duplicate=9,507` 為各來源**內部**精確去重。跨來源冗餘已實測為 **0.0%**（`audit_quality.py`，見〈修復狀態〉🔴①）。
