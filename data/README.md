# 資安語料庫 (Cybersecurity Corpus)

依 [`../dataset.md`](../dataset.md) 下載並整理，分成三層：**fine-tuning**(訓練)、**rag**(知識庫檢索)、**eval**(評測,**嚴禁拿來訓練**)。

重新同步：

```bash
export HF_TOKEN=hf_xxx          # 下載 gated 的 Primus-Instruct 才需要
./scripts/download_datasets.sh            # 全部
./scripts/download_datasets.sh rag        # 只更新 RAG 來源 (NVD 每日更新、KEV 變動時更新)
./scripts/download_datasets.sh fine-tuning eval
```

---

## 目錄結構

```
data/
├── fine-tuning/                         訓練資料 (11M)
│   ├── primus-instruct/data/*.parquet   trendmicro-ailab/Primus-Instruct (gated)
│   └── attackqa/attackqa.parquet        sambanovasystems/attackqa
├── rag/                                 知識庫 (268M)
│   ├── mitre-attack-stix/               最新 STIX 2.1 collections + index.json
│   │   ├── enterprise-attack.json (51M)
│   │   ├── mobile-attack.json
│   │   ├── ics-attack.json
│   │   └── index.json
│   ├── nvd/                             NVD JSON 2.0 年度 feed (gzip)
│   │   ├── nvdcve-2.0-{2002..2026}.json.gz (+ .meta)
│   │   ├── nvdcve-2.0-modified.json.gz  每 2 小時更新 (增量同步用)
│   │   └── nvdcve-2.0-recent.json.gz
│   ├── cisa-kev/                        git mirror (csv/json/schema) + live 目錄
│   │   └── known_exploited_vulnerabilities.json
│   └── cwe/
│       ├── cwec_v4.20.xml (18M)         完整 weakness 目錄
│       ├── 2000.csv                     Comprehensive View (969 條)
│       └── *.zip                        原始壓縮檔
└── eval/                                評測基準 (88M) — ⚠️ 不可訓練
    ├── secqa/data/*.csv                 zefang-liu/secqa
    └── cyberseceval-purplellama/        meta-llama/PurpleLlama (sparse: CybersecurityBenchmarks/)
```

---

## fine-tuning

| Dataset | 來源 | 內容 | 規模 (本地驗證) |
|---|---|---|---|
| **Primus-Instruct** | [trendmicro-ailab/Primus-Instruct](https://huggingface.co/datasets/trendmicro-ailab/Primus-Instruct) (gated) | 專家策劃的資安業務情境指令,回覆由 GPT-4o 生成。涵蓋告警解釋、可疑指令分析、風險建議、查詢語言生成等 | 6 個情境 parquet:`alert_explanation`、`cmd_analysis`、`general`、`security_doc_qa`、`security_event_query_generation`、`terraform_misconfiguration_scan` |
| **AttackQA** | [sambanovasystems/attackqa](https://huggingface.co/datasets/sambanovasystems/attackqa) | SOC QA、ATT&CK reasoning、RAG answer style,約 25,335 組 Q&A 並附 rationale | `attackqa.parquet` (7.2M) |

> 讀取:`pandas.read_parquet(...)` 或 `datasets.load_dataset("parquet", data_files=...)`(需 `pip install pyarrow pandas datasets`)。

## rag

| 來源 | 內容 | 規模 (本地驗證) | 更新頻率 |
|---|---|---|---|
| **MITRE ATT&CK STIX** | TTP / tactic / technique / mitigation / group / software mapping (STIX 2.1) | Enterprise 858 techniques · Mobile 190 · ICS 118 | 隨 ATT&CK 版本 |
| **NVD** | CVE / CVSS / CPE / 弱點描述 | 25 年度 feed (2024: 39,094 CVE · 2025: 44,406 CVE) | year feed 每日;modified/recent 每 2 小時 |
| **CISA KEV** | 已知遭實際利用漏洞,用於修補優先級 | catalog v2026.05.22,1,602 筆 | 變動時 |
| **CWE** | weakness taxonomy / 定義 / mitigation | `cwec_v4.20.xml` 完整目錄 · CSV 969 條 | 隨 CWE 版本 |

> NVD feed 為 `.json.gz`,串流讀取:`zcat nvdcve-2.0-2025.json.gz | jq ...` 或 Python `gzip.open(...)`。
> 切記 NVD 改用 [API 2.0](https://nvd.nist.gov/developers),legacy 1.1 feed 已停用;此處用的是 JSON **2.0** 年度 feed。

## eval — ⚠️ 僅供 benchmark,**不可訓練主模型**

| 來源 | 內容 | 規模 (本地驗證) |
|---|---|---|
| **SecQA** | 資安基礎知識 multiple-choice | v1 127 題 · v2 115 題 |
| **CyberSecEval** | LLM 資安風險與防禦能力評測 (secure code、prompt injection、SOC、malware、threat intel reasoning) | `CybersecurityBenchmarks/` 全套:`mitre`、`prompt_injection`、`spear_phishing`、`autocomplete`、`interpreter`、`autonomous_uplift`、`autopatch`、`canary_exploit` 等 |

---

## 授權與引用

各資料集授權不同(NVD/CWE/ATT&CK 為公開政府/MITRE 資料;CISA KEV 為公領域;HF 資料集見其 dataset card)。再散佈或商用前請依各來源 license 確認。出處與連結見 [`../dataset.md`](../dataset.md)。
