# 通用型資安 LLM Dataset 清單(Fine-tuning + RAG + Eval)

| 類別          | Dataset / Source                                                                                                                                | 用途                                                                                                                                                                       |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| fine-tuning | [Trendyol-Cybersecurity-Instruction-Tuning-Dataset](https://huggingface.co/datasets/Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset) | 通用資安指令微調資料,涵蓋 200+ 個子領域,包含雲端原生威脅、AI/ML 安全、量子運算風險、事件應變等。System/user/assistant 格式,由 Trendyol Security Team 整理。([Hugging Face][1])<br>⚠️ **前處理**:經 md5 全文比對確認為 Fenrir 的**嚴格子集(100% 重複)**,**預設建構已排除**(`config.DEFAULT_EXCLUDED_SOURCES`),零損失移除 ~30% 冗餘。                                       |
| fine-tuning | [Cybersecurity-Dataset-Fenrir-v2.0](https://huggingface.co/datasets/AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.0)                             | 資安指令資料,內容對應 OWASP / ATT&CK / NIST / CIS 標準,涵蓋 IAM、secrets 管理、CI/CD、container/k8s 強化、SIEM、incident response。([Hugging Face][2])<br>⚠️ **前處理**:實際抓到的 raw ≈ **99,869 筆**(較連結標示的 v2.0 / 83,920 新且大,疑為 v2.1);去重/長度過濾後 fenrir=**90,654**(佔總量 74.8%、佔 assistant-token ~90%)。**安全姿態實測**:拒答率僅 **~0.6%**,99.4% 照常作答;解釋型拒絕只集中於「武器化/攻擊能力建構」請求(kernel rootkit、bootkit/韌體持久化、多型惡意碼、反鑑識),大量 dual-use 內容仍給實質答案 → 與本專案 abliterated 模型「**不設限 dual-use**」定位一致(原 doc「攻擊性請求會收到解釋性拒絕」之描述方向對但嚴重高估)。 |
| fine-tuning | [Primus-Instruct](https://huggingface.co/datasets/trendmicro-ailab/Primus-Instruct)                                                             | Trend Micro 出品的資安指令微調資料,業務情境導向(告警解釋、可疑指令分析、風險建議、查詢語言生成),由 GPT-4o 生成回覆。可與 Trendyol/Fenrir 互補增加風格多樣性。([Hugging Face][3])<br>⚠️ **前處理**:去冗餘+剔離題後僅 **568 筆(~0.47%**;SOC5 五場景 485＋general 83),assistant-token 佔比更低(~0.79%)。「增加風格多樣性」需靠**訓練加權**(見 stats.json 的 `assistant_tokens_per_source`)放大,否則會被 Fenrir 淹沒。 |
| fine-tuning | [Primus-Reasoning](https://huggingface.co/datasets/trendmicro-ailab/Primus-Reasoning)                                                           | 資安推理蒸餾資料,加入後可讓模型「會思考」而非單純背答案。論文回報在 CISSP 認證測試提升 10%,適合做 chain-of-thought 風格的訓練。([Hugging Face][4])<br>⚠️ **前處理**:原始檔名即 `ctibench_*.parquet`,題型指紋確認為 **CTIBench** 子任務(CVE→CWE ~55%、MCQ ~44%、CVSS、ATT&CK 抽取)。本專案決策**保留於訓練**(4,611 筆,唯一的 `<think>` 推理訊號),故**評測不得使用 CTIBench**(否則＝考古題),改用 SecQA / CyberSecEval 等。 |
| fine-tuning | [AttackQA](https://huggingface.co/datasets/sambanovasystems/attackqa)                                                                           | 訓練 SOC QA、ATT&CK reasoning、RAG answer style。約 25,335 組 Q&A pairs 並附 rationales,原為 SOC analyst 的 fine-tuned + RAG QA pipeline 設計;前處理以 document 作 context 組成 RAG 式問答。([arXiv][19]) |
| RAG         | [zeroshot/cybersecurity-corpus](https://huggingface.co/datasets/zeroshot/cybersecurity-corpus)                                                  | 通用資安語料庫,適合做 dense retrieval index 的底層通用文本來源。([Hugging Face][5])                                                                                                          |
| RAG         | [Primus-Seed](https://huggingface.co/datasets/trendmicro-ailab/Primus-Seed)                                                                     | Trend Micro 整理的高品質資安預訓練語料(包含 CVE、威脅情報、技術文件),也可拆 chunk 後當 RAG 索引使用。ODC-BY 授權,商用相對友善。([Hugging Face][6])                                                                   |
| RAG         | [Primus-FineWeb](https://huggingface.co/datasets/trendmicro-ailab/Primus-FineWeb)                                                               | 從 FineWeb 篩選出的資安相關網頁語料,規模較大,適合補充廣度。([Hugging Face][7])                                                                                                                   |
| RAG         | [MITRE ATT&CK STIX Data](https://github.com/mitre-attack/attack-stix-data)                                                                      | TTP、tactic、technique、mitigation、detection、group、software 完整對應。Repo 內為 STIX 2.1 JSON collections,涵蓋 Enterprise、Mobile、ICS 三套 matrix。([GitHub][8])                          |
| RAG         | [MITRE D3FEND](https://d3fend.mitre.org/resources/)                                                                                             | ATT&CK 的防禦端對應(攻防成對使用)。提供 JSON / OWL / CSV 下載,搭配 ATT&CK 做 RAG 時可在「攻擊技術 → 防禦對策」之間建立連結。([MITRE D3FEND][9])                                                                  |
| RAG         | [nist-cybersecurity-training](https://huggingface.co/datasets/ethanolivertroy/nist-cybersecurity-training)                                      | NIST CSF / 800-53 / 800-61 等標準的 JSONL 格式,適合合規問答的事實檢索層。([Hugging Face][10])                                                                                                |
| RAG         | [OWASP Top 10](https://github.com/OWASP/Top10)                                                                                                  | Web、API、LLM 三套 Top 10 的 Markdown 原始檔,結構乾淨易切 chunk,涵蓋 Web/API/LLM 安全的權威來源。([GitHub][11])                                                                                   |
| RAG         | [CIS Benchmarks](https://www.cisecurity.org/cis-benchmarks)                                                                                     | 各作業系統與服務(Linux、Windows、Kubernetes、AWS 等)的強化指南 PDF,系統強化問答必備。需註冊免費下載。([CIS][12])                                                                                          |
| eval        | [CyberMetric](https://github.com/cybermetric/CyberMetric)                                                                                       | 約 10,000 題 MCQ,涵蓋滲透測試、密碼學、網路、資訊安全等領域,提供 -80 / -500 / -2000 / -10000 四個尺寸。2025 年學界引用最多的資安 benchmark。([GitHub][13])                                                       |
| eval        | [SecEval](https://github.com/XuanwuAI/SecEval)                                                                                                  | 涵蓋軟體、網路、Web 安全的 MCQ 套件(約 2k+ 題),跟 CyberMetric 互補,測知識廣度。([GitHub][14])                                                                                                    |
| eval        | [SecQA](https://huggingface.co/datasets/zefang-liu/secqa)                                                                                       | 資安基礎概念 MCQ,題目較簡單,適合快速跑 baseline 看模型有沒有崩。不要拿來訓練主模型,保留當 benchmark。([Hugging Face][15])                                                                                    |
| eval        | [SecBench](https://github.com/SecBench/SecBench)                                                                                                | 含 short-answer question (SAQ) 而非單純 MCQ,測模型的生成能力,比 MCQ 更接近實際使用情境。([GitHub][16])                                                                                          |
| eval        | [CTIBench](https://github.com/xashru/cti-bench)                                                                                                 | 威脅情報通用評估,五個子任務(MCQ、CVE→CWE、CVSS、ATT&CK 抽取、威脅歸因)。通用模型可只跑 CTI-MCQ 子集。已被 Google Sec-Gemini、Cisco Foundation-Sec 採用。([GitHub][17])<br>🚫 **本專案不可用作評測**:Primus-Reasoning 實為 CTIBench 衍生且已納入訓練,用 CTIBench 評測＝訓練/評測污染。 |
| eval        | [CyberSecEval (PurpleLlama)](https://github.com/meta-llama/PurpleLlama/tree/main/CybersecurityBenchmarks)                                       | Meta 出品,評估 LLM 的安全韌性,包含 prompt injection、insecure code generation、cyber attack helpfulness、spear phishing 等。**上線前必跑**,確保 fine-tune 後沒崩。([GitHub][18])                     |

---

## 前處理 / 建構說明

詳細品質稽核見 [`QUALITY_REPORT.md`](./QUALITY_REPORT.md);可重跑 `python3 preprocessing/audit_quality.py`。

**正式建構命令**(覆寫 `data/processed/`,用目標模型 tokenizer 計長):

```bash
uv sync --extra tokenizer
uv run python -m preprocessing.build_dataset --drop-offtopic \
  --hf-tokenizer huihui-ai/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated
```

產出 **121,168 筆**真實獨立內容(原始 collect 184,132;排除 Trendyol 53,201、離題 231、結構/長度/去重 ~9,532)。

**建構旗標:**

| 旗標 | 作用 |
|---|---|
| `--exclude-sources PATTERN...` | 排除來源(精確或前綴,如 `trendyol`、`primus/reasoning`);**給定即取代**預設。 |
| `--no-default-excludes` | 不套用 `config.DEFAULT_EXCLUDED_SOURCES`(含 Trendyol),用於重現舊版做對照。 |
| `--drop-offtopic` | 剔除 `primus/general` 中無資安關鍵字的離題樣本(opt-in)。 |

**CTIBench 變體**(若改為保留 CTIBench 當乾淨 eval,需從訓練移除 reasoning):

```bash
uv run python -m preprocessing.build_dataset --exclude-sources trendyol primus/reasoning \
  --hf-tokenizer huihui-ai/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated
```

**來源分佈(去冗餘後)與加權:** 來源嚴重失衡(Fenrir 74.8% 筆數 / ~90% assistant-token、AttackQA 20.9%、Primus-Reasoning 3.8%、Primus-Instruct 0.64%)。`stats.json` 已輸出 `assistant_tokens_per_source` / `total_tokens_per_source` 供訓練階段重算加權(weighted sampler 屬訓練階段,本 repo 尚未實作)。

---

[1]: https://huggingface.co/datasets/Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset "Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset · Datasets at Hugging Face"
[2]: https://huggingface.co/datasets/AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.0 "AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.0 · Datasets at Hugging Face"
[3]: https://huggingface.co/datasets/trendmicro-ailab/Primus-Instruct "trendmicro-ailab/Primus-Instruct · Datasets at Hugging Face"
[4]: https://huggingface.co/datasets/trendmicro-ailab/Primus-Reasoning "trendmicro-ailab/Primus-Reasoning · Datasets at Hugging Face"
[5]: https://huggingface.co/datasets/zeroshot/cybersecurity-corpus "zeroshot/cybersecurity-corpus · Datasets at Hugging Face"
[6]: https://huggingface.co/datasets/trendmicro-ailab/Primus-Seed "trendmicro-ailab/Primus-Seed · Datasets at Hugging Face"
[7]: https://huggingface.co/datasets/trendmicro-ailab/Primus-FineWeb "trendmicro-ailab/Primus-FineWeb · Datasets at Hugging Face"
[8]: https://github.com/mitre-attack/attack-stix-data "GitHub - mitre-attack/attack-stix-data: STIX data representing MITRE ATT&CK"
[9]: https://d3fend.mitre.org/resources/ "D3FEND Resources - MITRE D3FEND"
[10]: https://huggingface.co/datasets/ethanolivertroy/nist-cybersecurity-training "ethanolivertroy/nist-cybersecurity-training · Datasets at Hugging Face"
[11]: https://github.com/OWASP/Top10 "GitHub - OWASP/Top10: Official OWASP Top 10 Document Repository"
[12]: https://www.cisecurity.org/cis-benchmarks "CIS Benchmarks - Center for Internet Security"
[13]: https://github.com/cybermetric/CyberMetric "GitHub - CyberMetric: A Benchmark Dataset for Evaluating LLMs in Cybersecurity"
[14]: https://github.com/XuanwuAI/SecEval "GitHub - XuanwuAI/SecEval: A Benchmark for Evaluating Cybersecurity Knowledge of LLMs"
[15]: https://huggingface.co/datasets/zefang-liu/secqa "zefang-liu/secqa · Datasets at Hugging Face"
[16]: https://github.com/SecBench/SecBench "GitHub - SecBench: A Comprehensive Multi-Dimensional Benchmarking Dataset for LLMs in Cybersecurity"
[17]: https://github.com/xashru/cti-bench "GitHub - xashru/cti-bench: A Benchmark for Evaluating LLMs in Cyber Threat Intelligence"
[18]: https://github.com/meta-llama/PurpleLlama/tree/main/CybersecurityBenchmarks "PurpleLlama/CybersecurityBenchmarks at main · meta-llama/PurpleLlama · GitHub"
[19]: https://arxiv.org/abs/2411.01073 "AttackQA: Development and Adoption of a Dataset for Enhancing SOC Analyst Productivity"