#!/usr/bin/env python3
"""Fine-tuning 資料品質稽核（stdlib only，可重跑）。

對 `data/processed/sources/*.jsonl` 做五項檢驗，結果對應 QUALITY_REPORT.md：
  ① 跨來源重複（md5 內容指紋）+ 全域去重後獨立內容量
  ② system prompt 多樣性
  ③ Primus 各檔離題率（粗略關鍵字偵測，有假陽性）
  ④ Fenrir 安全姿態（拒答率 + 拒答觸發樣貌）
  ⑤ Primus-Reasoning 題型指紋（是否＝CTIBench 子任務）

用法：
    python3 preprocessing/audit_quality.py [SOURCES_DIR]
    # 預設 SOURCES_DIR = data/processed/sources
"""
from __future__ import annotations

import collections
import glob
import hashlib
import json
import os
import re
import sys

DEFAULT_SRC_DIR = "data/processed/sources"

# NOTE: 與 preprocessing/config.py 的 CYBER_HINT_PATTERN 逐字同步（test_build_filters 守住）。
# 本檔刻意維持 stdlib-only / standalone，故不從 config import，改各保留一份。
CYBER_HINT = re.compile(
    r"(secur|attack|threat|vulnerab|malware|cyber|exploit|cve|cwe|mitre|att&ck|"
    r"ransom|phish|firewall|encrypt|crypto|siem|\bsoc\b|incident|defen|payload|"
    r"injection|backdoor|privilege|reconnaiss|c2\b|command and control|"
    r"安全|漏洞|攻击|威胁|加密|防御)",
    re.I,
)

# assistant 開頭 ~300 字內出現拒答語句
REFUSE = re.compile(
    r"^(.{0,300}?)(I (can'?t|cannot|won'?t|am unable|am not able)|"
    r"cannot (assist|help|provide|comply|in good conscience)|"
    r"I must (decline|refuse)|I won'?t provide|unable to (assist|provide)|"
    r"I do not provide|against (my (guidelines|ethics)|ethical))",
    re.I | re.S,
)

CTIBENCH_SIG = {
    "CVE→CWE (CTI-RCM)": re.compile(r"map it to the appropriate CWE|only the CWE ID", re.I),
    "CVSS 評分 (CTI-VSP)": re.compile(r"CVSS|base score|vector string", re.I),
    "ATT&CK 抽取 (CTI-TAA)": re.compile(r"ATT&CK|technique ID|T\d{4}", re.I),
    "MCQ (CTI-MCQ)": re.compile(r"\bA\)|\bB\)|\bC\)|\bD\)|which of the following", re.I),
}


def norm(text: str) -> str:
    """正規化空白，避免尾隨換行造成假性不重複。"""
    return re.sub(r"\s+", " ", (text or "").strip())


def load(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def roles_joined(record: dict) -> dict:
    """回傳 {role: 該角色所有 content 串接}。"""
    buckets: dict[str, list[str]] = collections.defaultdict(list)
    for msg in record["messages"]:
        buckets[msg["role"]].append(msg["content"])
    return {role: " ".join(parts) for role, parts in buckets.items()}


def md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def collect(src_dir: str):
    """單次掃描各來源，回傳指紋集合與統計。"""
    ua_hashes: dict[str, set] = {}
    full_hashes: dict[str, set] = {}
    sys_counts: dict[str, collections.Counter] = {}
    counts: dict[str, int] = {}

    for path in sorted(glob.glob(os.path.join(src_dir, "*.jsonl"))):
        src = os.path.basename(path)[:-6]
        ua, full = set(), set()
        sysc: collections.Counter = collections.Counter()
        n = 0
        for rec in load(path):
            n += 1
            r = roles_joined(rec)
            u, a, s = norm(r.get("user", "")), norm(r.get("assistant", "")), norm(r.get("system", ""))
            ua.add(md5(u + "\x00" + a))
            full.add(md5(s + "\x00" + u + "\x00" + a))
            sysc[md5(s)] += 1
        ua_hashes[src], full_hashes[src], sys_counts[src], counts[src] = ua, full, sysc, n
    return ua_hashes, full_hashes, sys_counts, counts


def report_dedup(ua_hashes, full_hashes, counts):
    print("=" * 70)
    print("① 各來源筆數 / 跨來源重複")
    print("=" * 70)
    srcs = list(ua_hashes)
    for src in sorted(counts, key=lambda k: -counts[k]):
        others = set().union(*(ua_hashes[o] for o in srcs if o != src)) if len(srcs) > 1 else set()
        inter = ua_hashes[src] & others
        pct = 100.0 * len(inter) / max(1, len(ua_hashes[src]))
        print(f"{src:42s} n={counts[src]:6d}  與其他來源重疊 {len(inter):6d} ({pct:5.1f}%)")

    if "trendyol" in ua_hashes and "fenrir" in ua_hashes:
        t, fe = ua_hashes["trendyol"], ua_hashes["fenrir"]
        tf, ff = full_hashes["trendyol"], full_hashes["fenrir"]
        print(f"\n重點配對 trendyol vs fenrir：")
        print(f"  trendyol 落在 fenrir (user+assistant) = {100.0*len(t&fe)/max(1,len(t)):.1f}%")
        print(f"  trendyol 落在 fenrir (含 system)       = {100.0*len(tf&ff)/max(1,len(tf)):.1f}%")

    global_ua = set().union(*ua_hashes.values()) if ua_hashes else set()
    total = sum(counts.values())
    print(f"\n全部合計 {total} 筆 → 全域去重後獨立內容 {len(global_ua)} 筆 "
          f"(冗餘 {total-len(global_ua)}, {100.0*(total-len(global_ua))/max(1,total):.1f}%)")


def report_system(sys_counts, counts):
    print("\n" + "=" * 70)
    print("② system prompt 多樣性")
    print("=" * 70)
    for src in sorted(counts, key=lambda k: -counts[k]):
        c = sys_counts[src]
        top = c.most_common(1)[0][1] if c else 0
        print(f"{src:42s} distinct={len(c):4d}  top_share={100.0*top/max(1,counts[src]):5.1f}%")


def report_offtopic(src_dir):
    print("\n" + "=" * 70)
    print("③ Primus 離題率（無資安關鍵字；粗略、有假陽性）")
    print("=" * 70)
    for path in sorted(glob.glob(os.path.join(src_dir, "primus_*.jsonl"))):
        src = os.path.basename(path)[:-6]
        n = off = 0
        samples: list[str] = []
        for rec in load(path):
            n += 1
            r = roles_joined(rec)
            if not CYBER_HINT.search(r.get("user", "") + " " + r.get("assistant", "")):
                off += 1
                if len(samples) < 5:
                    samples.append(r.get("user", "")[:70].replace("\n", " "))
        print(f"{src:42s} off={off:4d}/{n:4d} ({100.0*off/max(1,n):4.1f}%)")
        for s in samples:
            print(f"      · {s!r}")


def report_fenrir_safety(src_dir):
    print("\n" + "=" * 70)
    print("④ Fenrir 安全姿態（拒答率 + 觸發樣貌）")
    print("=" * 70)
    path = os.path.join(src_dir, "fenrir.jsonl")
    if not os.path.exists(path):
        print("  (找不到 fenrir.jsonl，略過)")
        return
    n = refuse = 0
    samples: list[str] = []
    for rec in load(path):
        n += 1
        r = roles_joined(rec)
        if REFUSE.match(r.get("assistant", "")):
            refuse += 1
            if len(samples) < 15:
                samples.append(r.get("user", "")[:100].replace("\n", " "))
    print(f"總筆數 {n}；開頭即拒答 {refuse} ({100.0*refuse/max(1,n):.2f}%) → 照答 {100.0*(n-refuse)/max(1,n):.1f}%")
    print("拒答觸發的 prompt 樣貌（前 15）：")
    for s in samples:
        print(f"  · {s!r}")


def report_ctibench(src_dir):
    print("\n" + "=" * 70)
    print("⑤ Primus-Reasoning 題型指紋（是否＝CTIBench）")
    print("=" * 70)
    for path in sorted(glob.glob(os.path.join(src_dir, "primus_reasoning-*.jsonl"))):
        src = os.path.basename(path)[:-6]
        n = 0
        hits: collections.Counter = collections.Counter()
        for rec in load(path):
            n += 1
            u = roles_joined(rec).get("user", "")
            for name, rx in CTIBENCH_SIG.items():
                if rx.search(u):
                    hits[name] += 1
        print(f"{src} (n={n}):")
        for name in CTIBENCH_SIG:
            print(f"    {name:24s} {hits[name]:5d} ({100.0*hits[name]/max(1,n):4.1f}%)")


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    src_dir = argv[0] if argv else DEFAULT_SRC_DIR
    if not os.path.isdir(src_dir):
        print(f"錯誤：找不到來源目錄 {src_dir!r}（請先跑 build_dataset，或指定路徑）", file=sys.stderr)
        return 1

    ua_hashes, full_hashes, sys_counts, counts = collect(src_dir)
    report_dedup(ua_hashes, full_hashes, counts)
    report_system(sys_counts, counts)
    report_offtopic(src_dir)
    report_fenrir_safety(src_dir)
    report_ctibench(src_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
