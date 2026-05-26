"""前處理主流程 CLI。

load → convert → validate → length-filter → dedup → split → write JSONL + stats.json

用法：
    uv run python -m preprocessing.build_dataset --dry-run
    uv run python -m preprocessing.build_dataset
    uv run python -m preprocessing.build_dataset --max-total-tokens 16384 --hf-tokenizer Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Optional

from . import config
from .converters import attackqa_to_record, primus_to_record
from .loaders import load_attackqa, load_primus
from .quality import cjk_char_ratio, count_record_tokens, exact_dedup, validate_record
from .split import stratified_split

# 任一 message 的 CJK 佔比超過此值即視為含中文（語言分佈統計用）。
_CJK_FLAG_RATIO = 0.1


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    if lo == hi:
        return float(sorted_vals[lo])
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def _make_token_counter(hf_tokenizer: Optional[str]) -> Optional[Callable[[str], int]]:
    """需要精確計數時載入 HF tokenizer；否則回傳 None（使用 heuristic）。"""
    if not hf_tokenizer:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError:
        sys.exit("--hf-tokenizer 需要 transformers：先執行 `uv sync --extra tokenizer`")
    tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer)
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False))


def _collect_records(no_system: bool) -> list[dict]:
    system_attackqa = None if no_system else config.ATTACKQA_SYSTEM_PROMPT
    system_primus = None if no_system else config.SECURITY_SYSTEM_PROMPT

    records: list[dict] = []
    for row in load_attackqa():
        records.append(attackqa_to_record(row, system_attackqa))
    for scenario, row in load_primus():
        records.append(primus_to_record(scenario, row, system_primus))
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """寫 JSONL；剔除底線開頭的內部欄位（如 _tokens）。"""
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            clean = {k: v for k, v in record.items() if not k.startswith("_")}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def _build_stats(records: list[dict], raw_total: int, dropped: Counter, args) -> dict:
    tokens = sorted(r.get("_tokens", 0) for r in records)
    per_source = Counter(r["source"] for r in records)
    per_category = Counter(r.get("category") for r in records)
    with_cjk = sum(
        1
        for r in records
        if any(cjk_char_ratio(m["content"]) > _CJK_FLAG_RATIO for m in r["messages"])
    )
    return {
        "raw_total": raw_total,
        "kept_total": len(records),
        "dropped": dict(dropped),
        "max_total_tokens": args.max_total_tokens,
        "token_counter": args.hf_tokenizer or f"heuristic(chars/{config.CHARS_PER_TOKEN})",
        "system_prompt_injected": not args.no_system,
        "token_length": {
            "p50": round(_percentile(tokens, 0.50), 1),
            "p95": round(_percentile(tokens, 0.95), 1),
            "p99": round(_percentile(tokens, 0.99), 1),
            "max": tokens[-1] if tokens else 0,
        },
        "records_with_cjk": with_cjk,
        "per_source": dict(per_source.most_common()),
        "per_category": dict(per_category.most_common()),
    }


def _print_summary(stats: dict) -> None:
    print("=" * 60)
    print("前處理統計")
    print("=" * 60)
    print(f"原始筆數：{stats['raw_total']}")
    print(f"保留筆數：{stats['kept_total']}")
    print(f"丟棄：{stats['dropped']}")
    print(f"長度上限：{stats['max_total_tokens']} tok（計數：{stats['token_counter']}）")
    tl = stats["token_length"]
    print(f"長度分佈：p50={tl['p50']} p95={tl['p95']} p99={tl['p99']} max={tl['max']}")
    print(f"含 CJK 的筆數：{stats['records_with_cjk']}")
    print("各來源筆數：")
    for source, n in stats["per_source"].items():
        print(f"  {source:40} {n}")


def build(args) -> None:
    counter = _make_token_counter(args.hf_tokenizer)
    dropped: Counter = Counter()

    records = _collect_records(args.no_system)
    raw_total = len(records)

    # 1) 結構驗證
    valid: list[dict] = []
    for record in records:
        if validate_record(record):
            dropped["invalid"] += 1
            continue
        valid.append(record)

    # 2) 長度過濾（順便把估算長度暫存到 _tokens 供統計用）
    length_kept: list[dict] = []
    for record in valid:
        n_tokens = count_record_tokens(record, counter)
        record["_tokens"] = n_tokens
        if n_tokens > args.max_total_tokens:
            dropped["too_long"] += 1
            continue
        length_kept.append(record)

    # 3) 精確去重
    deduped, n_dup = exact_dedup(length_kept)
    dropped["duplicate"] = n_dup

    stats = _build_stats(deduped, raw_total, dropped, args)

    if args.dry_run:
        _print_summary(stats)
        print("\n[dry-run] 未寫入任何檔案。")
        return

    out_dir = Path(args.output_dir)
    sources_dir = out_dir / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)

    # 各來源獨立檔
    by_source: dict = defaultdict(list)
    for record in deduped:
        by_source[record["source"]].append(record)
    for source, recs in by_source.items():
        filename = source.replace("/", "_") + ".jsonl"
        _write_jsonl(sources_dir / filename, recs)

    # 合併 train/val（依來源分層）
    train, val = stratified_split(deduped, args.val_size, args.seed)
    _write_jsonl(out_dir / "train.jsonl", train)
    _write_jsonl(out_dir / "val.jsonl", val)

    stats["train_size"] = len(train)
    stats["val_size"] = len(val)
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    _print_summary(stats)
    print(f"\n輸出目錄：{out_dir}")
    print(f"  sources/*.jsonl（{len(by_source)} 檔）")
    print(f"  train.jsonl（{len(train)}） val.jsonl（{len(val)}） stats.json")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="資安語料 fine-tuning 前處理：parquet → chat messages JSONL",
    )
    parser.add_argument("--output-dir", default=str(config.OUTPUT_DIR))
    parser.add_argument("--val-size", type=float, default=config.DEFAULT_VAL_SIZE)
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument(
        "--max-total-tokens", type=int, default=config.DEFAULT_MAX_TOTAL_TOKENS
    )
    parser.add_argument(
        "--no-system", action="store_true", help="不注入 system prompt"
    )
    parser.add_argument(
        "--hf-tokenizer",
        default=None,
        help="用指定 HF tokenizer 精確計算長度（需先 `uv sync --extra tokenizer`）",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="只驗證並印統計，不寫任何檔"
    )
    build(parser.parse_args(argv))


if __name__ == "__main__":
    main()
