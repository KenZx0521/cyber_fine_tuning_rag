#!/usr/bin/env python3
"""從 stats.json 計算 per-source WeightedRandomSampler 權重（stdlib only，可重跑）。

問題：Fenrir 佔 ~90% assistant-token，直接訓練會用「超長顧問腔」主導學習訊號。
方法：對 assistant-token 分佈做溫度平滑（alpha），再對重複倍率設 cap，
      避免極小來源（<100 筆）被重複抽樣到過擬合。

原理：WeightedRandomSampler 控制「每筆被抽樣的次數」，而 loss 落在 assistant token
上，故某來源的有效學習量 ∝ (抽樣次數) × (每筆 assistant token) = (n_s·w_s)·avg_s
= w_s·T_s。令每筆權重 w_s ∝ T_s^(alpha-1)：
    alpha=1 → w 均等（維持原始 token 分佈，Fenrir 90%）
    alpha=0 → 各來源 token 貢獻均等（過度上採樣小來源）
    alpha≈0.5 → 折衷
權重正規化成「平均每筆=1」，故 w_s 數值即該來源每筆的 **epoch 重複倍率**；
再 clip 到 [floor, cap]，壓掉小來源的過擬合風險。

用法：
    python3 scripts/compute_sampler_weights.py --compare
    python3 scripts/compute_sampler_weights.py --alpha 0.5 --cap 5 \
        --out data/processed/sampler_weights.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def compute(tokens: dict, counts: dict, alpha: float, cap: float, floor: float):
    sources = list(tokens)
    n_total = sum(counts.values())
    raw = {s: tokens[s] ** (alpha - 1) for s in sources}  # ∝ T^(alpha-1)
    # 正規化成 Σ n_s·w_s = N（平均每筆權重=1 → w_s 即重複倍率）
    scale = n_total / sum(counts[s] * raw[s] for s in sources)
    weights = {s: min(cap, max(floor, raw[s] * scale)) for s in sources}

    tok_total = sum(tokens.values())
    adj_total = sum(weights[s] * tokens[s] for s in sources)
    table = [
        {
            "source": s,
            "n": counts[s],
            "tokens": tokens[s],
            "orig_tok_pct": 100 * tokens[s] / tok_total,
            "weight": weights[s],  # = epoch 重複倍率
            "adj_tok_pct": 100 * weights[s] * tokens[s] / adj_total,
        }
        for s in sources
    ]
    table.sort(key=lambda r: -r["adj_tok_pct"])
    return table, weights


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stats", default="data/processed/stats.json")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--cap", type=float, default=5.0, help="重複倍率上限（防小來源過擬合）")
    ap.add_argument("--floor", type=float, default=0.0, help="重複倍率下限")
    ap.add_argument("--out", default=None, help="輸出 {source: weight} JSON 路徑")
    ap.add_argument("--compare", action="store_true", help="額外印 alpha 敏感度對照")
    args = ap.parse_args(argv)

    stats = json.loads(Path(args.stats).read_text(encoding="utf-8"))
    tokens = stats["assistant_tokens_per_source"]
    counts = stats["per_source"]

    table, weights = compute(tokens, counts, args.alpha, args.cap, args.floor)
    print(f"# weighted-sampler 權重  alpha={args.alpha} cap={args.cap} floor={args.floor}")
    print(f"# 來源={len(table)}  總筆數={sum(counts.values())}  "
          f"總assistant-token={sum(tokens.values())}")
    print(f"{'source':42s}{'n':>7}{'orig%':>8}{'調整後%':>9}{'重複x/每筆權重':>16}")
    print("-" * 82)
    for r in table:
        print(f"{r['source']:42s}{r['n']:7d}{r['orig_tok_pct']:8.2f}"
              f"{r['adj_tok_pct']:9.2f}{r['weight']:16.3f}")

    if args.compare:
        alphas = (0.3, 0.5, 0.7)
        cache = {al: compute(tokens, counts, al, args.cap, args.floor)[1] for al in alphas}
        print("\n# alpha 敏感度（數值＝epoch 重複倍率）")
        print(f"{'source':42s}" + "".join(f"{f'a={al}':>8}" for al in alphas))
        for s in sorted(tokens, key=lambda k: -tokens[k]):
            print(f"{s:42s}" + "".join(f"{cache[al][s]:8.2f}" for al in alphas))

    if args.out:
        Path(args.out).write_text(
            json.dumps(weights, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n已寫出權重：{args.out}")


if __name__ == "__main__":
    main()
