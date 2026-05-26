"""依來源分層的 train/val 切分（不依賴 sklearn）。"""

from __future__ import annotations

import random
from collections import defaultdict


def stratified_split(
    records: list[dict],
    val_size: float = 0.02,
    seed: int = 42,
    key: str = "source",
) -> tuple[list[dict], list[dict]]:
    """依 `key` 分層切分，確保每個來源都依比例出現在 val。

    - 每組以固定 seed shuffle 後切出 round(len*val_size) 筆作 val。
    - 組別夠大（>=50）但比例算出 0 時，至少留 1 筆給 val。
    - 永不把整組都丟進 val（train 至少保留 1 筆）。
    """
    groups: dict = defaultdict(list)
    for record in records:
        groups[record.get(key)].append(record)

    rng = random.Random(seed)
    train: list[dict] = []
    val: list[dict] = []

    for group_key in sorted(groups, key=lambda k: (k is None, k)):
        group = groups[group_key][:]
        rng.shuffle(group)
        n_val = round(len(group) * val_size)
        if n_val == 0 and len(group) >= 50:
            n_val = 1
        if len(group) > 1:
            n_val = min(n_val, len(group) - 1)
        else:
            n_val = 0
        val.extend(group[:n_val])
        train.extend(group[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val
