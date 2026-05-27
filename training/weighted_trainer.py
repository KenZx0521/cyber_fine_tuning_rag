"""WeightedSFTTrainer：把 per-example 取樣權重接上 SFTTrainer 的訓練 sampler。"""

from __future__ import annotations

import torch
from torch.utils.data import Sampler, WeightedRandomSampler
from trl import SFTTrainer


def make_weighted_sampler(
    example_weights: list[float], dataset_len: int
) -> WeightedRandomSampler:
    """建 per-example 權重的 WeightedRandomSampler（抽出為純函式以便單元測試）。

    replacement=True 讓小來源能被重複抽樣；num_samples=dataset_len 維持 epoch 步數。
    長度不符直接報錯（避免權重與資料錯位卻靜默訓練）。
    """
    if len(example_weights) != dataset_len:
        raise ValueError(
            f"example_weights 長度 {len(example_weights)} "
            f"≠ train_dataset 長度 {dataset_len}（資料順序或筆數可能被改動）"
        )
    weights = torch.as_tensor(example_weights, dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=dataset_len, replacement=True)


class WeightedSFTTrainer(SFTTrainer):
    """覆寫 _get_train_sampler，改用 per-example 權重的 WeightedRandomSampler。

    sampler_weights.json 是 per-source 重複倍率，已由 data.build_example_weights
    展開成 per-example 權重 list（長度 = len(train_dataset)）後傳入。
      - replacement=True：讓 cap 觸頂的小來源能被重複抽樣（上採樣）。
      - num_samples=len(ds)：維持每 epoch 的步數規模不變。

    transformers 5.9：Trainer._get_train_sampler(self, train_dataset=None)（已驗證簽名）；
    SFTTrainer 未覆寫此方法，故子類覆寫無衝突。
    """

    def __init__(self, *args, example_weights: list[float] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._example_weights = example_weights

    def _get_train_sampler(self, train_dataset=None) -> Sampler | None:
        # 未提供權重時退回原生行為（一般 shuffle / sequential）。
        if self._example_weights is None:
            return super()._get_train_sampler(train_dataset)

        ds = train_dataset if train_dataset is not None else self.train_dataset
        return make_weighted_sampler(self._example_weights, len(ds))
