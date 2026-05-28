"""WeightedSFTTrainer：把 per-example 取樣權重接上 SFTTrainer 的訓練 sampler。

提供兩種 sampler（皆抽成純函式以便單元測試）：
  - make_weighted_sampler：純加權取樣（WeightedRandomSampler）。
  - weighted_length_grouped_indices / WeightedLengthGroupedSampler：
    先加權抽樣、再長度分組重排（砍 padding 浪費）。僅重排不改抽樣分佈 →
    per-source 上採樣語意完整保留，無品質損失。
"""

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


def weighted_length_grouped_indices(
    example_weights: list[float],
    lengths: list[int],
    batch_size: int,
    num_samples: int,
    mega_batch_mult: int = 50,
    generator: torch.Generator | None = None,
) -> list[int]:
    """先按 per-example 權重抽樣（replacement），再做長度分組重排。

    1) torch.multinomial 依權重抽 num_samples 個 index（上採樣語意同 WeightedRandomSampler）。
    2) 切成 mega-batch（= batch_size × mega_batch_mult），每個 mega-batch 內依長度降序排序，
       使 DataLoader 取連續 batch_size 時長度相近 → dynamic padding 逼近實際長度、砍 padding。
    3) 把含「全域最長樣本」的 mega-batch 換到最前、其最長樣本置頂 → 第 0 步即觸發記憶體峰值，
       OOM 會立刻發生而非訓到一半才爆（方便 batch size 實測）。

    僅重排不改抽樣分佈，per-source 加權／上採樣語意完整保留 → 無品質損失。
    """
    if len(example_weights) != len(lengths):
        raise ValueError(
            f"example_weights 長度 {len(example_weights)} ≠ lengths 長度 {len(lengths)}"
        )
    weights = torch.as_tensor(example_weights, dtype=torch.double)
    drawn = torch.multinomial(
        weights, num_samples, replacement=True, generator=generator
    ).tolist()

    mega = max(1, batch_size * mega_batch_mult)
    megabatches = [drawn[i : i + mega] for i in range(0, len(drawn), mega)]
    megabatches = [
        sorted(mb, key=lambda idx: lengths[idx], reverse=True) for mb in megabatches
    ]
    # 把含全域最長樣本的 mega-batch 換到最前、最長樣本置頂（OOM-early）。
    if megabatches:
        head_max = [lengths[mb[0]] for mb in megabatches]
        max_mb = int(torch.argmax(torch.tensor(head_max)).item())
        megabatches[0][0], megabatches[max_mb][0] = (
            megabatches[max_mb][0],
            megabatches[0][0],
        )
    return [idx for mb in megabatches for idx in mb]


class WeightedLengthGroupedSampler(Sampler[int]):
    """加權抽樣 + 長度分組重排；每次 __iter__ 重抽（行為對齊 WeightedRandomSampler）。"""

    def __init__(
        self,
        example_weights: list[float],
        lengths: list[int],
        batch_size: int,
        num_samples: int,
        mega_batch_mult: int = 50,
        generator: torch.Generator | None = None,
    ):
        if len(example_weights) != len(lengths):
            raise ValueError(
                f"example_weights 長度 {len(example_weights)} ≠ lengths 長度 {len(lengths)}"
            )
        self._weights = example_weights
        self._lengths = lengths
        self._batch_size = batch_size
        self._num_samples = num_samples
        self._mega_batch_mult = mega_batch_mult
        self._generator = generator

    def __len__(self) -> int:
        return self._num_samples

    def __iter__(self):
        return iter(
            weighted_length_grouped_indices(
                self._weights,
                self._lengths,
                self._batch_size,
                self._num_samples,
                self._mega_batch_mult,
                self._generator,
            )
        )


class WeightedSFTTrainer(SFTTrainer):
    """覆寫 _get_train_sampler，改用 per-example 權重的取樣器。

    sampler_weights.json 是 per-source 重複倍率，已由 data.build_example_weights
    展開成 per-example 權重 list（長度 = len(train_dataset)）後傳入。
      - replacement=True：讓 cap 觸頂的小來源能被重複抽樣（上採樣）。
      - num_samples=len(ds)：維持每 epoch 的步數規模不變。
      - group_by_length=True：抽樣後再長度分組重排，砍 padding 浪費（不改抽樣分佈）。

    transformers 5.9：Trainer._get_train_sampler(self, train_dataset=None)（已驗證簽名）；
    SFTTrainer 未覆寫此方法，故子類覆寫無衝突。
    """

    def __init__(
        self,
        *args,
        example_weights: list[float] | None = None,
        group_by_length: bool = True,
        mega_batch_mult: int = 50,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._example_weights = example_weights
        self._group_by_length = group_by_length
        self._mega_batch_mult = mega_batch_mult
        self._cached_lengths: list[int] | None = None

    def _example_lengths(self, ds) -> list[int] | None:
        """從已 tokenize 的 train_dataset 取每筆 token 長度（供長度分組）。

        取不到 input_ids 欄位時回傳 None → 退回不分組的純加權取樣。
        以 pyarrow 由 list offsets 直接算長度（不實體化 token 值，快且省記憶體）。
        """
        if self._cached_lengths is not None:
            return self._cached_lengths
        cols = getattr(ds, "column_names", None) or []
        if "input_ids" not in cols:
            return None
        try:
            import pyarrow.compute as pc

            self._cached_lengths = pc.list_value_length(
                ds.data.column("input_ids")
            ).to_pylist()
        except Exception:  # noqa: BLE001 — 退回逐筆計長（仍正確，僅較慢）
            self._cached_lengths = [len(x) for x in ds["input_ids"]]
        return self._cached_lengths

    def _get_train_sampler(self, train_dataset=None) -> Sampler | None:
        # 未提供權重時退回原生行為（一般 shuffle / sequential）。
        if self._example_weights is None:
            return super()._get_train_sampler(train_dataset)

        ds = train_dataset if train_dataset is not None else self.train_dataset
        n = len(ds)
        if len(self._example_weights) != n:
            raise ValueError(
                f"example_weights 長度 {len(self._example_weights)} "
                f"≠ train_dataset 長度 {n}（資料順序或筆數可能被改動）"
            )

        lengths = self._example_lengths(ds) if self._group_by_length else None
        if lengths is None:
            return make_weighted_sampler(self._example_weights, n)
        return WeightedLengthGroupedSampler(
            self._example_weights,
            lengths,
            batch_size=self.args.per_device_train_batch_size,
            num_samples=n,
            mega_batch_mult=self._mega_batch_mult,
        )
