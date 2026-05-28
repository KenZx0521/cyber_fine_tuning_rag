"""training/ 模組單元測試：per-source 權重展開、加權 sampler、訓練模板 mask。

純函式測試不載真模型；模板 mask 測試用 tokenizer（無法載入則 skip）。
"""

from __future__ import annotations

import argparse
import json

import pytest

from training import config
from training.data import build_example_weights, load_sampler_weights
from training.train import build_sft_config, configure_wandb
from training.weighted_trainer import (
    make_weighted_sampler,
    weighted_length_grouped_indices,
)


# --- build_example_weights（per-source 倍率 → per-example 權重） ---


def test_build_example_weights_expands_per_source():
    # Arrange
    sources = ["fenrir", "attackqa", "fenrir", "primus/general"]
    weights = {"fenrir": 0.416, "attackqa": 2.543, "primus/general": 5.0}
    # Act
    result = build_example_weights(sources, weights)
    # Assert：依每筆 source 查表展開，順序對應
    assert result == [0.416, 2.543, 0.416, 5.0]


def test_build_example_weights_raises_on_missing_source():
    # Arrange：sources 含權重表沒有的來源
    sources = ["fenrir", "unknown_source"]
    weights = {"fenrir": 0.416}
    # Act / Assert：fail fast，避免靜默用錯權重
    with pytest.raises(KeyError, match="unknown_source"):
        build_example_weights(sources, weights)


def test_load_sampler_weights_reads_json(tmp_path):
    # Arrange
    path = tmp_path / "w.json"
    path.write_text(json.dumps({"fenrir": 0.416, "attackqa": 2.5}), encoding="utf-8")
    # Act
    result = load_sampler_weights(path)
    # Assert
    assert result == {"fenrir": 0.416, "attackqa": 2.5}


# --- make_weighted_sampler ---


def test_make_weighted_sampler_length_and_type():
    # Arrange / Act
    from torch.utils.data import WeightedRandomSampler

    sampler = make_weighted_sampler([1.0, 2.0, 3.0], 3)
    # Assert：型別正確、每 epoch 步數 = dataset 長度、可重複抽樣
    assert isinstance(sampler, WeightedRandomSampler)
    assert len(sampler) == 3
    assert sampler.replacement is True


def test_make_weighted_sampler_raises_on_length_mismatch():
    # Act / Assert：權重與資料筆數錯位時報錯
    with pytest.raises(ValueError, match="長度"):
        make_weighted_sampler([1.0, 2.0], 3)


def test_make_weighted_sampler_upsamples_high_weight_source():
    # Arrange：index 0 權重極高、其餘極低
    import torch

    torch.manual_seed(0)
    weights = [100.0] + [0.01] * 9
    # Act
    draws = list(make_weighted_sampler(weights, 10))
    # Assert：高權重 index 應佔絕大多數（驗證加權方向正確）
    assert draws.count(0) >= 7


# --- weighted_length_grouped_indices（加權抽樣 + 長度分組重排） ---


def test_weighted_length_grouped_preserves_upsampling():
    # Arrange：index 0 權重極高 → 分組重排後仍應主導（驗證加權方向未被破壞）
    import torch

    torch.manual_seed(0)
    weights = [100.0] + [0.01] * 9
    lengths = list(range(10))
    # Act
    out = weighted_length_grouped_indices(
        weights, lengths, batch_size=2, num_samples=10, mega_batch_mult=50
    )
    # Assert：步數不變、高權重 index 仍佔絕大多數
    assert len(out) == 10
    assert out.count(0) >= 7


def test_weighted_length_grouped_sorts_within_megabatch():
    # Arrange：均一權重、長度打散；單一 mega-batch 內應依長度降序（padding 分組生效）
    import torch

    n = 20
    weights = [1.0] * n
    lengths = [(i * 7) % 20 for i in range(n)]
    torch.manual_seed(0)
    # Act：mega_batch_mult 夠大 → 全部落在一個 mega-batch
    out = weighted_length_grouped_indices(
        weights, lengths, batch_size=2, num_samples=n, mega_batch_mult=n
    )
    # Assert：取出的長度序列為降序（含 replacement 重複亦成立）
    out_lengths = [lengths[i] for i in out]
    assert out_lengths == sorted(out_lengths, reverse=True)


def test_weighted_length_grouped_raises_on_length_mismatch():
    # Act / Assert：weights 與 lengths 長度錯位時 fail fast
    with pytest.raises(ValueError, match="長度"):
        weighted_length_grouped_indices(
            [1.0, 2.0], [1, 2, 3], batch_size=2, num_samples=2
        )


# --- 訓練模板 assistant-only mask（需 tokenizer；無法載入則 skip） ---


@pytest.fixture(scope="module")
def train_tokenizer():
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(config.MODEL_ID)
    except Exception as exc:  # noqa: BLE001 — 無網路/快取時 skip 而非 fail
        pytest.skip(f"無法載入 tokenizer（需網路或本地快取）：{exc}")
    tok.chat_template = config.TRAIN_TEMPLATE_PATH.read_text(encoding="utf-8")
    return tok


def _masked_text(tok, messages):
    """回傳「被標記為 assistant（算 loss）」的 token 解碼字串。"""
    out = tok.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        add_generation_prompt=False,
    )
    masked = [i for i, m in zip(out["input_ids"], out["assistant_masks"]) if m == 1]
    return tok.decode(masked)


def test_template_masks_assistant_with_think_only(train_tokenizer):
    # Arrange：單輪 reasoning 樣本（think 內嵌在 content）
    messages = [
        {"role": "system", "content": "SYSTEM_MARKER"},
        {"role": "user", "content": "USER_MARKER"},
        {"role": "assistant", "content": "<think>\nREASONING_MARKER\n</think>\n\nANSWER_MARKER."},
    ]
    # Act
    masked = _masked_text(train_tokenizer, messages)
    # Assert：think + 答案算 loss；system/user 不算
    assert "REASONING_MARKER" in masked
    assert "ANSWER_MARKER" in masked
    assert "SYSTEM_MARKER" not in masked
    assert "USER_MARKER" not in masked


def test_template_masks_all_assistant_turns_in_multiturn(train_tokenizer):
    # Arrange：多輪（無 think）
    messages = [
        {"role": "system", "content": "SYSTEM_MARKER"},
        {"role": "user", "content": "USER_ONE"},
        {"role": "assistant", "content": "ANSWER_ONE"},
        {"role": "user", "content": "USER_TWO"},
        {"role": "assistant", "content": "ANSWER_TWO"},
    ]
    # Act
    masked = _masked_text(train_tokenizer, messages)
    # Assert：兩輪 assistant 都算 loss；user 不算
    assert "ANSWER_ONE" in masked
    assert "ANSWER_TWO" in masked
    assert "USER_ONE" not in masked
    assert "USER_TWO" not in masked


# --- wandb 監控設定（configure_wandb / build_sft_config 的 report_to、run_name） ---


def _train_args(**overrides):
    """造帶齊欄位的 Namespace（對應 parse_args 預設），供 build_sft_config/configure_wandb 用。"""
    base = dict(
        smoke=False, epochs=None, batch_size=None, grad_accum=None,
        max_length=None, output_dir=None, no_weighted=False,
        extended_targets=False, no_wandb=False, quantize="4bit",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _build_or_skip(args):
    """建 SFTConfig；若此環境（如無 GPU/bf16 支援）無法建構則 skip 而非 fail。"""
    try:
        return build_sft_config(args)
    except (ValueError, ImportError, RuntimeError) as exc:  # noqa: BLE001
        pytest.skip(f"SFTConfig 無法在此環境建構：{exc}")


def test_build_sft_config_reports_to_wandb_by_default():
    # Act
    cfg = _build_or_skip(_train_args())
    # Assert：預設只上報 wandb、專案名正確、run 名以 qlora-cyber- 開頭
    assert cfg.report_to == ["wandb"]
    assert cfg.project == config.WANDB_PROJECT
    assert cfg.run_name.startswith("qlora-cyber-")


def test_build_sft_config_no_wandb_disables_reporting():
    # Act
    cfg = _build_or_skip(_train_args(no_wandb=True))
    # Assert：--no-wandb 時不上報任何後端
    assert cfg.report_to == []


def test_build_sft_config_smoke_uses_smoke_run_name():
    # Act
    cfg = _build_or_skip(_train_args(smoke=True))
    # Assert：煙霧測試 run 名以 smoke- 開頭，與正式訓練區隔
    assert cfg.run_name.startswith("smoke-")


def test_configure_wandb_sets_nonsecret_defaults(monkeypatch):
    # Arrange：隔離環境變數、停用真實 .env 載入；提供金鑰使其通過 fail-fast
    import os

    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr(os, "environ", {"WANDB_API_KEY": "dummy"})
    # Act
    configure_wandb(_train_args())
    # Assert：非機密預設被 setdefault 補上
    assert os.environ["WANDB_PROJECT"] == config.WANDB_PROJECT
    assert os.environ["WANDB_BASE_URL"] == config.WANDB_BASE_URL


def test_configure_wandb_missing_key_exits(monkeypatch):
    # Arrange：無金鑰、停用 .env 載入
    import os

    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr(os, "environ", {})
    # Act / Assert：缺 WANDB_API_KEY 時 fail fast，不進入訓練
    with pytest.raises(SystemExit, match="WANDB_API_KEY"):
        configure_wandb(_train_args())


def test_configure_wandb_no_wandb_skips_setup(monkeypatch):
    # Arrange
    import os

    monkeypatch.setattr(os, "environ", {})
    # Act：--no-wandb 應直接略過，不檢查金鑰、不設環境變數
    configure_wandb(_train_args(no_wandb=True))
    # Assert
    assert "WANDB_PROJECT" not in os.environ
