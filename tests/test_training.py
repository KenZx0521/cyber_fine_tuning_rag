"""training/ 模組單元測試：per-source 權重展開、加權 sampler、訓練模板 mask。

純函式測試不載真模型；模板 mask 測試用 tokenizer（無法載入則 skip）。
"""

from __future__ import annotations

import json

import pytest

from training import config
from training.data import build_example_weights, load_sampler_weights
from training.weighted_trainer import make_weighted_sampler


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
