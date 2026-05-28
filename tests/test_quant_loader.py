"""training/quant_loader.py 單元測試：BitsAndBytesConfig 屬性 + 命中數防呆。

不需 GPU；bitsandbytes 已是 pyproject dependency，可直接 import。
"""

from __future__ import annotations

import pytest


# --- build_bnb_config ---


def test_build_bnb_config_defaults():
    # Arrange / Act
    from training.quant_loader import build_bnb_config

    cfg = build_bnb_config()

    # Assert：QLoRA 標準屬性 + 我們的 skip 清單
    import torch

    assert cfg.load_in_4bit is True
    assert cfg.bnb_4bit_quant_type == "nf4"
    assert cfg.bnb_4bit_compute_dtype == torch.bfloat16
    assert cfg.bnb_4bit_use_double_quant is True
    assert cfg.bnb_4bit_quant_storage == torch.bfloat16  # storage 對齊 compute（R6 緩解）
    assert "visual" in cfg.llm_int8_skip_modules
    assert "lm_head" in cfg.llm_int8_skip_modules
    assert "linear_attn" in cfg.llm_int8_skip_modules  # R1 緩解：SSM 子樹整段保 bf16


def test_build_bnb_config_8bit_disallowed():
    # Act / Assert：本 pipeline 明確不支援 8-bit
    from training.quant_loader import build_bnb_config

    with pytest.raises(ValueError, match="僅支援 4-bit"):
        build_bnb_config(load_in_4bit=False)


def test_build_bnb_config_custom_skip_modules():
    # Arrange / Act：允許覆寫 skip_modules
    from training.quant_loader import build_bnb_config

    cfg = build_bnb_config(skip_modules=["foo", "bar"])
    # Assert
    assert cfg.llm_int8_skip_modules == ["foo", "bar"]


# --- assert_quantization_applied ---


def _make_linear4bit():
    """無 GPU 環境也可建立 Linear4bit 實例（forward 才需要 GPU）。"""
    from bitsandbytes.nn import Linear4bit

    return Linear4bit(input_features=4, output_features=4)


class _FakeModel:
    """最小化的假 model：只實作 named_modules() 給 assert_quantization_applied 用。"""

    def __init__(self, modules):
        self._modules_list = modules

    def named_modules(self):
        return iter(self._modules_list)


def test_assert_quantization_applied_passes_when_enough():
    # Arrange：200 個 Linear4bit
    from training.quant_loader import assert_quantization_applied

    fake = _FakeModel([(f"layer.{i}.proj", _make_linear4bit()) for i in range(200)])
    # Act / Assert：不應 raise
    assert_quantization_applied(fake, expected_min_4bit_modules=200)


def test_assert_quantization_applied_raises_when_too_few():
    # Arrange：只給 10 個（< 預設 200）
    from training.quant_loader import assert_quantization_applied

    fake = _FakeModel([(f"layer.{i}.proj", _make_linear4bit()) for i in range(10)])
    # Act / Assert：低於門檻應 raise（防 silently 全 bf16 載入）
    with pytest.raises(AssertionError, match="量化未生效或命中過少"):
        assert_quantization_applied(fake, expected_min_4bit_modules=200)


def test_assert_quantization_applied_custom_threshold():
    # Arrange：5 個就夠（自訂門檻）
    from training.quant_loader import assert_quantization_applied

    fake = _FakeModel([(f"layer.{i}.proj", _make_linear4bit()) for i in range(5)])
    # Act / Assert
    assert_quantization_applied(fake, expected_min_4bit_modules=5)
