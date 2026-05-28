"""LoRA target_modules 命中數測試（meta-device，無 GPU、不下載權重）。

驗證 training.config.TARGET_MODULES 在目標模型上的 nn.Linear 命中數符合預期，
catch target_modules 拼錯造成 LoRA silently 0 命中的回歸。

無網路 / config 未 cache 時 skip，行為對齊 tests/test_training.py 的 train_tokenizer fixture。
"""

from __future__ import annotations

import pytest
import torch.nn as nn


@pytest.fixture(scope="module")
def skeleton_model():
    try:
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoModelForImageTextToText

        from modeling.config import MODEL_ID

        cfg = AutoConfig.from_pretrained(MODEL_ID)
        with init_empty_weights():
            model = AutoModelForImageTextToText.from_config(cfg)
        return model
    except Exception as exc:  # noqa: BLE001 — 無網路/cache 時 skip 而非 fail
        pytest.skip(f"無法建 meta-device skeleton（需網路或本地 config cache）：{exc}")


def _count_endswith(model, name: str) -> int:
    """endswith 模擬 PEFT 的子模組名稱匹配；僅算 nn.Linear。"""
    return sum(
        1 for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and (n.endswith("." + name) or n == name)
    )


def test_target_modules_attention_hits(skeleton_model):
    # Act / Assert：16 層 full-attention 各 4 投影
    assert _count_endswith(skeleton_model, "q_proj") == 16
    assert _count_endswith(skeleton_model, "k_proj") == 16
    assert _count_endswith(skeleton_model, "v_proj") == 16
    assert _count_endswith(skeleton_model, "o_proj") == 16


def test_target_modules_mlp_hits(skeleton_model):
    # Act / Assert：全 64 層 dense MLP 各 3 投影
    assert _count_endswith(skeleton_model, "gate_proj") == 64
    assert _count_endswith(skeleton_model, "up_proj") == 64
    assert _count_endswith(skeleton_model, "down_proj") == 64


def test_target_modules_no_vision_collision(skeleton_model):
    # Act：列出所有命中 target_modules 的模組名稱
    from training.config import TARGET_MODULES

    visual_hits = []
    for name, module in skeleton_model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        for tgt in TARGET_MODULES:
            if (name.endswith("." + tgt) or name == tgt) and "visual" in name:
                visual_hits.append(name)
                break
    # Assert：vision 用 linear_fc1/linear_fc2，與 target_modules 名稱不衝突
    assert visual_hits == [], f"vision 模組被誤命中：{visual_hits}"


def test_target_modules_extended_linear_attn(skeleton_model):
    # Act / Assert：--extended 模式下 48 層 GatedDeltaNet 各投影
    assert _count_endswith(skeleton_model, "in_proj_qkv") == 48
    assert _count_endswith(skeleton_model, "in_proj_z") == 48
    assert _count_endswith(skeleton_model, "in_proj_b") == 48
    assert _count_endswith(skeleton_model, "in_proj_a") == 48
    assert _count_endswith(skeleton_model, "out_proj") == 48


def test_no_mtp_params(skeleton_model):
    # 確認 transformers 載入時已忽略 MTP（R4 緩解）
    mtp_modules = [n for n, _ in skeleton_model.named_modules() if n.startswith("mtp")]
    assert mtp_modules == [], (
        f"預期 MTP 被 _keys_to_ignore_on_load_unexpected 丟棄，但找到 {len(mtp_modules)} 個"
    )
