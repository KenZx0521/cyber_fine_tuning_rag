"""loaders 工具函式的單元測試（AAA 結構）。"""

import json

from preprocessing.loaders import unescape_text


def _escape_n_times(text: str, n: int) -> str:
    """把 text 轉義 n 層（模擬 Trendyol/Fenrir 的 *_escaped.jsonl 欄位內容）。"""
    out = text
    for _ in range(n):
        out = json.dumps(out)[1:-1]  # 去掉外層引號，留下轉義過的內容
    return out


def test_unescape_single_level_like_fenrir():
    # Arrange — Fenrir 風格：一層轉義（真換行被存成字面 \n）
    original = "## Causal Analysis\n\n**Direct Answer:** ...\n- bullet"
    escaped = _escape_n_times(original, 1)
    assert escaped != original  # 確認確實被轉義

    # Act / Assert
    assert unescape_text(escaped) == original


def test_unescape_double_level_like_trendyol():
    # Arrange — Trendyol 風格：兩層轉義
    original = "Line one\nLine two\n\nLine four"
    escaped = _escape_n_times(original, 2)

    # Act / Assert
    assert unescape_text(escaped) == original


def test_unescape_plain_text_unchanged():
    # Arrange — 無轉義內容應原樣返回
    s = "no escapes here, just plain text"

    # Act / Assert
    assert unescape_text(s) == s


def test_unescape_does_not_touch_real_newlines():
    # Arrange — 已是真換行（未轉義）不應被破壞
    s = "already\nreal\nnewlines"

    # Act / Assert
    assert unescape_text(s) == s


def test_unescape_preserves_non_ascii():
    # Arrange — 系統提示常含 em-dash 等非 ASCII，還原須保留（不可用 unicode_escape）
    original = "cyber‑defense — confidentiality\nintegrity"
    escaped = _escape_n_times(original, 1)

    # Act / Assert
    assert unescape_text(escaped) == original
