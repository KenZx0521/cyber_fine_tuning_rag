"""quality 的單元測試（AAA 結構）。"""

from preprocessing.quality import (
    cjk_char_ratio,
    count_record_tokens,
    estimate_tokens,
    exact_dedup,
    is_valid,
    validate_record,
)


def _record(messages, source="attackqa"):
    return {"messages": messages, "source": source}


def test_estimate_tokens_rounds_up():
    # Arrange / Act / Assert
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1  # 4 chars / 4
    assert estimate_tokens("abcde") == 2  # 5 chars / 4 -> ceil


def test_count_record_tokens_sums_all_messages():
    # Arrange
    record = _record(
        [
            {"role": "user", "content": "abcd"},      # 1
            {"role": "assistant", "content": "abcdefgh"},  # 2
        ]
    )

    # Act / Assert
    assert count_record_tokens(record) == 3


def test_count_record_tokens_with_custom_counter():
    # Arrange
    record = _record([{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}])

    # Act
    total = count_record_tokens(record, counter=lambda s: len(s) * 10)

    # Assert
    assert total == 20


def test_valid_record_passes():
    # Arrange
    record = _record(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
    )

    # Act / Assert
    assert is_valid(record)
    assert validate_record(record) == []


def test_empty_content_is_invalid():
    # Arrange
    record = _record([{"role": "user", "content": "  "}, {"role": "assistant", "content": "a"}])

    # Act / Assert
    assert not is_valid(record)


def test_invalid_role_is_rejected():
    # Arrange
    record = _record([{"role": "human", "content": "q"}, {"role": "assistant", "content": "a"}])

    # Act / Assert
    assert any("invalid role" in e for e in validate_record(record))


def test_non_alternating_turns_rejected():
    # Arrange
    record = _record(
        [
            {"role": "user", "content": "q1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a"},
        ]
    )

    # Act / Assert
    assert any("alternating" in e for e in validate_record(record))


def test_must_end_with_assistant():
    # Arrange
    record = _record([{"role": "user", "content": "q"}])

    # Act / Assert
    errors = validate_record(record)
    assert any("assistant" in e for e in errors)


def test_system_not_first_is_rejected():
    # Arrange
    record = _record(
        [
            {"role": "user", "content": "q"},
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "a"},
        ]
    )

    # Act / Assert
    assert any("system role not at position 0" in e for e in validate_record(record))


def test_exact_dedup_removes_same_source_and_first_user():
    # Arrange
    a = _record([{"role": "user", "content": "dup"}, {"role": "assistant", "content": "a1"}])
    b = _record([{"role": "user", "content": "dup"}, {"role": "assistant", "content": "a2"}])
    c = _record([{"role": "user", "content": "unique"}, {"role": "assistant", "content": "a3"}])

    # Act
    kept, removed = exact_dedup([a, b, c])

    # Assert
    assert removed == 1
    assert len(kept) == 2


def test_exact_dedup_keeps_same_text_across_different_sources():
    # Arrange
    a = _record([{"role": "user", "content": "same"}, {"role": "assistant", "content": "x"}], source="attackqa")
    b = _record([{"role": "user", "content": "same"}, {"role": "assistant", "content": "x"}], source="primus/general")

    # Act
    kept, removed = exact_dedup([a, b])

    # Assert
    assert removed == 0
    assert len(kept) == 2


def test_cjk_char_ratio_detects_chinese():
    # Arrange / Act / Assert
    assert cjk_char_ratio("香港旅游") > 0.9
    assert cjk_char_ratio("hello world") == 0.0
    assert cjk_char_ratio("") == 0.0
