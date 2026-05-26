"""converters 的單元測試（AAA 結構）。"""

from preprocessing.converters import (
    attackqa_to_record,
    primus_reasoning_to_record,
    primus_to_record,
    triplet_to_record,
)

REASON_OPEN = "<|reserved_special_token_0|>"
REASON_CLOSE = "<|reserved_special_token_1|>"


def test_attackqa_assembles_rag_style_user_turn():
    # Arrange
    row = {
        "question": "What detects T1539?",
        "answer": "Monitor browser memory access.",
        "document": "Process Access can detect T1539.",
        "source": "relationships_detects",
        "_idx": 7,
    }

    # Act
    record = attackqa_to_record(row)

    # Assert
    roles = [m["role"] for m in record["messages"]]
    assert roles == ["system", "user", "assistant"]
    user_content = record["messages"][1]["content"]
    assert "Context:\nProcess Access can detect T1539." in user_content
    assert "Question: What detects T1539?" in user_content
    assert record["messages"][2]["content"] == "Monitor browser memory access."
    assert record["source"] == "attackqa"
    assert record["category"] == "relationships_detects"
    assert record["id"] == "attackqa-7"


def test_attackqa_without_system_prompt_omits_system_turn():
    # Arrange
    row = {"question": "q", "answer": "a", "document": "d", "source": "techniques", "_idx": 0}

    # Act
    record = attackqa_to_record(row, system_prompt=None)

    # Assert
    assert record["messages"][0]["role"] == "user"
    assert all(m["role"] != "system" for m in record["messages"])


def test_attackqa_does_not_use_thought_field():
    # Arrange — thought present but must be ignored in RAG-style assembly
    row = {
        "question": "q",
        "answer": "a",
        "document": "d",
        "source": "techniques",
        "thought": "SECRET-REASONING",
        "_idx": 1,
    }

    # Act
    record = attackqa_to_record(row)

    # Assert
    assert all("SECRET-REASONING" not in m["content"] for m in record["messages"])


def test_primus_preserves_multiturn_and_prepends_system():
    # Arrange
    row = {
        "prompt_id": "abc-123",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "again"},
            {"role": "assistant", "content": "sure"},
        ],
        "_idx": 0,
    }

    # Act
    record = primus_to_record("general", row)

    # Assert
    roles = [m["role"] for m in record["messages"]]
    assert roles == ["system", "user", "assistant", "user", "assistant"]
    assert record["source"] == "primus/general"
    assert record["category"] == "general"
    assert record["id"] == "abc-123"


def test_primus_generates_id_when_prompt_id_missing():
    # Arrange
    row = {
        "prompt_id": None,
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ],
        "_idx": 5,
    }

    # Act
    record = primus_to_record("cmd_analysis", row)

    # Assert
    assert record["id"] == "primus-cmd_analysis-5"


def test_triplet_preserves_native_system_prompt():
    # Arrange — Trendyol / Fenrir 自帶 system，需保留
    row = {"system": "You are a defender.", "user": "U", "assistant": "A", "_idx": 3}

    # Act
    record = triplet_to_record(row, "trendyol")

    # Assert
    roles = [m["role"] for m in record["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert record["messages"][0]["content"] == "You are a defender."
    assert record["messages"][1]["content"] == "U"
    assert record["messages"][2]["content"] == "A"
    assert record["source"] == "trendyol"
    assert record["category"] is None
    assert record["id"] == "trendyol-3"


def test_triplet_no_system_flag_omits_system_turn():
    # Arrange
    row = {"system": "SYS", "user": "U", "assistant": "A", "_idx": 0}

    # Act
    record = triplet_to_record(row, "fenrir", no_system=True)

    # Assert
    assert record["messages"][0]["role"] == "user"
    assert all(m["role"] != "system" for m in record["messages"])


def test_triplet_empty_system_omits_system_turn():
    # Arrange — 空白 system 不應注入
    row = {"system": "   ", "user": "U", "assistant": "A", "_idx": 1}

    # Act
    record = triplet_to_record(row, "fenrir")

    # Assert
    assert record["messages"][0]["role"] == "user"


def test_primus_reasoning_converts_special_tokens_to_think():
    # Arrange — Llama 風格特殊 token 包住推理
    row = {
        "messages": [
            {"role": "user", "content": "Map this CVE to a CWE."},
            {
                "role": "assistant",
                "content": f"{REASON_OPEN}First inspect the root cause.{REASON_CLOSE}CWE-79 (XSS).",
            },
        ],
        "variant": "o1",
        "_idx": 4,
    }

    # Act
    record = primus_reasoning_to_record(row)

    # Assert — 轉成 Qwen 原生 <think>
    assistant = record["messages"][-1]["content"]
    assert assistant == "<think>\nFirst inspect the root cause.\n</think>\n\nCWE-79 (XSS)."
    assert REASON_OPEN not in assistant and REASON_CLOSE not in assistant
    assert record["source"] == "primus/reasoning-o1"
    assert record["category"] == "reasoning"
    assert record["messages"][0]["role"] == "system"  # 注入 reasoning system


def test_primus_reasoning_without_delimiters_keeps_content():
    # Arrange — 無特殊 token 時原樣保留（fallback）
    row = {
        "messages": [
            {"role": "user", "content": "Q?"},
            {"role": "assistant", "content": "Plain answer without reasoning tokens."},
        ],
        "variant": "deepseek-r1",
        "_idx": 0,
    }

    # Act
    record = primus_reasoning_to_record(row)

    # Assert
    assert record["messages"][-1]["content"] == "Plain answer without reasoning tokens."
    assert record["source"] == "primus/reasoning-deepseek-r1"


def test_primus_reasoning_no_system_when_prompt_none():
    # Arrange
    row = {
        "messages": [
            {"role": "user", "content": "Q?"},
            {"role": "assistant", "content": "A."},
        ],
        "variant": "o1",
        "_idx": 0,
    }

    # Act
    record = primus_reasoning_to_record(row, system_prompt=None)

    # Assert
    assert all(m["role"] != "system" for m in record["messages"])
    assert record["messages"][0]["role"] == "user"
