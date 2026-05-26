"""converters 的單元測試（AAA 結構）。"""

from preprocessing.converters import attackqa_to_record, primus_to_record


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
