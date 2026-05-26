"""把原始 record 轉成統一的 chat messages 訓練 record。

每個輸出 record 形如：
    {"messages": [{"role","content"}...], "source": str, "category": str, "id": str}

純函式、無 I/O，便於測試。
"""

from __future__ import annotations

from typing import Optional

from . import config


def _with_system(system_prompt: Optional[str], turns: list[dict]) -> list[dict]:
    """需要時在最前面注入 system turn。"""
    if system_prompt:
        return [{"role": "system", "content": system_prompt}, *turns]
    return list(turns)


def attackqa_to_record(
    row: dict,
    system_prompt: Optional[str] = config.ATTACKQA_SYSTEM_PROMPT,
) -> dict:
    """AttackQA → RAG 式 record：document 當 context 放進 user turn，answer 為 assistant。

    不使用 thought 欄位（依使用者選定的 RAG 式組裝）。
    """
    document = str(row.get("document", "")).strip()
    question = str(row.get("question", "")).strip()
    answer = str(row.get("answer", "")).strip()

    user_content = f"Context:\n{document}\n\nQuestion: {question}"
    turns = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": answer},
    ]
    category = str(row.get("source", "")).strip() or None
    return {
        "messages": _with_system(system_prompt, turns),
        "source": "attackqa",
        "category": category,
        "id": f"attackqa-{row.get('_idx', 0)}",
    }


def primus_to_record(
    scenario: str,
    row: dict,
    system_prompt: Optional[str] = config.SECURITY_SYSTEM_PROMPT,
) -> dict:
    """Primus → record：沿用既有 messages（含 general 多輪），前面注入 system prompt。"""
    turns = [
        {"role": str(t["role"]), "content": str(t["content"])}
        for t in row.get("messages", [])
    ]
    prompt_id = row.get("prompt_id")
    record_id = str(prompt_id) if prompt_id else f"primus-{scenario}-{row.get('_idx', 0)}"
    return {
        "messages": _with_system(system_prompt, turns),
        "source": f"primus/{scenario}",
        "category": scenario,
        "id": record_id,
    }
