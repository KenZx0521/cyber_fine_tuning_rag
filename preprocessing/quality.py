"""品質驗證、長度估算與去重。

設計為輕量、無外部相依（不需 transformers/sklearn/datasketch）。
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional

from . import config

_VALID_ROLES = set(config.VALID_ROLES)


def estimate_tokens(text: str, chars_per_token: int = config.CHARS_PER_TOKEN) -> int:
    """以字元數/N 粗估 token 數（向上取整）。"""
    length = len(text)
    return (length + chars_per_token - 1) // chars_per_token


def count_record_tokens(
    record: dict,
    counter: Optional[Callable[[str], int]] = None,
) -> int:
    """估算整筆對話的 token 數；counter 為 None 時用 heuristic。"""
    messages = record.get("messages", [])
    if counter is None:
        return sum(estimate_tokens(m["content"]) for m in messages)
    return sum(counter(m["content"]) for m in messages)


def validate_record(record: dict) -> list[str]:
    """檢查單筆 record 結構；回傳錯誤原因清單（空清單代表通過）。"""
    errors: list[str] = []
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        return ["missing or empty messages"]

    non_system_roles: list[str] = []
    for i, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content")
        if role not in _VALID_ROLES:
            errors.append(f"invalid role at {i}: {role!r}")
        if not isinstance(content, str) or not content.strip():
            errors.append(f"empty content at {i}")
        if role == "system" and i != 0:
            errors.append(f"system role not at position 0 (index {i})")
        if role != "system":
            non_system_roles.append(role)

    if not non_system_roles:
        errors.append("no user/assistant turns")
        return errors
    if non_system_roles[0] != "user":
        errors.append("first non-system turn is not 'user'")
    if non_system_roles[-1] != "assistant":
        errors.append("conversation does not end with 'assistant'")
    if "assistant" not in non_system_roles:
        errors.append("missing assistant turn")
    for prev, curr in zip(non_system_roles, non_system_roles[1:]):
        if prev == curr:
            errors.append("consecutive same-role turns (not alternating)")
            break

    return errors


def is_valid(record: dict) -> bool:
    return not validate_record(record)


def exact_dedup(records: Iterable[dict]) -> tuple[list[dict], int]:
    """依 (source, 首個 user content) 做精確去重，回傳 (保留清單, 移除數)。"""
    seen: set[tuple] = set()
    kept: list[dict] = []
    removed = 0
    for record in records:
        messages = record.get("messages", [])
        first_user = next(
            (m["content"] for m in messages if m.get("role") == "user"), ""
        )
        key = (record.get("source"), first_user)
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        kept.append(record)
    return kept, removed


def cjk_char_ratio(text: str) -> float:
    """CJK 漢字字元佔比，用於語言分佈統計（非過濾）。"""
    if not text:
        return 0.0
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    return cjk / len(text)
