"""來源排除 / 離題過濾 / assistant-token 統計的單元測試（AAA 結構）。"""

from argparse import Namespace
from collections import Counter

from preprocessing import audit_quality, config
from preprocessing.build_dataset import (
    _build_stats,
    _resolve_excludes,
    _source_excluded,
)
from preprocessing.quality import count_assistant_tokens, count_record_tokens, is_offtopic


def _record(messages, source="attackqa", category=None, tokens=0, assistant_tokens=0):
    return {
        "messages": messages,
        "source": source,
        "category": category,
        "_tokens": tokens,
        "_assistant_tokens": assistant_tokens,
    }


# --- A. _source_excluded（精確或前綴邊界）---

def test_source_excluded_exact_match():
    # Arrange / Act / Assert
    assert _source_excluded("trendyol", {"trendyol"})


def test_source_excluded_no_substring_false_positive():
    # "attack" 不應誤殺 "attackqa"（無 / 或 - 邊界）
    assert not _source_excluded("attackqa", {"attack"})


def test_source_excluded_prefix_matches_both_reasoning_variants():
    # "primus/reasoning" 須同時命中 -o1 與 -deepseek-r1
    assert _source_excluded("primus/reasoning-o1", {"primus/reasoning"})
    assert _source_excluded("primus/reasoning-deepseek-r1", {"primus/reasoning"})


def test_source_excluded_reasoning_prefix_does_not_match_general():
    assert not _source_excluded("primus/general", {"primus/reasoning"})


def test_source_excluded_namespace_prefix_matches_all_primus():
    # "primus" 以 / 邊界命中整個命名空間
    assert _source_excluded("primus/general", {"primus"})
    assert _source_excluded("primus/reasoning-o1", {"primus"})


def test_source_excluded_empty_patterns():
    assert not _source_excluded("trendyol", set())


# --- A. _resolve_excludes（解析規則）---

def _excl_args(exclude_sources=None, no_default_excludes=False):
    return Namespace(
        exclude_sources=exclude_sources, no_default_excludes=no_default_excludes
    )


def test_resolve_excludes_defaults_when_unset():
    assert _resolve_excludes(_excl_args()) == set(config.DEFAULT_EXCLUDED_SOURCES)


def test_resolve_excludes_explicit_list_replaces_defaults():
    assert _resolve_excludes(_excl_args(exclude_sources=["a", "b"])) == {"a", "b"}


def test_resolve_excludes_empty_list_excludes_nothing():
    assert _resolve_excludes(_excl_args(exclude_sources=[])) == set()


def test_resolve_excludes_no_default_flag_empties():
    assert _resolve_excludes(_excl_args(no_default_excludes=True)) == set()


def test_resolve_excludes_explicit_wins_over_no_default_flag():
    args = _excl_args(exclude_sources=["x"], no_default_excludes=True)
    assert _resolve_excludes(args) == {"x"}


# --- B. is_offtopic（system 不納入掃描）---

def test_is_offtopic_travel_with_security_system_is_offtopic():
    # system 含 "cybersecurity" 也不應讓旅遊文判為非離題
    record = _record(
        [
            {"role": "system", "content": config.SECURITY_SYSTEM_PROMPT},
            {"role": "user", "content": "Plan a 3-day trip to Hong Kong with food spots"},
            {"role": "assistant", "content": "Day 1: Victoria Peak, then dim sum ..."},
        ],
        source="primus/general",
    )

    assert is_offtopic(record)


def test_is_offtopic_security_content_is_not_offtopic():
    record = _record(
        [
            {"role": "user", "content": "Explain CVE-2021-44228 (Log4Shell)"},
            {"role": "assistant", "content": "It is a remote code execution vulnerability ..."},
        ]
    )

    assert not is_offtopic(record)


def test_offtopic_filter_scoped_to_primus_general_only():
    # build() 僅對 OFFTOPIC_FILTER_SOURCES 套用離題過濾；fenrir 不在其中
    assert "primus/general" in config.OFFTOPIC_FILTER_SOURCES
    assert "fenrir" not in config.OFFTOPIC_FILTER_SOURCES
    assert "attackqa" not in config.OFFTOPIC_FILTER_SOURCES


# --- C. count_assistant_tokens ---

def test_count_assistant_tokens_heuristic_counts_assistant_only():
    # Arrange
    record = _record(
        [
            {"role": "user", "content": "abcd"},  # 1 tok (4 chars / 4)
            {"role": "assistant", "content": "abcdefgh"},  # 2 tok (8 chars / 4)
        ]
    )

    # Act / Assert
    assert count_assistant_tokens(record) == 2
    assert count_record_tokens(record) == 3  # 對比：含 user 的全量


def test_count_assistant_tokens_with_custom_counter():
    record = _record([{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}])

    total = count_assistant_tokens(record, counter=lambda s: len(s) * 10)

    assert total == 10  # 只算 assistant "y"


# --- C. _build_stats（per-source token 統計）---

def test_build_stats_emits_per_source_token_sums():
    # Arrange
    records = [
        _record(
            [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
            source="fenrir",
            tokens=100,
            assistant_tokens=80,
        ),
        _record(
            [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
            source="attackqa",
            tokens=20,
            assistant_tokens=5,
        ),
    ]
    args = Namespace(max_total_tokens=8192, hf_tokenizer=None, no_system=False)

    # Act
    stats = _build_stats(records, raw_total=10, dropped=Counter(), args=args)

    # Assert
    assert stats["assistant_tokens_per_source"] == {"fenrir": 80, "attackqa": 5}
    assert stats["total_tokens_per_source"] == {"fenrir": 100, "attackqa": 20}
    for src in stats["per_source"]:
        assert stats["assistant_tokens_per_source"][src] <= stats["total_tokens_per_source"][src]


# --- B. regex 同步守門 ---

def test_cyber_hint_regex_in_sync_between_config_and_audit():
    assert config.CYBER_HINT.pattern == audit_quality.CYBER_HINT.pattern
