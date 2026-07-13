from __future__ import annotations

import pytest

from llm_from_scratch.research.inference import (
    KVCacheConfig,
    kv_cache_bytes,
    kv_cache_bytes_per_layer,
    kv_cache_bytes_per_token,
    resolve_num_kv_heads,
    simulate_autoregressive_inference,
)


def test_resolve_num_kv_heads_matches_attention_variant() -> None:
    assert resolve_num_kv_heads(num_query_heads=8, attention_variant="mha") == 8
    assert resolve_num_kv_heads(num_query_heads=8, attention_variant="mqa") == 1
    assert resolve_num_kv_heads(
        num_query_heads=8,
        attention_variant="gqa",
        num_kv_heads=2,
    ) == 2


@pytest.mark.parametrize("bytes_per_element", [4, 2, 1])
def test_kv_cache_byte_formulas_cover_fp32_fp16_and_int8_like_storage(bytes_per_element: int) -> None:
    assert (
        kv_cache_bytes_per_token(
            batch_size=2,
            num_kv_heads=4,
            head_dim=8,
            bytes_per_element=bytes_per_element,
        )
        == 2 * 2 * 4 * 8 * bytes_per_element
    )
    assert (
        kv_cache_bytes_per_layer(
            seq_len=16,
            batch_size=2,
            num_kv_heads=4,
            head_dim=8,
            bytes_per_element=bytes_per_element,
        )
        == 16 * 2 * 2 * 4 * 8 * bytes_per_element
    )
    assert (
        kv_cache_bytes(
            seq_len=16,
            num_layers=3,
            batch_size=2,
            num_query_heads=8,
            head_dim=8,
            bytes_per_element=bytes_per_element,
            attention_variant="gqa",
            num_kv_heads=4,
        )
        == 3 * 16 * 2 * 2 * 4 * 8 * bytes_per_element
    )


def test_attention_variant_changes_kv_cache_size() -> None:
    common = dict(
        seq_len=32,
        num_layers=2,
        batch_size=1,
        num_query_heads=8,
        head_dim=16,
        bytes_per_element=2,
    )

    mha = kv_cache_bytes(attention_variant="mha", **common)
    mqa = kv_cache_bytes(attention_variant="mqa", **common)
    gqa = kv_cache_bytes(attention_variant="gqa", num_kv_heads=2, **common)

    assert mha == 2 * 32 * 2 * 1 * 8 * 16 * 2
    assert mqa == 2 * 32 * 2 * 1 * 1 * 16 * 2
    assert gqa == 2 * 32 * 2 * 1 * 2 * 16 * 2
    assert mha > gqa > mqa


def test_simulator_separates_prefill_from_decode() -> None:
    config = KVCacheConfig(
        num_layers=2,
        num_query_heads=6,
        num_kv_heads=2,
        head_dim=8,
        batch_size=1,
        bytes_per_element=2,
        attention_variant="gqa",
    )

    steps = simulate_autoregressive_inference(
        config=config,
        prompt_len=4,
        generated_tokens=3,
    )

    assert [step.phase for step in steps] == ["prefill", "decode", "decode", "decode"]

    prefill = steps[0]
    assert prefill.cache_tokens == 4
    assert prefill.attention_score_elements == 1 * 6 * 4 * 4
    assert prefill.kv_read_bytes_total == 0
    assert prefill.cache_bytes_total == kv_cache_bytes(
        seq_len=4,
        num_layers=2,
        batch_size=1,
        num_query_heads=6,
        head_dim=8,
        bytes_per_element=2,
        attention_variant="gqa",
        num_kv_heads=2,
    )

    first_decode = steps[1]
    assert first_decode.context_tokens_before_step == 4
    assert first_decode.cache_tokens == 5
    assert first_decode.attention_score_elements == 1 * 6 * 4
    assert first_decode.kv_read_bytes_total == kv_cache_bytes(
        seq_len=4,
        num_layers=2,
        batch_size=1,
        num_query_heads=6,
        head_dim=8,
        bytes_per_element=2,
        attention_variant="gqa",
        num_kv_heads=2,
    )
    assert first_decode.cache_growth_bytes_total == 2 * 2 * 1 * 2 * 8 * 2
    assert first_decode.cache_bytes_per_layer == kv_cache_bytes_per_layer(
        seq_len=5,
        batch_size=1,
        num_kv_heads=2,
        head_dim=8,
        bytes_per_element=2,
    )


def test_gqa_validation_rejects_missing_or_incompatible_kv_heads() -> None:
    with pytest.raises(ValueError, match="num_kv_heads"):
        resolve_num_kv_heads(num_query_heads=8, attention_variant="gqa")

    with pytest.raises(ValueError, match="divide"):
        resolve_num_kv_heads(
            num_query_heads=8,
            attention_variant="gqa",
            num_kv_heads=3,
        )
