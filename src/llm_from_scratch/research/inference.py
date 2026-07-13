from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


AttentionVariant = Literal["mha", "mqa", "gqa"]
InferencePhase = Literal["prefill", "decode"]


@dataclass(frozen=True)
class KVCacheConfig:
    num_layers: int
    num_query_heads: int
    head_dim: int
    batch_size: int
    bytes_per_element: int
    attention_variant: AttentionVariant = "mha"
    num_kv_heads: int | None = None


@dataclass(frozen=True)
class InferenceStep:
    phase: InferencePhase
    step_index: int
    context_tokens_before_step: int
    cache_tokens: int
    attention_score_elements: int
    cache_bytes_per_layer: int
    cache_bytes_total: int
    cache_growth_bytes_per_layer: int
    cache_growth_bytes_total: int
    kv_read_bytes_per_layer: int
    kv_read_bytes_total: int


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def resolve_num_kv_heads(
    *,
    num_query_heads: int,
    attention_variant: AttentionVariant,
    num_kv_heads: int | None = None,
) -> int:
    _validate_positive("num_query_heads", num_query_heads)

    if attention_variant == "mha":
        return num_query_heads
    if attention_variant == "mqa":
        return 1
    if attention_variant != "gqa":
        raise ValueError("attention_variant must be one of: mha, mqa, gqa")

    if num_kv_heads is None:
        raise ValueError("num_kv_heads is required for gqa")
    _validate_positive("num_kv_heads", num_kv_heads)
    if num_kv_heads > num_query_heads:
        raise ValueError("num_kv_heads cannot exceed num_query_heads")
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("num_kv_heads must divide num_query_heads for gqa")
    return num_kv_heads


def kv_cache_bytes_per_token(
    *,
    batch_size: int,
    num_kv_heads: int,
    head_dim: int,
    bytes_per_element: int,
) -> int:
    _validate_positive("batch_size", batch_size)
    _validate_positive("num_kv_heads", num_kv_heads)
    _validate_positive("head_dim", head_dim)
    _validate_positive("bytes_per_element", bytes_per_element)
    return 2 * batch_size * num_kv_heads * head_dim * bytes_per_element


def kv_cache_bytes_per_layer(
    *,
    seq_len: int,
    batch_size: int,
    num_kv_heads: int,
    head_dim: int,
    bytes_per_element: int,
) -> int:
    _validate_positive("seq_len", seq_len)
    return seq_len * kv_cache_bytes_per_token(
        batch_size=batch_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        bytes_per_element=bytes_per_element,
    )


def kv_cache_bytes(
    *,
    seq_len: int,
    num_layers: int,
    batch_size: int,
    num_query_heads: int,
    head_dim: int,
    bytes_per_element: int,
    attention_variant: AttentionVariant,
    num_kv_heads: int | None = None,
) -> int:
    _validate_positive("num_layers", num_layers)
    resolved_num_kv_heads = resolve_num_kv_heads(
        num_query_heads=num_query_heads,
        attention_variant=attention_variant,
        num_kv_heads=num_kv_heads,
    )
    return num_layers * kv_cache_bytes_per_layer(
        seq_len=seq_len,
        batch_size=batch_size,
        num_kv_heads=resolved_num_kv_heads,
        head_dim=head_dim,
        bytes_per_element=bytes_per_element,
    )


def simulate_autoregressive_inference(
    *,
    config: KVCacheConfig,
    prompt_len: int,
    generated_tokens: int,
) -> list[InferenceStep]:
    _validate_positive("prompt_len", prompt_len)
    if generated_tokens < 0:
        raise ValueError("generated_tokens must be nonnegative")

    num_kv_heads = resolve_num_kv_heads(
        num_query_heads=config.num_query_heads,
        attention_variant=config.attention_variant,
        num_kv_heads=config.num_kv_heads,
    )
    cache_growth_bytes_per_layer = kv_cache_bytes_per_token(
        batch_size=config.batch_size,
        num_kv_heads=num_kv_heads,
        head_dim=config.head_dim,
        bytes_per_element=config.bytes_per_element,
    )

    steps = [
        InferenceStep(
            phase="prefill",
            step_index=0,
            context_tokens_before_step=0,
            cache_tokens=prompt_len,
            attention_score_elements=config.batch_size
            * config.num_query_heads
            * prompt_len
            * prompt_len,
            cache_bytes_per_layer=kv_cache_bytes_per_layer(
                seq_len=prompt_len,
                batch_size=config.batch_size,
                num_kv_heads=num_kv_heads,
                head_dim=config.head_dim,
                bytes_per_element=config.bytes_per_element,
            ),
            cache_bytes_total=kv_cache_bytes(
                seq_len=prompt_len,
                num_layers=config.num_layers,
                batch_size=config.batch_size,
                num_query_heads=config.num_query_heads,
                head_dim=config.head_dim,
                bytes_per_element=config.bytes_per_element,
                attention_variant=config.attention_variant,
                num_kv_heads=config.num_kv_heads,
            ),
            cache_growth_bytes_per_layer=prompt_len * cache_growth_bytes_per_layer,
            cache_growth_bytes_total=prompt_len * cache_growth_bytes_per_layer * config.num_layers,
            kv_read_bytes_per_layer=0,
            kv_read_bytes_total=0,
        )
    ]

    for step_index in range(1, generated_tokens + 1):
        context_tokens = prompt_len + step_index - 1
        cache_bytes_before_per_layer = kv_cache_bytes_per_layer(
            seq_len=context_tokens,
            batch_size=config.batch_size,
            num_kv_heads=num_kv_heads,
            head_dim=config.head_dim,
            bytes_per_element=config.bytes_per_element,
        )
        cache_tokens = context_tokens + 1
        cache_bytes_after_per_layer = kv_cache_bytes_per_layer(
            seq_len=cache_tokens,
            batch_size=config.batch_size,
            num_kv_heads=num_kv_heads,
            head_dim=config.head_dim,
            bytes_per_element=config.bytes_per_element,
        )
        steps.append(
            InferenceStep(
                phase="decode",
                step_index=step_index,
                context_tokens_before_step=context_tokens,
                cache_tokens=cache_tokens,
                attention_score_elements=config.batch_size
                * config.num_query_heads
                * context_tokens,
                cache_bytes_per_layer=cache_bytes_after_per_layer,
                cache_bytes_total=config.num_layers * cache_bytes_after_per_layer,
                cache_growth_bytes_per_layer=cache_growth_bytes_per_layer,
                cache_growth_bytes_total=config.num_layers * cache_growth_bytes_per_layer,
                kv_read_bytes_per_layer=cache_bytes_before_per_layer,
                kv_read_bytes_total=config.num_layers * cache_bytes_before_per_layer,
            )
        )
    return steps
