from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from typing import Literal


MetadataValue = str | int | bool
CorrectSource = Literal["near", "far"]


@dataclass(frozen=True)
class SequenceTaskExample:
    task_name: str
    context: str
    query: str
    prompt: str
    answer: str
    answer_span: tuple[int, int]
    distance_tokens: int
    metadata: dict[str, MetadataValue] = field(default_factory=dict)


@dataclass(frozen=True)
class DistanceBucketAccuracy:
    label: str
    min_distance: int
    max_distance: int | None
    count: int
    accuracy: float


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_nonnegative(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _unique_token(
    rng: random.Random,
    used_tokens: set[str],
    prefix: str,
    *,
    width: int = 6,
) -> str:
    alphabet = string.ascii_uppercase + string.digits
    while True:
        suffix = "".join(rng.choice(alphabet) for _ in range(width))
        token = f"{prefix}_{suffix}"
        if token not in used_tokens:
            used_tokens.add(token)
            return token


def _noise_tokens(
    rng: random.Random,
    used_tokens: set[str],
    count: int,
) -> list[str]:
    _validate_nonnegative("count", count)
    return [_unique_token(rng, used_tokens, "NOISE") for _ in range(count)]


def _token_offsets(tokens: list[str]) -> list[int]:
    offsets: list[int] = []
    cursor = 0
    for token in tokens:
        offsets.append(cursor)
        cursor += len(token) + 1
    return offsets


def _build_example(
    *,
    task_name: str,
    context_tokens: list[str],
    query_tokens: list[str],
    answer: str,
    answer_index: int,
    metadata: dict[str, MetadataValue] | None = None,
) -> SequenceTaskExample:
    if answer_index < 0 or answer_index >= len(context_tokens):
        raise ValueError("answer_index must point to a token inside context_tokens")
    if context_tokens[answer_index] != answer:
        raise ValueError("context_tokens[answer_index] must equal answer")

    prompt_tokens = context_tokens + query_tokens
    offsets = _token_offsets(prompt_tokens)
    answer_start = offsets[answer_index]
    answer_span = (answer_start, answer_start + len(answer))
    prompt = " ".join(prompt_tokens)
    query = " ".join(query_tokens)
    context = " ".join(context_tokens)
    distance_tokens = len(context_tokens) - answer_index - 1

    if prompt.count(answer) != 1:
        raise ValueError("generated example must contain exactly one answer-bearing span")

    return SequenceTaskExample(
        task_name=task_name,
        context=context,
        query=query,
        prompt=prompt,
        answer=answer,
        answer_span=answer_span,
        distance_tokens=distance_tokens,
        metadata={} if metadata is None else dict(metadata),
    )


def generate_passkey_retrieval_example(
    *,
    seed: int,
    noise_tokens: int = 64,
) -> SequenceTaskExample:
    _validate_positive("noise_tokens", noise_tokens)
    rng = random.Random(seed)
    used_tokens: set[str] = set()
    answer = _unique_token(rng, used_tokens, "PASS")
    before_count = max(1, (2 * noise_tokens) // 3)
    after_count = noise_tokens - before_count

    context_tokens = (
        _noise_tokens(rng, used_tokens, before_count)
        + ["PASSKEY", answer]
        + _noise_tokens(rng, used_tokens, after_count)
    )
    query_tokens = ["QUESTION", "What", "is", "the", "passkey", "?"]

    return _build_example(
        task_name="passkey_retrieval",
        context_tokens=context_tokens,
        query_tokens=query_tokens,
        answer=answer,
        answer_index=before_count + 1,
        metadata={"noise_tokens": noise_tokens},
    )


def generate_delayed_copy_example(
    *,
    seed: int,
    delay_tokens: int = 64,
) -> SequenceTaskExample:
    _validate_positive("delay_tokens", delay_tokens)
    rng = random.Random(seed)
    used_tokens: set[str] = set()
    answer = _unique_token(rng, used_tokens, "COPY")
    context_tokens = ["MEMORIZE", answer] + _noise_tokens(rng, used_tokens, delay_tokens)
    query_tokens = ["QUESTION", "Repeat", "the", "memorized", "token", "."]

    return _build_example(
        task_name="delayed_copy",
        context_tokens=context_tokens,
        query_tokens=query_tokens,
        answer=answer,
        answer_index=1,
        metadata={"delay_tokens": delay_tokens},
    )


def generate_repeated_key_induction_example(
    *,
    seed: int,
    noise_pairs: int = 8,
    target_repetitions: int = 3,
) -> SequenceTaskExample:
    _validate_positive("noise_pairs", noise_pairs)
    _validate_positive("target_repetitions", target_repetitions)
    rng = random.Random(seed)
    used_tokens: set[str] = set()

    target_key = _unique_token(rng, used_tokens, "KEY")
    answer = _unique_token(rng, used_tokens, "VAL")
    context_tokens = ["PAIR", target_key, answer]
    answer_index = 2

    for _ in range(noise_pairs):
        context_tokens.extend(
            [
                "PAIR",
                _unique_token(rng, used_tokens, "KEY"),
                _unique_token(rng, used_tokens, "VAL"),
            ]
        )

    for _ in range(target_repetitions):
        context_tokens.extend(["CUE", target_key])
    context_tokens.extend(_noise_tokens(rng, used_tokens, noise_pairs))

    query_tokens = [
        "QUESTION",
        "Which",
        "value",
        "follows",
        target_key,
        "?",
    ]

    return _build_example(
        task_name="repeated_key_induction",
        context_tokens=context_tokens,
        query_tokens=query_tokens,
        answer=answer,
        answer_index=answer_index,
        metadata={
            "noise_pairs": noise_pairs,
            "target_repetitions": target_repetitions,
            "target_key": target_key,
        },
    )


def generate_conflicting_evidence_example(
    *,
    seed: int,
    noise_tokens: int = 48,
    correct_source: CorrectSource = "far",
) -> SequenceTaskExample:
    _validate_positive("noise_tokens", noise_tokens)
    if correct_source not in {"near", "far"}:
        raise ValueError("correct_source must be 'near' or 'far'")

    rng = random.Random(seed)
    used_tokens: set[str] = set()
    entity = _unique_token(rng, used_tokens, "ITEM")
    far_answer = _unique_token(rng, used_tokens, "FAR")
    near_answer = _unique_token(rng, used_tokens, "NEAR")
    middle_noise = max(1, noise_tokens // 2)
    tail_noise = noise_tokens - middle_noise

    context_tokens = [
        "ARCHIVE",
        entity,
        far_answer,
    ]
    context_tokens.extend(_noise_tokens(rng, used_tokens, middle_noise))
    near_answer_index = len(context_tokens) + 2
    context_tokens.extend(["RECENT", entity, near_answer])
    context_tokens.extend(_noise_tokens(rng, used_tokens, tail_noise))

    if correct_source == "far":
        answer = far_answer
        answer_index = 2
        distractor_answer = near_answer
    else:
        answer = near_answer
        answer_index = near_answer_index
        distractor_answer = far_answer

    query_tokens = [
        "QUESTION",
        "According",
        "to",
        "the",
        "ARCHIVE" if correct_source == "far" else "RECENT",
        "record",
        ",",
        "what",
        "is",
        entity,
        "?",
    ]

    return _build_example(
        task_name="conflicting_evidence",
        context_tokens=context_tokens,
        query_tokens=query_tokens,
        answer=answer,
        answer_index=answer_index,
        metadata={
            "correct_source": correct_source,
            "entity": entity,
            "distractor_answer": distractor_answer,
            "noise_tokens": noise_tokens,
        },
    )


def distance_conditioned_accuracy(
    examples: list[SequenceTaskExample],
    predictions: list[str],
    *,
    bucket_boundaries: list[int],
) -> list[DistanceBucketAccuracy]:
    if len(examples) != len(predictions):
        raise ValueError("examples and predictions must have the same length")
    if any(boundary < 0 for boundary in bucket_boundaries):
        raise ValueError("bucket_boundaries must be nonnegative")
    if bucket_boundaries != sorted(bucket_boundaries):
        raise ValueError("bucket_boundaries must be sorted in ascending order")
    if len(set(bucket_boundaries)) != len(bucket_boundaries):
        raise ValueError("bucket_boundaries must be unique")

    boundaries = [0, *bucket_boundaries]
    buckets: list[DistanceBucketAccuracy] = []
    for lower, upper in zip(boundaries, [*bucket_boundaries, None]):
        matches = [
            predicted.strip() == example.answer.strip()
            for example, predicted in zip(examples, predictions)
            if example.distance_tokens >= lower
            and (upper is None or example.distance_tokens < upper)
        ]
        label = f"[{lower}, inf)" if upper is None else f"[{lower}, {upper})"
        accuracy = 0.0 if not matches else sum(matches) / len(matches)
        buckets.append(
            DistanceBucketAccuracy(
                label=label,
                min_distance=lower,
                max_distance=upper,
                count=len(matches),
                accuracy=accuracy,
            )
        )
    return buckets
