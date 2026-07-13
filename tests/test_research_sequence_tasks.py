from __future__ import annotations

import pytest

from llm_from_scratch.research.sequence_tasks import (
    SequenceTaskExample,
    distance_conditioned_accuracy,
    generate_conflicting_evidence_example,
    generate_delayed_copy_example,
    generate_passkey_retrieval_example,
    generate_repeated_key_induction_example,
)


def assert_unique_answer_span(example: SequenceTaskExample) -> None:
    start, end = example.answer_span
    assert example.prompt[start:end] == example.answer
    assert example.prompt.count(example.answer) == 1
    assert example.distance_tokens > 0


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [
        (generate_passkey_retrieval_example, {"noise_tokens": 48}),
        (generate_delayed_copy_example, {"delay_tokens": 40}),
        (generate_repeated_key_induction_example, {"noise_pairs": 6, "target_repetitions": 3}),
        (generate_conflicting_evidence_example, {"noise_tokens": 36, "correct_source": "far"}),
    ],
)
def test_generators_emit_exactly_one_answer_bearing_span(
    factory,
    kwargs: dict[str, int | str],
) -> None:
    example = factory(seed=7, **kwargs)

    assert_unique_answer_span(example)


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [
        (generate_passkey_retrieval_example, {"noise_tokens": 32}),
        (generate_delayed_copy_example, {"delay_tokens": 28}),
        (generate_repeated_key_induction_example, {"noise_pairs": 5, "target_repetitions": 2}),
        (generate_conflicting_evidence_example, {"noise_tokens": 24, "correct_source": "near"}),
    ],
)
def test_generators_are_deterministic_for_seed(
    factory,
    kwargs: dict[str, int | str],
) -> None:
    first = factory(seed=11, **kwargs)
    second = factory(seed=11, **kwargs)
    third = factory(seed=12, **kwargs)

    assert first == second
    assert first != third


def test_repeated_key_induction_repeats_the_target_key_without_repeating_the_answer() -> None:
    example = generate_repeated_key_induction_example(
        seed=5,
        noise_pairs=4,
        target_repetitions=4,
    )

    target_key = str(example.metadata["target_key"])
    assert example.prompt.count(target_key) >= 3
    assert_unique_answer_span(example)


def test_conflicting_evidence_tracks_the_distractor_and_requested_source() -> None:
    example = generate_conflicting_evidence_example(
        seed=3,
        noise_tokens=20,
        correct_source="far",
    )

    assert example.metadata["correct_source"] == "far"
    assert example.metadata["distractor_answer"] != example.answer
    assert example.prompt.count(str(example.metadata["distractor_answer"])) == 1
    assert_unique_answer_span(example)


def test_distance_conditioned_accuracy_groups_exact_match_by_distance_bucket() -> None:
    examples = [
        SequenceTaskExample(
            task_name="passkey",
            context="ctx",
            query="q1",
            prompt="ctx q1",
            answer="A1",
            answer_span=(0, 2),
            distance_tokens=12,
        ),
        SequenceTaskExample(
            task_name="passkey",
            context="ctx",
            query="q2",
            prompt="ctx q2",
            answer="A2",
            answer_span=(0, 2),
            distance_tokens=28,
        ),
        SequenceTaskExample(
            task_name="passkey",
            context="ctx",
            query="q3",
            prompt="ctx q3",
            answer="A3",
            answer_span=(0, 2),
            distance_tokens=40,
        ),
    ]

    summary = distance_conditioned_accuracy(
        examples,
        predictions=["A1", "wrong", "A3"],
        bucket_boundaries=[16, 32],
    )

    assert [bucket.label for bucket in summary] == ["[0, 16)", "[16, 32)", "[32, inf)"]
    assert [bucket.count for bucket in summary] == [1, 1, 1]
    assert [bucket.accuracy for bucket in summary] == [1.0, 0.0, 1.0]
