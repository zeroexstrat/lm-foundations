import torch

from llm_from_scratch.data import get_batch, split_tokens, toy_instruction_examples


def test_split_tokens_preserves_total_length() -> None:
    tokens = torch.arange(100)
    train, val = split_tokens(tokens, train_fraction=0.8)
    assert len(train) == 80
    assert len(val) == 20


def test_get_batch_returns_shifted_inputs_and_targets() -> None:
    tokens = torch.tensor([3, 10, 4, 11, 5, 12, 6, 13, 7, 14, 8, 15])
    x, y = get_batch(tokens, block_size=4, batch_size=4, device=torch.device("cpu"))
    assert x.shape == (4, 4)
    assert y.shape == (4, 4)
    for row_x, row_y in zip(x, y, strict=True):
        matches = [
            start
            for start in range(len(tokens) - 4)
            if torch.equal(row_x.cpu(), tokens[start : start + 4])
        ]
        assert len(matches) == 1
        start = matches[0]
        assert torch.equal(row_y.cpu(), tokens[start + 1 : start + 5])


def test_toy_instruction_examples_have_prompt_and_response() -> None:
    examples = toy_instruction_examples()
    assert examples
    prompt, response = examples[0]
    assert isinstance(prompt, str)
    assert isinstance(response, str)
    assert prompt
    assert response
