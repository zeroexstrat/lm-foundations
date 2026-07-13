from llm_from_scratch.tokenization import CharTokenizer


def test_char_tokenizer_round_trip() -> None:
    tokenizer = CharTokenizer.from_text("banana")
    ids = tokenizer.encode("banana")
    assert tokenizer.decode(ids) == "banana"
    assert tokenizer.vocab_size == 3


def test_char_tokenizer_rejects_unknown_character() -> None:
    tokenizer = CharTokenizer.from_text("abc")
    try:
        tokenizer.encode("abd")
    except KeyError as exc:
        assert "Unknown character" in str(exc)
    else:
        raise AssertionError("encoding an unknown character should fail")


def test_train_bpe_tokenizer_encodes_and_decodes_text() -> None:
    from llm_from_scratch.tokenization import train_bpe_tokenizer

    tokenizer = train_bpe_tokenizer(["hello world", "hello token"], vocab_size=64)
    encoded = tokenizer.encode("hello world")
    decoded = tokenizer.decode(encoded.ids)
    assert encoded.ids
    assert "hello" in decoded
