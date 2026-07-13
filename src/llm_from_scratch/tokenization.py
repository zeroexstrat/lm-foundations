from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from tokenizers import Tokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers


@dataclass(frozen=True)
class CharTokenizer:
    stoi: dict[str, int]
    itos: dict[int, str]

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        stoi = {ch: idx for idx, ch in enumerate(chars)}
        itos = {idx: ch for ch, idx in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for ch in text:
            try:
                ids.append(self.stoi[ch])
            except KeyError as exc:
                raise KeyError(f"Unknown character: {ch!r}") from exc
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[idx] for idx in ids)


def train_bpe_tokenizer(texts: Iterable[str], vocab_size: int = 256) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<unk>"])
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer
