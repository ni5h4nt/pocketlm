"""Char-level tokenizer. Index 0 is a reserved <unk>; round-trip is lossless
for in-vocab text and lossy (-> the sentinel) for out-of-vocab characters."""
from __future__ import annotations

UNK = "�"

class CharTokenizer:
    def __init__(self, chars: list[str]) -> None:
        self.itos = [UNK, *chars]
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.unk_id = 0

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        # Exclude the sentinel so it can never collide with a real corpus char;
        # if the sentinel appears in the text it is treated as OOV (encodes to <unk>).
        return cls(sorted(set(text) - {UNK}))

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, s: str) -> list[int]:
        return [self.stoi.get(c, self.unk_id) for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)
