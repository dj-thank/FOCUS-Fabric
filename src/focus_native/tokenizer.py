"""Deterministic compatibility tokenizer for newly trained prototype models.

This byte tokenizer does not reproduce the lost symbolic mapping used by the
bundled legacy checkpoints.  It exists to make the package API complete and to
provide a stable mapping for new training runs.
"""
from __future__ import annotations

import re
from typing import Iterable


class HybridTokenizer:
    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"
    UNK = "<unk>"

    def __init__(self, vocab_size: int = 1058) -> None:
        if vocab_size < 260:
            raise ValueError("vocab_size must contain four specials and all bytes")
        specials = [self.PAD, self.BOS, self.EOS, self.UNK]
        bytes_ = [f"<byte:{value:02x}>" for value in range(256)]
        symbols = [
            f"<sym:{index:04d}>"
            for index in range(vocab_size - len(specials) - len(bytes_))
        ]
        self.tokens = specials + bytes_ + symbols
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}
        self.pad_id = self.token_to_id[self.PAD]
        self.bos_id = self.token_to_id[self.BOS]
        self.eos_id = self.token_to_id[self.EOS]
        self.unk_id = self.token_to_id[self.UNK]
        self._special_pattern = re.compile(
            r"<sym:\d{4}>|<pad>|<bos>|<eos>|<unk>"
        )

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        result: list[int] = [self.bos_id] if add_bos else []
        cursor = 0
        for match in self._special_pattern.finditer(text):
            result.extend(4 + byte for byte in text[cursor : match.start()].encode("utf-8"))
            result.append(self.token_to_id.get(match.group(0), self.unk_id))
            cursor = match.end()
        result.extend(4 + byte for byte in text[cursor:].encode("utf-8"))
        if add_eos:
            result.append(self.eos_id)
        return result

    def decode(self, ids: Iterable[int], *, skip_special: bool = False) -> str:
        pieces: list[str] = []
        buffer = bytearray()

        def flush() -> None:
            if buffer:
                pieces.append(buffer.decode("utf-8", errors="replace"))
                buffer.clear()

        for raw in ids:
            index = int(raw)
            if 4 <= index < 260:
                buffer.append(index - 4)
                continue
            flush()
            token = self.tokens[index] if 0 <= index < len(self.tokens) else self.UNK
            if not skip_special or token not in {self.PAD, self.BOS, self.EOS, self.UNK}:
                pieces.append(token)
        flush()
        return "".join(pieces)
