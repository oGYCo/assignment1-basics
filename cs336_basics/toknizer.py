import json
import os
from pathlib import Path
from typing import Iterable, Iterator

import regex as re

from .train_bpe import PAT

class BPETokenizer():
    def __init__(self, vocab, merges, special_tokens=None):
        """
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
        special_tokens: list[str] | None = None
        """
        self.vocab = vocab
        self.merges = merges
        self.bytes_to_id = {token_bytes: token_id for token_id, token_bytes in vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.special_tokens = special_tokens if special_tokens is not None else []
        self._pretoken_re = re.compile(PAT)
        self._special_tokens_set = frozenset(self.special_tokens)
        if self.special_tokens:
            special_tokens_pattern = "|".join(
                re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)
            )
            self._special_split_re = re.compile(f"({special_tokens_pattern})")
        else:
            self._special_split_re = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike[str],
        merges_filepath: str | os.PathLike[str],
        special_tokens: list[str] | None = None,
    ):
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)
        with open(merges_filepath, "r", encoding="utf-8") as f:
            raw_merges = json.load(f)
        
        vocab = {int(token_id): token_str.encode("utf-8") for token_id, token_str in raw_vocab.items()}
        merges = [(left.encode("utf-8"), right.encode("utf-8")) for left, right in raw_merges]

        if special_tokens:
            existing = set(vocab.values())
            for token in special_tokens:
                token_bytes = token.encode("utf-8")
                if token_bytes not in existing:
                    vocab[len(vocab)] = token_bytes
                    existing.add(token_bytes)
        
        return cls(vocab, merges, special_tokens)

    def _pre_tokenize(self, text: str) -> list[bytes]:
        pretokens: list[bytes] = []

        if self._special_split_re is None:
            segments = [text]
        else:
            segments = self._special_split_re.split(text)

        for piece in segments:
            if not piece:
                continue

            if piece in self._special_tokens_set:
                pretokens.append(piece.encode("utf-8"))
                continue

            for match in self._pretoken_re.finditer(piece):
                pretokens.append(match.group(0).encode("utf-8"))

        return pretokens
    
    def _encode_pre_token(self, pretoken_bytes: bytes) -> list[int]:
        if pretoken_bytes in self.vocab.values():
            return [self.bytes_to_id[pretoken_bytes]]

        current_tokens = [pretoken_bytes[i:i+1] for i in range(len(pretoken_bytes))]

        while len(current_tokens) > 1:
            best_index = None
            best_rank = None

            for i in range(len(current_tokens) - 1):
                pair = (current_tokens[i], current_tokens[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_index = i

            if best_index is None:
                break

            merged_token = current_tokens[best_index] + current_tokens[best_index + 1]
            current_tokens[best_index : best_index + 2] = [merged_token]

        return [self.bytes_to_id[token] for token in current_tokens]

    def encode(self, text: str) -> list[int]:
        result_ids: list[int] = []
        seq = self._pre_tokenize(text)

        for pretoken_bytes in seq:
            result_ids.extend(self._encode_pre_token(pretoken_bytes))

        return result_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        tokens = [self.vocab[token_id] for token_id in ids]
        return b"".join(tokens).decode("utf-8", errors="replace")


def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    vocab_filepath = data_dir / "owt_bpe_vocab.json"
    merges_filepath = data_dir / "owt_bpe_merges.json"
    special_tokens = ["<|endoftext|>"]
    tokenizer = BPETokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)
    text = "Hello, world!  How are you?, I am fine., thanks!, <|endoftext|> How are you doing today? I hope you are doing well. I am doing great, thank you for asking! <|endoftext|>"
    token_ids = tokenizer.encode(text)
    print(token_ids)
    string = tokenizer.decode(token_ids)
    print(string)


if __name__ == "__main__":
    main()
